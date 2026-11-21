"""
Costureiro — núcleo.

Monta uma IA nova a partir de PARTES (faixas de camadas) de IAs diferentes, ligadas por
"super agentes tradutores" que aprendem a mudança de base entre os espaços internos de cada uma.

    texto ──► [modelo 1: camadas 0..c1] ──T1──► [modelo 2: camadas c2..fim] ──► próxima palavra
                (lê com o próprio tokenizer)        (fala com o próprio vocabulário)

Mais de duas partes também funcionam:  modelo 1 ──T1──► modelo 2 (meio) ──T2──► modelo 3 (fim).

Ideias principais
- Grade de tokens: a sequência é a do tokenizer da ÚLTIMA parte (é ela que prevê a próxima palavra).
  A primeira parte lê o texto com o próprio tokenizer; seus vetores são levados para a grade
  alinhando pelos caracteres (para cada token da grade, o último token da 1ª parte que termina antes
  ou junto -> nada do futuro vaza).
- Partes do meio e do fim rodam sobre a grade: injeta-se o vetor traduzido na camada de início
  (gancho antes da camada) e lê-se a saída da camada de fim (gancho depois da camada).
- Tradutor = mudança de base: começa como rotação de Procrustes (SVD) e depois aprende por gradiente
  (linear + rede pequena residual), com os modelos congelados, minimizando o erro da IA nova
  no texto do foco. Para pelo melhor resultado no texto de teste.
- Comparação justa entre tokenizers diferentes: bits por caractere (BPC), por assunto.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32


# ============================================================================ receita

@dataclass
class Parte:
    modelo: str          # id do Hugging Face ou pasta local
    inicio: int          # primeira camada usada (0 = desde o começo, com o embedding do próprio modelo)
    fim: int             # última camada usada (inclusiva). Na última parte tem que ser a última camada.


@dataclass
class Receita:
    nome: str
    partes: list[Parte]
    assuntos: dict[str, list[str]] = field(default_factory=dict)   # assunto -> arquivos de texto do foco
    max_tok: int = 192
    epocas: int = 6
    lr: float = 3e-4
    oculto_mlp: int = 1024
    max_trechos: int = 500          # por assunto (limita o tempo de treino)
    criado: str = ""
    resultados: dict = field(default_factory=dict)

    def salvar(self, pasta: Path):
        pasta.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        (pasta / "receita.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def abrir(pasta: Path) -> "Receita":
        d = json.loads((Path(pasta) / "receita.json").read_text(encoding="utf-8"))
        d["partes"] = [Parte(**p) for p in d["partes"]]
        return Receita(**d)


# ============================================================================ modelos

_cache: dict[str, tuple] = {}


def carregar_modelo(nome: str, dispositivo: str = DEV):
    """Carrega (tokenizer, modelo) congelado; guarda em cache para não carregar duas vezes."""
    if nome in _cache:
        return _cache[nome]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nome)
    if not getattr(tok, "is_fast", False):
        raise ValueError(f"{nome}: o tokenizer precisa ser 'fast' (com offsets de caracteres).")
    m = AutoModelForCausalLM.from_pretrained(nome, dtype=DT if dispositivo == "cuda" else torch.float32).to(dispositivo).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    _cache[nome] = (tok, m)
    return tok, m


def liberar_cache():
    _cache.clear()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def camadas(m):
    for caminho in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = m
        try:
            for p in caminho.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise ValueError("arquitetura não suportada: não achei a lista de camadas")


def info_modelo(nome: str) -> dict:
    """Só lê a configuração (rápido, sem baixar os pesos)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(nome)
    n = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", None)
    d = getattr(c, "hidden_size", None) or getattr(c, "n_embd", None)
    return {"camadas": n, "dimensao": d, "arquitetura": (c.architectures or ["?"])[0]}


# ============================================================================ texto

def ler_textos(arquivos: list[str]) -> str:
    partes = []
    for a in arquivos:
        p = Path(a)
        lista = sorted(p.rglob("*")) if p.is_dir() else [p]
        for f in lista:
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            try:
                if suf == ".pdf":
                    from pypdf import PdfReader
                    partes.append("\n".join((pg.extract_text() or "") for pg in PdfReader(str(f)).pages))
                elif suf in (".txt", ".md", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".java", ".tex", ".rs", ".go",
                             ".lua", ".json", ".csv", ".html", ".css", ".kt", ".cs", ".m", ".r", ".sql"):
                    partes.append(f.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:  # arquivo ruim não derruba o treino
                print(f"  (pulei {f.name}: {e})")
    return "\n\n".join(partes)


def fatiar(texto: str, rng: random.Random, min_chars=300, max_chars=900) -> list[str]:
    """Pedaços de 300-900 caracteres, quebrando em fim de linha quando possível (bom para código)."""
    out, i = [], 0
    while i < len(texto):
        tam = rng.randint(min_chars, max_chars)
        j = min(len(texto), i + tam)
        quebra = texto.rfind("\n", i + min_chars // 2, j)
        if quebra > i and j < len(texto):
            j = quebra + 1
        trecho = texto[i:j].strip()
        if len(trecho) >= min_chars // 2:
            out.append(trecho)
        i = j
    return out


# ============================================================================ alinhamento e execução

def alinhar(offs_src, offs_grade):
    fins = [e for _, e in offs_src]
    idx, j = [], -1
    for _, fim in offs_grade:
        while j + 1 < len(fins) and fins[j + 1] <= fim:
            j += 1
        idx.append(max(j, 0))   # o 1º token da grade, se não tiver par, usa o 1º da origem
    return idx


class _Parar(Exception):
    def __init__(self, h):
        self.h = h


def _saida_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def rodar_parte(m, ids, inicio: int, fim: int, entrada: torch.Tensor | None):
    """Roda as camadas [inicio..fim] do modelo m sobre uma sequência de len(ids) posições.
    entrada=None => usa o próprio embedding (só quando inicio == 0).
    Retorna o vetor na SAÍDA da camada fim (antes da norma final)."""
    L = camadas(m)
    ganchos = []
    if entrada is not None:
        def pre(mod, a, kw):
            hs = kw.get("hidden_states", a[0] if a else None)
            novo = entrada.to(hs.dtype).unsqueeze(0)
            if "hidden_states" in kw:
                kw["hidden_states"] = novo
                return a, kw
            return (novo,) + tuple(a[1:]), kw
        ganchos.append(L[inicio].register_forward_pre_hook(pre, with_kwargs=True))

    def pos(mod, a, out):
        raise _Parar(_saida_tensor(out)[0].float())
    ganchos.append(L[fim].register_forward_hook(pos))
    try:
        m(torch.tensor([ids], device=next(m.parameters()).device))
    except _Parar as p:
        return p.h.to(DEV)
    finally:
        for g in ganchos:
            g.remove()
    raise RuntimeError("a camada de fim não rodou")


def logits_final(m, ids, inicio: int, entrada: torch.Tensor | None):
    """Última parte: injeta na camada inicio e roda até as probabilidades."""
    L = camadas(m)
    g = None
    if entrada is not None:
        def pre(mod, a, kw):
            hs = kw.get("hidden_states", a[0] if a else None)
            novo = entrada.to(hs.dtype).unsqueeze(0)
            if "hidden_states" in kw:
                kw["hidden_states"] = novo
                return a, kw
            return (novo,) + tuple(a[1:]), kw
        g = L[inicio].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        return m(torch.tensor([ids], device=next(m.parameters()).device)).logits[0].float().to(DEV)
    finally:
        if g is not None:
            g.remove()


@torch.no_grad()
def estado_nativo(m, ids, camada_entrada: int):
    """Vetor que entra na camada `camada_entrada` quando o modelo lê o texto sozinho."""
    o = m(torch.tensor([ids], device=next(m.parameters()).device), output_hidden_states=True)
    return o.hidden_states[camada_entrada][0].float().to(DEV)


# ============================================================================ super agente tradutor

class Tradutor(nn.Module):
    """y = W·(x - mx) + MLP(x - mx) + my     (W começa como rotação de Procrustes com escala)"""

    def __init__(self, da, db, oculto=1024):
        super().__init__()
        self.register_buffer("mx", torch.zeros(da))
        self.register_buffer("my", torch.zeros(db))
        self.W = nn.Parameter(torch.zeros(da, db))
        self.mlp = nn.Sequential(nn.Linear(da, oculto), nn.GELU(), nn.Linear(oculto, db))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.escala = nn.Parameter(torch.ones(()))

    @torch.no_grad()
    def iniciar_procrustes(self, x, y):
        self.mx.copy_(x.mean(0)); self.my.copy_(y.mean(0))
        xc, yc = x - self.mx, y - self.my
        U, S, Vh = torch.linalg.svd(xc.T @ yc, full_matrices=False)
        self.W.copy_((U @ Vh) * (S.sum() / xc.pow(2).sum()))
        pred = xc @ self.W
        return 1 - (pred - yc).pow(2).sum().item() / yc.pow(2).sum().item()   # R²

    def forward(self, x):
        z = x - self.mx
        return self.escala * (z @ self.W) + self.mlp(z) + self.my


# ============================================================================ a IA costurada

class IACosturada(nn.Module):
    def __init__(self, receita: Receita, log: Callable[[str], None] = print):
        super().__init__()
        self.r = receita
        self.log = log
        self.mods = [carregar_modelo(p.modelo) for p in receita.partes]
        for i, (p, (tok, m)) in enumerate(zip(receita.partes, self.mods)):
            n = len(camadas(m))
            if not (0 <= p.inicio <= p.fim < n):
                raise ValueError(f"parte {i + 1} ({p.modelo}): camadas {p.inicio}..{p.fim} fora de 0..{n - 1}")
            if i == 0 and p.inicio != 0:
                raise ValueError("a 1ª parte precisa começar na camada 0 (é ela que lê o texto)")
            if i == len(receita.partes) - 1 and p.fim != n - 1:
                raise ValueError(f"a última parte precisa ir até a última camada ({n - 1}): é ela que fala")
        dims = [m.config.hidden_size for _, m in self.mods]
        self.tradutores = nn.ModuleList([Tradutor(dims[i], dims[i + 1], receita.oculto_mlp).to(DEV)
                                         for i in range(len(self.mods) - 1)])

    # ---------------------------------------------------------------- preparo de um trecho
    def preparar(self, texto: str):
        tok_f = self.mods[-1][0]
        eg = tok_f(texto, return_offsets_mapping=True, add_special_tokens=False)
        ids_g, offs_g = eg["input_ids"][:self.r.max_tok], eg["offset_mapping"][:self.r.max_tok]
        if not ids_g:
            return None
        corte = offs_g[-1][1]
        tok0 = self.mods[0][0]
        e0 = tok0(texto[:corte], return_offsets_mapping=True, add_special_tokens=False)
        if not e0["input_ids"]:
            return None
        return {"ids0": e0["input_ids"], "ids_g": ids_g, "al": alinhar(e0["offset_mapping"], offs_g),
                "chars": corte, "texto": texto[:corte]}

    def _cadeia(self, d, ate: int | None = None):
        """Roda a cadeia. Retorna logits da última parte (ou o vetor que entra na parte `ate`)."""
        p0, (_, m0) = self.r.partes[0], self.mods[0]
        with torch.no_grad():
            h = rodar_parte(m0, d["ids0"], 0, p0.fim, None)
        h = h[torch.tensor(d["al"], device=DEV)]          # leva para a grade da última parte
        pad = [0] * len(d["ids_g"])
        for i in range(1, len(self.mods)):
            h = self.tradutores[i - 1](h)
            if ate == i:
                return h
            p, (_, m) = self.r.partes[i], self.mods[i]
            if i == len(self.mods) - 1:
                return logits_final(m, d["ids_g"], p.inicio, h)
            h = rodar_parte(m, pad, p.inicio, p.fim, h)
        raise RuntimeError

    def perda(self, d):
        lg = self._cadeia(d)
        alvo = torch.tensor(d["ids_g"][1:], device=DEV)
        return nn.functional.cross_entropy(lg[:-1], alvo, reduction="sum")

    # ---------------------------------------------------------------- iniciar tradutores (Procrustes)
    @torch.no_grad()
    def iniciar(self, trechos):
        """Para cada junção: x = saída da parte anterior (na grade), y = o que a próxima parte espera
        (o vetor que ela mesma produziria lendo o texto, alinhado à grade)."""
        for i in range(1, len(self.mods)):
            xs, ys = [], []
            tok_i, m_i = self.mods[i]
            for d in trechos:
                ei = tok_i(d["texto"], return_offsets_mapping=True, add_special_tokens=False)
                if not ei["input_ids"]:
                    continue
                y = estado_nativo(m_i, ei["input_ids"], self.r.partes[i].inicio)
                tok_g = self.mods[-1][0]
                eg = tok_g(d["texto"], return_offsets_mapping=True, add_special_tokens=False)
                al = alinhar(ei["offset_mapping"], eg["offset_mapping"][:len(d["ids_g"])])
                xs.append(self._antes_do_tradutor(d, i)); ys.append(y[torch.tensor(al, device=DEV)])
            r2 = self.tradutores[i - 1].iniciar_procrustes(torch.cat(xs), torch.cat(ys))
            self.log(f"  junção {i}: rotação de Procrustes explica {100 * r2:.0f}% (R²)")

    @torch.no_grad()
    def _antes_do_tradutor(self, d, i):
        """Vetor na grade que entra no tradutor i (saída da parte i-1)."""
        p0, (_, m0) = self.r.partes[0], self.mods[0]
        h = rodar_parte(m0, d["ids0"], 0, p0.fim, None)[torch.tensor(d["al"], device=DEV)]
        pad = [0] * len(d["ids_g"])
        for j in range(1, i):
            h = self.tradutores[j - 1](h)
            p, (_, m) = self.r.partes[j], self.mods[j]
            h = rodar_parte(m, pad, p.inicio, p.fim, h)
        return h

    # ---------------------------------------------------------------- medir
    @torch.no_grad()
    def bpc(self, trechos):
        nll, chars = 0.0, 0
        for d in trechos:
            nll += self.perda(d).item()
            chars += d["chars"]
        return nll / max(chars, 1) / math.log(2)

    # ---------------------------------------------------------------- gerar
    @torch.no_grad()
    def gerar(self, prompt: str, n: int = 120, temperatura: float = 0.0):
        tok_f = self.mods[-1][0]
        s, gerados = prompt, []
        for _ in range(n):
            d = self.preparar(s)
            if d is None:
                break
            if len(d["ids_g"]) >= self.r.max_tok:     # janela deslizante
                self.r.max_tok, guarda = 4096, self.r.max_tok
                d = self.preparar(s); self.r.max_tok = guarda
            lg = self._cadeia(d)[-1]
            if temperatura > 0:
                prox = torch.multinomial(torch.softmax(lg / temperatura, -1), 1).item()
            else:
                prox = lg.argmax().item()
            if prox == tok_f.eos_token_id:
                break
            gerados.append(prox)
            # decodifica tudo junto: decodificar token por token quebra letras com acento ("resistância")
            s = prompt + tok_f.decode(gerados).rstrip("\ufffd")
        return tok_f.decode(gerados)

    # ---------------------------------------------------------------- salvar / abrir
    def salvar(self, pasta: Path):
        pasta = Path(pasta)
        self.r.criado = time.strftime("%Y-%m-%d %H:%M")
        self.r.salvar(pasta)
        torch.save(self.tradutores.state_dict(), pasta / "tradutores.pt")

    @staticmethod
    def abrir(pasta: Path, log=print) -> "IACosturada":
        r = Receita.abrir(pasta)
        ia = IACosturada(r, log)
        ia.tradutores.load_state_dict(torch.load(Path(pasta) / "tradutores.pt", map_location=DEV))
        return ia


# ============================================================================ referência: modelo sozinho

@torch.no_grad()
def bpc_sozinho(nome: str, textos: list[str], max_tok=192):
    tok, m = carregar_modelo(nome)
    nll, chars = 0.0, 0
    for t in textos:
        e = tok(t, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = e["input_ids"][:max_tok], e["offset_mapping"][:max_tok]
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=next(m.parameters()).device)
        lg = m(x).logits[0, :-1].float()
        nll += nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()
        chars += offs[-1][1]
    return nll / max(chars, 1) / math.log(2)


# ============================================================================ treino completo

def dados_por_assunto(receita: Receita, rng: random.Random):
    treino, teste = {}, {}
    for assunto, arquivos in receita.assuntos.items():
        tr = fatiar(ler_textos(arquivos), rng)
        rng.shuffle(tr)
        tr = tr[:receita.max_trechos + max(5, receita.max_trechos // 6)]
        n_te = max(5, len(tr) // 6)
        treino[assunto], teste[assunto] = tr[:-n_te], tr[-n_te:]
    return treino, teste


def costurar(receita: Receita, pasta_saida: Path, log: Callable[[str], None] = print, parar=lambda: False):
    """Cria a IA nova: inicia os tradutores, treina, mede e salva. Retorna o dicionário de resultados."""
    t0 = time.time()
    rng = random.Random(0)
    torch.manual_seed(0)
    def L(msg):
        log(f"[{(time.time() - t0) / 60:5.1f} min] {msg}")

    if not receita.assuntos:
        raise ValueError("indique pelo menos um assunto com textos de foco")
    treino_txt, teste_txt = dados_por_assunto(receita, rng)
    for a in receita.assuntos:
        L(f"assunto '{a}': {len(treino_txt[a])} trechos de treino, {len(teste_txt[a])} de teste")

    L("carregando os modelos...")
    ia = IACosturada(receita, L)
    for i, p in enumerate(receita.partes):
        n = len(camadas(ia.mods[i][1]))
        L(f"  parte {i + 1}: {p.modelo}  camadas {p.inicio}..{p.fim} de {n}")

    prep = lambda lista: [d for d in (ia.preparar(t) for t in lista) if d and len(d["ids_g"]) > 8]
    tr = [d for a in treino_txt for d in prep(treino_txt[a])]
    te = {a: prep(teste_txt[a]) for a in teste_txt}

    L("super agente: começando pela rotação de Procrustes...")
    ia.iniciar(rng.sample(tr, min(len(tr), 150)))

    res = {"assuntos": {}, "curva": []}
    for a in te:
        res["assuntos"][a] = {"costurada_inicio": ia.bpc(te[a])}
    for i, p in enumerate(receita.partes):
        for a in te:
            res["assuntos"][a][f"sozinho_{i + 1}"] = bpc_sozinho(p.modelo, [d["texto"] for d in te[a]], receita.max_tok)
    L("antes do treino (bits por caractere, menor = melhor): " + "; ".join(
        f"{a}: costurada {v['costurada_inicio']:.3f}" for a, v in res["assuntos"].items()))

    opt = torch.optim.AdamW(ia.tradutores.parameters(), lr=receita.lr)
    passos = max(1, receita.epocas * len(tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=receita.lr, total_steps=passos, pct_start=0.1)
    todos_te = [d for a in te for d in te[a]]
    melhor, melhor_estado = float("inf"), None
    for ep in range(receita.epocas):
        rng.shuffle(tr)
        acc, ch = 0.0, 0
        for k, d in enumerate(tr):
            if parar():
                L("parado pelo usuário")
                break
            l = ia.perda(d)
            (l / len(d["ids_g"])).backward()
            torch.nn.utils.clip_grad_norm_(ia.tradutores.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            acc += l.item(); ch += d["chars"]
        b = ia.bpc(todos_te)
        res["curva"].append({"epoca": ep + 1, "treino_bpc": acc / max(ch, 1) / math.log(2), "teste_bpc": b})
        marca = ""
        if b < melhor:
            melhor, melhor_estado, marca = b, {k: v.detach().clone() for k, v in ia.tradutores.state_dict().items()}, "  ← melhor"
        L(f"época {ep + 1}/{receita.epocas}: treino {res['curva'][-1]['treino_bpc']:.3f}  teste {b:.3f} bpc{marca}")
        if parar():
            break
    if melhor_estado:
        ia.tradutores.load_state_dict(melhor_estado)
    for a in te:
        res["assuntos"][a]["costurada"] = ia.bpc(te[a])
    res["minutos"] = (time.time() - t0) / 60
    receita.resultados = res
    ia.salvar(pasta_saida)
    (Path(pasta_saida) / "resultado.md").write_text(relatorio(receita), encoding="utf-8")
    L(f"IA nova salva em {pasta_saida}")
    return ia, res


def relatorio(r: Receita) -> str:
    res = r.resultados
    nomes = [f"{i + 1}. `{p.modelo}` (camadas {p.inicio}–{p.fim})" for i, p in enumerate(r.partes)]
    linhas = [f"# {r.nome}", "", "Partes:", *[f"- {n}" for n in nomes], "",
              "Bits por caractere no texto de teste de cada assunto (menor = melhor):", "",
              "| assunto | " + " | ".join(f"só a IA {i + 1}" for i in range(len(r.partes))) + " | costurada (rotação) | **costurada (treinada)** |",
              "|---|" + "---|" * (len(r.partes) + 2)]
    for a, v in res.get("assuntos", {}).items():
        sos = " | ".join(f"{v.get(f'sozinho_{i + 1}', float('nan')):.3f}" for i in range(len(r.partes)))
        linhas.append(f"| {a} | {sos} | {v.get('costurada_inicio', float('nan')):.3f} | **{v.get('costurada', float('nan')):.3f}** |")
    if res.get("curva"):
        linhas += ["", "Curva do treino (bpc): " + ", ".join(f"{c['teste_bpc']:.3f}" for c in res["curva"])]
    linhas += ["", f"Tempo: {res.get('minutos', 0):.1f} min. Criada em {r.criado}."]
    return "\n".join(linhas)


# ============================================================================ busca do melhor corte

@torch.no_grad()
def sugerir_corte(modelo_a: str, modelo_b: str, textos: list[str], log=print, fracoes=(0.3, 0.4, 0.5, 0.6, 0.7), max_tok=160):
    """Testa vários pontos de corte só com a rotação (rápido) e devolve o melhor (inicio_b, fim_a)."""
    na = len(camadas(carregar_modelo(modelo_a)[1]))
    nb = len(camadas(carregar_modelo(modelo_b)[1]))
    rng = random.Random(1)
    if len(textos) < 6:
        raise ValueError("texto de foco curto demais para testar cortes (precisa de pelo menos ~5 mil caracteres)")
    amostra = rng.sample(textos, min(len(textos), 60))
    n_med = max(2, len(amostra) // 3)
    medir, ajustar = amostra[:n_med], amostra[n_med:]
    melhor, curva = None, []
    for f in fracoes:
        ca, cb = max(0, round(f * na) - 1), max(1, round(f * nb))
        rec = Receita("teste", [Parte(modelo_a, 0, ca), Parte(modelo_b, cb, nb - 1)], max_tok=max_tok)
        ia = IACosturada(rec, log=lambda *_: None)
        prep = lambda lst: [d for d in (ia.preparar(t) for t in lst) if d and len(d["ids_g"]) > 8]
        ia.iniciar(prep(ajustar))
        b = ia.bpc(prep(medir))
        log(f"  corte em {int(100 * f)}%: A até a camada {ca}, B a partir da {cb} → {b:.3f} bpc")
        curva.append({"fracao": f, "fim_a": ca, "inicio_b": cb, "bpc": b})
        if melhor is None or b < melhor[0]:
            melhor = (b, ca, cb, f)
    return {"bpc": melhor[0], "fim_a": melhor[1], "inicio_b": melhor[2], "fracao": melhor[3], "camadas_a": na, "camadas_b": nb, "curva": curva}


# ============================================================================ análise: no que cada IA é boa

@torch.no_grad()
def _bpc_textos(tok, m, textos, max_tok, pular: int | None = None):
    """bits por caractere do modelo sozinho; pular=ℓ desliga a camada ℓ (a saída vira a própria entrada)."""
    ganchos, guarda = [], {}
    if pular is not None:
        L = camadas(m)
        def pre(mod, a, kw):
            guarda["h"] = kw.get("hidden_states", a[0] if a else None)
        def pos(mod, a, out):
            if isinstance(out, (tuple, list)):
                return (guarda["h"],) + tuple(out[1:])
            return guarda["h"]
        ganchos += [L[pular].register_forward_pre_hook(pre, with_kwargs=True), L[pular].register_forward_hook(pos)]
    try:
        nll, chars = 0.0, 0
        for t in textos:
            e = tok(t, return_offsets_mapping=True, add_special_tokens=False)
            ids, offs = e["input_ids"][:max_tok], e["offset_mapping"][:max_tok]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], device=next(m.parameters()).device)
            lg = m(x).logits[0, :-1].float()
            nll += nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()
            chars += offs[-1][1]
        return nll / max(chars, 1) / math.log(2)
    finally:
        for g in ganchos:
            g.remove()


def analisar(modelos: list[str], assuntos: dict[str, list[str]], n_trechos=10, max_tok=128, log=print):
    """Para cada IA: (1) bits/caractere em cada assunto; (2) importância de cada camada para cada assunto
    = quanto o erro sobe quando aquela camada é desligada. Camadas importantes para um assunto
    são as 'melhores partes' daquela IA para ele."""
    rng = random.Random(3)
    textos = {}
    for a, arqs in assuntos.items():
        tr = fatiar(ler_textos(arqs), rng)
        rng.shuffle(tr)
        textos[a] = tr[:n_trechos]
    out = {"assuntos": list(textos), "modelos": {}}
    for nome in modelos:
        log(f"analisando {nome}...")
        tok, m = carregar_modelo(nome)
        L = len(camadas(m))
        base = {a: _bpc_textos(tok, m, t, max_tok) for a, t in textos.items()}
        imp = {a: [] for a in textos}
        for l in range(L):
            for a, t in textos.items():
                imp[a].append(_bpc_textos(tok, m, t, max_tok, pular=l) - base[a])
            if l % 4 == 3:
                log(f"  camada {l + 1}/{L}")
        out["modelos"][nome] = {"camadas": L, "bpc": base, "importancia": imp}
    return out
