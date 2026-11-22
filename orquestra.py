"""Modo orquestra: a fusão que tenta somar as HABILIDADES, não só o estilo do texto.

O que a avaliação ensinou (HumanEval/MBPP/GSM8K): costurar em fila (A lê → B escreve) baixa a perplexidade
mas não passa a habilidade adiante — a parte que "sabe programar" do Coder mora nas camadas do FIM, e na
fila elas eram jogadas fora. A orquestra muda quatro coisas:

1. Em paralelo, não em fila. A BASE (ex.: Math) roda inteira, intacta. Cada SOLISTA (ex.: Coder) lê o
   mesmo texto sozinho, com o próprio vocabulário, até uma camada FUNDA (onde a habilidade já foi calculada).
2. Ponte + portão. Na camada c da base, o pensamento vira
        h ← w₀·h + Σⱼ wⱼ·Tⱼ(sⱼ)          com (w₀, w₁, …) = softmax(portão(h, T₁(s₁), …))
   Tⱼ é o super agente tradutor (começa pela rotação de Procrustes, depois aprende). O portão decide,
   palavra por palavra, de quem é a vez. Começa com w₀ ≈ 1: no início a orquestra É a base — não tem
   como começar pior que ela.
3. Aprende com o melhor professor de cada palavra. Para cada posição, olhamos qual modelo original
   (a base ou um solista, rodando inteiros) dava mais chance à palavra certa, e a orquestra aprende a
   imitar a distribuição DELE. Assim o alvo é "o melhor dos dois em cada ponto", não o estilo médio do texto.
   O portão também recebe essa resposta como dica (quem era o melhor ali).
4. Dados no formato das tarefas e contexto longo: problemas de programação com solução (MBPP — só as
   partes de treino, nunca a de teste), problemas de matemática resolvidos (GSM8K treino) e código real,
   em trechos de até 512 tokens.

As camadas (até onde o solista lê, onde a base recebe) podem ser escolhidas sozinhas: o super agente testa
vários pares e fica com o que a rotação explica melhor.

    python orquestra.py --base Qwen/Qwen2.5-Math-1.5B-Instruct --solista Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --nome math-com-coder
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

import nucleo
from nucleo import DEV, Tradutor, alinhar, camadas, fatiar, ler_textos

AQUI = Path(__file__).parent
DADOS = AQUI / "dados_treino"


def _dt():
    if DEV != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ============================================================================ receita

@dataclass
class Solista:
    modelo: str
    ate: int = -1            # última camada que o solista roda (-1 = escolher sozinho)
    nome: str = ""


@dataclass
class ReceitaOrquestra:
    nome: str
    base: str
    solistas: list[Solista]
    camada: int = -1                 # camada da base que recebe os solistas (-1 = escolher sozinho)
    assuntos: dict[str, list[str]] = field(default_factory=dict)   # textos extras do usuário
    dados_de_tarefa: bool = True     # MBPP (treino) + GSM8K (treino) + código real
    max_tok: int = 512
    epocas: int = 3
    lr: float = 1e-3
    oculto_mlp: int = 1024
    max_trechos: int = 2500
    peso_ce: float = 0.2
    peso_portao: float = 0.2
    criado: str = ""
    resultados: dict = field(default_factory=dict)

    def salvar(self, pasta: Path):
        pasta.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["tipo"] = "orquestra"
        (pasta / "receita.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def abrir(pasta: Path) -> "ReceitaOrquestra":
        d = json.loads((Path(pasta) / "receita.json").read_text(encoding="utf-8"))
        d.pop("tipo", None)
        d["solistas"] = [Solista(**s) for s in d["solistas"]]
        return ReceitaOrquestra(**d)


# ============================================================================ modelos

def carregar(nome: str, ate: int | None = None):
    """(tokenizer, modelo) congelado. ate=k: joga fora as camadas depois de k e a cabeça (economiza placa)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nome)
    m = AutoModelForCausalLM.from_pretrained(nome, dtype=_dt(), low_cpu_mem_usage=True).eval()
    if ate is not None:
        L = camadas(m)
        while len(L) > ate + 1:
            del L[len(L) - 1]
        m.lm_head = nn.Identity()
    m = m.to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def _mesmo_vocab(t1, t2) -> bool:
    return t1.get_vocab() == t2.get_vocab()


@torch.no_grad()
def estado_saida(m, ids, k: int) -> torch.Tensor:
    """Vetor na SAÍDA da camada k (o solista para ali)."""
    return nucleo.rodar_parte(m, ids, 0, k, None)


@torch.no_grad()
def estados_todos(m, ids):
    o = m(torch.tensor([ids], device=next(m.parameters()).device), output_hidden_states=True)
    return [h[0].float() for h in o.hidden_states]          # [embed, saída da 0, saída da 1, ...]


# ============================================================================ ponte e portão

class Ponte(nn.Module):
    """Um tradutor (solista → base) por solista e um portão que divide a vez entre a base e os solistas."""

    def __init__(self, d_base: int, d_solistas: list[int], oculto: int):
        super().__init__()
        self.trads = nn.ModuleList([Tradutor(ds, d_base, oculto) for ds in d_solistas])
        n = len(d_solistas)
        self.portao = nn.Sequential(nn.Linear(d_base * (n + 1), 256), nn.GELU(), nn.Linear(256, n + 1))
        nn.init.zeros_(self.portao[2].weight)
        with torch.no_grad():
            self.portao[2].bias.zero_()
            self.portao[2].bias[0] = 4.0        # começa com ~98% para a base: a orquestra nasce igual à base

    def forward(self, hb: torch.Tensor, estados: list[torch.Tensor]):
        ts = [t(s) for t, s in zip(self.trads, estados)]
        unit = lambda v: v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        z = torch.cat([unit(hb)] + [unit(t) for t in ts], -1) * math.sqrt(hb.shape[-1])
        lg = self.portao(z)                                   # (T, n+1)
        w = torch.softmax(lg, -1)
        h = w[:, :1] * hb
        for j, t in enumerate(ts):
            h = h + w[:, j + 1:j + 2] * t
        return h, lg


# ============================================================================ a orquestra

class IAOrquestra(nn.Module):
    def __init__(self, r: ReceitaOrquestra, log=print, preparar: bool = False, textos_ajuste: list[str] | None = None):
        super().__init__()
        self.r, self.log = r, log
        self.tok, self.base = carregar(r.base)
        nb = len(camadas(self.base))
        self.sol = []
        dims = []
        for s in r.solistas:
            if preparar and (s.ate < 0 or r.camada < 0):
                self._escolher_camadas(s, textos_ajuste or [])
            tok_s, m_s = carregar(s.modelo, ate=s.ate)
            self.sol.append((tok_s, m_s, _mesmo_vocab(tok_s, self.tok)))
            dims.append(m_s.config.hidden_size)
        if not 1 <= r.camada < nb:
            raise ValueError(f"camada da base deve estar entre 1 e {nb - 1}")
        self.ponte = Ponte(self.base.config.hidden_size, dims, r.oculto_mlp).to(DEV)
        self.ultimo_portao = None
        if preparar and textos_ajuste:
            self._iniciar_pontes(textos_ajuste)

    # ---------------------------------------------------------------- camadas e rotação inicial
    @torch.no_grad()
    def _escolher_camadas(self, s: Solista, textos):
        """Testa pares (até onde o solista lê, onde a base recebe) e fica com o que a rotação explica melhor."""
        tok_s, m_s = carregar(s.modelo)
        ns, nb = len(camadas(m_s)), len(camadas(self.base))
        cand_k = sorted({max(1, round(f * ns) - 1) for f in (0.6, 0.7, 0.8, 0.9)})
        cand_c = sorted({max(1, round(f * nb)) for f in (0.55, 0.65, 0.75, 0.85)}) if self.r.camada < 0 else [self.r.camada]
        if s.ate >= 0:
            cand_k = [s.ate]
        xs = {k: [] for k in cand_k}
        ys = {c: [] for c in cand_c}
        for t in textos[:40]:
            pb, ps = self._ids(self.tok, t), self._ids(tok_s, t)
            if pb is None or ps is None:
                continue
            al = torch.tensor(alinhar(ps[1], pb[1]), device=DEV)
            hs, hb = estados_todos(m_s, ps[0]), estados_todos(self.base, pb[0])
            for k in cand_k:
                xs[k].append(hs[k + 1][al])
            for c in cand_c:
                ys[c].append(hb[c])
        melhor = None
        for k in cand_k:
            X = torch.cat(xs[k])
            for c in cand_c:
                Y = torch.cat(ys[c])
                tr = Tradutor(X.shape[1], Y.shape[1]).to(DEV)
                r2 = tr.iniciar_procrustes(X, Y)
                self.log(f"  {Path(s.modelo).name} até a camada {k} → base na camada {c}: rotação explica {100 * r2:.0f}%")
                if melhor is None or r2 > melhor[0]:
                    melhor = (r2, k, c)
        _, s.ate, c = melhor
        self.r.camada = c
        self.log(f"  escolhido: {Path(s.modelo).name} lê até a camada {s.ate}; a base recebe na camada {c}")
        del m_s
        gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None

    @torch.no_grad()
    def _iniciar_pontes(self, textos):
        for j, (tok_s, m_s, _) in enumerate(self.sol):
            xs, ys, normas = [], [], []
            for t in textos[:150]:
                pb, ps = self._ids(self.tok, t), self._ids(tok_s, t)
                if pb is None or ps is None:
                    continue
                al = torch.tensor(alinhar(ps[1], pb[1]), device=DEV)
                xs.append(estado_saida(m_s, ps[0], self.r.solistas[j].ate)[al])
                yb = nucleo.estado_nativo(self.base, pb[0], self.r.camada)
                ys.append(yb); normas.append(yb.norm(dim=-1).mean())
            r2 = self.ponte.trads[j].iniciar_procrustes(torch.cat(xs), torch.cat(ys))
            self.log(f"  ponte {self.r.solistas[j].nome or Path(self.r.solistas[j].modelo).name} → base: "
                     f"rotação explica {100 * r2:.0f}% (R²)")

    def _ids(self, tok, texto, max_tok=None):
        e = tok(texto, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = e["input_ids"][:max_tok or self.r.max_tok], e["offset_mapping"][:max_tok or self.r.max_tok]
        return (ids, offs) if len(ids) >= 2 else None

    # ---------------------------------------------------------------- rodar
    def preparar(self, texto: str, max_tok: int | None = None):
        pb = self._ids(self.tok, texto, max_tok)
        if pb is None:
            return None
        corte = pb[1][-1][1]
        sol = []
        for tok_s, m_s, _ in self.sol:
            ps = self._ids(tok_s, texto[:corte], 10 ** 9)
            if ps is None:
                return None
            sol.append((ps[0], alinhar(ps[1], pb[1])))
        return {"ids": pb[0], "sol": sol, "chars": corte}

    def logits(self, d) -> torch.Tensor:
        with torch.no_grad():
            estados = []
            for (ids_s, al), (tok_s, m_s, _), s in zip(d["sol"], self.sol, self.r.solistas):
                estados.append(estado_saida(m_s, ids_s, s.ate)[torch.tensor(al, device=DEV)])
        guarda = {}

        def pre(mod, a, kw):
            hs = kw.get("hidden_states", a[0] if a else None)
            h, lg = self.ponte(hs[0].float(), estados)
            guarda["lg"] = lg
            novo = h.to(hs.dtype).unsqueeze(0)
            if "hidden_states" in kw:
                kw["hidden_states"] = novo
                return a, kw
            return (novo,) + tuple(a[1:]), kw
        g = camadas(self.base)[self.r.camada].register_forward_pre_hook(pre, with_kwargs=True)
        try:
            out = self.base(torch.tensor([d["ids"]], device=DEV)).logits[0].float()
        finally:
            g.remove()
        self.ultimo_portao = guarda.get("lg")
        return out

    @torch.no_grad()
    def completar(self, prompt: str, max_novos: int, paradas: list[str]) -> str:
        texto = ""
        gerados = []
        for _ in range(max_novos):
            d = self.preparar(prompt + texto, max_tok=10 ** 9)
            prox = int(self.logits(d)[-1].argmax())
            if prox == self.tok.eos_token_id:
                break
            gerados.append(prox)
            texto = self.tok.decode(gerados, skip_special_tokens=True).rstrip("�")
            if any(p in texto for p in paradas):
                break
        cortes = [texto.find(p) for p in paradas if texto.find(p) >= 0]
        return texto[:min(cortes)] if cortes else texto

    def uso(self) -> list[float]:
        """Fração da vez de cada um no último texto: [base, solista1, ...]."""
        if self.ultimo_portao is None:
            return []
        return torch.softmax(self.ultimo_portao, -1).mean(0).tolist()

    def salvar(self, pasta: Path):
        self.r.criado = time.strftime("%Y-%m-%d %H:%M")
        self.r.salvar(Path(pasta))
        torch.save(self.ponte.state_dict(), Path(pasta) / "ponte.pt")

    @staticmethod
    def abrir(pasta: Path, log=print) -> "IAOrquestra":
        r = ReceitaOrquestra.abrir(pasta)
        ia = IAOrquestra(r, log)
        ia.ponte.load_state_dict(torch.load(Path(pasta) / "ponte.pt", map_location=DEV))
        return ia


# ============================================================================ dados

def _baixar(url: str, nome: str) -> Path:
    DADOS.mkdir(exist_ok=True)
    arq = DADOS / nome
    if not arq.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "costureiro"})
        arq.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    return arq


def dados_de_tarefa(rng: random.Random, log=print) -> dict[str, list[str]]:
    """Texto no formato das tarefas, SEM as partes de teste (MBPP teste, HumanEval e GSM8K teste ficam de fora)."""
    import pandas as pd
    out = {"programação": [], "matemática": []}
    hf = "https://huggingface.co/datasets"
    for parte in ("train", "validation", "prompt"):      # MBPP completo: tarefas 1-10 e 511-974 (o teste é 11-510)
        try:
            df = pd.read_parquet(_baixar(f"{hf}/google-research-datasets/mbpp/resolve/main/full/{parte}-00000-of-00001.parquet",
                                         f"mbpp_{parte}.parquet"))
            for r in df.to_dict("records"):
                testes = list(r["test_list"])
                out["programação"].append(f'"""\n{r["text"]}\n{testes[0]}\n"""\n{r["code"]}\n')
        except Exception as e:
            log(f"  (sem MBPP {parte}: {e})")
    try:
        df = pd.read_parquet(_baixar(f"{hf}/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet", "gsm8k_train.parquet"))
        out["matemática"] = [f"Problem: {q}\nSolution: {a.replace('####', 'Answer:')}" for q, a in zip(df["question"], df["answer"])]
    except Exception as e:
        log(f"  (sem GSM8K: {e})")
    cod = AQUI / "textos" / "programacao"
    if cod.exists():
        out["programação"] += fatiar(ler_textos([str(cod)]), rng, 800, 2000)
    return out


# ============================================================================ treino

@torch.no_grad()
def _professores(ia: IAOrquestra, d, k=32):
    """Para cada posição: top-k (ids, logprob) e logprob da palavra certa de cada professor
    [base inteira, solista1 inteiro (se o vocabulário for o mesmo), ...]."""
    ids = d["ids"]
    alvo = torch.tensor(ids[1:], device=DEV)
    profs = [ia.base(torch.tensor([ids], device=DEV)).logits[0, :-1].float()]
    for (tok_s, _, mesmo), m_cheio in zip(ia.sol, ia._cheios):
        if mesmo and m_cheio is not None:
            profs.append(m_cheio(torch.tensor([ids], device=DEV)).logits[0, :-1].float())
    tops_i, tops_v, certos = [], [], []
    V = profs[0].shape[-1]
    for lg in profs:
        lg = lg[:, :V]                                      # vocabulários com "folga" de tamanhos diferentes
        lp = torch.log_softmax(lg, -1)
        v, i = lp.topk(k, -1)
        tops_i.append(i.to(torch.int32).cpu()); tops_v.append(v.half().cpu())
        certos.append(lp.gather(-1, alvo[:, None])[:, 0].cpu())
    return {"i": torch.stack(tops_i), "v": torch.stack(tops_v), "certo": torch.stack(certos)}


def criar(r: ReceitaOrquestra, pasta: Path, log=print, parar=lambda: False, progresso=lambda f, t: None):
    t0 = time.time()
    rng = random.Random(0)
    torch.manual_seed(0)
    L = lambda m: log(f"[{(time.time() - t0) / 60:5.1f} min] {m}")
    nucleo.liberar_cache()

    # ---- dados: tarefas + textos do usuário; separa teste
    textos = {}
    if r.dados_de_tarefa:
        textos = dados_de_tarefa(rng, L)
    for a, arqs in r.assuntos.items():
        textos.setdefault(a, [])
        textos[a] += fatiar(ler_textos(arqs), rng, 800, 2000)
    treino, teste = {}, {}
    for a, lst in textos.items():
        if not lst:
            continue
        rng.shuffle(lst)
        lst = lst[:r.max_trechos + max(20, r.max_trechos // 10)]
        n_te = max(10, min(60, len(lst) // 10))
        treino[a], teste[a] = lst[:-n_te], lst[-n_te:]
        L(f"assunto '{a}': {len(treino[a])} trechos de treino, {len(teste[a])} de teste")
    todos_tr = [t for a in treino for t in treino[a]]
    if not todos_tr:
        raise ValueError("sem textos de treino: não consegui baixar o MBPP/GSM8K (sem internet?) e não há textos em textos/. "
                         "Rode preparar_textos.py ou indique assuntos com textos.")

    # ---- modelos, camadas e pontes
    progresso(0.02, "carregando e escolhendo as camadas")
    ia = IAOrquestra(r, L, preparar=True, textos_ajuste=rng.sample(todos_tr, min(150, len(todos_tr))))
    L(f"base: {r.base} (recebe na camada {r.camada})")
    for s in r.solistas:
        L(f"solista: {s.modelo} (lê até a camada {s.ate})")

    # ---- professores: base inteira e cada solista inteiro
    progresso(0.08, "professores: o que cada modelo original preveria")
    ia._cheios = []
    for (tok_s, _, mesmo), s in zip(ia.sol, r.solistas):
        ia._cheios.append(carregar(s.modelo)[1] if mesmo else None)
        if not mesmo:
            L(f"  {s.modelo}: vocabulário diferente da base — aprende só com a base e com o texto")
    ds_tr = []
    for j, t in enumerate(todos_tr):
        d = ia.preparar(t)
        if d is None or len(d["ids"]) < 16:
            continue
        d["prof"] = _professores(ia, d)
        ds_tr.append(d)
        if j % 50 == 0:
            progresso(0.08 + 0.17 * j / len(todos_tr), f"professores: {j}/{len(todos_tr)} trechos")
    ds_te = {a: [d for d in (ia.preparar(t) for t in lst) if d and len(d["ids"]) >= 16] for a, lst in teste.items()}
    # quem é o melhor professor em cada assunto (só para mostrar)
    for a in ds_te:
        for d in ds_te[a]:
            d["prof"] = _professores(ia, d)
    nomes = ["base"] + [s.nome or Path(s.modelo).name for s in r.solistas]
    res = {"assuntos": {}, "curva": [], "nomes": nomes}

    def medir(lista, tambem_prof=False):
        nll, chars, uso = 0.0, 0, None
        with torch.no_grad():
            for d in lista:
                lg = ia.logits(d)
                nll += nn.functional.cross_entropy(lg[:-1], torch.tensor(d["ids"][1:], device=DEV), reduction="sum").item()
                chars += d["chars"]
                u = torch.tensor(ia.uso())
                uso = u if uso is None else uso + u
        out = {"bpc": nll / max(chars, 1) / math.log(2), "uso": (uso / max(1, len(lista))).tolist()}
        if tambem_prof:
            for j in range(len(lista[0]["prof"]["certo"])):
                s = -sum(float(d["prof"]["certo"][j].sum()) for d in lista)
                out[f"prof_{nomes[j] if j < len(nomes) else j}"] = s / max(chars, 1) / math.log(2)
            melhor = -sum(float(d["prof"]["certo"].max(0).values.sum()) for d in lista)
            out["melhor_de_cada_palavra"] = melhor / max(chars, 1) / math.log(2)
        return out

    for a in ds_te:
        res["assuntos"][a] = {"inicio": medir(ds_te[a], tambem_prof=True)}
        v = res["assuntos"][a]["inicio"]
        L(f"'{a}' antes do treino: orquestra {v['bpc']:.3f} bpc · " +
          " · ".join(f"{k[5:]} {x:.3f}" for k, x in v.items() if k.startswith("prof_")) +
          f" · melhor de cada palavra {v['melhor_de_cada_palavra']:.3f}")

    # ---- treino
    params = list(ia.ponte.parameters())
    opt = torch.optim.AdamW(params, lr=r.lr, weight_decay=0.01)
    passos = max(1, r.epocas * len(ds_tr))
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=r.lr, total_steps=passos, pct_start=0.05) if passos >= 50
             else torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0))
    todos_te = [d for a in ds_te for d in ds_te[a]]
    melhor, estado, feitos = float("inf"), None, 0
    for ep in range(r.epocas):
        rng.shuffle(ds_tr)
        t_ep, acc = time.time(), 0.0
        for k, d in enumerate(ds_tr):
            if parar():
                break
            lg = ia.logits(d)[:-1]
            lp = torch.log_softmax(lg, -1)
            P = d["prof"]
            certo = P["certo"].to(DEV)                       # (n_prof, T-1)
            quem = certo.argmax(0)                           # melhor professor em cada posição
            idx = quem[None, :, None].expand(1, -1, P["i"].shape[-1])
            ti = P["i"].to(DEV).long().gather(0, idx)[0]
            tv = P["v"].to(DEV).float().gather(0, idx)[0]
            kd = -(torch.softmax(tv, -1) * lp.gather(-1, ti)).sum(-1).mean()
            ce = nn.functional.cross_entropy(lg, torch.tensor(d["ids"][1:], device=DEV))
            perda = kd + r.peso_ce * ce
            if r.peso_portao > 0 and ia.ultimo_portao is not None and certo.shape[0] == ia.ultimo_portao.shape[1]:
                # dica para o portão: só onde a diferença entre os professores é clara (> 0,5 nat)
                ordem = certo.sort(0, descending=True).values
                claro = (ordem[0] - ordem[1]) > 0.5 if certo.shape[0] > 1 else torch.zeros_like(quem, dtype=torch.bool)
                if claro.any():
                    perda = perda + r.peso_portao * nn.functional.cross_entropy(ia.ultimo_portao[:-1][claro], quem[claro])
            perda.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            acc += perda.item(); feitos += 1
            if k % 10 == 0:
                pp = (time.time() - t_ep) / (k + 1)
                progresso(0.27 + 0.65 * feitos / passos,
                          f"época {ep + 1}/{r.epocas} · trecho {k + 1}/{len(ds_tr)} · perda {acc / (k + 1):.3f} · "
                          f"faltam ~{(passos - feitos) * pp / 60:.0f} min")
        m = medir(todos_te)
        res["curva"].append({"epoca": ep + 1, "perda_treino": acc / max(1, len(ds_tr)), "teste_bpc": m["bpc"], "uso": m["uso"]})
        marca = ""
        if m["bpc"] < melhor:
            melhor, estado, marca = m["bpc"], {k: v.detach().clone() for k, v in ia.ponte.state_dict().items()}, "  ← melhor"
        L(f"época {ep + 1}: teste {m['bpc']:.3f} bpc · vez de cada um: " +
          ", ".join(f"{n} {100 * u:.0f}%" for n, u in zip(nomes, m["uso"])) + marca)
        if parar():
            break
    if estado:
        ia.ponte.load_state_dict(estado)
    for a in ds_te:
        res["assuntos"][a]["fim"] = medir(ds_te[a])
        v = res["assuntos"][a]
        L(f"'{a}' depois: orquestra {v['fim']['bpc']:.3f} bpc (antes {v['inicio']['bpc']:.3f}; melhor de cada palavra "
          f"{v['inicio']['melhor_de_cada_palavra']:.3f}) · vez: " +
          ", ".join(f"{n} {100 * u:.0f}%" for n, u in zip(nomes, v["fim"]["uso"])))
    res["minutos"] = (time.time() - t0) / 60
    r.resultados = res
    for m_cheio in ia._cheios:
        del m_cheio
    ia._cheios = []
    gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    ia.salvar(Path(pasta))
    (Path(pasta) / "resultado.md").write_text(relatorio(r), encoding="utf-8")
    progresso(1.0, "pronto")
    L(f"salva em {pasta}")
    return ia, res


def relatorio(r: ReceitaOrquestra) -> str:
    res = r.resultados
    nomes = res.get("nomes", [])
    lin = [f"# {r.nome} (orquestra)", "", f"Base: `{r.base}` (recebe na camada {r.camada})"]
    lin += [f"Solista: `{s.modelo}` (lê até a camada {s.ate})" for s in r.solistas]
    lin += ["", "Bits por caractere no texto de teste (menor = melhor):", "",
            "| assunto | " + " | ".join(f"só {n}" for n in nomes) + " | melhor de cada palavra | orquestra antes | **orquestra** | vez de cada um |",
            "|---|" + "---|" * (len(nomes) + 4)]
    for a, v in res.get("assuntos", {}).items():
        ini, fim = v.get("inicio", {}), v.get("fim", {})
        profs = " | ".join(f"{ini.get('prof_' + n, float('nan')):.3f}" for n in nomes)
        uso = ", ".join(f"{n} {100 * u:.0f}%" for n, u in zip(nomes, fim.get("uso", [])))
        lin.append(f"| {a} | {profs} | {ini.get('melhor_de_cada_palavra', float('nan')):.3f} | {ini.get('bpc', float('nan')):.3f} | "
                   f"**{fim.get('bpc', float('nan')):.3f}** | {uso} |")
    lin += ["", f"Tempo: {res.get('minutos', 0):.0f} min. Criada em {r.criado}."]
    return "\n".join(lin)


# ============================================================================ linha de comando

def main(argv=None):
    ap = argparse.ArgumentParser(description="Costureiro · modo orquestra")
    ap.add_argument("--base", required=True)
    ap.add_argument("--solista", action="append", required=True, help="modelo[:até_camada] (repita)")
    ap.add_argument("--camada", type=int, default=-1, help="camada da base que recebe (-1 = escolher)")
    ap.add_argument("--assunto", action="append", default=[], help="nome=arquivo_ou_pasta[;outro] (textos extras)")
    ap.add_argument("--sem-dados-de-tarefa", action="store_true")
    ap.add_argument("--epocas", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_tok", type=int, default=512)
    ap.add_argument("--max_trechos", type=int, default=2500)
    ap.add_argument("--nome", default="orquestra")
    ap.add_argument("--saida", default=str(AQUI / "ias_novas"))
    a = ap.parse_args(argv)
    sol = []
    for s in a.solista:
        if ":" in s.rsplit("/", 1)[-1]:
            m, k = s.rsplit(":", 1)
            sol.append(Solista(m, int(k)))
        else:
            sol.append(Solista(s))
    assuntos = {}
    for s in a.assunto:
        n, c = s.split("=", 1)
        assuntos[n] = [x for x in c.split(";") if x]
    r = ReceitaOrquestra(a.nome, a.base, sol, a.camada, assuntos, not a.sem_dados_de_tarefa, a.max_tok, a.epocas, a.lr,
                         max_trechos=a.max_trechos)
    pasta = Path(a.saida) / a.nome
    criar(r, pasta, log=lambda m: print(m, flush=True), progresso=lambda f, t: None)
    print("\n" + (pasta / "resultado.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
