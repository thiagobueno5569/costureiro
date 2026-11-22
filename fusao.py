"""Fusão direta: junta camadas de 2 ou mais modelos num modelo comum, sem treinar nada.

Feita para qualquer tamanho e qualquer máquina:
  * os pesos são lidos e escritos camada por camada, direto dos arquivos .safetensors
    (a memória usada é a de UMA camada, não a do modelo inteiro);
  * para achar a rotação, cada modelo é carregado sozinho, um de cada vez, na placa de vídeo,
    em várias placas (device_map="auto") ou só na RAM;
  * funciona com Llama, Mistral, Qwen2/2.5, Qwen3 e os MoE (Mixtral, Qwen2-MoE, Qwen3-MoE).

A conta (ver também exportar.py): se a parte B espera y = s·x·Q (Q ortogonal) e a base do modelo final é a
da parte A, cada peso de B é reescrito assim
    entradas que leem a norma (q, k, v, gate, up, roteador, w1, w3):  W' = W·diag(g)·Qᵀ   e a norma vira 1
    saídas que escrevem no fluxo residual (o, down, w2):             W' = Q·W / s
    cabeça final:                                                    W' = W·diag(g_final)·Qᵀ
A norma RMS não muda com rotação nem com escala, então o resultado é EXATO para essa parte linear.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

# arquiteturas com norma RMS "comum" (peso multiplica, sem +1) e sem normas depois das subcamadas
TIPOS = {"llama", "mistral", "qwen2", "qwen3", "mixtral", "qwen2_moe", "qwen3_moe"}
# chaves de config que podem ser diferentes sem problema
INOFENSIVAS = {"_name_or_path", "architectures", "torch_dtype", "dtype", "transformers_version", "bos_token_id",
               "eos_token_id", "pad_token_id", "max_position_embeddings", "use_cache", "num_hidden_layers",
               "layer_types", "max_window_layers", "use_sliding_window", "sliding_window", "initializer_range",
               "tie_word_embeddings", "_attn_implementation", "_attn_implementation_autoset", "output_router_logits",
               "router_aux_loss_coef", "model_type", "auto_map", "quantization_config", "attention_dropout",
               "use_mrope", "label2id", "id2label", "task_specific_params", "rope_parameters"}
ENTRADAS = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "gate_up_proj", "w1", "w3", "gate",
            "shared_expert_gate", "router"}
SAIDAS = {"o_proj", "down_proj", "w2"}
IGUAIS = ("q_norm.", "k_norm.", "e_score_correction_bias")


@dataclass
class Peca:
    modelo: str      # nome no Hugging Face ou pasta
    inicio: int      # 1ª camada usada
    fim: int         # última camada usada (inclusiva)

    @staticmethod
    def de_texto(s: str) -> "Peca":
        """'Qwen/Qwen2.5-3B-Instruct:0-19'"""
        nome, faixa = s.rsplit(":", 1)
        a, b = faixa.split("-")
        return Peca(nome.strip(), int(a), int(b))


# ============================================================================ arquivos

def pasta_do_modelo(nome: str) -> Path:
    p = Path(nome)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(nome, allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken",
                                                        "*.jinja", "tokenizer*", "LICENSE*", "README.md"]))


class Leitor:
    """Lê um tensor de cada vez dos .safetensors de um modelo."""

    def __init__(self, nome: str):
        from safetensors import safe_open
        self.pasta = pasta_do_modelo(nome)
        arquivos = sorted(self.pasta.glob("*.safetensors"))
        if not arquivos:
            raise ValueError(f"{nome}: não achei arquivos .safetensors (modelos só em .bin não são suportados)")
        self.onde = {}
        self._abertos = {}
        for a in arquivos:
            f = safe_open(str(a), "pt")
            self._abertos[a] = f
            for k in f.keys():
                self.onde[k] = a

    def chaves(self):
        return self.onde.keys()

    def get(self, k: str) -> torch.Tensor:
        return self._abertos[self.onde[k]].get_tensor(k)

    def tem(self, k: str) -> bool:
        return k in self.onde


class Escritor:
    """Escreve os tensores em pedaços de ~4 GB, no formato do Hugging Face."""

    def __init__(self, pasta: Path, max_bytes: int = 4 * 2 ** 30):
        self.pasta, self.max = Path(pasta), max_bytes
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.atual, self.tam, self.pedacos, self.total = {}, 0, [], 0

    def add(self, k: str, t: torch.Tensor):
        t = t.contiguous()
        n = t.numel() * t.element_size()
        if self.atual and self.tam + n > self.max:
            self._fechar_pedaco()
        self.atual[k] = t
        self.tam += n
        self.total += n

    def _fechar_pedaco(self):
        from safetensors.torch import save_file
        nome = f"_parte{len(self.pedacos):05d}.safetensors"
        save_file(self.atual, str(self.pasta / nome), metadata={"format": "pt"})
        self.pedacos.append((nome, list(self.atual)))
        self.atual, self.tam = {}, 0
        gc.collect()

    def fechar(self):
        if self.atual:
            self._fechar_pedaco()
        n = len(self.pedacos)
        mapa = {}
        for i, (tmp, chaves) in enumerate(self.pedacos):
            final = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            (self.pasta / tmp).rename(self.pasta / final)
            for k in chaves:
                mapa[k] = final
        if n > 1:
            (self.pasta / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": self.total}, "weight_map": mapa}, indent=1), encoding="utf-8")


# ============================================================================ checagem

def _cfg(nome: str) -> dict:
    return json.loads((pasta_do_modelo(nome) / "config.json").read_text(encoding="utf-8"))


def _cfg_texto(c: dict) -> dict:
    """Modelos multimodais guardam a parte de texto em text_config."""
    return c.get("text_config", c)


def checar(modelos: list[str]) -> tuple[bool, str]:
    """Dá para fundir exatamente? Mesma arquitetura (incluindo RoPE e MoE), mesma dimensão, mesmo tokenizador."""
    try:
        cfgs = [_cfg_texto(_cfg(m)) for m in modelos]
    except Exception as e:
        return False, f"não consegui ler a configuração: {e}"
    c0 = cfgs[0]
    if c0.get("model_type") not in TIPOS:
        return False, (f"arquitetura '{c0.get('model_type')}' ainda não suportada na fusão exata "
                       f"(suportadas: {', '.join(sorted(TIPOS))})")
    for m, c in zip(modelos[1:], cfgs[1:]):
        if c.get("model_type") != c0.get("model_type"):
            return False, f"famílias diferentes ({c0.get('model_type')} × {c.get('model_type')} em {m})"
        for k in sorted(set(c0) | set(c)):
            if k in INOFENSIVAS:
                continue
            if json.dumps(c0.get(k), sort_keys=True) != json.dumps(c.get(k), sort_keys=True):
                if k in ("rope_theta", "rope_scaling"):
                    return False, (f"a posição das palavras (RoPE) é calculada diferente em {m}: {k} = "
                                   f"{c0.get(k)} × {c.get(k)}")
                return False, f"estrutura diferente em {m}: {k} = {c0.get(k)} × {c.get(k)}"
        # rope_parameters (transformers novos) também carrega o θ
        r0, r1 = c0.get("rope_parameters") or {}, c.get("rope_parameters") or {}
        if r0.get("rope_theta") != r1.get("rope_theta") or r0.get("rope_type") != r1.get("rope_type"):
            return False, f"a posição das palavras (RoPE) é calculada diferente em {m}"
    if any(c.get("mlp_only_layers") for c in cfgs):
        return False, "modelos com algumas camadas MoE e outras não (mlp_only_layers) ainda não são suportados"
    from transformers import AutoTokenizer
    v0 = AutoTokenizer.from_pretrained(str(pasta_do_modelo(modelos[0]))).get_vocab()
    for m in modelos[1:]:
        if AutoTokenizer.from_pretrained(str(pasta_do_modelo(m))).get_vocab() != v0:
            return False, f"tokenizadores diferentes ({modelos[0]} × {m})"
    return True, "mesma arquitetura, mesmo tamanho, mesma RoPE e mesmo tokenizador"


# ============================================================================ estados internos

class _Parar(Exception):
    def __init__(self, h):
        self.h = h


def _camadas(m):
    for caminho in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = m
        try:
            for p in caminho.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise ValueError("não achei a lista de camadas do modelo")


def carregar_para_medir(nome: str, dispositivo: str = "auto"):
    """Carrega UM modelo para ler os estados internos. auto = placas de vídeo (e RAM se não couber)."""
    from transformers import AutoModelForCausalLM
    pasta = str(pasta_do_modelo(nome))
    usar_cuda = torch.cuda.is_available() and dispositivo in ("auto", "cuda")
    if usar_cuda:
        try:
            import accelerate  # noqa: F401
            m = AutoModelForCausalLM.from_pretrained(pasta, dtype=torch.float16, device_map="auto")
        except ImportError:
            m = AutoModelForCausalLM.from_pretrained(pasta, dtype=torch.float16).to("cuda")
    else:
        m = AutoModelForCausalLM.from_pretrained(pasta, dtype=torch.float32, low_cpu_mem_usage=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _disp(m):
    return m.get_input_embeddings().weight.device


@torch.no_grad()
def estados(m, ids_lista, entrada_na: int | None = None, saida_de: int | None = None, injetar=None):
    """Para cada frase: o vetor que ENTRA na camada `entrada_na`, ou o que SAI da camada `saida_de`.
    injetar[k] (opcional) substitui o vetor que entra na camada `entrada_na` (usado com saida_de)."""
    L = _camadas(m)
    out = []
    for k, ids in enumerate(ids_lista):
        ganchos = []
        if injetar is not None:
            novo = injetar[k]

            def pre_inj(mod, a, kw, novo=novo):
                hs = kw.get("hidden_states", a[0] if a else None)
                t = novo.to(hs.device, hs.dtype).unsqueeze(0)
                if "hidden_states" in kw:
                    kw["hidden_states"] = t
                    return a, kw
                return (t,) + tuple(a[1:]), kw
            ganchos.append(L[entrada_na].register_forward_pre_hook(pre_inj, with_kwargs=True))
        if saida_de is not None:
            def pos(mod, a, out_):
                h = out_[0] if isinstance(out_, (tuple, list)) else out_
                raise _Parar(h[0].float().cpu())
            ganchos.append(L[saida_de].register_forward_hook(pos))
        else:
            def pre(mod, a, kw):
                hs = kw.get("hidden_states", a[0] if a else None)
                raise _Parar(hs[0].float().cpu())
            ganchos.append(L[entrada_na].register_forward_pre_hook(pre, with_kwargs=True))
        try:
            m(input_ids=torch.tensor([ids], device=_disp(m)))
            raise RuntimeError("a camada pedida não rodou")
        except _Parar as p:
            out.append(p.h)
        finally:
            for g in ganchos:
                g.remove()
    return out


def procrustes(x: torch.Tensor, y: torch.Tensor):
    """y ≈ s·x·Q, Q ortogonal, sem centralizar (o deslocamento de média não cabe nos pesos)."""
    x, y = x.double(), y.double()
    U, S, Vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    Q = U @ Vh
    s = S.sum() / x.pow(2).sum()
    yc = y - y.mean(0)
    r2 = 1 - (s * x @ Q - y).pow(2).sum() / yc.pow(2).sum()
    return Q.float(), float(s), float(r2)


# ============================================================================ reescrever pesos

def _papel(nome: str) -> str:
    """O que um peso da camada faz: 'norma1', 'norma2', 'entrada', 'saida' ou 'igual'."""
    if nome == "input_layernorm.weight":
        return "norma1"
    if nome == "post_attention_layernorm.weight":
        return "norma2"
    if any(x in nome for x in IGUAIS):
        return "igual"
    partes = nome.split(".")
    tipo = partes[-1]
    modulo = partes[-2] if tipo in ("weight", "bias") else partes[-1]
    if modulo in SAIDAS or (tipo not in ("weight", "bias") and tipo in SAIDAS):
        return "saida"
    if modulo in ENTRADAS or (tipo not in ("weight", "bias") and tipo in ENTRADAS):
        return "entrada_bias" if tipo == "bias" else "entrada"
    raise ValueError(f"peso inesperado na camada: '{nome}' (arquitetura diferente do esperado)")


def girar_tensor(nome: str, w: torch.Tensor, Q: torch.Tensor, s: float, g1: torch.Tensor, g2: torch.Tensor):
    papel = _papel(nome)
    d = Q.shape[0]
    dt = w.dtype
    if papel in ("norma1", "norma2"):
        return torch.ones_like(w)
    if papel in ("igual", "entrada_bias"):
        return w
    w = w.float()
    if papel == "entrada":
        g = g1 if nome.startswith("self_attn.") else g2
        if w.shape[-1] == d:                       # (…, saída, entrada): padrão do PyTorch
            return ((w * g) @ Q.T).to(dt)
        if w.dim() == 3 and w.shape[-2] == d:      # (especialistas, entrada, saída)
            return (Q @ (g[:, None] * w)).to(dt)
        raise ValueError(f"{nome}: formato {tuple(w.shape)} não bate com a dimensão {d}")
    # saída: escreve no fluxo residual
    if w.dim() == 1:                               # viés
        return (Q @ w / s).to(dt)
    if w.shape[-2] == d:                           # (…, saída=d, entrada)
        return (Q @ w / s).to(dt)
    if w.dim() == 3 and w.shape[-1] == d:          # (especialistas, entrada, saída=d)
        return (w @ Q.T / s).to(dt)
    raise ValueError(f"{nome}: formato {tuple(w.shape)} não bate com a dimensão {d}")


def _prefixo(leitor: Leitor) -> str:
    for k in leitor.chaves():
        m = re.match(r"(.*?)layers\.\d+\.", k)
        if m:
            return m.group(1)
    raise ValueError("não achei 'layers.N.' nos pesos")


# ============================================================================ a fusão

def _ids(tok, textos, max_tok):
    out = []
    for t in textos:
        ids = tok(t, add_special_tokens=False)["input_ids"][:max_tok]
        if len(ids) >= 16:
            out.append(ids)
    return out


@torch.no_grad()
def fundir(pecas: list[Peca], saida: Path, textos: list[str], log=print, progresso=lambda f, t: None,
           dispositivo: str = "auto", max_tok: int = 256, max_textos: int = 96, checar_antes: bool = True) -> dict:
    """Gera, em `saida`, um modelo comum do Hugging Face com as camadas de cada peça."""
    from transformers import AutoTokenizer
    t0 = time.time()
    modelos = [p.modelo for p in pecas]
    if len(pecas) < 2:
        raise ValueError("precisa de pelo menos 2 peças")
    if checar_antes:
        pode, motivo = checar(modelos)
        if not pode:
            raise ValueError(f"fusão exata não dá: {motivo}")
    cfg0 = _cfg(modelos[0])
    ct = _cfg_texto(cfg0)
    n_por = [_cfg_texto(_cfg(m))["num_hidden_layers"] for m in modelos]
    for p, n in zip(pecas, n_por):
        if not (0 <= p.inicio <= p.fim < n):
            raise ValueError(f"{p.modelo}: camadas {p.inicio}..{p.fim} fora de 0..{n - 1}")
    if pecas[0].inicio != 0:
        log("  aviso: a 1ª peça não começa na camada 0 (as primeiras camadas dela serão puladas)")

    tok = AutoTokenizer.from_pretrained(str(pasta_do_modelo(modelos[0])))
    textos = list(textos)
    random.Random(0).shuffle(textos)
    ids = _ids(tok, textos, max_tok)
    if len(ids) < 8:
        raise ValueError("texto de calibração curto demais (precisa de pelo menos ~10 mil caracteres)")
    n_med = max(4, min(16, len(ids) // 6))
    ids_medir, ids_ajuste = ids[:n_med], ids[n_med:n_med + max_textos]
    n_tok = sum(len(i) - 1 for i in ids_ajuste)
    log(f"  calibração: {len(ids_ajuste)} trechos ({n_tok} tokens) para achar as rotações, {len(ids_medir)} para medir")
    if n_tok < 2 * ct["hidden_size"]:
        log(f"  aviso: poucos tokens para a dimensão {ct['hidden_size']}; use mais texto para uma rotação mais precisa")

    # ---- 1) rotações, um modelo de cada vez
    giros, juncoes = [], []
    progresso(0.03, f"lendo {modelos[0]}")
    m = carregar_para_medir(modelos[0], dispositivo)
    x = estados(m, ids_ajuste, saida_de=pecas[0].fim)
    del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    for i in range(1, len(pecas)):
        p = pecas[i]
        progresso(0.03 + 0.4 * i / len(pecas), f"lendo {p.modelo}")
        m = carregar_para_medir(p.modelo, dispositivo)
        y = estados(m, ids_ajuste, entrada_na=p.inicio)
        Q, s, r2 = procrustes(torch.cat([h[1:] for h in x]), torch.cat([h[1:] for h in y]))
        parecida = float(torch.trace(Q)) / Q.shape[0]
        log(f"  junção {i}: {Path(modelos[i - 1]).name} → {Path(p.modelo).name}: rotação explica {100 * r2:.1f}% · "
            f"escala {s:.3f} · {100 * parecida:.0f}% igual a 'sem rotação'")
        giros.append((Q, s))
        juncoes.append({"de": modelos[i - 1], "para": p.modelo, "r2": r2, "escala": s, "identidade": parecida})
        if i < len(pecas) - 1:
            ent = [s * (h @ Q) for h in x]
            y_fim = estados(m, ids_ajuste, entrada_na=p.inicio, saida_de=p.fim, injetar=ent)
            x = [(h @ Q.T) / s for h in y_fim]
        del m, y; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 2) pesos, em streaming
    saida = Path(saida)
    shutil.rmtree(saida, ignore_errors=True)
    esc = Escritor(saida)
    total = sum(p.fim - p.inicio + 1 for p in pecas)
    k_novo, tipos_camada = 0, []
    for i, p in enumerate(pecas):
        progresso(0.45 + 0.4 * i / len(pecas), f"reescrevendo as camadas de {p.modelo}")
        log(f"  camadas {p.inicio}..{p.fim} de {p.modelo}" + (" (giradas para a base da 1ª peça)" if i else ""))
        lei = Leitor(p.modelo)
        pre = _prefixo(lei)
        cfg_i = _cfg_texto(_cfg(p.modelo))
        lt = cfg_i.get("layer_types")
        if i == 0:
            for k in lei.chaves():
                if not re.search(r"layers\.\d+\.", k) and "embed_tokens" in k:
                    esc.add("model.embed_tokens.weight", lei.get(k))
        for l in range(p.inicio, p.fim + 1):
            pv = f"{pre}layers.{l}."
            chaves = [k for k in lei.chaves() if k.startswith(pv)]
            if i == 0:
                for k in chaves:
                    esc.add(f"model.layers.{k_novo}." + k[len(pv):], lei.get(k))
            else:
                Q, s = giros[i - 1]
                g1 = lei.get(pv + "input_layernorm.weight").float()
                g2 = lei.get(pv + "post_attention_layernorm.weight").float()
                for k in chaves:
                    nome = k[len(pv):]
                    esc.add(f"model.layers.{k_novo}.{nome}", girar_tensor(nome, lei.get(k), Q, s, g1, g2))
            if lt:
                tipos_camada.append(lt[l])
            k_novo += 1
        if i == len(pecas) - 1:
            Q, s = giros[-1]
            k_norma = next(k for k in lei.chaves() if k.endswith("norm.weight") and not re.search(r"layers\.\d+\.", k))
            gf = lei.get(k_norma)
            cab = next((k for k in lei.chaves() if k.endswith("lm_head.weight")), None)
            if cab is None:
                cab = next(k for k in lei.chaves() if "embed_tokens" in k and not re.search(r"layers\.\d+\.", k))
            W = lei.get(cab)
            esc.add("model.norm.weight", torch.ones_like(gf))
            esc.add("lm_head.weight", ((W.float() * gf.float()) @ Q.T).to(W.dtype))
        del lei
        gc.collect()
    progresso(0.86, "gravando os arquivos")
    esc.fechar()

    # ---- 3) configuração e tokenizador
    cfg_novo = dict(cfg0)
    alvo = cfg_novo.get("text_config", cfg_novo)
    alvo["num_hidden_layers"] = total
    alvo["tie_word_embeddings"] = False
    cfg_novo["tie_word_embeddings"] = False
    if tipos_camada:
        alvo["layer_types"] = tipos_camada
    if "max_window_layers" in alvo:
        alvo["max_window_layers"] = total
    (saida / "config.json").write_text(json.dumps(cfg_novo, indent=2, ensure_ascii=False), encoding="utf-8")
    origem = pasta_do_modelo(modelos[0])
    for f in origem.iterdir():
        if f.is_file() and (f.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "chat_template",
                                                "generation_config")) or f.suffix in (".model", ".tiktoken", ".jinja")):
            shutil.copy2(f, saida / f.name)
    res = {"modo": "fusão exata", "pecas": [p.__dict__ for p in pecas], "camadas": total, "juncoes": juncoes,
           "minutos_fusao": (time.time() - t0) / 60, "_ids_medir": ids_medir}
    return res


@torch.no_grad()
def bpc(nome_ou_pasta: str, ids_lista, tok, dispositivo="auto") -> float:
    m = carregar_para_medir(nome_ou_pasta, dispositivo)
    nll, chars = 0.0, 0
    for ids in ids_lista:
        x = torch.tensor([ids], device=_disp(m))
        lg = m(input_ids=x).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(lg, x[0, 1:].to(lg.device), reduction="sum").item()
        chars += len(tok.decode(ids))
    del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return nll / max(chars, 1) / math.log(2)


def medir(res: dict, saida: Path, log=print, progresso=lambda f, t: None, dispositivo="auto") -> dict:
    """bits por caractere do modelo fundido e de cada original no mesmo texto (menor = melhor)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(saida))
    ids = res.get("_ids_medir") or []
    if not ids:
        return res
    medidas = {}
    for i, p in enumerate(res["pecas"]):
        progresso(0.87 + 0.04 * i / len(res["pecas"]), f"medindo {p['modelo']}")
        medidas[p["modelo"]] = bpc(p["modelo"], ids, tok, dispositivo)
    progresso(0.91, "medindo o modelo fundido")
    medidas["fundido"] = bpc(str(saida), ids, tok, dispositivo)
    res["bpc"] = medidas
    log("  bits por caractere (menor = melhor): " + " · ".join(
        f"{'FUNDIDO' if k == 'fundido' else Path(k).name} {v:.3f}" for k, v in medidas.items()))
    return res


def _licenca(nome: str) -> str:
    try:
        from huggingface_hub import model_info
        cd = model_info(nome).card_data
        return (cd.get("license") if cd else None) or "?"
    except Exception:
        return "?"


def ficha(res: dict, saida: Path, nome: str):
    """README.md (model card) do modelo fundido, com as licenças herdadas."""
    pecas = res["pecas"]
    lic = {p["modelo"]: _licenca(p["modelo"]) for p in pecas}
    base = [p["modelo"] for p in pecas if not Path(p["modelo"]).exists()]
    linhas = ["---", "library_name: transformers", "tags: [costureiro, merge, model-stitching]"]
    if base:
        linhas += ["base_model:"] + [f"- {b}" for b in base]
    uniq = sorted({v for v in lic.values() if v != "?"})
    if len(uniq) == 1:
        linhas.append(f"license: {uniq[0]}")
    linhas += ["---", "", f"# {nome}", "",
               "Modelo criado pela **fusão direta** do [Costureiro](https://github.com/thiagobueno5569/costureiro): "
               "camadas de modelos diferentes, com a mudança de base entre eles embutida nos pesos.", "",
               "| peça | camadas | licença |", "|---|---|---|"]
    for p in pecas:
        linhas.append(f"| `{p['modelo']}` | {p['inicio']}–{p['fim']} | {lic[p['modelo']]} |")
    linhas += ["", "| junção | rotação explica | escala |", "|---|---|---|"]
    for j in res.get("juncoes", []):
        linhas.append(f"| {Path(j['de']).name} → {Path(j['para']).name} | {100 * j['r2']:.1f}% | {j['escala']:.3f} |")
    if res.get("bpc"):
        linhas += ["", "Bits por caractere no texto de calibração reservado (menor = melhor):", "",
                   "| modelo | bpc |", "|---|---|"]
        for k, v in res["bpc"].items():
            linhas.append(f"| {'**fundido**' if k == 'fundido' else '`' + k + '`'} | {v:.3f} |")
    linhas += ["", "⚠️ **Licença:** um modelo fundido herda as condições de TODAS as peças. Confira cada licença "
               "acima antes de usar comercialmente."]
    (Path(saida) / "README.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    res["licencas"] = lic
    return lic


def textos_de(caminhos: list[str]) -> list[str]:
    import nucleo
    rng = random.Random(0)
    return nucleo.fatiar(nucleo.ler_textos(caminhos), rng, 600, 1400)


def fundir_completo(pecas: list[Peca], pasta_saida: Path, textos: list[str], nome: str, log=print,
                    progresso=lambda f, t: None, dispositivo="auto", gguf: str | None = "q8_0",
                    lmstudio: bool = False, medir_qualidade: bool = True) -> dict:
    """Fusão + medição + ficha + .gguf (+ cópia para o LM Studio)."""
    import exportar
    t0 = time.time()
    def L(msg):
        log(f"[{(time.time() - t0) / 60:5.1f} min] {msg}")
    pasta_saida = Path(pasta_saida)
    hf = pasta_saida / "hf"
    res = fundir(pecas, hf, textos, log=L, progresso=progresso, dispositivo=dispositivo)
    if medir_qualidade:
        res = medir(res, hf, log=L, progresso=progresso, dispositivo=dispositivo)
    res.pop("_ids_medir", None)
    ficha(res, hf, nome)
    if gguf:
        progresso(0.93, "convertendo para .gguf")
        arq = exportar.para_gguf(hf, pasta_saida / f"{exportar._nome_arquivo(nome)}.{gguf}.gguf", gguf, log=L)
        res["gguf"] = str(arq)
        res["tamanho_mb"] = arq.stat().st_size / 2 ** 20
        if lmstudio:
            destino = exportar.pasta_lmstudio() / "costureiro" / exportar._nome_arquivo(nome) / arq.name
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(arq, destino)
            res["lmstudio"] = str(destino)
            L(f"copiado para o LM Studio: {destino}")
    res["minutos"] = (time.time() - t0) / 60
    (pasta_saida / "fusao.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    progresso(1.0, "pronto")
    L("pronto")
    return res


def relatorio(res: dict) -> str:
    lin = []
    if res.get("gguf"):
        lin.append(f"**Arquivo:** `{Path(res['gguf']).name}` ({res.get('tamanho_mb', 0):.0f} MB) · {res.get('minutos', 0):.0f} min")
    if res.get("lmstudio"):
        lin.append("Já está no LM Studio, em **My Models → costureiro**.")
    lin += ["", "| junção | rotação explica | igual a 'sem rotação' |", "|---|---|---|"]
    for j in res.get("juncoes", []):
        lin.append(f"| {Path(j['de']).name} → {Path(j['para']).name} | {100 * j['r2']:.1f}% | {100 * j['identidade']:.0f}% |")
    if res.get("bpc"):
        lin += ["", "| modelo | bits por caractere (menor = melhor) |", "|---|---|"]
        for k, v in res["bpc"].items():
            lin.append(f"| {'**fundido**' if k == 'fundido' else Path(k).name} | {'**' if k == 'fundido' else ''}{v:.3f}{'**' if k == 'fundido' else ''} |")
    if res.get("licencas"):
        lin += ["", "**Licenças herdadas:** " + ", ".join(f"{Path(k).name}: {v}" for k, v in res["licencas"].items())
                + ". O modelo fundido precisa respeitar todas."]
    return "\n".join(lin)


# ============================================================================ linha de comando

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Costureiro · fusão direta: junta camadas de modelos da mesma arquitetura num modelo comum.",
        epilog="Exemplo: python fusao.py --peca Qwen/Qwen2.5-3B-Instruct:0-19 "
               "--peca Qwen/Qwen2.5-Coder-3B-Instruct:20-35 --texto textos/ --saida fusoes/instruct-coder")
    ap.add_argument("--peca", action="append", required=True, help="modelo:inicio-fim (repita para cada peça, em ordem)")
    ap.add_argument("--texto", action="append", help="arquivos ou pastas de texto para calibrar (repita)")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--nome", default=None)
    ap.add_argument("--gguf", default="q8_0", help="q8_0, f16, bf16 ou 'nao'")
    ap.add_argument("--lmstudio", action="store_true", help="copia o .gguf para a pasta do LM Studio")
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sem-medir", action="store_true")
    ap.add_argument("--so-checar", action="store_true")
    a = ap.parse_args(argv)
    pecas = [Peca.de_texto(s) for s in a.peca]
    pode, motivo = checar([p.modelo for p in pecas])
    print(("✅ " if pode else "❌ ") + motivo)
    if a.so_checar or not pode:
        return 0 if pode else 1
    caminhos = a.texto or [str(Path(__file__).parent / "textos")]
    textos = textos_de(caminhos)
    nome = a.nome or Path(a.saida).name
    res = fundir_completo(pecas, Path(a.saida), textos, nome, dispositivo=a.dispositivo,
                          gguf=None if a.gguf == "nao" else a.gguf, lmstudio=a.lmstudio, medir_qualidade=not a.sem_medir)
    print(relatorio(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
