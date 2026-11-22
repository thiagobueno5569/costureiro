"""Exportar uma IA costurada para um modelo normal (.gguf) que abre no LM Studio, Ollama, llama.cpp…

Por que não dá para salvar a costurada direto em .gguf: um .gguf descreve UMA pilha de camadas iguais
com UM tokenizador. A costurada tem um tradutor no meio (rotação + rede pequena) e pode juntar
modelos de tamanhos e tokenizadores diferentes. Então há dois caminhos:

1. FUSÃO EXATA (rápida, minutos) — quando todas as partes são da mesma família e do mesmo tamanho
   (mesmo tokenizador, mesma dimensão, mesma RoPE; ex.: Qwen2.5-1.5B + Qwen2.5-Math-1.5B + Coder-1.5B).
   A mudança de base vira parte dos pesos: se a parte B espera  y = s·x·Q  (Q ortogonal), reescrevemos
   as camadas de B para trabalharem direto na base de A:
       norma RMS:            RMS(s·x·Q) = RMS(x)·Q          (rotação e escala não mudam o tamanho relativo)
       entradas (q,k,v,gate,up):   W' = W · diag(g) · Qᵀ     (o ganho g da norma é absorvido)
       saídas (o_proj, down_proj):  W' = Q · W / s
       cabeça final:               W' = W · diag(g_final) · Qᵀ
   O resultado é um modelo Qwen/Llama comum, com as camadas de A seguidas das camadas de B.
   Só a parte LINEAR do tradutor cabe nos pesos (sem a rede pequena e sem o deslocamento de média),
   por isso o Costureiro mede de novo e mostra quanto custou.

2. DESTILAÇÃO (qualquer par, de dezenas de minutos a horas) — a IA costurada vira "professora":
   para cada trecho de treino guardamos as 64 palavras mais prováveis que ela prevê. Depois um modelo
   normal (o da última parte, que tem o mesmo tokenizador) aprende com LoRA a imitar essas previsões.
   No fim a LoRA é somada aos pesos e o modelo vira .gguf.
"""
from __future__ import annotations

import gc
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

import nucleo
from nucleo import DEV, IACosturada, Receita, camadas, dados_por_assunto, _bpc_textos

AQUI = Path(__file__).parent
CONVERSOR = AQUI / "conversor_gguf"
# versão fixa do conversor do llama.cpp (MIT) — baixada só na primeira exportação
LLAMA_CPP_COMMIT = "cbb1c1c50bba81929387dea473873d1df87a6212"
TIPOS_SUPORTADOS = {"qwen2", "qwen3", "llama", "mistral"}
ENTRADAS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
SAIDAS_LIN = ("o_proj", "down_proj")


def _nome_arquivo(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "ia-costurada"


# ============================================================================ 1. fusão exata

def checar_fusao(receita: Receita) -> tuple[bool, str]:
    """A fusão exata só funciona se as partes 'falam a mesma língua' por fora: mesmo tokenizador,
    mesma dimensão e mesma RoPE. Devolve (pode, motivo)."""
    from transformers import AutoConfig, AutoTokenizer
    cfgs = [AutoConfig.from_pretrained(p.modelo) for p in receita.partes]
    c0 = cfgs[0]
    if c0.model_type not in TIPOS_SUPORTADOS:
        return False, f"arquitetura {c0.model_type} ainda não suportada na fusão exata"

    def rope(c):
        return (getattr(c, "rope_theta", None), json.dumps(getattr(c, "rope_scaling", None), sort_keys=True, default=str),
                json.dumps(getattr(c, "rope_parameters", None), sort_keys=True, default=str))

    campos = ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "hidden_act")
    for p, c in zip(receita.partes[1:], cfgs[1:]):
        if c.model_type != c0.model_type:
            return False, f"famílias diferentes ({c0.model_type} × {c.model_type})"
        for k in campos:
            if getattr(c, k, None) != getattr(c0, k, None):
                return False, f"tamanhos diferentes: {k} = {getattr(c0, k, None)} × {getattr(c, k, None)} ({p.modelo})"
        if getattr(c, "head_dim", None) != getattr(c0, "head_dim", None):
            return False, "tamanho das cabeças de atenção diferente"
        if rope(c) != rope(c0):
            return False, (f"a posição das palavras (RoPE) é calculada diferente em {p.modelo} "
                           f"(θ = {getattr(c0, 'rope_theta', '?')} × {getattr(c, 'rope_theta', '?')})")
        if getattr(c, "attention_bias", None) != getattr(c0, "attention_bias", None):
            return False, "um modelo tem viés na atenção e o outro não"
    toks = [AutoTokenizer.from_pretrained(p.modelo) for p in receita.partes]
    v0 = toks[0].get_vocab()
    for p, t in zip(receita.partes[1:], toks[1:]):
        if t.get_vocab() != v0:
            return False, f"tokenizadores diferentes ({receita.partes[0].modelo} × {p.modelo})"
    return True, "mesma família, mesmo tamanho e mesmo tokenizador"


def procrustes_sem_media(x: torch.Tensor, y: torch.Tensor):
    """Melhor y ≈ s·x·Q com Q ortogonal (sem centralizar: a média não cabe nos pesos)."""
    x, y = x.double(), y.double()
    U, S, Vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    Q = U @ Vh
    s = S.sum() / x.pow(2).sum()
    yc = y - y.mean(0)
    r2 = 1 - (s * x @ Q - y).pow(2).sum() / yc.pow(2).sum()
    return Q.float(), float(s), float(r2)


@torch.no_grad()
def _estados(m, ids_lista, camada_entrada=None, saida_de=None, entrada=None, inicio=0):
    """Junta vetores de várias frases (pula a 1ª posição, que costuma ter ativações gigantes)."""
    out = []
    for k, ids in enumerate(ids_lista):
        if saida_de is not None:
            h = nucleo.rodar_parte(m, ids, inicio, saida_de, None if entrada is None else entrada[k])
        else:
            h = nucleo.estado_nativo(m, ids, camada_entrada)
        out.append(h)
    return out


def _girar_camada(sd: dict, prefixo: str, Q: torch.Tensor, s: float) -> dict:
    """Reescreve uma camada para trabalhar na base nova (ver explicação no topo do arquivo)."""
    novo = {}
    g1 = sd[prefixo + "input_layernorm.weight"].float()
    g2 = sd[prefixo + "post_attention_layernorm.weight"].float()
    Qt = Q.T.contiguous()
    for k, v in sd.items():
        if not k.startswith(prefixo):
            continue
        nome = k[len(prefixo):]
        v = v.float()
        if nome in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            novo[k] = torch.ones_like(v)
        elif any(nome == f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj")):
            novo[k] = (v * g1[None, :]) @ Qt
        elif any(nome == f"mlp.{n}.weight" for n in ("gate_proj", "up_proj")):
            novo[k] = (v * g2[None, :]) @ Qt
        elif nome in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
            novo[k] = Q @ v / s
        elif nome in ("self_attn.o_proj.bias", "mlp.down_proj.bias"):
            novo[k] = Q @ v / s
        elif nome.startswith(("self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
                              "self_attn.q_norm.", "self_attn.k_norm.")):
            novo[k] = v
        else:
            raise ValueError(f"peso inesperado na camada: {nome} (arquitetura diferente do esperado)")
    return novo


def _sd_cpu(nome: str) -> dict:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(nome, dtype=torch.float32, low_cpu_mem_usage=True)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    del m
    gc.collect()
    return sd


@torch.no_grad()
def fundir(receita: Receita, pasta_hf: Path, textos_ajuste: list[str], log=print, progresso=lambda f, t: None) -> dict:
    """Constrói um modelo normal (pasta do Hugging Face) com as camadas de cada parte, já girado."""
    from transformers import AutoConfig, AutoTokenizer
    partes = receita.partes
    tok = AutoTokenizer.from_pretrained(partes[0].modelo)
    ids_lista = []
    for t in textos_ajuste:
        ids = tok(t, add_special_tokens=False)["input_ids"][:receita.max_tok]
        if len(ids) > 8:
            ids_lista.append(ids)
    if len(ids_lista) < 4:
        raise ValueError("texto de ajuste curto demais para achar a rotação")

    # --- 1) acha a rotação de cada junção, em cadeia, sempre na base da 1ª parte
    progresso(0.05, "achando a rotação de cada junção")
    _, m0 = nucleo.carregar_modelo(partes[0].modelo)
    x = _estados(m0, ids_lista, saida_de=partes[0].fim)          # base da parte 1, saída da última camada usada
    giros, r2s = [], []
    for i in range(1, len(partes)):
        p = partes[i]
        _, mi = nucleo.carregar_modelo(p.modelo)
        y = _estados(mi, ids_lista, camada_entrada=p.inicio)     # o que a parte i espera receber
        Q, s, r2 = procrustes_sem_media(torch.cat([h[1:] for h in x]).cpu(), torch.cat([h[1:] for h in y]).cpu())
        giros.append((Q, s)); r2s.append(r2)
        parecida = float(torch.trace(Q)) / Q.shape[0]
        log(f"  junção {i}: rotação explica {100 * r2:.0f}% · escala {s:.3f} · parecença com 'nenhuma rotação' {100 * parecida:.0f}%")
        if i < len(partes) - 1:                                   # leva o estado até a próxima junção
            Qd = Q.to(DEV)
            ent = [s * (h @ Qd) for h in x]
            y_fim = _estados(mi, ids_lista, saida_de=p.fim, entrada=ent, inicio=p.inicio)
            x = [(h @ Qd.T) / s for h in y_fim]
    nucleo.liberar_cache()

    # --- 2) monta os pesos (na CPU, um modelo de cada vez)
    cfg = AutoConfig.from_pretrained(partes[0].modelo)
    total = sum(p.fim - p.inicio + 1 for p in partes)
    tipos_camada = []
    sd_novo, k_novo = {}, 0
    for i, p in enumerate(partes):
        progresso(0.15 + 0.6 * i / len(partes), f"reescrevendo as camadas de {p.modelo}")
        log(f"  copiando camadas {p.inicio}..{p.fim} de {p.modelo}" + (" (giradas para a base da 1ª parte)" if i else ""))
        sd = _sd_cpu(p.modelo)
        ci = AutoConfig.from_pretrained(p.modelo)
        lt = getattr(ci, "layer_types", None)
        if i == 0:
            sd_novo["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"]
        for l in range(p.inicio, p.fim + 1):
            pre_v, pre_n = f"model.layers.{l}.", f"model.layers.{k_novo}."
            if i == 0:
                camada = {k: v for k, v in sd.items() if k.startswith(pre_v)}
            else:
                camada = _girar_camada(sd, pre_v, *giros[i - 1])
            for k, v in camada.items():
                sd_novo[pre_n + k[len(pre_v):]] = v
            if lt:
                tipos_camada.append(lt[l])
            k_novo += 1
        if i == len(partes) - 1:
            Q, s = giros[-1]
            gf = sd["model.norm.weight"].float()
            cab = sd.get("lm_head.weight", sd["model.embed_tokens.weight"]).float()
            sd_novo["model.norm.weight"] = torch.ones_like(gf)
            sd_novo["lm_head.weight"] = (cab * gf[None, :]) @ Q.T
        del sd
        gc.collect()

    cfg.num_hidden_layers = total
    cfg.tie_word_embeddings = False
    if tipos_camada and hasattr(cfg, "layer_types"):
        cfg.layer_types = tipos_camada
    if hasattr(cfg, "max_window_layers"):
        cfg.max_window_layers = total

    # --- 3) salva no formato do Hugging Face
    progresso(0.8, "salvando o modelo novo")
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        modelo = AutoModelForCausalLM.from_config(cfg)
    modelo.load_state_dict(sd_novo, assign=True, strict=True)
    modelo = modelo.to(torch.bfloat16)
    pasta_hf = Path(pasta_hf)
    pasta_hf.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(pasta_hf, safe_serialization=True)
    tok.save_pretrained(pasta_hf)
    del modelo, sd_novo
    gc.collect()
    return {"modo": "fusão exata", "r2_juncoes": r2s, "camadas": total}


# ============================================================================ 2. destilação

def _professora_topk(ia: IACosturada, ds: list[dict], k=64, log=print, progresso=lambda f, t: None):
    alvos = []
    for j, d in enumerate(ds):
        with torch.no_grad():
            lp = torch.log_softmax(ia._cadeia(d).float(), -1)
            v, i = lp.topk(k, dim=-1)
        alvos.append({"ids": d["ids_g"], "top_i": i.to(torch.int32).cpu(), "top_v": v.half().cpu(), "chars": d["chars"]})
        if j % 20 == 0:
            progresso(0.05 + 0.25 * j / len(ds), f"a costurada está ensinando: {j}/{len(ds)} trechos")
    return alvos


def destilar(pasta_ia: Path, pasta_hf: Path, log=print, progresso=lambda f, t: None, parar=lambda: False,
             epocas: int = 3, lr: float = 2e-4, rank: int = 16) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    r = Receita.abrir(pasta_ia)
    rng = random.Random(0)                    # mesma divisão treino/teste usada na costura
    treino_txt, teste_txt = dados_por_assunto(r, rng)
    ia = IACosturada.abrir(pasta_ia, log=lambda *_: None)
    prep = lambda lst: [d for d in (ia.preparar(t) for t in lst) if d and len(d["ids_g"]) > 8]
    tr = [d for a in treino_txt for d in prep(treino_txt[a])]
    te = {a: prep(teste_txt[a]) for a in teste_txt}
    log(f"  {len(tr)} trechos de treino; a costurada vai mostrar o que preveria em cada um")
    alvos = _professora_topk(ia, tr, log=log, progresso=progresso)
    res = {"modo": "destilação", "assuntos": {}}
    for a in te:
        res["assuntos"][a] = {"costurada": ia.bpc(te[a])}
    textos_te = {a: [d["texto"] for d in te[a]] for a in te}
    aluna = r.partes[-1].modelo
    del ia
    nucleo.liberar_cache(); gc.collect()

    progresso(0.32, "preparando a aluna (modelo normal + LoRA)")
    tok = AutoTokenizer.from_pretrained(aluna)
    dt = torch.float16 if DEV == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(aluna, dtype=dt).to(DEV)
    for a in te:
        res["assuntos"][a]["aluna_antes"] = _bpc_textos(tok, m, textos_te[a], r.max_tok)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    alvo_mods = sorted({n.split(".")[-1] for n, mod in m.named_modules()
                        if isinstance(mod, nn.Linear) and n.split(".")[-1] in ENTRADAS + SAIDAS_LIN})
    m = get_peft_model(m, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=alvo_mods,
                                     task_type="CAUSAL_LM"))
    params = [p for p in m.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    passos = max(1, epocas * len(alvos))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos, pct_start=0.05)
    escala = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
    melhor, melhor_estado, curva = float("inf"), None, []
    todos_te = [t for a in textos_te for t in textos_te[a]]
    feitos = 0
    for ep in range(epocas):
        m.train()
        rng.shuffle(alvos)
        t_ep, acc = time.time(), 0.0
        for j, al in enumerate(alvos):
            if parar():
                break
            x = torch.tensor([al["ids"]], device=DEV)
            lg = m(input_ids=x).logits[0].float()
            lp = torch.log_softmax(lg, -1)
            ti, tv = al["top_i"].to(DEV).long(), al["top_v"].to(DEV).float()
            pt = torch.softmax(tv, -1)                                    # previsão da professora (top-64)
            kd = -(pt * lp.gather(-1, ti)).sum(-1).mean()
            ce = nn.functional.cross_entropy(lg[:-1], x[0, 1:])
            perda = 0.9 * kd + 0.1 * ce
            escala.scale(perda).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            escala.step(opt); escala.update(); opt.zero_grad(set_to_none=True); sched.step()
            acc += perda.item(); feitos += 1
            if j % 10 == 0:
                por = (time.time() - t_ep) / (j + 1)
                progresso(0.35 + 0.5 * feitos / passos,
                          f"aluna aprendendo: época {ep + 1}/{epocas}, trecho {j + 1}/{len(alvos)} · "
                          f"faltam ~{max(0, (passos - feitos) * por) / 60:.0f} min")
        m.eval()
        b = _bpc_textos(tok, m, todos_te, r.max_tok)
        curva.append(b)
        marca = ""
        if b < melhor:
            melhor = b
            melhor_estado = {k: v.detach().clone() for k, v in m.state_dict().items() if "lora_" in k}
            marca = "  ← melhor"
        log(f"  época {ep + 1}/{epocas}: perda {acc / max(1, len(alvos)):.3f} · aluna no teste {b:.3f} bpc{marca}")
        if parar():
            log("  parado pelo usuário (vou salvar a melhor versão até aqui)")
            break
    if melhor_estado:
        m.load_state_dict(melhor_estado, strict=False)
    for a in te:
        res["assuntos"][a]["aluna"] = _bpc_textos(tok, m, textos_te[a], r.max_tok)
    res["curva"] = curva

    progresso(0.87, "somando a LoRA aos pesos e salvando")
    m = m.merge_and_unload()
    m.gradient_checkpointing_disable()
    m = m.to(torch.bfloat16) if DEV != "cuda" else m
    pasta_hf = Path(pasta_hf)
    pasta_hf.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(pasta_hf, safe_serialization=True)
    tok.save_pretrained(pasta_hf)
    del m
    nucleo.liberar_cache(); gc.collect()
    res["minutos"] = (time.time() - t0) / 60
    return res


# ============================================================================ 3. .gguf

def obter_conversor(log=print) -> Path:
    """Baixa (só uma vez) o conversor oficial do llama.cpp: convert_hf_to_gguf.py + conversion/ + gguf-py/."""
    script = CONVERSOR / "convert_hf_to_gguf.py"
    if script.exists():
        return script
    url = f"https://codeload.github.com/ggml-org/llama.cpp/zip/{LLAMA_CPP_COMMIT}"
    log("  baixando o conversor do llama.cpp (só na primeira vez, ~40 MB)...")
    dados = urllib.request.urlopen(url, timeout=300).read()
    tmp = CONVERSOR.with_name("conversor_gguf_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(dados)) as z:
        raiz = z.namelist()[0].split("/")[0]
        for n in z.namelist():
            rel = n[len(raiz) + 1:]
            if not rel or n.endswith("/"):
                continue
            if rel == "convert_hf_to_gguf.py" or rel.startswith(("conversion/", "gguf-py/")):
                destino = tmp / rel
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(z.read(n))
    shutil.rmtree(CONVERSOR, ignore_errors=True)
    tmp.rename(CONVERSOR)
    return script


def para_gguf(pasta_hf: Path, arquivo: Path, tipo: str = "q8_0", log=print) -> Path:
    script = obter_conversor(log)
    for pacote in ("numpy", "yaml", "tqdm", "requests", "safetensors"):
        try:
            __import__(pacote)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml" if pacote == "yaml" else pacote], check=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CONVERSOR / "gguf-py") + os.pathsep + str(CONVERSOR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    arquivo = Path(arquivo)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), str(pasta_hf), "--outfile", str(arquivo), "--outtype", tipo]
    log(f"  convertendo para .gguf ({tipo})...")
    p = subprocess.run(cmd, env=env, cwd=str(CONVERSOR), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not arquivo.exists():
        raise RuntimeError("o conversor do llama.cpp falhou:\n" + (p.stderr or p.stdout)[-3000:])
    return arquivo


def pasta_lmstudio() -> Path:
    base = Path.home() / ".lmstudio"
    try:
        d = json.loads((base / "settings.json").read_text(encoding="utf-8")).get("downloadsFolder")
        if d:
            return Path(d)
    except Exception:
        pass
    return base / "models"


# ============================================================================ tudo junto

def exportar(pasta_ia: Path, modo: str = "auto", log=print, progresso=lambda f, t: None, parar=lambda: False,
             tipo: str = "q8_0", epocas: int = 3, para_lmstudio: bool = True) -> dict:
    """modo: 'auto' (fusão se der, senão destilação), 'fusao' ou 'destilacao'."""
    pasta_ia = Path(pasta_ia)
    if json.loads((pasta_ia / "receita.json").read_text(encoding="utf-8")).get("tipo") == "cerebro":
        raise ValueError("IAs do modo cérebro ainda não podem ser exportadas (só as costuradas)")
    r = Receita.abrir(pasta_ia)
    t0 = time.time()
    def L(msg):
        log(f"[{(time.time() - t0) / 60:5.1f} min] {msg}")

    pode, motivo = checar_fusao(r)
    if modo == "fusao" and not pode:
        raise ValueError(f"fusão exata não dá para estas partes: {motivo}. Use a destilação.")
    usar_fusao = pode and modo in ("auto", "fusao")
    L(("fusão exata: " if usar_fusao else "destilação: ") + motivo)
    pasta_hf = pasta_ia / "exportado_hf"
    shutil.rmtree(pasta_hf, ignore_errors=True)

    if usar_fusao:
        rng = random.Random(0)
        treino_txt, teste_txt = dados_por_assunto(r, rng)
        ajuste = [t for a in treino_txt for t in treino_txt[a][:60]]
        res = fundir(r, pasta_hf, ajuste, log=L, progresso=progresso)
        # mede: o modelo fundido sozinho, como vai rodar no LM Studio
        from transformers import AutoModelForCausalLM, AutoTokenizer
        progresso(0.85, "medindo o modelo fundido")
        tok = AutoTokenizer.from_pretrained(pasta_hf)
        m = AutoModelForCausalLM.from_pretrained(pasta_hf, dtype=nucleo.DT).to(DEV).eval()
        res["assuntos"] = {}
        for a in teste_txt:
            res["assuntos"][a] = {"exportada": _bpc_textos(tok, m, teste_txt[a], r.max_tok)}
            if r.resultados.get("assuntos", {}).get(a, {}).get("costurada") is not None:
                res["assuntos"][a]["costurada"] = r.resultados["assuntos"][a]["costurada"]
        del m
        nucleo.liberar_cache()
    else:
        res = destilar(pasta_ia, pasta_hf, log=L, progresso=progresso, parar=parar, epocas=epocas)
        for a, v in res["assuntos"].items():
            v["exportada"] = v.pop("aluna")

    progresso(0.92, "convertendo para .gguf")
    nome = _nome_arquivo(r.nome)
    gguf = para_gguf(pasta_hf, pasta_ia / f"{nome}.{tipo}.gguf", tipo, log=L)
    res["gguf"] = str(gguf)
    res["tamanho_mb"] = gguf.stat().st_size / 2 ** 20
    if para_lmstudio:
        destino = pasta_lmstudio() / "costureiro" / nome / gguf.name
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gguf, destino)
        res["lmstudio"] = str(destino)
        L(f"copiado para o LM Studio: {destino}")
    res["minutos"] = (time.time() - t0) / 60
    (pasta_ia / "exportacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    progresso(1.0, "pronto")
    L(f"pronto: {gguf.name} ({res['tamanho_mb']:.0f} MB)")
    return res


def relatorio_exportacao(res: dict) -> str:
    linhas = [f"**Modo:** {res.get('modo')} · **arquivo:** `{Path(res.get('gguf', '?')).name}` "
              f"({res.get('tamanho_mb', 0):.0f} MB) · {res.get('minutos', 0):.0f} min", ""]
    if res.get("lmstudio"):
        linhas += [f"Já está no LM Studio, em **My Models → costureiro**. (`{res['lmstudio']}`)", ""]
    linhas += ["| assunto | IA costurada | **exportada (.gguf)** |", "|---|---|---|"]
    for a, v in res.get("assuntos", {}).items():
        c = v.get("costurada")
        linhas.append(f"| {a} | {c:.3f} | **{v.get('exportada', float('nan')):.3f}** |" if c is not None
                      else f"| {a} | — | **{v.get('exportada', float('nan')):.3f}** |")
    linhas += ["", "Bits por caractere no texto de teste (menor = melhor). A exportada é um modelo comum: "
               "a diferença para a costurada é o que se perde ao caber num .gguf."]
    return "\n".join(linhas)
