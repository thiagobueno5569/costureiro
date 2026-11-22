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

import torch
import torch.nn as nn

import nucleo
from nucleo import DEV, IACosturada, Receita, dados_por_assunto, _bpc_textos

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
    """A fusão exata só funciona se as partes têm a mesma arquitetura, dimensão, RoPE e tokenizador."""
    import fusao
    return fusao.checar([p.modelo for p in receita.partes])


def fundir(receita: Receita, pasta_hf: Path, textos_ajuste: list[str], log=print, progresso=lambda f, t: None) -> dict:
    """Fusão exata a partir de uma receita (o trabalho pesado está em fusao.py)."""
    import fusao
    nucleo.liberar_cache()
    res = fusao.fundir([fusao.Peca(p.modelo, p.inicio, p.fim) for p in receita.partes], pasta_hf, textos_ajuste,
                       log=log, progresso=progresso, max_tok=max(receita.max_tok, 128))
    res.pop("_ids_medir", None)
    res["r2_juncoes"] = [j["r2"] for j in res["juncoes"]]
    return res


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
    for pacote in ("numpy", "yaml", "tqdm", "requests", "safetensors", "sentencepiece", "google.protobuf"):
        try:
            __import__(pacote)
        except ImportError:
            nome = {"yaml": "pyyaml", "google.protobuf": "protobuf"}.get(pacote, pacote)
            subprocess.run([sys.executable, "-m", "pip", "install", nome], check=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CONVERSOR / "gguf-py") + os.pathsep + str(CONVERSOR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    pasta_hf = Path(pasta_hf).resolve()       # o conversor roda em outra pasta: caminhos relativos quebrariam
    arquivo = Path(arquivo).resolve()
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
