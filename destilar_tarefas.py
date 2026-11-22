"""IA única por destilação com respostas CONFERIDAS.

Professores resolvem os problemas de TREINO (nunca os de teste) e só ficam as respostas certas:
  * matemática: o professor (ex.: Qwen2.5-Math-1.5B ou 7B) resolve o GSM8K-treino no formato de chat dele;
    fica só quem acertou o número final do gabarito;
  * código: o professor (ex.: o próprio Coder) escreve soluções do MBPP-treino; fica só o que passa nos testes.
Depois o ALUNO (ex.: Coder-1.5B) aprende essas respostas com LoRA (só nas palavras da resposta), a LoRA é
somada aos pesos e sai um modelo comum → .gguf. O professor pode ser bem maior que o aluno: ele só é usado
no treino.

    python destilar_tarefas.py --aluno Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --prof-matematica Qwen/Qwen2.5-Math-1.5B-Instruct --prof-codigo Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --saida fusoes/destilada
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import avaliar
import orquestra

AQUI = Path(__file__).parent
SISTEMA_MATH = "Please reason step by step, and put your final answer within \\boxed{}."


def _dev_dt():
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return "cpu", torch.float32


def carregar(nome):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dt = _dev_dt()
    tok = AutoTokenizer.from_pretrained(nome)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(nome, dtype=dt).to(dev).eval()
    return tok, m


@torch.no_grad()
def gerar(tok, m, prompts: list[str], max_novos: int, paradas: list[str], lote: int = 48, temperatura: float = 0.0,
          log=print, rotulo="") -> list[str]:
    out = []
    t0 = time.time()
    for i in range(0, len(prompts), lote):
        x = tok(prompts[i:i + lote], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        o = m.generate(**x, max_new_tokens=max_novos, do_sample=temperatura > 0, temperature=max(temperatura, 1e-5),
                       top_p=0.95 if temperatura > 0 else None, pad_token_id=tok.pad_token_id)
        for seq in o[:, x["input_ids"].shape[1]:]:
            t = tok.decode(seq, skip_special_tokens=True)
            cortes = [t.find(p) for p in paradas if t.find(p) >= 0]
            out.append(t[:min(cortes)] if cortes else t)
        if (i // lote) % 5 == 0:
            log(f"  {rotulo} {min(i + lote, len(prompts))}/{len(prompts)} ({(time.time() - t0) / 60:.1f} min)")
    return out


def _resposta_boxed(t: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", t)
    if m:
        return m[-1]
    return avaliar.numero_final(t)


def _numero(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None


def solucoes_matematica(prof: str, n: int, rng: random.Random, log=print, lote=48) -> list[str]:
    import pandas as pd
    df = pd.read_parquet(orquestra._baixar("https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/"
                                           "train-00000-of-00001.parquet", "gsm8k_train.parquet"))
    itens = list(zip(df["question"], df["answer"]))
    rng.shuffle(itens)
    itens = itens[:n]
    tok, m = carregar(prof)
    prompts = []
    for q, _ in itens:
        msgs = [{"role": "system", "content": SISTEMA_MATH}, {"role": "user", "content": q}]
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        except Exception:
            prompts.append(f"Problem: {q}\nSolution:")
    resp = gerar(tok, m, prompts, 512, ["<|im_end|>", "\n\nProblem:"], lote=lote, log=log, rotulo=f"matemática ({Path(prof).name})")
    del m
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    bons = []
    for (q, a), r in zip(itens, resp):
        gab = _numero(a.split("####")[-1])
        dada = _numero(_resposta_boxed(r))
        if gab is None or dada is None or abs(gab - dada) > 1e-6:
            continue
        sol = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", r).strip()
        final = a.split("####")[-1].strip()
        bons.append(f"Problem: {q}\nSolution: {sol}\nAnswer: {final}")
    log(f"  matemática: {len(bons)} de {len(itens)} soluções certas ({100 * len(bons) / max(1, len(itens)):.0f}%)")
    return bons


def solucoes_codigo(prof: str, rng: random.Random, amostras: int = 4, log=print, lote=48) -> list[str]:
    import pandas as pd
    itens = []
    for parte in ("train", "validation", "prompt"):
        df = pd.read_parquet(orquestra._baixar(f"https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/"
                                               f"full/{parte}-00000-of-00001.parquet", f"mbpp_{parte}.parquet"))
        for r in df.to_dict("records"):
            testes = list(r["test_list"])
            itens.append({"prompt": f'"""\n{r["text"]}\n{testes[0]}\n"""\n',
                          "teste": "\n".join([r.get("test_setup_code") or ""] + testes)})
    tok, m = carregar(prof)
    prompts = [it["prompt"] for it in itens]
    candidatos = [gerar(tok, m, prompts, 384, avaliar.PARADAS["mbpp"], lote=lote, log=log, rotulo="código (guloso)")]
    for _ in range(max(0, amostras - 1)):
        candidatos.append(gerar(tok, m, prompts, 384, avaliar.PARADAS["mbpp"], lote=lote, temperatura=0.8, log=log,
                                rotulo="código (amostra)"))
    del m
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    tarefas = [(k, c) for c in candidatos for k in range(len(itens))]
    with ThreadPoolExecutor(max_workers=32) as ex:
        ok = list(ex.map(lambda kc: avaliar.rodar_codigo(kc[1][kc[0]] + "\n\n" + itens[kc[0]]["teste"] + "\n", 10), tarefas))
    bons, vistos = [], set()
    for (k, c), passou in zip(tarefas, ok):
        sol = c[k].strip()
        if passou and (k, sol) not in vistos:
            vistos.add((k, sol))
            bons.append(itens[k]["prompt"] + c[k].rstrip() + "\n")
    resolvidos = len({k for k, _ in vistos})
    log(f"  código: {resolvidos} de {len(itens)} problemas resolvidos ({len(bons)} soluções diferentes que passam)")
    return bons


def treinar_aluno(aluno: str, exemplos: list[tuple[str, str]], saida: Path, epocas: int = 2, lr: float = 2e-4,
                  rank: int = 32, max_tok: int = 768, log=print) -> dict:
    """exemplos: (prompt, resposta). Perda só nas palavras da resposta."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, dt = _dev_dt()
    tok = AutoTokenizer.from_pretrained(aluno)
    m = AutoModelForCausalLM.from_pretrained(aluno, dtype=dt).to(dev)
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in m.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    passos = epocas * len(exemplos)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(passos, 2), pct_start=0.03)
    rng = random.Random(0)
    t0, feitos, acc = time.time(), 0, 0.0
    m.train()
    for ep in range(epocas):
        rng.shuffle(exemplos)
        for p_txt, r_txt in exemplos:
            ip = tok(p_txt, add_special_tokens=False)["input_ids"]
            ir = tok(r_txt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (ip + ir)[:max_tok]
            alvo = ([-100] * len(ip) + ir)[:max_tok]
            if sum(1 for a in alvo if a != -100) < 2:
                continue
            x = torch.tensor([ids], device=dev)
            y = torch.tensor([alvo], device=dev)
            with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
                perda = m(input_ids=x, labels=y).loss
            perda.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            feitos += 1; acc += perda.item()
            if feitos % 200 == 0:
                log(f"  aluno: passo {feitos}/{passos} · perda {acc / 200:.3f} · {(time.time() - t0) / 60:.0f} min")
                acc = 0.0
    m = m.merge_and_unload()
    m.gradient_checkpointing_disable()
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    m = m.to(dt)
    m.save_pretrained(saida, safe_serialization=True)
    tok.save_pretrained(saida)
    del m
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"exemplos": len(exemplos), "epocas": epocas, "minutos_treino": (time.time() - t0) / 60}


def destilar(aluno: str, profs_mat: list[str], prof_cod: str | None, saida: Path, n_mat: int = 3000, amostras_cod: int = 4,
             epocas: int = 2, log=print) -> dict:
    rng = random.Random(0)
    t0 = time.time()
    mat = []
    for p in profs_mat:
        mat += solucoes_matematica(p, n_mat, random.Random(len(mat)), log=log)
    cod = solucoes_codigo(prof_cod, rng, amostras_cod, log=log) if prof_cod else []
    exemplos = []
    for t in mat:
        p, r = t.split("\nSolution:", 1)
        exemplos.append((p + "\nSolution:", r))
    for t in cod:
        i = t.index('"""\n', 4) + 4
        exemplos.append((t[:i], t[i:]))
    Path(saida).mkdir(parents=True, exist_ok=True)
    (Path(saida) / "exemplos.jsonl").write_text("\n".join(json.dumps({"prompt": p, "resposta": r}, ensure_ascii=False)
                                                         for p, r in exemplos), encoding="utf-8")
    log(f"  {len(exemplos)} exemplos conferidos ({len(mat)} de matemática, {len(cod)} de código); treinando o aluno")
    res = treinar_aluno(aluno, exemplos, Path(saida), epocas=epocas, log=log)
    res.update({"modo": "destilação conferida", "aluno": aluno, "prof_matematica": profs_mat, "prof_codigo": prof_cod,
                "n_matematica": len(mat), "n_codigo": len(cod), "minutos": (time.time() - t0) / 60})
    (Path(saida) / "destilacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Costureiro · IA única por destilação com respostas conferidas")
    ap.add_argument("--aluno", required=True)
    ap.add_argument("--prof-matematica", action="append", default=[])
    ap.add_argument("--prof-codigo", default=None)
    ap.add_argument("--n-matematica", type=int, default=3000)
    ap.add_argument("--amostras-codigo", type=int, default=4)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--gguf", default="q8_0")
    a = ap.parse_args(argv)
    res = destilar(a.aluno, a.prof_matematica, a.prof_codigo, Path(a.saida), a.n_matematica, a.amostras_codigo, a.epocas,
                   log=lambda m: print(m, flush=True))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.gguf and a.gguf != "nao":
        import exportar
        arq = exportar.para_gguf(Path(a.saida), Path(a.saida).parent / f"{Path(a.saida).name}.{a.gguf}.gguf", a.gguf)
        print(f".gguf: {arq}")


if __name__ == "__main__":
    main()
