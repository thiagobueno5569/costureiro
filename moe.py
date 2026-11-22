"""IA única em forma de MoE: a "parte de conhecimento" (FFN) de cada modelo vira um ESPECIALISTA, em
todas as camadas, e um roteador escolhe, palavra por palavra e camada por camada, quanto usar de cada um.

    Coder-1.5B  ─┐                          camada l:  atenção (do Coder)
    Math-1.5B   ─┼─►  um modelo só (MoE)  ─►           roteador → FFN do Coder · FFN do Math
                 ┘                                     (os dois, misturados pelos pesos do roteador)

O resultado é um modelo `qwen2_moe` comum: abre no Transformers e vira .gguf (LM Studio, Ollama, llama.cpp).

Regras:
  * os modelos precisam ter a mesma "anatomia" (dimensão, nº de camadas, tamanho da FFN, cabeças) e o
    mesmo tokenizador — na prática, versões do mesmo modelo-base (ex.: Qwen2.5-Coder-1.5B e Qwen2.5-Math-1.5B).
    A RoPE pode ser diferente: atenção e posições vêm do modelo BASE, os especialistas só trazem a FFN.
  * a norma antes da FFN é a do base; para cada especialista de outro modelo dobramos a diferença de ganho
    nos pesos de entrada (W·diag(g_dele/g_base)), então ele recebe a entrada na escala em que foi treinado.
  * o roteador começa sabendo os assuntos: em cada camada, a direção média do pensamento em textos de cada
    assunto vira a linha do roteador do especialista daquele assunto. Depois ele é treinado (só ele: ~2×d
    números por camada) com textos misturados.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

import fusao
from fusao import Escritor, Leitor, pasta_do_modelo

AQUI = Path(__file__).parent
CAMPOS = ("hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads", "num_key_value_heads",
          "vocab_size", "hidden_act")


def checar(modelos: list[str]) -> tuple[bool, str]:
    cfgs = [fusao._cfg_texto(fusao._cfg(m)) for m in modelos]
    if cfgs[0].get("model_type") not in ("qwen2",):
        return False, f"por enquanto o MoE é montado a partir de modelos qwen2 (veio {cfgs[0].get('model_type')})"
    for m, c in zip(modelos[1:], cfgs[1:]):
        if c.get("model_type") != cfgs[0].get("model_type"):
            return False, f"arquiteturas diferentes ({m})"
        for k in CAMPOS:
            if c.get(k) != cfgs[0].get(k):
                return False, f"{m}: {k} = {c.get(k)} × {cfgs[0].get(k)}"
    from transformers import AutoTokenizer
    v0 = AutoTokenizer.from_pretrained(str(pasta_do_modelo(modelos[0]))).get_vocab()
    for m in modelos[1:]:
        if AutoTokenizer.from_pretrained(str(pasta_do_modelo(m))).get_vocab() != v0:
            return False, f"tokenizadores diferentes ({m})"
    return True, "mesma anatomia e mesmo tokenizador (a RoPE pode ser diferente: atenção vem do modelo base)"


@torch.no_grad()
def direcoes_por_assunto(base: str, textos: dict[str, list[str]], max_tok=256, n=60, dispositivo="auto"):
    """Média do vetor que ENTRA na FFN (depois da norma) de cada camada, por assunto."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(pasta_do_modelo(base)))
    m = fusao.carregar_para_medir(base, dispositivo)
    L = fusao._camadas(m)
    medias = {}
    for assunto, lista in textos.items():
        soma, cont = [None] * len(L), 0
        guardado = {}
        ganchos = [L[i].post_attention_layernorm.register_forward_hook(
            lambda mod, a, o, i=i: guardado.__setitem__(i, o[0].float().mean(0).cpu())) for i in range(len(L))]
        try:
            for t in lista[:n]:
                ids = tok(t, add_special_tokens=False)["input_ids"][:max_tok]
                if len(ids) < 8:
                    continue
                m(input_ids=torch.tensor([ids], device=fusao._disp(m)))
                for i in range(len(L)):
                    soma[i] = guardado[i] if soma[i] is None else soma[i] + guardado[i]
                cont += 1
        finally:
            for g in ganchos:
                g.remove()
        medias[assunto] = torch.stack([s / max(cont, 1) for s in soma])        # (camadas, d)
    del m
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return medias


def roteador_inicial(medias: dict[str, torch.Tensor], ordem: list[str], gap: float = 2.0) -> torch.Tensor:
    """(camadas, n_especialistas, d): linha j = direção do assunto j (menos a média), escalada para que, na média
    do próprio assunto, o especialista certo tenha `gap` de vantagem no logit."""
    M = torch.stack([medias[a] for a in ordem], 1)                   # (L, E, d)
    centro = M.mean(1, keepdim=True)
    U = M - centro
    U = U / (U.norm(dim=-1, keepdim=True) + 1e-6)
    logits = torch.einsum("led,lkd->lek", U, M)                       # logit do especialista e na média do assunto k
    E = len(ordem)
    if E > 1:
        certo = logits.diagonal(dim1=1, dim2=2)                       # (L, E)
        outros = (logits.sum(1) - certo) / (E - 1)
        vant = (certo - outros).mean(1, keepdim=True).clamp(min=1e-3)  # (L, 1)
        U = U * (gap / vant)[:, :, None]
    return U


def montar(base: str, outros: list[str], assuntos_dos: list[str], saida: Path, textos: dict[str, list[str]],
           topk: int = 2, log=print, progresso=lambda f, t: None, dispositivo="auto") -> dict:
    """base: o modelo que dá atenção, normas, embeddings e cabeça (e é o especialista 0).
    outros: modelos que entram só com a FFN (especialistas 1..). assuntos_dos[j] = assunto do especialista j."""
    modelos = [base] + list(outros)
    pode, motivo = checar(modelos)
    if not pode:
        raise ValueError(f"não dá para montar o MoE: {motivo}")
    log(f"  {motivo}")
    cfg = fusao._cfg(base)
    n_camadas, d, ff = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]

    progresso(0.05, "roteador: direção de cada assunto em cada camada")
    medias = direcoes_por_assunto(base, {a: textos[a] for a in dict.fromkeys(assuntos_dos)}, dispositivo=dispositivo)
    W_rot = roteador_inicial(medias, assuntos_dos)                     # (L, E, d)

    progresso(0.3, "montando os pesos")
    saida = Path(saida)
    import shutil
    shutil.rmtree(saida, ignore_errors=True)
    esc = Escritor(saida)
    leitores = [Leitor(m) for m in modelos]
    pre = [fusao._prefixo(l) for l in leitores]
    lb, pb = leitores[0], pre[0]
    for k in lb.chaves():
        if not k.startswith(pb + "layers."):
            esc.add(k, lb.get(k))
    if not lb.tem("lm_head.weight"):          # o llama.cpp exige a cabeça separada no qwen2moe (não aceita "amarrada")
        esc.add("lm_head.weight", lb.get(next(k for k in lb.chaves() if k.endswith("embed_tokens.weight"))).clone())
    SH = 32                                                           # especialista compartilhado "vazio" (o formato exige)
    for l in range(n_camadas):
        pv = f"{pb}layers.{l}."
        g_base = lb.get(pv + "post_attention_layernorm.weight").float()
        for k in lb.chaves():
            if k.startswith(pv) and ".mlp." not in k:
                esc.add(f"model.layers.{l}." + k[len(pv):], lb.get(k))
        dt = lb.get(pv + "mlp.gate_proj.weight").dtype
        for j, (lei, p) in enumerate(zip(leitores, pre)):
            pj = f"{p}layers.{l}.mlp."
            g = lei.get(f"{p}layers.{l}.post_attention_layernorm.weight").float()
            ajuste = (g / torch.where(g_base.abs() < 1e-4, torch.full_like(g_base, 1e-4), g_base))
            for nome in ("gate_proj", "up_proj"):
                w = lei.get(pj + f"{nome}.weight").float()
                esc.add(f"model.layers.{l}.mlp.experts.{j}.{nome}.weight", (w * ajuste[None, :] if j else w).to(dt))
            esc.add(f"model.layers.{l}.mlp.experts.{j}.down_proj.weight", lei.get(pj + "down_proj.weight"))
        esc.add(f"model.layers.{l}.mlp.gate.weight", W_rot[l].to(dt))
        esc.add(f"model.layers.{l}.mlp.shared_expert.gate_proj.weight", torch.zeros(SH, d, dtype=dt))
        esc.add(f"model.layers.{l}.mlp.shared_expert.up_proj.weight", torch.zeros(SH, d, dtype=dt))
        esc.add(f"model.layers.{l}.mlp.shared_expert.down_proj.weight", torch.zeros(d, SH, dtype=dt))
        esc.add(f"model.layers.{l}.mlp.shared_expert_gate.weight", torch.full((1, d), 0.0, dtype=dt))
        if l % 4 == 0:
            progresso(0.3 + 0.4 * l / n_camadas, f"montando a camada {l + 1} de {n_camadas}")
    esc.fechar()
    del leitores
    gc.collect()

    novo = {k: v for k, v in cfg.items() if k not in ("architectures", "model_type", "auto_map", "max_window_layers",
                                                      "use_sliding_window", "sliding_window", "layer_types")}
    novo.update({"architectures": ["Qwen2MoeForCausalLM"], "model_type": "qwen2_moe", "num_experts": len(modelos),
                 "num_experts_per_tok": min(topk, len(modelos)), "moe_intermediate_size": ff,
                 "shared_expert_intermediate_size": SH, "norm_topk_prob": True, "decoder_sparse_step": 1,
                 "mlp_only_layers": [], "qkv_bias": True, "output_router_logits": False,
                 "use_sliding_window": False, "tie_word_embeddings": False})
    (saida / "config.json").write_text(json.dumps(novo, indent=2), encoding="utf-8")
    origem = pasta_do_modelo(base)
    for f in origem.iterdir():
        if f.is_file() and (f.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "generation_config"))
                            or f.suffix in (".jinja",)):
            shutil.copy2(f, saida / f.name)
    return {"modo": "MoE", "especialistas": [{"modelo": m, "assunto": a} for m, a in zip(modelos, assuntos_dos)],
            "topk": topk}


def treinar_roteador(pasta: Path, textos: dict[str, list[str]], passos: int = 600, lr: float = 3e-3, max_tok: int = 512,
                     log=print, progresso=lambda f, t: None, dtype=None) -> dict:
    """Treina SÓ os roteadores (o resto fica congelado) com textos misturados de todos os assuntos."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = dtype or (torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32))
    tok = AutoTokenizer.from_pretrained(pasta)
    m = AutoModelForCausalLM.from_pretrained(pasta, dtype=dt).to(dev)
    params, ganchos = [], []

    def _entrada(mod, args):              # o roteador treina em float32; o resto do modelo segue em bf16/fp16
        mod._dt_entrada = args[0].dtype
        return (args[0].to(mod.weight.dtype),) + tuple(args[1:])

    def _saida(mod, args, out):
        volta = lambda t: t.to(mod._dt_entrada) if torch.is_tensor(t) and t.is_floating_point() else t
        return tuple(volta(t) for t in out) if isinstance(out, tuple) else volta(out)

    for n, p in m.named_parameters():
        treina = n.endswith("mlp.gate.weight")
        p.requires_grad_(treina)
        if treina:
            p.data = p.data.float()
            params.append(p)
            mod = m.get_submodule(n[: -len(".weight")])
            ganchos += [mod.register_forward_pre_hook(_entrada), mod.register_forward_hook(_saida)]
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    rng = random.Random(0)
    todos = [t for a in textos for t in textos[a]]
    rng.shuffle(todos)
    n_te = min(60, len(todos) // 10)
    teste, treino = todos[:n_te], todos[n_te:]

    def ids_de(t):
        return tok(t, add_special_tokens=False)["input_ids"][:max_tok]

    @torch.no_grad()
    def medir():
        m.eval()
        nll, n = 0.0, 0
        for t in teste:
            ids = ids_de(t)
            if len(ids) < 8:
                continue
            x = torch.tensor([ids], device=dev)
            nll += m(input_ids=x, labels=x).loss.item() * (len(ids) - 1)
            n += len(ids) - 1
        m.train()
        return nll / max(n, 1)

    antes = medir()
    log(f"  roteador: perda no teste antes {antes:.4f}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    m.train()
    melhor, estado = antes, {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    for k in range(passos):
        ids = ids_de(treino[k % len(treino)])
        if len(ids) < 8:
            continue
        x = torch.tensor([ids], device=dev)
        perda = m(input_ids=x, labels=x).loss
        perda.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (k + 1) % 100 == 0 or k + 1 == passos:
            p_te = medir()
            marca = ""
            if p_te < melhor:
                melhor, marca = p_te, "  ← melhor"
                estado = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
            log(f"  roteador: passo {k + 1}/{passos} · perda no teste {p_te:.4f}{marca}")
            progresso(0.72 + 0.2 * (k + 1) / passos, f"treinando o roteador: passo {k + 1} de {passos}")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in estado:
                p.copy_(estado[n])
    m.gradient_checkpointing_disable()
    for g in ganchos:
        g.remove()
    for p in params:
        p.data = p.data.to(dt)
    m.save_pretrained(pasta, safe_serialization=True)
    del m
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"perda_antes": antes, "perda_depois": melhor}


def textos_padrao(rng=None) -> dict[str, list[str]]:
    """Textos de tarefa (treino do MBPP e do GSM8K, nunca o teste) + código real."""
    import orquestra
    return orquestra.dados_de_tarefa(rng or random.Random(0))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Costureiro · monta uma IA única em forma de MoE (FFN de cada modelo = um especialista)")
    ap.add_argument("--base", required=True, help="modelo que dá a atenção (e é o especialista do 1º assunto)")
    ap.add_argument("--assunto-base", default="programação")
    ap.add_argument("--especialista", action="append", required=True, help="modelo=assunto (repita)")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--passos-roteador", type=int, default=600)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--gguf", default="q8_0")
    a = ap.parse_args(argv)
    outros, assuntos = [], [a.assunto_base]
    for e in a.especialista:
        mdl, ass = e.rsplit("=", 1)
        outros.append(mdl); assuntos.append(ass)
    t0 = time.time()
    textos = textos_padrao()
    saida = Path(a.saida)
    res = montar(a.base, outros, assuntos, saida, textos, topk=a.topk, log=lambda m: print(m, flush=True))
    if a.passos_roteador > 0:
        res["roteador"] = treinar_roteador(saida, textos, passos=a.passos_roteador, log=lambda m: print(m, flush=True))
    res["minutos"] = (time.time() - t0) / 60
    (saida / "moe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.gguf and a.gguf != "nao":
        import exportar
        arq = exportar.para_gguf(saida, saida.parent / f"{saida.name}.{a.gguf}.gguf", a.gguf)
        print(f".gguf: {arq} ({arq.stat().st_size / 2 ** 20:.0f} MB)")


if __name__ == "__main__":
    main()
