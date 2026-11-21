"""
Costura de modelos por mudança de base (experimento do fatiador).

Modelo A (Qwen3-0.6B) lê o texto até a camada k. Uma matriz W "traduz" o vetor do espaço de A
para o espaço de B (SmolLM2-1.7B), e B continua da camada k' até o fim. Mede-se a perplexidade
de B com a costura, comparada com:
  - B inteiro (o melhor possível),
  - B "sem pensamento" (entrada da camada k' trocada pelo vetor médio: o que se perde sem tradutor).

Tradutores testados:
  - mínimos quadrados (afim, com regularização),
  - Procrustes (rotação pura + escala: mudança de base que preserva ângulos e distâncias).

Tokenizers diferentes: para cada token de B usamos o último token de A que termina antes (ou junto)
do fim do token de B -> nada do futuro vaza.

Uso: python costura.py [--a Qwen/Qwen3-0.6B] [--b HuggingFaceTB/SmolLM2-1.7B] [--texto arquivo]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--a", default="Qwen/Qwen3-0.6B")
p.add_argument("--b", default="HuggingFaceTB/SmolLM2-1.7B")
p.add_argument("--texto", default="avaliacao_mista.txt")
p.add_argument("--n_treino", type=int, default=500)
p.add_argument("--n_teste", type=int, default=80)
p.add_argument("--max_tok", type=int, default=192)
p.add_argument("--saida", default="costura_resultado")
p.add_argument("--profundidades", default="0.25,0.4,0.5,0.6,0.75")
p.add_argument("--tiny", action="store_true", help="teste rápido com modelos aleatórios")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
random.seed(0)
torch.manual_seed(0)
SAIDA = Path(args.saida)
SAIDA.mkdir(exist_ok=True)


def carregar(nome):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nome)
    m = AutoModelForCausalLM.from_pretrained(nome, dtype=dt).to(dev).eval()
    return tok, m


def carregar_tiny():
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    amostra = ["O capacitor armazena energia no campo elétrico. A lei de Ohm relaciona tensão e corrente."] * 50
    def treinar(n):
        t = Tokenizer(models.BPE(unk_token="[UNK]"))
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.train_from_iterator(amostra, trainers.BpeTrainer(vocab_size=n, special_tokens=["[UNK]"],
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        return PreTrainedTokenizerFast(tokenizer_object=t)
    ta, tb = treinar(300), treinar(400)
    ma = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(ta), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2)).to(dev).eval()
    mb = LlamaForCausalLM(LlamaConfig(vocab_size=len(tb), hidden_size=96, intermediate_size=192, num_hidden_layers=6,
                                      num_attention_heads=4, num_key_value_heads=2)).to(dev).eval()
    return (ta, ma), (tb, mb)


def camadas(m):
    return m.model.layers


def trechos(texto, n):
    palavras = texto.split()
    out = []
    i = 0
    while i < len(palavras) and len(out) < n:
        tam = random.randint(90, 140)
        out.append(" ".join(palavras[i:i + tam]))
        i += tam + random.randint(0, 40)
    return out


def tokenizar(tok, s, limite_chars=None):
    e = tok(s, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = e["input_ids"], e["offset_mapping"]
    return ids, offs


@torch.no_grad()
def estados(m, ids):
    x = torch.tensor([ids], device=dev)
    o = m(x, output_hidden_states=True)
    return [h[0].float() for h in o.hidden_states]  # hidden_states[i] = entrada da camada i


def alinhar(offs_a, offs_b):
    """para cada token de B: índice do último token de A com fim <= fim do token de B (ou -1)."""
    fins_a = [e for _, e in offs_a]
    idx, j = [], -1
    for _, fim_b in offs_b:
        while j + 1 < len(fins_a) and fins_a[j + 1] <= fim_b:
            j += 1
        idx.append(j)
    return idx


def preparar(tok_a, tok_b, s):
    ids_b, offs_b = tokenizar(tok_b, s)
    ids_b, offs_b = ids_b[:args.max_tok], offs_b[:args.max_tok]
    fim = offs_b[-1][1]
    ids_a, offs_a = tokenizar(tok_a, s[:fim])
    al = alinhar(offs_a, offs_b)
    return ids_a, ids_b, al


def main():
    t0 = time.time()
    if args.tiny:
        (ta, ma), (tb, mb) = carregar_tiny()
        texto = ("O capacitor armazena energia no campo elétrico. A lei de Ohm relaciona tensão e corrente. " * 400)
    else:
        ta, ma = carregar(args.a)
        tb, mb = carregar(args.b)
        texto = Path(args.texto).read_text(encoding="utf-8", errors="replace")
    LA, LB = len(camadas(ma)), len(camadas(mb))
    print(f"A={args.a} ({LA} camadas, d={ma.config.hidden_size})  B={args.b} ({LB} camadas, d={mb.config.hidden_size})", flush=True)

    todos = trechos(texto, args.n_treino + args.n_teste)
    random.shuffle(todos)
    n_te = min(args.n_teste, max(20, len(todos) // 5))  # o texto pode render menos trechos que o pedido
    treino, teste = todos[:-n_te], todos[-n_te:]
    profs = [float(x) for x in args.profundidades.split(",")]
    pares = [(max(1, round(r * LA)), max(1, round(r * LB)), r) for r in profs]

    # ---------------- coleta dos pares (vetor de A, vetor de B) para cada costura
    X = {i: [] for i in range(len(pares))}
    Y = {i: [] for i in range(len(pares))}
    for n, s in enumerate(treino):
        ids_a, ids_b, al = preparar(ta, tb, s)
        if len(ids_b) < 8 or len(ids_a) < 4:
            continue
        ha, hb = estados(ma, ids_a), estados(mb, ids_b)
        ok = torch.tensor([i for i, a in enumerate(al) if a >= 0], device=dev)
        ia = torch.tensor([al[i] for i in ok.tolist()], device=dev)
        for i, (k, k2, _) in enumerate(pares):
            X[i].append(ha[k][ia].cpu())
            Y[i].append(hb[k2][ok].cpu())
        if n % 100 == 0:
            print(f"  coletando {n}/{len(treino)}  ({time.time() - t0:.0f}s)", flush=True)

    tradutores = {}
    for i, (k, k2, r) in enumerate(pares):
        x = torch.cat(X[i]).to(dev)
        y = torch.cat(Y[i]).to(dev)
        mx, my = x.mean(0), y.mean(0)
        xc, yc = x - mx, y - my
        # mínimos quadrados com ridge
        lam = 1e-2 * xc.pow(2).sum(0).mean()
        A = xc.T @ xc + lam * torch.eye(x.shape[1], device=dev)
        W_ls = torch.linalg.solve(A, xc.T @ yc)
        # Procrustes (semi-ortogonal: dA -> dB) + escala
        U, S, Vh = torch.linalg.svd(xc.T @ yc, full_matrices=False)
        W_pr = U @ Vh
        esc = (S.sum() / xc.pow(2).sum()).item()
        # qualidade do ajuste (R² no treino)
        def r2(W, e=1.0):
            pred = e * (xc @ W)
            return 1 - (pred - yc).pow(2).sum().item() / yc.pow(2).sum().item()
        tradutores[(k, k2)] = {"mx": mx, "my": my, "ls": W_ls, "pr": W_pr, "esc": esc}
        print(f"  costura A{k}->B{k2} (prof {r}): {x.shape[0]} pares, R² mín.quadrados={r2(W_ls):.3f}  Procrustes={r2(W_pr, esc):.3f}", flush=True)
    del X, Y

    # ---------------- avaliação
    @torch.no_grad()
    def perda_b(ids_b, k2=None, injetar=None):
        h = []
        if k2 is not None:
            def gancho(mod, a, kw):
                hs = kw.get("hidden_states", a[0] if a else None)
                novo = injetar.to(hs.dtype).unsqueeze(0)
                if "hidden_states" in kw:
                    kw["hidden_states"] = novo
                    return a, kw
                return (novo,) + tuple(a[1:]), kw
            h.append(camadas(mb)[k2].register_forward_pre_hook(gancho, with_kwargs=True))
        try:
            x = torch.tensor([ids_b], device=dev)
            logits = mb(x).logits[0, :-1].float()
            alvo = x[0, 1:]
            return torch.nn.functional.cross_entropy(logits, alvo, reduction="sum").item(), len(alvo)
        finally:
            for g in h:
                g.remove()

    res = {"B inteiro": [0.0, 0]}
    for k, k2, r in pares:
        for nome in (f"{r}|costura mín.quadrados", f"{r}|costura Procrustes", f"{r}|sem tradutor (vetor médio)"):
            res[nome] = [0.0, 0]
    for n, s in enumerate(teste):
        ids_a, ids_b, al = preparar(ta, tb, s)
        if len(ids_b) < 8 or len(ids_a) < 4:
            continue
        l, c = perda_b(ids_b)
        res["B inteiro"][0] += l; res["B inteiro"][1] += c
        ha, hb = estados(ma, ids_a), estados(mb, ids_b)
        for k, k2, r in pares:
            T = tradutores[(k, k2)]
            ia = torch.tensor([max(a, 0) for a in al], device=dev)
            xa = ha[k][ia] - T["mx"]
            base = hb[k2].clone()  # tokens sem par em A (início) ficam com o vetor do próprio B
            sem_par = torch.tensor([a < 0 for a in al], device=dev)
            for nome, inj in ((f"{r}|costura mín.quadrados", xa @ T["ls"] + T["my"]),
                              (f"{r}|costura Procrustes", T["esc"] * (xa @ T["pr"]) + T["my"]),
                              (f"{r}|sem tradutor (vetor médio)", T["my"].expand_as(base).clone())):
                inj = torch.where(sem_par.unsqueeze(1), base, inj)
                l, c = perda_b(ids_b, k2, inj)
                res[nome][0] += l; res[nome][1] += c
        if n % 20 == 0:
            print(f"  avaliando {n}/{len(teste)}", flush=True)

    ppl = {k: math.exp(v[0] / max(v[1], 1)) for k, v in res.items()}
    base = ppl["B inteiro"]
    linhas = ["# Costura de modelos por mudança de base", "",
              f"A = `{args.a}` ({LA} camadas) → tradutor W → B = `{args.b}` ({LB} camadas)",
              f"Texto: {args.texto}, {len(treino)} trechos para achar W, {len(teste)} para medir.", "",
              f"**B inteiro: perplexidade {base:.2f}** (referência)", "",
              "| profundidade | camada A → B | mínimos quadrados | Procrustes (rotação) | sem tradutor |",
              "|---|---|---|---|---|"]
    tabela = []
    for k, k2, r in pares:
        a, b, c = ppl[f"{r}|costura mín.quadrados"], ppl[f"{r}|costura Procrustes"], ppl[f"{r}|sem tradutor (vetor médio)"]
        tabela.append({"prof": r, "camada_a": k, "camada_b": k2, "ls": a, "procrustes": b, "sem": c})
        linhas.append(f"| {r:.2f} | {k} → {k2} | {a:.2f} | {b:.2f} | {c:.2f} |")
    linhas += ["", f"Tempo total: {(time.time() - t0) / 60:.1f} min."]
    (SAIDA / "resultado.md").write_text("\n".join(linhas), encoding="utf-8")
    (SAIDA / "resultado.json").write_text(json.dumps({"a": args.a, "b": args.b, "b_inteiro": base, "costuras": tabela},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(linhas), flush=True)

    # exemplo de texto gerado pela costura do meio (greedy), para ver na prática
    try:
        k, k2, r = pares[len(pares) // 2]
        T = tradutores[(k, k2)]
        prompt = "A lei de Ohm diz que"
        s = prompt
        for _ in range(25):
            ids_a, ids_b, al = preparar(ta, tb, s)
            ha = estados(ma, ids_a)
            ia = torch.tensor([max(a, 0) for a in al], device=dev)
            inj = T["esc"] * ((ha[k][ia] - T["mx"]) @ T["pr"]) + T["my"]
            hb0 = estados(mb, ids_b)[k2]
            sem_par = torch.tensor([a < 0 for a in al], device=dev)
            inj = torch.where(sem_par.unsqueeze(1), hb0, inj)
            captura = {}
            def gancho(mod, a, kw):
                hs = kw.get("hidden_states", a[0] if a else None)
                novo = inj.to(hs.dtype).unsqueeze(0)
                if "hidden_states" in kw:
                    kw["hidden_states"] = novo
                    return a, kw
                return (novo,) + tuple(a[1:]), kw
            g = camadas(mb)[k2].register_forward_pre_hook(gancho, with_kwargs=True)
            with torch.no_grad():
                prox = mb(torch.tensor([ids_b], device=dev)).logits[0, -1].argmax().item()
            g.remove()
            s += tb.decode([prox])
        (SAIDA / "exemplo.txt").write_text(s, encoding="utf-8")
        print("\nExemplo (Procrustes, prof", r, "):", s, flush=True)
    except Exception as e:
        print("exemplo falhou:", e)


if __name__ == "__main__":
    main()
