"""
Costura treinada ("super agente tradutor").

Igual ao costura.py, mas o tradutor é TREINADO por gradiente para minimizar a perplexidade do
modelo B costurado (A e B ficam congelados; só o tradutor aprende).

Tradutores comparados (na mesma costura):
  1. Procrustes cru            (o melhor do experimento anterior)
  2. Procrustes normalizado    (z-score por dimensão antes/depois: tira o peso das dimensões gigantes)
  3. linear treinado           (parte do 2 e ajusta W por gradiente)
  4. linear + MLP treinado     (3 + uma rede pequena residual: mudança de base não linear)

Uso: python treinar_costura.py --texto arquivo [--prof 0.4] [--epocas 4]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--a", default="Qwen/Qwen3-0.6B")
p.add_argument("--b", default="HuggingFaceTB/SmolLM2-1.7B")
p.add_argument("--texto", default="avaliacao_mista.txt")
p.add_argument("--profs", default="0.4,0.6")
p.add_argument("--epocas", type=int, default=4)
p.add_argument("--max_tok", type=int, default=192)
p.add_argument("--saida", default="costura_treinada")
p.add_argument("--tiny", action="store_true")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
random.seed(0)
torch.manual_seed(0)
SAIDA = Path(args.saida)
SAIDA.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


# ------------------------------------------------------------------ modelos e dados

def carregar(nome):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(nome)
    m = AutoModelForCausalLM.from_pretrained(nome, dtype=dt).to(dev).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return tok, m


def carregar_tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
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
    for m in (ma, mb):
        for q in m.parameters():
            q.requires_grad_(False)
    return (ta, ma), (tb, mb)


def trechos(texto):
    pal = texto.split()
    out, i = [], 0
    while i < len(pal):
        tam = random.randint(90, 140)
        out.append(" ".join(pal[i:i + tam]))
        i += tam + random.randint(0, 40)
    return [t for t in out if len(t.split()) > 40]


def alinhar(offs_a, offs_b):
    fins_a = [e for _, e in offs_a]
    idx, j = [], -1
    for _, fim_b in offs_b:
        while j + 1 < len(fins_a) and fins_a[j + 1] <= fim_b:
            j += 1
        idx.append(j)
    return idx


def preparar(ta, tb, s):
    eb = tb(s, return_offsets_mapping=True, add_special_tokens=False)
    ids_b, offs_b = eb["input_ids"][:args.max_tok], eb["offset_mapping"][:args.max_tok]
    ea = ta(s[:offs_b[-1][1]], return_offsets_mapping=True, add_special_tokens=False)
    return ea["input_ids"], ids_b, alinhar(ea["offset_mapping"], offs_b)


@torch.no_grad()
def estado(m, ids, camada):
    o = m(torch.tensor([ids], device=dev), output_hidden_states=True)
    return o.hidden_states[camada][0].float()


# ------------------------------------------------------------------ tradutor

class Tradutor(nn.Module):
    """y = desnormaliza( W · normaliza(x) [+ MLP(normaliza(x))] )"""

    def __init__(self, mx, sx, my, sy, W, mlp=False):
        super().__init__()
        for nome, v in (("mx", mx), ("sx", sx), ("my", my), ("sy", sy)):
            self.register_buffer(nome, v.clone())
        self.W = nn.Parameter(W.clone())
        self.mlp = None
        if mlp:
            da, db = W.shape
            self.mlp = nn.Sequential(nn.Linear(da, 1024), nn.GELU(), nn.Linear(1024, db))
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        z = (x - self.mx) / self.sx
        y = z @ self.W
        if self.mlp is not None:
            y = y + self.mlp(z)
        return y * self.sy + self.my


def procrustes(xc, yc):
    U, S, Vh = torch.linalg.svd(xc.T @ yc, full_matrices=False)
    return U @ Vh, (S.sum() / xc.pow(2).sum()).item()


# ------------------------------------------------------------------ costura (injeção na camada k2 de B)

def perda_costurada(mb, ids_b, k2, inj):
    def gancho(mod, a, kw):
        hs = kw.get("hidden_states", a[0] if a else None)
        novo = inj.to(hs.dtype).unsqueeze(0)
        if "hidden_states" in kw:
            kw["hidden_states"] = novo
            return a, kw
        return (novo,) + tuple(a[1:]), kw
    g = mb.model.layers[k2].register_forward_pre_hook(gancho, with_kwargs=True)
    try:
        x = torch.tensor([ids_b], device=dev)
        logits = mb(x).logits[0, :-1].float()
        return nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum"), x.shape[1] - 1
    finally:
        g.remove()


def injecao(trad, ha, al, hb_proprio):
    ia = torch.tensor([max(a, 0) for a in al], device=dev)
    y = trad(ha[ia])
    sem_par = torch.tensor([a < 0 for a in al], device=dev).unsqueeze(1)
    return torch.where(sem_par, hb_proprio, y)


def avaliar(mb, dados_teste, k2, trad):
    tot, n = 0.0, 0
    with torch.no_grad():
        for d in dados_teste:
            l, c = perda_costurada(mb, d["ids_b"], k2, injecao(trad, d["ha"], d["al"], d["hb"]))
            tot += l.item(); n += c
    return math.exp(tot / n)


def main():
    if args.tiny:
        (ta, ma), (tb, mb) = carregar_tiny()
        texto = "O capacitor armazena energia no campo elétrico. A lei de Ohm relaciona tensão e corrente. " * 600
    else:
        ta, ma = carregar(args.a)
        tb, mb = carregar(args.b)
        texto = Path(args.texto).read_text(encoding="utf-8", errors="replace")
    LA, LB = len(ma.model.layers), len(mb.model.layers)
    todos = trechos(texto)
    random.shuffle(todos)
    n_te = max(10, len(todos) // 5)
    treino_txt, teste_txt = todos[:-n_te], todos[-n_te:]
    log(f"A={args.a} ({LA} camadas)  B={args.b} ({LB} camadas)  treino={len(treino_txt)} teste={len(teste_txt)}")

    # perplexidade de B inteiro
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in teste_txt:
            _, ids_b, _ = preparar(ta, tb, s)
            x = torch.tensor([ids_b], device=dev)
            lg = mb(x).logits[0, :-1].float()
            tot += nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item(); n += x.shape[1] - 1
    ppl_b = math.exp(tot / n)
    log(f"B inteiro: {ppl_b:.2f}")

    resultados = []
    for r in [float(v) for v in args.profs.split(",")]:
        k, k2 = max(1, round(r * LA)), max(1, round(r * LB))
        log(f"=== costura {r}: A{k} -> B{k2}")

        def coletar(lista):
            out = []
            for s in lista:
                ids_a, ids_b, al = preparar(ta, tb, s)
                if len(ids_b) < 8 or len(ids_a) < 4:
                    continue
                out.append({"ids_b": ids_b, "al": al, "ha": estado(ma, ids_a, k), "hb": estado(mb, ids_b, k2)})
            return out
        tr, te = coletar(treino_txt), coletar(teste_txt)

        xs, ys = [], []
        for d in tr:
            ok = [i for i, a in enumerate(d["al"]) if a >= 0]
            xs.append(d["ha"][[d["al"][i] for i in ok]]); ys.append(d["hb"][ok])
        x, y = torch.cat(xs), torch.cat(ys)
        mx, my = x.mean(0), y.mean(0)
        um_a, um_b = torch.ones_like(mx), torch.ones_like(my)

        # 1. Procrustes cru (escala global)
        W1, esc = procrustes(x - mx, y - my)
        t1 = Tradutor(mx, um_a, my, um_b, W1 * esc).to(dev)
        # 2. Procrustes normalizado por dimensão
        sx, sy = x.std(0).clamp_min(1e-4), y.std(0).clamp_min(1e-4)
        zx, zy = (x - mx) / sx, (y - my) / sy
        W2, esc2 = procrustes(zx, zy)
        t2 = Tradutor(mx, sx, my, sy, W2 * esc2).to(dev)
        top = (y - my).pow(2).mean(0)
        frac = (top.sort(descending=True).values[:8].sum() / top.sum()).item()
        log(f"  as 8 maiores dimensões de B concentram {100 * frac:.0f}% da variância (dimensões gigantes)")
        del x, y, xs, ys, zx, zy

        res = {"prof": r, "camada_a": k, "camada_b": k2}
        res["procrustes_cru"] = avaliar(mb, te, k2, t1)
        res["procrustes_normalizado"] = avaliar(mb, te, k2, t2)
        log(f"  Procrustes cru {res['procrustes_cru']:.1f} | normalizado {res['procrustes_normalizado']:.1f}")

        # 3 e 4. treinados por gradiente
        for nome, mlp, lr in (("linear_treinado", False, 3e-4), ("linear_mais_mlp_treinado", True, 3e-4)):
            trad = Tradutor(mx, sx, my, sy, W2 * esc2, mlp=mlp).to(dev)
            opt = torch.optim.AdamW(trad.parameters(), lr=lr, weight_decay=0.0)
            passos = args.epocas * len(tr)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=passos, pct_start=0.1)
            curva = []
            for ep in range(args.epocas):
                random.shuffle(tr)
                acc, cnt = 0.0, 0
                for d in tr:
                    l, c = perda_costurada(mb, d["ids_b"], k2, injecao(trad, d["ha"], d["al"], d["hb"]))
                    (l / c).backward()
                    torch.nn.utils.clip_grad_norm_(trad.parameters(), 1.0)
                    opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                    acc += l.item(); cnt += c
                ppl_te = avaliar(mb, te, k2, trad)
                curva.append({"epoca": ep + 1, "treino": math.exp(acc / cnt), "teste": ppl_te})
                log(f"  {nome} época {ep + 1}: treino {math.exp(acc / cnt):.1f}  teste {ppl_te:.1f}")
            res[nome] = curva[-1]["teste"]
            res[nome + "_curva"] = curva
            if nome == "linear_mais_mlp_treinado":
                torch.save(trad.state_dict(), SAIDA / f"tradutor_{r}.pt")
                melhor = trad
        resultados.append(res)

        # exemplo de geração com o melhor tradutor
        try:
            s = "A lei de Ohm diz que"
            for _ in range(30):
                ids_a, ids_b, al = preparar(ta, tb, s)
                with torch.no_grad():
                    inj = injecao(melhor, estado(ma, ids_a, k), al, estado(mb, ids_b, k2))
                    def gancho(mod, a, kw):
                        hs = kw.get("hidden_states", a[0] if a else None)
                        novo = inj.to(hs.dtype).unsqueeze(0)
                        if "hidden_states" in kw:
                            kw["hidden_states"] = novo
                            return a, kw
                        return (novo,) + tuple(a[1:]), kw
                    g = mb.model.layers[k2].register_forward_pre_hook(gancho, with_kwargs=True)
                    prox = mb(torch.tensor([ids_b], device=dev)).logits[0, -1].argmax().item()
                    g.remove()
                s += tb.decode([prox])
            res["exemplo"] = s
            log("  exemplo:", s)
        except Exception as e:
            log("  exemplo falhou:", e)

    linhas = ["# Costura treinada (mudança de base aprendida)", "",
              f"A = `{args.a}` → tradutor → B = `{args.b}`; {len(treino_txt)} trechos de treino, {len(teste_txt)} de teste.", "",
              f"**B inteiro: perplexidade {ppl_b:.2f}**", "",
              "| costura | Procrustes cru | Procrustes normalizado | linear treinado | linear + MLP treinado |",
              "|---|---|---|---|---|"]
    for r in resultados:
        linhas.append(f"| {r['prof']} (A{r['camada_a']}→B{r['camada_b']}) | {r['procrustes_cru']:.1f} | "
                      f"{r['procrustes_normalizado']:.1f} | {r['linear_treinado']:.1f} | {r['linear_mais_mlp_treinado']:.1f} |")
    linhas += [""] + [f"Exemplo ({r['prof']}): {r.get('exemplo', '')}" for r in resultados]
    linhas += ["", f"Tempo total: {(time.time() - T0) / 60:.1f} min."]
    (SAIDA / "resultado.md").write_text("\n".join(linhas), encoding="utf-8")
    (SAIDA / "resultado.json").write_text(json.dumps({"b_inteiro": ppl_b, "costuras": resultados}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print("\n".join(linhas), flush=True)


if __name__ == "__main__":
    main()
