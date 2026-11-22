"""Teste da fusão direta, sem internet.

Para cada arquitetura: pega um modelo A, escreve cópias dele em OUTRAS bases (y = s·x·R, R ortogonal
aleatória) e funde pedaços dessas cópias. Como todas são o mesmo modelo em bases diferentes, a fusão
tem que devolver exatamente o A — isso prova a conta para densos, MoE, Qwen3 e para 3 peças.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
from safetensors.torch import load_file, save_file

import fusao

torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp())


def tokenizador():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    amostra = ["def soma(a, b):\n    return a + b\n", "A derivada de x^2 é 2x. A integral de 2x é x^2 + C."] * 40
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.train_from_iterator(amostra, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=t)


TOK = tokenizador()
TEXTOS = ["def soma(a, b):\n    return a + b\n" * 4, "A derivada de x^2 é 2x. A integral de 2x é x^2 + C. " * 4] * 12


def modelo(tipo):
    from transformers import (MixtralConfig, MixtralForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2MoeConfig,
                              Qwen2MoeForCausalLM, Qwen3Config, Qwen3ForCausalLM)
    comum = dict(vocab_size=len(TOK), hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                 tie_word_embeddings=True)
    if tipo == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(intermediate_size=128, **comum))
    if tipo == "qwen3":
        return Qwen3ForCausalLM(Qwen3Config(intermediate_size=128, head_dim=16, **comum))
    if tipo == "mixtral":
        return MixtralForCausalLM(MixtralConfig(intermediate_size=96, num_local_experts=4, num_experts_per_tok=2, **comum))
    if tipo == "qwen2_moe":
        return Qwen2MoeForCausalLM(Qwen2MoeConfig(intermediate_size=128, moe_intermediate_size=48,
                                                  shared_expert_intermediate_size=64, num_experts=4,
                                                  num_experts_per_tok=2, **comum))
    raise ValueError(tipo)


def salvar_a(tipo):
    m = modelo(tipo).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n and "q_norm" not in n and "k_norm" not in n:
                p.copy_(0.5 + torch.rand_like(p))
            if n.endswith("bias"):
                p.normal_(0, 0.1)
            if "gate.weight" in n:                 # roteador com escolhas bem definidas
                p.mul_(4)
            if "embed_tokens" in n:                # vetores bem maiores que o eps da norma, como num modelo real
                p.mul_(30)
    pa = tmp / f"{tipo}_A"
    m.save_pretrained(pa); TOK.save_pretrained(pa)
    return m, pa


def em_outra_base(pa: Path, destino: Path, R: torch.Tensor, s: float):
    """Escreve o modelo de `pa` na base y = s·x·R (o inverso do que a fusão faz)."""
    sd = load_file(str(pa / "model.safetensors"))
    novo = {}
    emb = sd["model.embed_tokens.weight"]
    novo["model.embed_tokens.weight"] = s * emb @ R
    for k, v in sd.items():
        if ".layers." not in k:
            continue
        l = k.split(".layers.")[1].split(".")[0]
        nome = k.split(f".layers.{l}.")[1]
        g1 = sd[f"model.layers.{l}.input_layernorm.weight"]
        g2 = sd[f"model.layers.{l}.post_attention_layernorm.weight"]
        papel = fusao._papel(nome)
        if papel in ("norma1", "norma2"):
            novo[k] = torch.ones_like(v)
        elif papel == "entrada":
            g = g1 if nome.startswith("self_attn.") else g2
            novo[k] = (v * g) @ R
        elif papel == "saida":
            novo[k] = s * (R.T @ v)
        else:
            novo[k] = v
    gf = sd["model.norm.weight"]
    novo["model.norm.weight"] = torch.ones_like(gf)
    novo["lm_head.weight"] = (sd.get("lm_head.weight", emb) * gf) @ R
    destino.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in novo.items()}, str(destino / "model.safetensors"), metadata={"format": "pt"})
    cfg = json.loads((pa / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    (destino / "config.json").write_text(json.dumps(cfg))
    TOK.save_pretrained(destino)
    return destino


def logits(pasta_ou_modelo, x):
    from transformers import AutoModelForCausalLM
    m = pasta_ou_modelo if not isinstance(pasta_ou_modelo, Path) else \
        AutoModelForCausalLM.from_pretrained(pasta_ou_modelo, dtype=torch.float32).eval()
    with torch.no_grad():
        return m(x).logits


x = torch.tensor([TOK("def soma(a, b): return a + b. A derivada de x^2 é 2x")["input_ids"]])
for tipo in ("qwen2", "qwen3", "mixtral", "qwen2_moe"):
    A, pa = salvar_a(tipo)
    R1, _ = torch.linalg.qr(torch.randn(64, 64))
    R2, _ = torch.linalg.qr(torch.randn(64, 64))
    pb = em_outra_base(pa, tmp / f"{tipo}_B", R1, 2.7)
    pc = em_outra_base(pa, tmp / f"{tipo}_C", R2, 0.6)
    la = logits(A, x)
    assert torch.allclose(la, logits(pb, x), atol=2e-3), f"{tipo}: a cópia girada não bate (erro no teste)"
    ok, motivo = fusao.checar([str(pa), str(pb), str(pc)])
    assert ok, motivo
    for pecas in ([fusao.Peca(str(pa), 0, 1), fusao.Peca(str(pb), 2, 3)],
                  [fusao.Peca(str(pa), 0, 0), fusao.Peca(str(pb), 1, 2), fusao.Peca(str(pc), 3, 3)]):
        saida = tmp / f"{tipo}_fundido_{len(pecas)}"
        res = fusao.fundir(pecas, saida, TEXTOS, log=lambda *_: None, dispositivo="cpu", max_tok=64)
        erro = float((la - logits(saida, x)).abs().max())
        r2 = [round(j["r2"], 5) for j in res["juncoes"]]
        print(f"{tipo:10s} {len(pecas)} peças: erro máximo {erro:.4f} · R² {r2}")
        assert erro < 0.05, erro
        assert min(r2) > 0.999
    shutil.rmtree(tmp / f"{tipo}_fundido_2")

# medição + ficha + relatório (sem .gguf)
res = fusao.fundir([fusao.Peca(str(tmp / "qwen2_A"), 0, 1), fusao.Peca(str(tmp / "qwen2_B"), 2, 3)],
                   tmp / "f_med", TEXTOS, log=print, dispositivo="cpu", max_tok=64)
res = fusao.medir(res, tmp / "f_med", dispositivo="cpu")
res.pop("_ids_medir")
fusao.ficha(res, tmp / "f_med", "teste")
print((tmp / "f_med" / "README.md").read_text()[:600])
print(fusao.relatorio(res))
assert abs(res["bpc"]["fundido"] - res["bpc"][str(tmp / "qwen2_A")]) < 0.01

# deve recusar: dimensão diferente e RoPE diferente
from transformers import Qwen2Config, Qwen2ForCausalLM
outro = tmp / "qwen2_grande"
Qwen2ForCausalLM(Qwen2Config(vocab_size=len(TOK), hidden_size=96, intermediate_size=128, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2)).save_pretrained(outro)
TOK.save_pretrained(outro)
print("dimensão diferente:", fusao.checar([str(tmp / "qwen2_A"), str(outro)]))
assert not fusao.checar([str(tmp / "qwen2_A"), str(outro)])[0]
cfg = json.loads((tmp / "qwen2_B" / "config.json").read_text())
if "rope_parameters" in cfg:
    cfg["rope_parameters"] = {**cfg["rope_parameters"], "rope_theta": 10000.0}
cfg["rope_theta"] = 10000.0
(tmp / "qwen2_B" / "config.json").write_text(json.dumps(cfg))
print("RoPE diferente:", fusao.checar([str(tmp / "qwen2_A"), str(tmp / "qwen2_B")]))
assert not fusao.checar([str(tmp / "qwen2_A"), str(tmp / "qwen2_B")])[0]

# linha de comando
codigo = fusao.main(["--peca", f"{tmp / 'mixtral_A'}:0-1", "--peca", f"{tmp / 'mixtral_B'}:2-3", "--so-checar",
                     "--saida", str(tmp / "cli")])
assert codigo == 0
print("OK")
