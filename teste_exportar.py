"""Testes da exportação (sem internet, com modelos aleatórios minúsculos).

1. Fusão exata: B é o próprio A escrito em outra base (y = s·x·R). Fundir A[0..1] + B[2..3] tem que
   devolver exatamente o A original — prova que a conta de girar as camadas está certa.
2. Destilação: costura duas IAs de tokenizadores diferentes e destila numa normal.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
import nucleo
import exportar
from nucleo import Parte, Receita, costurar

torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp())


def tokenizador(vocab):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    amostra = ["def soma(a, b):\n    return a + b\n", "A derivada de x^2 é 2x. A integral de 2x é x^2 + C."] * 40
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.train_from_iterator(amostra, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["[UNK]"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=t)


def salvar(m, tok, nome):
    p = tmp / nome
    m.save_pretrained(p); tok.save_pretrained(p)
    return str(p)


(tmp / "cod.py").write_text("def soma(a, b):\n    return a + b\n\n" * 200)
(tmp / "mat.txt").write_text("A derivada de x^2 é 2x. A integral de 2x é x^2 + C.\n" * 200)
assuntos = {"programação": [str(tmp / "cod.py")], "matemática": [str(tmp / "mat.txt")]}

# ---------------------------------------------------------------- 1. fusão exata
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

tok = tokenizador(300)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
A = Qwen2ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in A.named_parameters():
        if "norm" in n:
            p.copy_(0.5 + torch.rand_like(p))            # ganhos ≠ 1 para testar a absorção
        if n.endswith("bias"):
            p.normal_(0, 0.1)
caminho_a = salvar(A, tok, "A")

R, _ = torch.linalg.qr(torch.randn(64, 64))
s = 2.7
sd = {k: v.clone() for k, v in A.state_dict().items()}
B = {}
B["model.embed_tokens.weight"] = s * sd["model.embed_tokens.weight"] @ R
for k, v in sd.items():
    if ".layers." not in k:
        continue
    pre, nome = k.split(".layers.")[0], k.split(".layers.")[1].split(".", 1)[1]
    l = k.split(".layers.")[1].split(".")[0]
    g1 = sd[f"model.layers.{l}.input_layernorm.weight"]; g2 = sd[f"model.layers.{l}.post_attention_layernorm.weight"]
    if nome.endswith("layernorm.weight"):
        B[k] = torch.ones_like(v)
    elif nome in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
        B[k] = (v * g1[None, :]) @ R
    elif nome in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        B[k] = (v * g2[None, :]) @ R
    elif nome in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
        B[k] = s * R.T @ v
    else:
        B[k] = v
gf = sd["model.norm.weight"]
B["model.norm.weight"] = torch.ones_like(gf)
B["lm_head.weight"] = (sd["model.embed_tokens.weight"] * gf[None, :]) @ R
cfgB = Qwen2Config(**{**cfg.to_dict(), "tie_word_embeddings": False})
mB = Qwen2ForCausalLM(cfgB).eval()
mB.load_state_dict(B, strict=True)
caminho_b = salvar(mB, tok, "B")

x = torch.tensor([tok("def soma(a, b): return a + b")["input_ids"]])
with torch.no_grad():
    la, lb = A(x).logits, mB(x).logits
print("B é o A em outra base:", torch.allclose(la, lb, atol=1e-3), float((la - lb).abs().max()))

r = Receita("fusao-teste", [Parte(caminho_a, 0, 1), Parte(caminho_b, 2, 3)], assuntos=assuntos, max_tok=64)
pode, motivo = exportar.checar_fusao(r)
print("pode fundir:", pode, motivo)
assert pode
textos = ["def soma(a, b):\n    return a + b\n" * 3, "A derivada de x^2 é 2x. A integral de 2x é x^2 + C. " * 3] * 6
res = exportar.fundir(r, tmp / "fundido", textos)
F = AutoModelForCausalLM.from_pretrained(tmp / "fundido", dtype=torch.float32).eval()
with torch.no_grad():
    lf = F(x).logits
erro = float((la - lf).abs().max())
print("fundido == A original:", erro, res)
assert erro < 0.05, erro          # salvo em bfloat16 → pequena diferença de arredondamento
assert res["r2_juncoes"][0] > 0.999

# não deve fundir quando a dimensão é diferente
cfgC = Qwen2Config(**{**cfg.to_dict(), "hidden_size": 96, "intermediate_size": 192})
caminho_c = salvar(Qwen2ForCausalLM(cfgC), tok, "C")
pode, motivo = exportar.checar_fusao(Receita("x", [Parte(caminho_a, 0, 1), Parte(caminho_c, 2, 3)]))
print("A + C (tamanhos diferentes):", pode, motivo)
assert not pode

# ---------------------------------------------------------------- 2. destilação
from transformers import LlamaConfig, LlamaForCausalLM
tok2 = tokenizador(350)
mat = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok2), hidden_size=96, intermediate_size=192, num_hidden_layers=6,
                                   num_attention_heads=4, num_key_value_heads=2)).eval()
caminho_mat = salvar(mat, tok2, "mat")
r = Receita("destila-teste", [Parte(caminho_a, 0, 1), Parte(caminho_mat, 3, 5)], assuntos=assuntos, epocas=1, max_tok=64,
            max_trechos=30)
pasta = tmp / "ia_destila"
costurar(r, pasta, log=lambda *_: None)
print("pode fundir (tokenizadores diferentes)?", exportar.checar_fusao(r))
exportar.para_gguf = lambda pasta_hf, arquivo, tipo="q8_0", log=print: (Path(arquivo).write_bytes(b"GGUF"), Path(arquivo))[1]
res = exportar.exportar(pasta, modo="auto", epocas=1, para_lmstudio=False, log=print)
print(json.dumps({k: v for k, v in res.items() if k != "curva"}, ensure_ascii=False, indent=1, default=str)[:800])
assert res["modo"] == "destilação"
D = AutoModelForCausalLM.from_pretrained(pasta / "exportado_hf")
print("aluna salva:", type(D).__name__, D.config.num_hidden_layers, "camadas")
print(exportar.relatorio_exportacao(res))
print("OK")
