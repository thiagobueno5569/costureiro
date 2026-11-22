"""Teste do MoE montado (sem internet).

Exato: B é o próprio A "disfarçado" (ganhos da norma diferentes e pesos de entrada da FFN compensados).
O MoE (A, B) precisa dar exatamente o A, qualquer que seja o roteador — prova a montagem e a correção de ganho.
Depois: roteador inicial por assunto, treino do roteador, abrir de novo.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

import moe

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
cfg = Qwen2Config(vocab_size=len(TOK), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, tie_word_embeddings=True)
A = Qwen2ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in A.named_parameters():
        if "norm" in n:
            p.copy_(0.5 + torch.rand_like(p))
        if "embed" in n:
            p.mul_(20)
pa = tmp / "A"
A.save_pretrained(pa); TOK.save_pretrained(pa)

# B = A disfarçado: ganho g' da norma antes da FFN e entradas da FFN compensadas (W' = W·g/g')
sd = load_file(str(pa / "model.safetensors"))
for l in range(4):
    g = sd[f"model.layers.{l}.post_attention_layernorm.weight"]
    g2 = g * (0.5 + torch.rand_like(g) * 2)
    sd[f"model.layers.{l}.post_attention_layernorm.weight"] = g2
    for nome in ("gate_proj", "up_proj"):
        k = f"model.layers.{l}.mlp.{nome}.weight"
        sd[k] = sd[k] * (g / g2)[None, :]
    # a atenção do B é bagunçada de propósito: o MoE usa a atenção do base, então não pode fazer diferença
    sd[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn_like(sd[f"model.layers.{l}.self_attn.q_proj.weight"])
pb = tmp / "B"
pb.mkdir()
save_file(sd, str(pb / "model.safetensors"), metadata={"format": "pt"})
(pb / "config.json").write_text((pa / "config.json").read_text())
TOK.save_pretrained(pb)

textos = {"programação": ["def soma(a, b):\n    return a + b\n" * 6] * 12,
          "matemática": ["A derivada de x^2 é 2x. A integral de 2x é x^2 + C. " * 6] * 12}
print(moe.checar([str(pa), str(pb)]))
x = torch.tensor([TOK("def soma(a, b): return a + b. A derivada de x^2 é 2x")["input_ids"]])
with torch.no_grad():
    la = A(x).logits
for topk in (1, 2):
    saida = tmp / f"moe_{topk}"
    res = moe.montar(str(pa), [str(pb)], ["programação", "matemática"], saida, textos, topk=topk, dispositivo="cpu")
    M = AutoModelForCausalLM.from_pretrained(saida, dtype=torch.float32).eval()
    print(type(M).__name__, json.loads((saida / "config.json").read_text())["num_experts"])
    with torch.no_grad():
        erro = float((M(x).logits - la).abs().max())
    print(f"topk={topk}: MoE(A, A disfarçado) == A? erro máximo {erro:.5f}")
    assert erro < 1e-3, erro

# o roteador inicial separa os assuntos?
medias = moe.direcoes_por_assunto(str(pa), textos, dispositivo="cpu")
W = moe.roteador_inicial(medias, ["programação", "matemática"])
lg_p = torch.einsum("led,ld->le", W, medias["programação"])
lg_m = torch.einsum("led,ld->le", W, medias["matemática"])
print("roteador inicial: programação prefere o especialista 0 em", int((lg_p[:, 0] > lg_p[:, 1]).sum()), "de 4 camadas;",
      "matemática prefere o 1 em", int((lg_m[:, 1] > lg_m[:, 0]).sum()))
assert (lg_p[:, 0] > lg_p[:, 1]).all() and (lg_m[:, 1] > lg_m[:, 0]).all()

# treino do roteador (com especialistas diferentes de verdade)
pc = tmp / "C"
Qwen2ForCausalLM(cfg).save_pretrained(pc); TOK.save_pretrained(pc)
saida = tmp / "moe_treino"
moe.montar(str(pa), [str(pc)], ["programação", "matemática"], saida, textos, topk=2, dispositivo="cpu")
r = moe.treinar_roteador(saida, textos, passos=40, max_tok=64)
print("roteador treinado:", r)
assert r["perda_depois"] <= r["perda_antes"]
# como na placa de vídeo: modelo em bf16, roteador em float32 (na RunPod isso quebrava)
r16 = moe.treinar_roteador(saida, textos, passos=20, max_tok=64, dtype=torch.bfloat16)
print("roteador treinado em bf16:", r16)
M = AutoModelForCausalLM.from_pretrained(saida)
print("abre de novo:", type(M).__name__)
print("OK")
