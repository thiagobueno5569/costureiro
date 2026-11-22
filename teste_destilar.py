"""Teste da destilação com respostas conferidas (modelos minúsculos, dados falsos, sem internet)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

import destilar_tarefas as dt
import orquestra

torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp())
src = open(Path(__file__).parent / "teste_moe.py", encoding="utf-8").read()
exec(src.split("TOK = tokenizador()")[0].split("import moe")[1])        # só a função tokenizador()
TOK = tokenizador()
TOK.add_special_tokens({"eos_token": "<|fim|>"})          # os Qwen reais têm fim de texto
cfg = Qwen2Config(vocab_size=len(TOK), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2)
Qwen2ForCausalLM(cfg).save_pretrained(tmp / "m"); TOK.save_pretrained(tmp / "m")

# dados falsos no lugar dos downloads
pd.DataFrame({"question": ["quanto é 2+2?"] * 6, "answer": ["2+2=4\n#### 4"] * 6}).to_parquet(tmp / "gsm8k_train.parquet")
for p in ("train", "validation", "prompt"):
    pd.DataFrame([{"text": "dobro de x", "code": "def dobro(x):\n    return 2*x", "test_list": ["assert dobro(3) == 6"],
                   "test_setup_code": ""}] * 3).to_parquet(tmp / f"mbpp_{p}.parquet")
orquestra._baixar = lambda url, nome: tmp / nome

# conferência: só respostas certas entram
assert dt._numero(dt._resposta_boxed("logo \\boxed{4}")) == 4.0
assert dt._numero(dt._resposta_boxed("Answer: 1,234")) == 1234.0
# o gerador minúsculo erra tudo -> nenhum exemplo; então injetamos respostas certas para testar o treino
mat = dt.solucoes_matematica(str(tmp / "m"), 6, __import__("random").Random(0), log=print, lote=4)
cod = dt.solucoes_codigo(str(tmp / "m"), __import__("random").Random(0), amostras=1, log=print, lote=4)
print("conferidos (modelo aleatório):", len(mat), len(cod))
exemplos = [("Problem: quanto é 2+2?\nSolution:", " 2+2=4\nAnswer: 4")] * 8 + \
           [('"""\ndobro de x\nassert dobro(3) == 6\n"""\n', "def dobro(x):\n    return 2*x\n")] * 8
res = dt.treinar_aluno(str(tmp / "m"), exemplos, tmp / "aluno", epocas=3, lr=5e-3, max_tok=64)
print(res)
A = AutoModelForCausalLM.from_pretrained(tmp / "aluno")
x = torch.tensor([TOK("Problem: quanto é 2+2?\nSolution:", add_special_tokens=False)["input_ids"]])
print("aluno abre:", type(A).__name__, "· gera:", repr(TOK.decode(A.generate(x, max_new_tokens=8, do_sample=False)[0][x.shape[1]:])))
print("OK")

# com um "professor" que acerta, as respostas certas entram (e as erradas não)
respostas = iter(["2+2 = 4. \\boxed{4}", "2+2 = 5. \\boxed{5}"] * 3)
dt.gerar = lambda tok, m, prompts, *a, **k: [next(respostas) for _ in prompts] if "Problem" in prompts[0] or "2+2" in prompts[0] \
    else ["def dobro(x):\n    return 2*x" if i % 2 == 0 else "def dobro(x):\n    return x" for i in range(len(prompts))]
mat = dt.solucoes_matematica(str(tmp / "m"), 6, __import__("random").Random(0), log=print)
cod = dt.solucoes_codigo(str(tmp / "m"), __import__("random").Random(0), amostras=1, log=print)
print(mat[0])
assert len(mat) == 3 and "Answer: 4" in mat[0] and "\\boxed" not in mat[0]
assert len(cod) == 5            # 9 problemas, os de índice par recebem a solução certa
print("OK conferência")
