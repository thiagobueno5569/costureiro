"""Teste do modo orquestra com modelos minúsculos (sem internet).

Base e solista com o mesmo vocabulário (aprende com os dois professores e o portão recebe a dica) e
com vocabulário diferente (aprende só com a base). Confere que a orquestra nasce igual à base, treina,
salva, abre de novo e gera texto.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
import nucleo
import orquestra
from orquestra import ReceitaOrquestra, Solista

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


def salvar(nome, tok, d, L):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=d, intermediate_size=2 * d, num_hidden_layers=L,
                                     num_attention_heads=4, num_key_value_heads=2))
    m.save_pretrained(tmp / nome); tok.save_pretrained(tmp / nome)
    return str(tmp / nome)


T1, T2 = tokenizador(300), tokenizador(260)
base = salvar("base", T1, 64, 6)
sol_mesmo = salvar("sol_mesmo", T1, 96, 5)
sol_outro = salvar("sol_outro", T2, 80, 5)
(tmp / "cod.py").write_text("".join(f"def f{i}(x):\n    return x * {i}\n\n" for i in range(3000)))
(tmp / "mat.txt").write_text("".join(f"Problem: quanto é {i} + {i}?\nSolution: {i} + {i} = {2 * i}. Answer: {2 * i}\n\n"
                                     for i in range(1500)))
assuntos = {"programação": [str(tmp / "cod.py")], "matemática": [str(tmp / "mat.txt")]}

for nome, sols in (("mesmo", [Solista(sol_mesmo, nome="código")]),
                   ("outro", [Solista(sol_outro, nome="outro")]),
                   ("dois", [Solista(sol_mesmo, nome="código"), Solista(sol_outro, 3, nome="outro")])):
    r = ReceitaOrquestra(f"orq-{nome}", base, sols, assuntos=assuntos, dados_de_tarefa=False, max_tok=96, epocas=2,
                         max_trechos=60)
    pasta = tmp / f"orq_{nome}"
    ia, res = orquestra.criar(r, pasta, log=print)
    v = res["assuntos"]["programação"]
    # nasce (quase) igual à base: com o portão em ~98% para a base
    assert abs(v["inicio"]["bpc"] - v["inicio"]["prof_base"]) < 0.25, v
    ia2 = orquestra.IAOrquestra.abrir(pasta, log=lambda *_: None)
    d = ia2.preparar("def f3(x):\n    return x")
    assert torch.allclose(ia2.logits(d), ia.logits(d), atol=1e-3)
    print("gera:", repr(ia2.completar("def f7(x):\n", 12, ["\n\n"])), "· vez:", [round(u, 2) for u in ia2.uso()])
    print((pasta / "resultado.md").read_text())
    nucleo.liberar_cache()

# avaliação com dados falsos: a orquestra e as réguas (roteador, oráculo)
import pandas as pd
import avaliar
avaliar.DADOS = tmp / "dados"; avaliar.DADOS.mkdir()
he = {"task_id": "H/0", "prompt": "def soma(a, b):\n", "test": "def check(f):\n    assert f(2, 3) == 5\n",
      "entry_point": "soma", "canonical_solution": ""}
pd.DataFrame([he] * 2).to_parquet(avaliar.DADOS / "humaneval.parquet")
pd.DataFrame([{"question": "Quanto é 2+2?", "answer": "#### 4"}] * 2).to_parquet(avaliar.DADOS / "gsm8k.parquet")
avaliar.TAMANHO = {"humaneval": 8, "gsm8k": 8}
res = avaliar.avaliar(tmp / "orq_mesmo", ["humaneval", "gsm8k"], log=print)
print(avaliar.relatorio(res))
assert any(k.startswith("roteador") for k in res["tarefas"]["humaneval"])
print("OK")
