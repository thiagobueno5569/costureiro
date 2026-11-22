"""Teste de fumaça do Costureiro com modelos aleatórios minúsculos (sem internet)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
import nucleo
from nucleo import Parte, Receita, costurar, IACosturada


def tiny(tipo, vocab, d, L):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    amostra = ["def soma(a, b):\n    return a + b\n", "A derivada de x^2 é 2x. A integral de 2x é x^2 + C."] * 40
    t = Tokenizer(models.BPE(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.train_from_iterator(amostra, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["[UNK]"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=t)
    Cfg, M = (Qwen2Config, Qwen2ForCausalLM) if tipo == "qwen" else (LlamaConfig, LlamaForCausalLM)
    m = M(Cfg(vocab_size=len(tok), hidden_size=d, intermediate_size=2 * d, num_hidden_layers=L,
              num_attention_heads=4, num_key_value_heads=2)).to(nucleo.DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return tok, m


MODELOS = {"codigo": tiny("qwen", 300, 64, 4), "mat": tiny("llama", 350, 96, 6), "meio": tiny("llama", 320, 80, 5)}
nucleo._cache.update(MODELOS)
tmp = Path(tempfile.mkdtemp())
(tmp / "cod.py").write_text("def soma(a, b):\n    return a + b\n\n" * 200)
(tmp / "mat.txt").write_text("A derivada de x^2 é 2x. A integral de 2x é x^2 + C.\n" * 200)

for partes in ([Parte("codigo", 0, 1), Parte("mat", 3, 5)],
               [Parte("codigo", 0, 1), Parte("meio", 2, 3), Parte("mat", 4, 5)]):
    nucleo._cache.update(MODELOS)      # cada tarefa tira da placa o que não usa; o teste devolve
    r = Receita("teste", partes, assuntos={"programação": [str(tmp / "cod.py")], "matemática": [str(tmp / "mat.txt")]},
                epocas=2, max_tok=64)
    ia, res = costurar(r, tmp / f"ia_{len(partes)}")
    print((tmp / f"ia_{len(partes)}" / "resultado.md").read_text())
    ia2 = IACosturada.abrir(tmp / f"ia_{len(partes)}")
    print("geração:", repr(ia2.gerar("def ", n=10)))

print(nucleo.sugerir_corte("codigo", "mat", ["def soma(a, b):\n    return a + b\n" * 5] * 60, fracoes=(0.5,), max_tok=48))
print("OK")
