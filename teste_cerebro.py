"""Teste de fumaça do modo cérebro com modelos aleatórios minúsculos (sem internet)."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "teste_rapido.py", encoding="utf-8").read().split("tmp = Path")[0])
import nucleo
from nucleo import Especialista, ReceitaCerebro, criar_cerebro, IACerebro
base = {"codigo": ("qwen", 300, 64, 4), "mat": ("llama", 350, 96, 6), "meio": ("llama", 320, 80, 5)}
nucleo._carregar_cru = lambda n, ate_camada=None: tiny(*base[n])
nucleo.info_modelo = lambda n: {"camadas": base[n][3], "dimensao": base[n][2], "arquitetura": "?"}
tmp = Path(tempfile.mkdtemp())
(tmp / "cod.py").write_text("".join(f"def f{i}(x):\n    return x * {i}\n" for i in range(600)))
(tmp / "mat.txt").write_text("".join(f"Problema: {i} + {i}?\nSolução: {2*i}\n" for i in range(600)))
r = ReceitaCerebro("cerebro-teste", base="meio", camada=2,
                   especialistas=[Especialista("codigo", 1, 2, "código"), Especialista("mat", 2, 4, "matemática")],
                   assuntos={"programação": [str(tmp / "cod.py")], "matemática": [str(tmp / "mat.txt")]},
                   epocas=2, max_tok=48, max_trechos=40)
ia, res = criar_cerebro(r, tmp / "c")
print((tmp / "c" / "resultado.md").read_text())
nucleo._cache = {k: v for k, v in nucleo._cache.items() if "|" not in k}   # força recarregar podado
ia2 = IACerebro.abrir(tmp / "c")
txt, pedacos, portoes = ia2.gerar("def ", 6)
print("geração:", repr(txt), len(pedacos), [len(p) for p in portoes])
print("OK")
