"""Baixa/junta textos de exemplo para o foco: programação (código da biblioteca padrão do Python)
e matemática (problemas resolvidos do GSM8K). Grava em textos/programacao e textos/matematica."""
import io
import random
import sysconfig
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).parent / "textos"
(RAIZ / "programacao").mkdir(parents=True, exist_ok=True)
(RAIZ / "matematica").mkdir(parents=True, exist_ok=True)

# programação: ~3 MB de código Python de verdade (biblioteca padrão)
lib = Path(sysconfig.get_paths()["stdlib"])          # Windows: ...\\Lib · Linux: /usr/lib/python3.x
arqs = [f for f in lib.rglob("*.py") if "test" not in f.parts and "site-packages" not in f.parts and "idlelib" not in f.parts]
random.Random(0).shuffle(arqs)
total, partes = 0, []
for f in arqs:
    try:
        t = f.read_text(encoding="utf-8")
    except Exception:
        continue
    partes.append(f"# arquivo: {f.name}\n{t}")
    total += len(t)
    if total > 3_000_000:
        break
if total < 100_000:
    raise SystemExit(f"achei pouco código Python em {lib} ({total} caracteres); coloque arquivos em textos/programacao")
(RAIZ / "programacao" / "python_stdlib.py.txt").write_text("\n\n".join(partes), encoding="utf-8")
print("programação:", total, "caracteres de", len(partes), "arquivos")

# matemática: GSM8K (problemas com solução passo a passo)
url = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
import pandas as pd
dados = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "costureiro"}), timeout=120).read()
df = pd.read_parquet(io.BytesIO(dados))
txt = "\n\n".join(f"Problem: {q}\nSolution: {a.replace('####', 'Answer:')}" for q, a in zip(df["question"], df["answer"]))
(RAIZ / "matematica" / "gsm8k.txt").write_text(txt, encoding="utf-8")
print("matemática:", len(txt), "caracteres,", len(df), "problemas")
