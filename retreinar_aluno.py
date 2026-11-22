"""Treina de novo o aluno da destilação a partir dos exemplos já conferidos (exemplos.jsonl), sem gerar tudo de novo.

Útil quando o treino completo não cabe no tempo/saldo: usa todos os exemplos de código (são poucos) e completa com
matemática até --max-exemplos, numa época. Depois gera o .gguf.

    python retreinar_aluno.py --pasta fusoes/destilada-coder-math --max-exemplos 3000 --epocas 1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import destilar_tarefas


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", required=True)
    ap.add_argument("--aluno", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--max-exemplos", type=int, default=3000)
    ap.add_argument("--epocas", type=int, default=1)
    ap.add_argument("--gguf", default="q8_0")
    a = ap.parse_args(argv)
    pasta = Path(a.pasta)
    todos = [json.loads(l) for l in (pasta / "exemplos.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    mat = [(e["prompt"], e["resposta"]) for e in todos if e["prompt"].startswith("Problem:")]
    cod = [(e["prompt"], e["resposta"]) for e in todos if not e["prompt"].startswith("Problem:")]
    rng = random.Random(0)
    rng.shuffle(mat)
    exemplos = cod + mat[: max(0, a.max_exemplos - len(cod))]
    print(f"{len(todos)} exemplos conferidos ({len(mat)} de matemática, {len(cod)} de código); "
          f"treinando com {len(exemplos)} ({len(cod)} de código), {a.epocas} época(s)", flush=True)
    t0 = time.time()
    res = destilar_tarefas.treinar_aluno(a.aluno, exemplos, pasta, epocas=a.epocas, log=lambda m: print(m, flush=True))
    res.update({"modo": "destilação conferida (treino reduzido)", "aluno": a.aluno, "n_matematica_usados": len(exemplos) - len(cod),
                "n_codigo": len(cod), "n_conferidos": len(todos), "minutos": (time.time() - t0) / 60})
    (pasta / "destilacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.gguf and a.gguf != "nao":
        import exportar
        print(".gguf:", exportar.para_gguf(pasta, pasta.parent / f"{pasta.name}.{a.gguf}.gguf", a.gguf), flush=True)


if __name__ == "__main__":
    main()
