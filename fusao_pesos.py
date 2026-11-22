"""IA única por soma de pesos (TIES): modelos que vieram do MESMO modelo-base aprenderam "diferenças"
em relação a ele. Somamos essas diferenças num modelo só.

    θ = θ_base + λ · junta( θ_coder − θ_base , θ_math − θ_base )

TIES (Yadav et al., 2023) junta sem que uma diferença apague a outra:
  1. aparar: em cada tensor, fica só a fração `densidade` das mudanças maiores (o resto é ruído);
  2. eleger o sinal: em cada número, vence o sinal com mais "massa" somada;
  3. média só das mudanças que concordam com o sinal eleito.
O resultado é um modelo comum (mesma arquitetura) → .gguf.

    python fusao_pesos.py --base Qwen/Qwen2.5-1.5B --modelo Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --modelo Qwen/Qwen2.5-Math-1.5B-Instruct --saida fusoes/ties-coder-math
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

import fusao
from fusao import Escritor, Leitor, pasta_do_modelo


def _aparar(d: torch.Tensor, densidade: float) -> torch.Tensor:
    if densidade >= 1:
        return d
    a = d.abs().flatten()
    k = max(1, int(a.numel() * (1 - densidade)))
    if a.numel() > 20_000_000:                                    # tensores enormes: limiar por amostra
        amostra = a[torch.randint(0, a.numel(), (2_000_000,))]
        lim = amostra.kthvalue(max(1, int(amostra.numel() * (1 - densidade)))).values
    else:
        lim = a.kthvalue(k).values
    return torch.where(d.abs() >= lim, d, torch.zeros_like(d))


def ties(base: torch.Tensor, outros: list[torch.Tensor], pesos: list[float], densidade: float, lam: float) -> torch.Tensor:
    b = base.float()
    ds = [_aparar(o.float() - b, densidade) * w for o, w in zip(outros, pesos)]
    D = torch.stack(ds)
    sinal = torch.sign(D.sum(0))
    concorda = (torch.sign(D) == sinal) & (D != 0)
    soma = (D * concorda).sum(0)
    n = concorda.sum(0).clamp(min=1)
    return (b + lam * soma / n).to(base.dtype)


def fundir_pesos(base: str, modelos: list[str], saida: Path, pesos: list[float] | None = None, densidade: float = 0.5,
                 lam: float = 1.0, config_de: int = 0, log=print, progresso=lambda f, t: None) -> dict:
    pesos = pesos or [1.0] * len(modelos)
    lb, lms = Leitor(base), [Leitor(m) for m in modelos]
    chaves = [k for k in lb.chaves() if all(l.tem(k) for l in lms)]
    faltam = [k for k in lb.chaves() if k not in chaves]
    if faltam:
        log(f"  {len(faltam)} pesos não existem em todos os modelos; ficam do base: {faltam[:3]}")
    saida = Path(saida)
    shutil.rmtree(saida, ignore_errors=True)
    esc = Escritor(saida)
    t0 = time.time()
    for i, k in enumerate(chaves):
        b = lb.get(k)
        outros = [l.get(k) for l in lms]
        if any(o.shape != b.shape for o in outros):
            raise ValueError(f"{k}: formatos diferentes {[tuple(o.shape) for o in outros]} × {tuple(b.shape)}")
        esc.add(k, ties(b, outros, pesos, densidade, lam))
        if i % 40 == 0:
            progresso(i / len(chaves), f"juntando os pesos: {i}/{len(chaves)}")
    for k in faltam:
        esc.add(k, lb.get(k))
    esc.fechar()
    # configuração e tokenizador do modelo escolhido (a RoPE vem dele)
    origem = pasta_do_modelo(modelos[config_de])
    cfg = fusao._cfg(modelos[config_de])
    (saida / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for f in origem.iterdir():
        if f.is_file() and (f.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "generation_config"))
                            or f.suffix == ".jinja"):
            shutil.copy2(f, saida / f.name)
    log(f"  fusão TIES pronta em {(time.time() - t0) / 60:.1f} min (densidade {densidade}, λ {lam})")
    return {"modo": "TIES", "base": base, "modelos": modelos, "pesos": pesos, "densidade": densidade, "lambda": lam}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Costureiro · IA única por soma de pesos (TIES)")
    ap.add_argument("--base", required=True, help="o ancestral comum (ex.: Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--modelo", action="append", required=True, help="modelos derivados dele (repita)")
    ap.add_argument("--peso", action="append", type=float, default=None)
    ap.add_argument("--densidade", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--gguf", default="q8_0")
    a = ap.parse_args(argv)
    res = fundir_pesos(a.base, a.modelo, Path(a.saida), a.peso, a.densidade, a.lam, log=lambda m: print(m, flush=True))
    (Path(a.saida) / "ties.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if a.gguf and a.gguf != "nao":
        import exportar
        arq = exportar.para_gguf(Path(a.saida), Path(a.saida).parent / f"{Path(a.saida).name}.{a.gguf}.gguf", a.gguf)
        print(f".gguf: {arq}")


if __name__ == "__main__":
    main()
