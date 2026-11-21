# Costura de modelos por mudança de base

A = `Qwen/Qwen3-0.6B` (28 camadas) → tradutor W → B = `HuggingFaceTB/SmolLM2-1.7B` (24 camadas)
Texto: D:\IA\avaliacao_mista.txt, 266 trechos para achar W, 66 para medir.

**B inteiro: perplexidade 9.33** (referência)

| profundidade | camada A → B | mínimos quadrados | Procrustes (rotação) | sem tradutor |
|---|---|---|---|---|
| 0.25 | 7 → 6 | 2001.11 | 9856.04 | 174596.52 |
| 0.40 | 11 → 10 | 392.46 | 164.13 | 12168.10 |
| 0.50 | 14 → 12 | 279.02 | 177.29 | 3862.92 |
| 0.60 | 17 → 14 | 221.15 | 200.03 | 3886.25 |
| 0.75 | 21 → 18 | 213.42 | 236.27 | 7024.66 |

Tempo total: 3.1 min.