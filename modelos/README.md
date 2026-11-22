# Orquestras treinadas (só as pontes; os modelos Qwen são públicos)

| pasta | base | solista | HumanEval | MBPP | GSM8K |
|---|---|---|---|---|---|
| `orq-coder-com-math` | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Math-1.5B-Instruct (camadas 0–16) | 81,7% | 68,3% | 30,0% |
| `orq-coder-com-coder` (controle) | Qwen2.5-Coder-1.5B-Instruct | o próprio Coder | 58,5% | 63,8% | 37,7% |
| `orq-coder-com-math-longo` | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Math-1.5B-Instruct, 3 épocas | 54,9% | 61,1% | 41,0% |
| `orq-math-com-coder` | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct (camadas 0–19) | 63,3% | 45,0% | 33,3% |

Números das 3 primeiras linhas: conjuntos completos (HumanEval 164, MBPP 257, GSM8K 300), `resultados/bateria/RESULTADO.md`. `orq-coder-com-math` no completo: 61,0% / 63,8% / 36,3%. `orq-math-com-coder`: n = 60 por tarefa (ver `resultados/avaliacao_orq-*.md`). Para usar: copie a pasta para `ias_novas/` e rode
`python avaliar.py --ia ias_novas/orq-coder-com-math`, ou carregue com `orquestra.IAOrquestra.abrir(...)`.
O primeiro uso baixa os dois modelos Qwen (~6 GB).
