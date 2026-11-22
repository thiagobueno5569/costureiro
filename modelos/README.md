# Orquestras treinadas (só as pontes; os modelos Qwen são públicos)

| pasta | base | solista | HumanEval | MBPP | GSM8K |
|---|---|---|---|---|---|
| `orq-coder-com-math` | Qwen2.5-Coder-1.5B-Instruct | Qwen2.5-Math-1.5B-Instruct (camadas 0–16) | 81,7% | 68,3% | 30,0% |
| `orq-math-com-coder` | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Coder-1.5B-Instruct (camadas 0–19) | 63,3% | 45,0% | 33,3% |

n = 60 por tarefa (ver `resultados/avaliacao_orq-*.md`). Para usar: copie a pasta para `ias_novas/` e rode
`python avaliar.py --ia ias_novas/orq-coder-com-math`, ou carregue com `orquestra.IAOrquestra.abrir(...)`.
O primeiro uso baixa os dois modelos Qwen (~6 GB).
