# orq-math-com-coder (orquestra)

Base: `Qwen/Qwen2.5-Math-1.5B-Instruct` (recebe na camada 15)
Solista: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (lê até a camada 19)

Bits por caractere no texto de teste (menor = melhor):

| assunto | só base | só Qwen2.5-Coder-1.5B-Instruct | melhor de cada palavra | orquestra antes | **orquestra** | vez de cada um |
|---|---|---|---|---|---|---|
| programação | 1.241 | 0.467 | 0.433 | 1.237 | **0.828** | base 24%, Qwen2.5-Coder-1.5B-Instruct 76% |
| matemática | 0.703 | 0.531 | 0.430 | 0.702 | **0.454** | base 52%, Qwen2.5-Coder-1.5B-Instruct 48% |

Tempo: 18 min. Criada em 2026-11-22 04:37.