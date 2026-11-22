# orq-coder-com-math-longo (orquestra)

Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (recebe na camada 21)
Solista: `Qwen/Qwen2.5-Math-1.5B-Instruct` (lê até a camada 16)

Bits por caractere no texto de teste (menor = melhor):

| assunto | só base | só Qwen2.5-Math-1.5B-Instruct | melhor de cada palavra | orquestra antes | **orquestra** | vez de cada um |
|---|---|---|---|---|---|---|
| programação | 0.491 | 1.229 | 0.454 | 0.491 | **0.460** | base 91%, Qwen2.5-Math-1.5B-Instruct 9% |
| matemática | 0.525 | 0.676 | 0.421 | 0.525 | **0.434** | base 75%, Qwen2.5-Math-1.5B-Instruct 25% |

Tempo: 42 min. Criada em 2026-11-22 14:43.