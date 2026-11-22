# orq-coder-com-math (orquestra)

Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (recebe na camada 21)
Solista: `Qwen/Qwen2.5-Math-1.5B-Instruct` (lê até a camada 16)

Bits por caractere no texto de teste (menor = melhor):

| assunto | só base | só Qwen2.5-Math-1.5B-Instruct | melhor de cada palavra | orquestra antes | **orquestra** | vez de cada um |
|---|---|---|---|---|---|---|
| programação | 0.467 | 1.241 | 0.433 | 0.467 | **0.447** | base 90%, Qwen2.5-Math-1.5B-Instruct 10% |
| matemática | 0.531 | 0.703 | 0.430 | 0.531 | **0.440** | base 74%, Qwen2.5-Math-1.5B-Instruct 26% |

Tempo: 16 min. Criada em 2026-11-22 05:53.