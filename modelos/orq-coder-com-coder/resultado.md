# orq-coder-com-coder (orquestra)

Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (recebe na camada 18)
Solista: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (lê até a camada 16)

Bits por caractere no texto de teste (menor = melhor):

| assunto | só base | só Qwen2.5-Coder-1.5B-Instruct | melhor de cada palavra | orquestra antes | **orquestra** | vez de cada um |
|---|---|---|---|---|---|---|
| programação | 0.467 | 0.467 | 0.467 | 0.467 | **0.445** | base 98%, Qwen2.5-Coder-1.5B-Instruct 2% |
| matemática | 0.531 | 0.531 | 0.531 | 0.531 | **0.434** | base 98%, Qwen2.5-Coder-1.5B-Instruct 2% |

Tempo: 17 min. Criada em 2026-11-22 14:01.