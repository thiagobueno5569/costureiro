# Avaliação em tarefas de verdade · `orq-coder-com-math`

Partes: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base) · `Qwen/Qwen2.5-Math-1.5B-Instruct` [0–16]

| modelo | HumanEval (pass@1) | MBPP sanitized (pass@1) | GSM8K (acerto) |
|---|---|---|---|
| **orquestra (orq-coder-com-math)** | 81.7% (n=60) | 68.3% (n=60) | 30.0% (n=60) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 73.3% (n=60) | 58.3% (n=60) | 45.0% (n=60) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 53.3% (n=60) | 30.0% (n=60) | 43.3% (n=60) |
| *roteador simples (IA inteira por pergunta)* | 73.3% (n=60) | 58.3% (n=60) | 43.3% (n=60) |
| *oráculo (alguma original acerta)* | 80.0% (n=60) | 65.0% (n=60) | 58.3% (n=60) |

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.
*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). *Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). Uma fusão só vale a pena se passar do roteador.

Respostas de cada item: arquivo `.json` ao lado.