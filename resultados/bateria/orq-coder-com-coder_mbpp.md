# Avaliação em tarefas de verdade · `orq-coder-com-coder`

Partes: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base) · `Qwen/Qwen2.5-Coder-1.5B-Instruct` [0–16]

| modelo | MBPP sanitized (pass@1) |
|---|---|
| **orquestra (orq-coder-com-coder)** | 63.8% (n=257) |

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.
*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). *Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). Uma fusão só vale a pena se passar do roteador.

Respostas de cada item: arquivo `.json` ao lado.