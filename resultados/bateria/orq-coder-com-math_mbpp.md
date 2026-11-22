# Avaliação em tarefas de verdade · `orq-coder-com-math`

Partes: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base) · `Qwen/Qwen2.5-Math-1.5B-Instruct` [0–16]

| modelo | MBPP sanitized (pass@1) |
|---|---|
| **orquestra (orq-coder-com-math)** | 63.8% (n=257) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 57.6% (n=257) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 29.2% (n=257) |
| *roteador simples (IA inteira por pergunta)* | 57.6% (n=257) |
| *oráculo (alguma original acerta)* | 63.8% (n=257) |

**Teste pareado (McNemar exato)**: em quantos itens só a IA nova acertou × só a outra acertou, e a chance de a diferença ser sorte (p < 0,05 = provavelmente real):

- MBPP sanitized (pass@1) · orquestra (orq-coder-com-math): contra Qwen2.5-Coder-1.5B-Instruct: +28/−12, p = 0.017; contra Qwen2.5-Math-1.5B-Instruct: +98/−9, p = 0.000; contra roteador: +28/−12, p = 0.017

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.
*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). *Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). Uma fusão só vale a pena se passar do roteador.

Respostas de cada item: arquivo `.json` ao lado.