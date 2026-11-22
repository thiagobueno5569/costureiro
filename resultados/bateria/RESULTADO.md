# Avaliação em tarefas de verdade · `orq-coder-com-coder`

Partes: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base) · `Qwen/Qwen2.5-Coder-1.5B-Instruct` [0–16]

| modelo | GSM8K (acerto) | HumanEval (pass@1) | MBPP sanitized (pass@1) |
|---|---|---|---|
| **orquestra (orq-coder-com-coder)** | 37.7% (n=300) | 58.5% (n=164) | 63.8% (n=257) |
| **orquestra (orq-coder-com-math-longo)** | 41.0% (n=300) | 54.9% (n=164) | 61.1% (n=257) |
| **orquestra (orq-coder-com-math)** | 36.3% (n=300) | 61.0% (n=164) | 63.8% (n=257) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 39.3% (n=300) | 46.3% (n=164) | 57.6% (n=257) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 48.0% (n=300) | 33.5% (n=164) | 29.2% (n=257) |
| *roteador simples (IA inteira por pergunta)* | 48.0% (n=300) | 46.3% (n=164) | 57.6% (n=257) |
| *oráculo (alguma original acerta)* | 61.3% (n=300) | 56.1% (n=164) | 63.8% (n=257) |

**Teste pareado (McNemar exato)**: em quantos itens só a IA nova acertou × só a outra acertou, e a chance de a diferença ser sorte (p < 0,05 = provavelmente real):

- GSM8K (acerto) · orquestra (orq-coder-com-coder): contra Qwen2.5-Coder-1.5B-Instruct: +33/−38, p = 0.635; contra Qwen2.5-Math-1.5B-Instruct: +42/−73, p = 0.005; contra roteador: +42/−73, p = 0.005
- GSM8K (acerto) · orquestra (orq-coder-com-math-longo): contra Qwen2.5-Coder-1.5B-Instruct: +50/−45, p = 0.682; contra Qwen2.5-Math-1.5B-Instruct: +45/−66, p = 0.057; contra roteador: +45/−66, p = 0.057
- GSM8K (acerto) · orquestra (orq-coder-com-math): contra Qwen2.5-Coder-1.5B-Instruct: +43/−52, p = 0.412; contra Qwen2.5-Math-1.5B-Instruct: +43/−78, p = 0.002; contra roteador: +43/−78, p = 0.002
- HumanEval (pass@1) · orquestra (orq-coder-com-coder): contra Qwen2.5-Coder-1.5B-Instruct: +29/−9, p = 0.002; contra Qwen2.5-Math-1.5B-Instruct: +51/−10, p = 0.000; contra roteador: +29/−9, p = 0.002
- HumanEval (pass@1) · orquestra (orq-coder-com-math-longo): contra Qwen2.5-Coder-1.5B-Instruct: +25/−11, p = 0.029; contra Qwen2.5-Math-1.5B-Instruct: +48/−13, p = 0.000; contra roteador: +25/−11, p = 0.029
- HumanEval (pass@1) · orquestra (orq-coder-com-math): contra Qwen2.5-Coder-1.5B-Instruct: +31/−7, p = 0.000; contra Qwen2.5-Math-1.5B-Instruct: +55/−10, p = 0.000; contra roteador: +31/−7, p = 0.000
- MBPP sanitized (pass@1) · orquestra (orq-coder-com-coder): contra Qwen2.5-Coder-1.5B-Instruct: +24/−8, p = 0.007; contra Qwen2.5-Math-1.5B-Instruct: +97/−8, p = 0.000; contra roteador: +24/−8, p = 0.007
- MBPP sanitized (pass@1) · orquestra (orq-coder-com-math-longo): contra Qwen2.5-Coder-1.5B-Instruct: +24/−15, p = 0.200; contra Qwen2.5-Math-1.5B-Instruct: +90/−8, p = 0.000; contra roteador: +24/−15, p = 0.200
- MBPP sanitized (pass@1) · orquestra (orq-coder-com-math): contra Qwen2.5-Coder-1.5B-Instruct: +28/−12, p = 0.017; contra Qwen2.5-Math-1.5B-Instruct: +98/−9, p = 0.000; contra roteador: +28/−12, p = 0.017

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.
*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). *Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). Uma fusão só vale a pena se passar do roteador.

Respostas de cada item: arquivo `.json` ao lado.