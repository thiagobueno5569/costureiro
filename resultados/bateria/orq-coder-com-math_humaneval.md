# Avaliação em tarefas de verdade · `orq-coder-com-math`

Partes: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base) · `Qwen/Qwen2.5-Math-1.5B-Instruct` [0–16]

| modelo | HumanEval (pass@1) |
|---|---|
| **orquestra (orq-coder-com-math)** | 61.0% (n=164) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 46.3% (n=164) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 33.5% (n=164) |
| *roteador simples (IA inteira por pergunta)* | 46.3% (n=164) |
| *oráculo (alguma original acerta)* | 56.1% (n=164) |

**Teste pareado (McNemar exato)**: em quantos itens só a IA nova acertou × só a outra acertou, e a chance de a diferença ser sorte (p < 0,05 = provavelmente real):

- HumanEval (pass@1) · orquestra (orq-coder-com-math): contra Qwen2.5-Coder-1.5B-Instruct: +31/−7, p = 0.000; contra Qwen2.5-Math-1.5B-Instruct: +55/−10, p = 0.000; contra roteador: +31/−7, p = 0.000

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.
*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). *Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). Uma fusão só vale a pena se passar do roteador.

Respostas de cada item: arquivo `.json` ao lado.