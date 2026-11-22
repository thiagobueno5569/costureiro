# Avaliação em tarefas de verdade · `orq-coder-com-math`

Partes: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base) · `Qwen/Qwen2.5-Math-1.5B-Instruct` [0–16]

| modelo | GSM8K (acerto) |
|---|---|
| **orquestra (orq-coder-com-math)** | 37.0% (n=100) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 39.0% (n=100) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 49.0% (n=100) |
| *roteador simples (IA inteira por pergunta)* | 49.0% (n=100) |
| *oráculo (alguma original acerta)* | 65.0% (n=100) |

**Teste pareado (McNemar exato)**: em quantos itens só a IA nova acertou × só a outra acertou, e a chance de a diferença ser sorte (p < 0,05 = provavelmente real):

- GSM8K (acerto) · orquestra (orq-coder-com-math): contra Qwen2.5-Coder-1.5B-Instruct: +15/−17, p = 0.860; contra Qwen2.5-Math-1.5B-Instruct: +12/−24, p = 0.065; contra roteador: +12/−24, p = 0.065

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.
*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). *Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). Uma fusão só vale a pena se passar do roteador.

Respostas de cada item: arquivo `.json` ao lado.