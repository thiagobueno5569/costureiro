# Orquestra: base Qwen2.5-Math-1.5B-Instruct + solista Qwen2.5-Coder-1.5B-Instruct (camadas 0–19)

Rodado no RunPod (RTX 4090), 2026-11-22. 2 épocas, até 1.500 trechos por assunto, contexto 512.

| modelo | HumanEval (pass@1) | MBPP sanitized (pass@1) | GSM8K (acerto) |
|---|---|---|---|
| **orquestra (orq-math-com-coder)** | 63.3% (n=60) | 45.0% (n=60) | 33.3% (n=60) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` (base) | 53.3% (n=60) | 31.7% (n=60) | 43.3% (n=60) |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 73.3% (n=60) | 58.3% (n=60) | 45.0% (n=60) |
| *roteador simples (IA inteira por pergunta)* | 73.3% (n=60) | 58.3% (n=60) | 43.3% (n=60) |
| *oráculo (alguma original acerta)* | 80.0% (n=60) | 66.7% (n=60) | 58.3% (n=60) |

Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos.

**Leitura:** primeira transferência de habilidade medida em tarefa real: a base sobe +10 pontos no HumanEval
e +13 no MBPP com o solista. Mas fica abaixo do Coder sozinho e do roteador, e o GSM8K cai 10 pontos.
Neste formato de prompt o Coder-1.5B é melhor que o Math-1.5B até no GSM8K, então a direção que importa é a
inversa (base Coder). n=60 → margem de uns ±12 pontos.
