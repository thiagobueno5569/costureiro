# Bateria completa no RunPod (RTX 4090, 2026-11-22) — 17 trabalhos, 194 min, 0 falhas

Conjuntos completos: HumanEval 164, MBPP sanitized 257, GSM8K 300 (primeiros 300 do teste).
Decodificação gulosa, mesmos prompts e paradas para todos.

| modelo | HumanEval | MBPP | GSM8K |
|---|---|---|---|
| **orquestra Coder + Math** (2 épocas) | **61,0%** | **63,8%** | 36,3% |
| **controle Coder + Coder** (mesmo treino, sem o Math) | 58,5% | 63,8% | 37,7% |
| **orquestra Coder + Math longa** (3 épocas, 2× dados) | 54,9% | 61,1% | 41,0% |
| Qwen2.5-Coder-1.5B-Instruct | 46,3% | 57,6% | 39,3% |
| Qwen2.5-Math-1.5B-Instruct | 33,5% | 29,2% | 48,0% |
| *roteador simples* | 46,3% | 57,6% | 48,0% |
| *oráculo* | 56,1% | 63,8% | 61,3% |

Teste pareado (McNemar exato), contra o Coder sozinho (só a nova acertou / só o Coder acertou):

| | HumanEval | MBPP | GSM8K |
|---|---|---|---|
| Coder + Math | +31/−7, p < 0,001 | +28/−12, p = 0,017 | +43/−52, p = 0,41 |
| controle Coder + Coder | +29/−9, p = 0,002 | +24/−8, p = 0,007 | +33/−38, p = 0,64 |
| Coder + Math longa | +25/−11, p = 0,029 | +24/−15, p = 0,20 | +50/−45, p = 0,68 |

## Conclusões
1. **O ganho em código é real, mas vem do treino da ponte, não do Math.** O controle (solista = cópia do
   próprio Coder, portão dá só 2% da vez a ele) ganha quase o mesmo: 58,5% contra 61,0% no HumanEval,
   empate no MBPP. A ponte funciona como um **ajuste fino barato**: modelo congelado, ~6 milhões de
   parâmetros, 17 minutos numa 4090, +12 pontos no HumanEval para o Coder-1.5B.
2. **A fusão com o Math não passou para a matemática.** Nenhuma orquestra chega aos 48% do Math sozinho
   no GSM8K; o roteador (48%) segue melhor. A versão longa usa mais o Math (25% da vez em matemática) e
   sobe para 41%, mas piora o código.
3. **Passar do roteador e do oráculo em código** vale para as três orquestras, mas como o controle também
   passa, isso mede o ajuste fino, não a soma de habilidades.

## Próximos passos
- Tratar a ponte como método de ajuste fino e compará-la com LoRA no mesmo orçamento (parâmetros, tempo).
- Para a soma de habilidades: portão por pergunta, solista com contexto de matemática no formato de chat
  do Math, e um par de modelos realmente complementares.
