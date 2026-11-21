# codigo-mais-matematica

Partes:
- 1. `Qwen/Qwen2.5-Coder-0.5B-Instruct` (camadas 0–9)
- 2. `Qwen/Qwen2.5-Math-1.5B-Instruct` (camadas 11–27)

Bits por caractere no texto de teste de cada assunto (menor = melhor):

| assunto | só a IA 1 | só a IA 2 | costurada (rotação) | **costurada (treinada)** |
|---|---|---|---|---|
| programação | 0.730 | 1.360 | 1.391 | **1.283** |
| matemática | 0.866 | 0.864 | 1.031 | **0.722** |

Curva do treino (bpc): 1.047, 1.029, 1.045

Tempo: 15.4 min. Criada em 2026-11-21 19:18.