# IA única (um só `.gguf`): MoE montado, fusão de pesos (TIES) e destilação conferida

Rodado em 22/11/2026 numa RTX PRO 6000 (96 GB) do RunPod, com o Coder-1.5B e o Math-1.5B (Qwen2.5, Instruct).
Arquivo atualizado aos poucos, conforme cada parte termina, porque o pod não tem disco permanente.

## Originais (referência)

| modelo | HumanEval (pass@1) |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 46.3% (n=164) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 32.9% (n=164) |
| *roteador simples (IA inteira por pergunta)* | 46.3% (n=164) |
| *oráculo (alguma original acerta)* | 56.1% (n=164) |

| modelo | MBPP sanitized (pass@1) |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 57.6% (n=257) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 28.8% (n=257) |
| *roteador simples (IA inteira por pergunta)* | 57.6% (n=257) |
| *oráculo (alguma original acerta)* | 63.4% (n=257) |

## Fusão de pesos (TIES): quebra o modelo

TIES sobre o ancestral comum `Qwen/Qwen2.5-1.5B` (densidade 0.5, λ 1.0). A fusão ficou pronta em 1.6 min e o
`.gguf` saiu (1.6 GB), mas o modelo gera texto sem sentido:

```
def soma(a, b):
    """Retorna a soma."""
� Chandler其中有qf(ern.ail 解_qu_ \nWe**Eb 十e tprPcTo作巢prime_chi  elementsp元入入这个页…
```

0 de 10 nos primeiros itens de HumanEval, MBPP e GSM8K. As avaliações foram interrompidas para não gastar a placa.

**Não é erro do código:** fundindo só o Coder (densidade 1.0), o resultado é idêntico ao Coder original
(diferença máxima 1.5e-8 em todos os 338 tensores). O que quebra é somar os dois: o Coder (5,5 trilhões de tokens
de código a mais) e o Math se afastaram demais do ancestral para que as mudanças deles se somem.
É por isso que o MoE mantém as duas partes separadas, como especialistas.

## MoE montado

Atenção do Coder; em cada camada, a FFN do Coder e a do Math viram dois especialistas (top-2), com um roteador
iniciado pelas médias de cada assunto e depois treinado (só ele; 800 passos em textos de treino do MBPP, do GSM8K e
código real). Perda no teste do roteador: **5.81 → 1.06**. Montagem e treino: 26 min. `.gguf` q8_0: 3.1 GB.

| modelo | HumanEval (pass@1) |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 46.3% (n=164) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 32.9% (n=164) |
| **IA única (MoE Coder+Math)** | **52.4%** (n=164) |
| *roteador simples (IA inteira por pergunta)* | 46.3% (n=164) |
| *oráculo (alguma original acerta)* | 56.1% (n=164) |

Teste pareado (McNemar exato), MoE contra o Coder: +22/−12 (22 problemas que só o MoE acertou, 12 que só o Coder
acertou), **p = 0.121**: provável ganho, mas ainda pode ser sorte. Contra o Math: +43/−11, p < 0.001.

Cuidado ao ler: o roteador foi treinado com textos de treino do MBPP e do GSM8K (nunca o teste). Só ele (2×1536
números por camada) foi treinado, mas parte do ganho pode vir desse ajuste, como aconteceu com a ponte da orquestra.

| modelo | MBPP sanitized (pass@1) |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 57.6% (n=257) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 28.8% (n=257) |
| **IA única (MoE Coder+Math)** | **49.8%** (n=257) |
| *roteador simples (IA inteira por pergunta)* | 57.6% (n=257) |
| *oráculo (alguma original acerta)* | 63.4% (n=257) |

No MBPP o MoE **perde** para o Coder sozinho: +15/−35, **p = 0.007** (diferença real). Somando as duas provas de
código (421 problemas), o MoE fica em +37/−47: no código, misturar o especialista de matemática mais atrapalha
do que ajuda.

| modelo | GSM8K (acerto) |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 40.3% (n=439) |
| `Qwen/Qwen2.5-Math-1.5B-Instruct` | 45.6% (n=439) |
| **IA única (MoE Coder+Math)** | **35.1%** (n=439) |
| *roteador simples (IA inteira por pergunta)* | 45.6% (n=439) |
| *oráculo (alguma original acerta)* | 60.4% (n=439) |

GSM8K: as 439 primeiras questões do teste (fatias 0/6 e 1/6 de 1.319, por falta de saldo). O MoE fica **abaixo
dos dois**: contra o Math +56/−102 (p < 0.001), contra o Coder +59/−82 (p = 0.064).

**Conclusão do MoE:** montado assim (atenção do Coder, FFNs como especialistas, só o roteador treinado), ele não
vira uma IA melhor que as partes. A atenção do Coder não "sabe" usar a FFN do Math: as duas foram treinadas juntas
em outro modelo, e trocar a metade de um modelo pela de outro quebra o acordo entre elas. Treinar só o roteador
não basta; seria preciso ajustar também a atenção (ou ao menos uma ponte por camada).

## Destilação conferida

Em andamento.
