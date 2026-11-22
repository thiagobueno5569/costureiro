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

## MoE montado e destilação

Em andamento.
