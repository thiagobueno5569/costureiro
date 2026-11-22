#!/usr/bin/env bash
# Do zero ao resultado, sem tela: recria a IA do artigo (Coder-0.5B camadas 0-6 → Math-1.5B camadas 8-27),
# e avalia em tarefas de verdade (HumanEval, MBPP, GSM8K) contra as duas originais.
#
#   cd /workspace && (git -C costureiro pull || git clone https://github.com/thiagobueno5569/costureiro) && bash costureiro/experimento.sh
#
# Roda em segundo plano (pode fechar o terminal). Acompanhe com:  tail -f /workspace/costureiro/resultados/experimento.log
# Tudo fica em /workspace, que não some quando o pod é parado (Stop). Terminate apaga tudo.
set -e
cd "$(dirname "$0")"
mkdir -p resultados
if [ -z "$COSTUREIRO_EM_FUNDO" ]; then
  COSTUREIRO_EM_FUNDO=1 nohup bash "$0" "$@" > resultados/experimento.log 2>&1 &
  echo "Rodando em segundo plano. Acompanhe com:"
  echo "  tail -f $(pwd)/resultados/experimento.log"
  exit 0
fi
[ -d /workspace ] && export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
N=${N:-40}                                  # itens por tarefa (0 = todos)
NOME=${NOME:-codigo-mais-matematica}
echo "== $(date) · preparando"
python -m pip install -q -r requirements.txt
if [ ! -s textos/programacao/python_stdlib.py.txt ] || [ ! -f textos/matematica/gsm8k.txt ]; then python preparar_textos.py; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ ! -f "ias_novas/$NOME/tradutores.pt" ]; then
  echo "== $(date) · costurando $NOME"
  python cli.py --a Qwen/Qwen2.5-Coder-0.5B-Instruct --b Qwen/Qwen2.5-Math-1.5B-Instruct --fim_a 6 --inicio_b 8 \
    --assunto programação=textos/programacao --assunto matemática=textos/matematica --epocas 6 --nome "$NOME"
else
  echo "== IA $NOME já existe, pulando a costura"
fi
echo "== $(date) · avaliando ($N itens por tarefa; 0 = todos)"
python avaliar.py --ia "ias_novas/$NOME" --n "$N"
echo "== $(date) · FIM. Resultado: $(pwd)/resultados/avaliacao_$NOME.md"
cat "resultados/avaliacao_$NOME.md"
