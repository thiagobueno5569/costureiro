#!/usr/bin/env bash
# Modo orquestra, do zero ao resultado (RunPod ou qualquer servidor com placa de 24 GB+).
#
#   cd /workspace && (git -C costureiro pull || git clone https://github.com/thiagobueno5569/costureiro) && bash costureiro/experimento_orquestra.sh
#
# Treina duas orquestras e avalia cada uma contra os originais, o roteador simples e o oráculo:
#   1) base Math-1.5B  + solista Coder-1.5B   (a de matemática chama a de código quando precisa)
#   2) base Coder-1.5B + solista Math-1.5B    (o contrário)
# Acompanhe:  tail -f /workspace/costureiro/resultados/orquestra.log
set -e
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SCRIPT")"
mkdir -p resultados
if [ -z "$COSTUREIRO_EM_FUNDO" ]; then
  COSTUREIRO_EM_FUNDO=1 nohup bash "$SCRIPT" "$@" > resultados/orquestra.log 2>&1 &
  echo "Rodando em segundo plano. Acompanhe com:"
  echo "  tail -f $(pwd)/resultados/orquestra.log"
  exit 0
fi
[ -d /workspace ] && export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
N=${N:-60}                    # itens por tarefa na avaliação (0 = todos)
EPOCAS=${EPOCAS:-2}
TRECHOS=${TRECHOS:-1500}      # trechos de treino por assunto
MATH=Qwen/Qwen2.5-Math-1.5B-Instruct
CODER=Qwen/Qwen2.5-Coder-1.5B-Instruct
echo "== $(date) · preparando"
python -m pip install -q -r requirements.txt
if [ ! -s textos/programacao/python_stdlib.py.txt ] || [ ! -f textos/matematica/gsm8k.txt ]; then python preparar_textos.py; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

treinar_e_avaliar () {   # nome base solista
  if [ ! -f "ias_novas/$1/ponte.pt" ]; then
    echo "== $(date) · treinando $1"
    python orquestra.py --base "$2" --solista "$3" --nome "$1" --epocas "$EPOCAS" --max_trechos "$TRECHOS"
  fi
  echo "== $(date) · avaliando $1 ($N itens por tarefa)"
  python avaliar.py --ia "ias_novas/$1" --n "$N"
}
treinar_e_avaliar orq-math-com-coder "$MATH" "$CODER"
treinar_e_avaliar orq-coder-com-math "$CODER" "$MATH"
echo "== $(date) · FIM"
for f in resultados/avaliacao_orq-*.md; do echo; cat "$f"; done
