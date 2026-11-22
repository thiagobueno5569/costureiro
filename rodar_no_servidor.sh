#!/usr/bin/env bash
# Costureiro num servidor com placa de vídeo (RunPod, Vast, Lambda...).
# Uso, no terminal do servidor:
#   git clone https://github.com/thiagobueno5569/costureiro && cd costureiro && bash rodar_no_servidor.sh
set -e
cd "$(dirname "$0")"
# modelos e cache no disco grande (no RunPod é o /workspace, que não some ao reiniciar)
if [ -d /workspace ]; then export HF_HOME=/workspace/hf; fi
python -m pip install -q -r requirements.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# refaz os textos se a parte de programação ficou vazia (versão antiga procurava a pasta do Python do Windows)
if [ ! -s textos/programacao/python_stdlib.py.txt ] || [ ! -f textos/matematica/gsm8k.txt ]; then python preparar_textos.py; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo
echo "Abrindo o Costureiro. Procure a linha 'Running on public URL: https://....gradio.live' e abra no navegador."
COSTUREIRO_HOST=0.0.0.0 COSTUREIRO_PORTA=${COSTUREIRO_PORTA:-7860} COSTUREIRO_LINK=1 python app.py
