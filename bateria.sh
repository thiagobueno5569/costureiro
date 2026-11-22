#!/usr/bin/env bash
# Roda a bateria inteira em segundo plano e desliga o pod no fim.
#   cd /workspace && (git -C costureiro pull || git clone https://github.com/thiagobueno5569/costureiro) && bash costureiro/bateria.sh
cd "$(cd "$(dirname "$0")" && pwd)"
mkdir -p resultados/bateria
[ -d /workspace ] && export HF_HOME=/workspace/hf
python -m pip install -q -r requirements.txt
[ -s textos/programacao/python_stdlib.py.txt ] && [ -f textos/matematica/gsm8k.txt ] || python preparar_textos.py
nohup python bateria.py --desligar "$@" > resultados/bateria/saida.txt 2>&1 &
echo "Bateria rodando. Acompanhe com:  tail -f $(pwd)/resultados/bateria/gerente.log"
