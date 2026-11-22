#!/usr/bin/env bash
# IA única: MoE montado, fusão de pesos (TIES) e destilação conferida — treina, avalia e gera o .gguf de cada.
#   cd /workspace && (git -C costureiro pull || git clone https://github.com/thiagobueno5569/costureiro) && bash costureiro/bateria_uma_ia.sh
cd "$(cd "$(dirname "$0")" && pwd)"
mkdir -p resultados/bateria
[ -d /workspace ] && export HF_HOME=/workspace/hf
setsid nohup bash -c 'python -m pip install -q -r requirements.txt; bash garantir_torch.sh; [ -s textos/programacao/python_stdlib.py.txt ] || python preparar_textos.py; python bateria.py --plano uma_ia "$@"' _ "$@" > resultados/bateria/saida.txt 2>&1 < /dev/null &
echo "Rodando. Acompanhe com:  watch -n 15 bash /workspace/costureiro/painel.sh"
