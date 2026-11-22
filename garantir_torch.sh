#!/usr/bin/env bash
# Placas novas (Blackwell: RTX PRO 6000, RTX 5090, B200) só funcionam com PyTorch feito para CUDA 12.8+.
# Se o PyTorch do pod não consegue rodar nada na placa, instala a versão nova.
teste='import torch; x = torch.ones(8, device="cuda", dtype=torch.bfloat16); print(torch.__version__, torch.cuda.get_device_name(0), float((x @ x).item()))'
if python -c "$teste" 2>/dev/null; then exit 0; fi
echo "PyTorch não roda nesta placa: instalando a versão para CUDA 12.8"
python -m pip install -q -U torch --index-url https://download.pytorch.org/whl/cu128
python -c "$teste" || { echo "ERRO: a placa continua sem funcionar com o PyTorch"; exit 1; }
