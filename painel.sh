#!/usr/bin/env bash
# Painel da bateria:  watch -n 15 bash painel.sh
cd "$(cd "$(dirname "$0")" && pwd)/resultados/bateria" || exit 1
{
echo "=== GERENTE (ultimas acoes) ==="; tail -n 8 gerente.log | cut -c1-110; echo
echo "=== CADA TRABALHO ==="
for f in *.log; do
  [ "$f" = gerente.log ] && continue
  printf "%-36s %s\n" "${f%.log}" "$(grep -E 'por item|poca|passo|matem|digo:|roteador|exemplos|gguf' "$f" | tail -n1 | cut -c1-74)"
done
echo; echo "=== PLACA ==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
} | iconv -f utf-8 -t ascii//TRANSLIT 2>/dev/null
