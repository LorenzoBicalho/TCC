#!/bin/bash

set -e

exec > >(tee -a /home/lab/elm.log) 2>&1

echo "[$(date)] Iniciando coleta ELM"

cd /home/lab/Desktop/ELM327 || exit 1

#sleep 5

for i in {5..1}; do
    echo "Aguardando $i..."
    sleep 1
done

echo "[$(date)] Iniciando elm_data.py"
./venv/bin/python -u elm_data.py
