#!/usr/bin/env bash
# Orquestador NOCTURNO de la serie ca3 (programa del algoritmo cultural).
# 1) Espera a que las campanas ca2 liberen el binario de build-release2 y lo
#    reconstruye (incluye KOBATHE_CA_SOURCES).
# 2) Corre EN SECUENCIA la ablacion de fuentes de conocimiento y la variante
#    de semilla, todas con el fitness v2 anclado a la meseta fisica:
#      ca3_w156_5src : ventana 156:12, belief space COMPLETO (5 fuentes)
#      ca3_w112_5src : ventana 112:12, belief space COMPLETO (5 fuentes)
#      ca3_w156_seedB: ventana 156:12, 2 fuentes, semilla alterna (varianza)
# Comparables 1:1 con ca2_w156 / ca2_w112 (2 fuentes, semilla 20260715).
set -u
export PATH="/c/msys64/ucrt64/bin:$PATH"
cd "$(dirname "$0")/.."

EXE=./build-release2/fall_n_reduced_rc_column_continuum_reference_benchmark.exe
echo "[ca3] esperando a que ca2 libere el binario... $(date +%H:%M:%S)"
until ninja -C build-release2 fall_n_reduced_rc_column_continuum_reference_benchmark >/dev/null 2>&1; do
  sleep 300
done
echo "[ca3] binario reconstruido con KOBATHE_CA_SOURCES $(date +%H:%M:%S)"

run_one() { # $1=out  $2=window  $3=sources  $4=seed
  echo "[ca3] === $1 window=$2 sources=$3 seed=$4 $(date +%H:%M:%S)"
  EXE="$EXE" KOBATHE_LM_LINESEARCH=1 \
  KOBATHE_CA_VTARGET=0.0237 KOBATHE_CA_WZZ=0.5 \
  KOBATHE_CA_SOURCES="$3" CA_SEED="$4" \
  CLOSURE_FRACS="1.0" WINDOW="$2" CA_POP=10 CA_GEN=12 \
  G0=data/output/p1_linesearch/cyc/lmnewton_hysteresis.csv \
  bash scripts/kobathe_ca_tuner.sh "data/output/$1"
  echo "[ca3] === fin $1 rc=$? $(date +%H:%M:%S)"
}

run_one ca3_w156_5src  156:12 5 20260715
run_one ca3_w112_5src  112:12 5 20260715
run_one ca3_w156_seedB 156:12 2 20260717
echo "[ca3] serie completa $(date +%H:%M:%S)"
