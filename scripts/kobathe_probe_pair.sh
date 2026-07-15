#!/usr/bin/env bash
# Lanza una pareja de sondas Ko-Bathe (mono 20mm, 2x2x4 Hex27) con η extremos
# sobre la ley indicada, en paralelo. Uso:
#   kobathe_probe_pair.sh <outdir> <softening-law> [tip_mm] [steps]
set -u
export PATH="/c/msys64/ucrt64/bin:$PATH"
cd "$(dirname "$0")/.."

OUT="${1:?outdir}"
LAW="${2:?softening-law}"
TIP="${3:-20}"
STEPS="${4:-10}"
EXE=${FALLN_EXE:-./build/fall_n_reduced_rc_column_continuum_reference_benchmark.exe}

mkdir -p "$OUT/logs"

common=(--analysis monotonic --monotonic-tip-mm "$TIP" --monotonic-steps "$STEPS"
  --continuum-kinematics corotational --hex-order hex27 --nx 2 --ny 2 --nz 4
  --longitudinal-bias-power 2.2 --longitudinal-bias-location fixed-end
  --top-cap-mode lateral-translation-only
  --reinforcement-mode embedded-longitudinal-bars
  --embedded-boundary-mode dirichlet-rebar-endcap
  --axial-preload-transfer-mode composite-section-force-split
  --axial-compression-mn 0.02 --axial-preload-steps 4
  --penalty-alpha-scale-over-ec 10 --solver-policy canonical-cascade
  --predictor-policy secant --continuation monolithic
  --concrete-profile production-stabilized
  --kobathe-crack-softening-law "$LAW"
  --concrete-characteristic-length-mode fixed-end-longitudinal-host-edge-mm
  --concrete-fracture-energy-nmm 0.14
  --kobathe-crack-closure-transition-strain 1e-4
  --max-bisections 12 --print-progress)

OMP_NUM_THREADS=5 "$EXE" --output-dir "$OUT/eta_lo" "${common[@]}" \
  --kobathe-crack-eta-n 0.0001 --kobathe-crack-eta-s 0.10 \
  > "$OUT/logs/eta_lo.log" 2>&1 &
PID_LO=$!

OMP_NUM_THREADS=5 "$EXE" --output-dir "$OUT/eta_hi" "${common[@]}" \
  --kobathe-crack-eta-n 0.90 --kobathe-crack-eta-s 0.90 \
  > "$OUT/logs/eta_hi.log" 2>&1 &
PID_HI=$!

wait $PID_LO; RC_LO=$?
wait $PID_HI; RC_HI=$?
echo "RC_LO=$RC_LO RC_HI=$RC_HI"
tail -1 "$OUT/logs/eta_lo.log"
tail -1 "$OUT/logs/eta_hi.log"
