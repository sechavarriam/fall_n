#!/usr/bin/env bash
# Escalera de sondas Ko-Bathe (ley damage-secant) con descarte automático.
# Cada peldaño corre EN SOLITARIO; las compuertas (kobathe_probe_gates.py)
# deciden si la config sigue al siguiente peldaño.
#
#   Peldaño 1: monótona 20 mm, 20 pasos   (rápida; G1+G2)
#   Peldaño 2: monótona 50 mm, 25 pasos   (G2 en la amplitud de la figura)
#   Peldaño 3: cíclica ±25/±50 mm         (lazo: reversas y disipación)
#
# Uso: kobathe_ladder.sh <outdir> <tag> [flags extra del benchmark...]
set -u
export PATH="/c/msys64/ucrt64/bin:$PATH"
cd "$(dirname "$0")/.."

OUT="${1:?outdir}"; TAG="${2:?tag}"; shift 2
EXTRA=("$@")
EXE=${FALLN_EXE:-./build/fall_n_reduced_rc_column_continuum_reference_benchmark.exe}
GATES="python scripts/kobathe_probe_gates.py"

mkdir -p "$OUT/logs"

base=(--continuum-kinematics corotational --hex-order hex27 --nx 2 --ny 2 --nz 4
  --longitudinal-bias-power 2.2 --longitudinal-bias-location fixed-end
  --top-cap-mode lateral-translation-only
  --reinforcement-mode embedded-longitudinal-bars
  --embedded-boundary-mode dirichlet-rebar-endcap
  --axial-preload-transfer-mode composite-section-force-split
  --axial-compression-mn 0.02 --axial-preload-steps 4
  --penalty-alpha-scale-over-ec 10 --solver-policy canonical-cascade
  --predictor-policy secant --continuation monolithic
  --concrete-profile production-stabilized
  --kobathe-crack-softening-law damage-secant
  --concrete-characteristic-length-mode fixed-end-longitudinal-host-edge-mm
  --concrete-fracture-energy-nmm 0.14
  --kobathe-crack-eta-n 0.01 --kobathe-crack-eta-s 0.25
  --kobathe-crack-closure-transition-strain 1e-4
  --max-bisections 12 --print-progress)

run() { # nombre args...
  local name="$1"; shift
  echo "[escalera] $TAG/$name  $(date +%H:%M:%S)"
  OMP_NUM_THREADS=10 "$EXE" --output-dir "$OUT/$name" "${base[@]}" "${EXTRA[@]}" "$@" \
    > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  tail -1 "$OUT/logs/$name.log"
  return $rc
}

# ── Peldaño 1: monótona 20 mm ────────────────────────────────────────────
run "${TAG}_mono20" --analysis monotonic --monotonic-tip-mm 20 --monotonic-steps 20 || {
  echo "[escalera] ${TAG}: mono20 NO COMPLETO -> DESCARTADA"; exit 10; }
$GATES "$OUT/${TAG}_mono20" | tee "$OUT/logs/${TAG}_mono20.gates.json"
grep -q '"veredicto": "CANDIDATA"' "$OUT/logs/${TAG}_mono20.gates.json" || {
  echo "[escalera] ${TAG}: compuertas mono20 FALLAN -> DESCARTADA"; exit 11; }

# ── Peldaño 2: monótona 50 mm ────────────────────────────────────────────
run "${TAG}_mono50" --analysis monotonic --monotonic-tip-mm 50 --monotonic-steps 25 || {
  echo "[escalera] ${TAG}: mono50 NO COMPLETO -> DESCARTADA"; exit 20; }
$GATES "$OUT/${TAG}_mono50" | tee "$OUT/logs/${TAG}_mono50.gates.json"
grep -q '"veredicto": "CANDIDATA"' "$OUT/logs/${TAG}_mono50.gates.json" || {
  echo "[escalera] ${TAG}: compuertas mono50 FALLAN -> DESCARTADA"; exit 21; }

# ── Peldaño 3: cíclica ±25/±50 mm ────────────────────────────────────────
run "${TAG}_cyc50" --analysis cyclic --amplitudes-mm 25,50 --steps-per-segment 10 || {
  echo "[escalera] ${TAG}: cíclica NO COMPLETA -> DESCARTADA"; exit 30; }
echo "[escalera] ${TAG}: CICLO COMPLETO — candidata a figura"
exit 0
