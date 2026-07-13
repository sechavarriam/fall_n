#!/usr/bin/env bash
# Prueba de la continuacion ARC-LENGTH (quasi-estatico + KOBATHE_ARCLEN=1).
# Objetivo inmediato: verificar que CRUZA el punto limite de carga (~88mm) que
# nl.step_to no cruza, trazando 0->+100mm. NO usa el path dinamico.
set -u
export PATH="/c/msys64/ucrt64/bin:$PATH"
cd "$(dirname "$0")/.."
O="${1:-data/output/kobathe_arclen_test_20260713}"
EXE=./build-release/fall_n_reduced_rc_column_continuum_reference_benchmark.exe
THREADS=${THREADS:-8}
mkdir -p "$O/logs"

export PETSC_OPTIONS="${PETSC_OPTIONS:--ksp_type preonly -pc_type lu}"
# NADA de KOBATHE_DYNAMIC (usa el quasi-estatico donde vive la rama arc-length).
export KOBATHE_ARCLEN=1
export KOBATHE_ARCLEN_DS="${DS:-5.0e-3}"
export KOBATHE_ARCLEN_DS_MAX="${DS_MAX:-3.0e-2}"
export KOBATHE_ARCLEN_DS_MIN="${DS_MIN:-1.0e-5}"
# PSI (load_scaling) > 0 = arc-length ESFERICO: hace dc/dlambda=2*psi^2*(l-l0)!=0,
#  evitando el pivote CERO en (n,n) del sistema aumentado que con psi=0
#  (cilindrico) hace fallar la factorizacion LU (KSP reason -11). Ver test.
export KOBATHE_ARCLEN_PSI="${PSI:-0.12}"
export KOBATHE_ARCLEN_MAXSTEPS="${MAXSTEPS:-20000}"
# Tolerancia del corrector bordered TIGHT (elastico converge de verdad) +
#  accept-floor para el ablandamiento (residuo estancado ~1e-3 por penalizacion).
export KOBATHE_ARCLEN_RTOL="${RTOL:-1.0e-7}"
export KOBATHE_ARCLEN_CTOL="${CTOL:-1.0e-8}"
export KOBATHE_ARCLEN_ACCEPT_FLOOR="${ACCEPT_FLOOR:-1.0e-3}"
export KOBATHE_ARCLEN_MAXIT="${MAXIT:-60}"
# Regularización LM POR ITERACION dentro del kernel bordered (arc-length + LM
#  combinado): MU>0 = semilla de mu_frac*||diag(K)||; el kernel la adapta (sube en
#  la tangente casi-singular, baja en lo bien condicionado). 0 = arc-length puro.
export KOBATHE_ARCLEN_MU="${MU:-0.0}"
export KOBATHE_ARCLEN_MU_MAX="${MU_MAX:-1.0e-1}"
export KOBATHE_ARCLEN_MU_GROW="${MU_GROW:-4.0}"
export KOBATHE_ARCLEN_MU_DROP="${MU_DROP:-0.25}"
# Orden de la columna de carga dR/dλ (1/2/4) y del predictor (1=secante,2=curvatura).
export KOBATHE_ARCLEN_FD_ORDER="${FD_ORDER:-2}"
export KOBATHE_ARCLEN_PRED_ORDER="${PRED_ORDER:-2}"
# Predictor consciente de la reversa: paso de control fijo a traves del giro del
#  protocolo (deriva prescrita triangular). 1=ON (cierra el lazo), 0=OFF.
export KOBATHE_ARCLEN_TURN_AWARE="${TURN_AWARE:-1}"

base=(--analysis "${ANALYSIS:-monotonic}" --amplitudes-mm "${AMPS:-120}" --steps-per-segment 8
  --hex-order hex27 --nx "${NX:-2}" --ny "${NY:-2}" --nz "${NZ:-4}"
  --longitudinal-bias-power 2.2 --longitudinal-bias-location "${BIAS_LOCATION:-fixed-end}"
  --top-cap-mode lateral-translation-only
  --reinforcement-mode embedded-longitudinal-bars
  --embedded-boundary-mode dirichlet-rebar-endcap
  --axial-preload-transfer-mode composite-section-force-split
  --axial-compression-mn 0.02 --axial-preload-steps 4
  --penalty-alpha-scale-over-ec "${ALPHA_PEN:-10}"
  --concrete-profile production-stabilized
  --concrete-characteristic-length-mode fixed-end-longitudinal-host-edge-mm
  --concrete-fracture-energy-nmm 0.14
  --kobathe-crack-softening-law damage-secant
  --kobathe-crack-eta-n 0.01 --kobathe-crack-eta-s 0.25
  --kobathe-crack-closure-transition-strain 1e-4
  --solver-policy canonical-cascade --predictor-policy secant
  --max-bisections 16 --print-progress)

echo "[arclen] arranca $(date +%H:%M:%S) ds=$KOBATHE_ARCLEN_DS psi=$KOBATHE_ARCLEN_PSI amps=${AMPS:-120} analysis=${ANALYSIS:-monotonic}"
OMP_NUM_THREADS=$THREADS "$EXE" --output-dir "$O/run" "${base[@]}" > "$O/logs/run.log" 2>&1
rc=$?
echo "[arclen] fin rc=$rc $(date +%H:%M:%S)"
echo "--- ultimas lineas (incluye [arclen] DONE) ---"; tail -15 "$O/logs/run.log"
C="$O/run/arclen_hysteresis.csv"
if [ -f "$C" ]; then
  echo "--- arclen_hysteresis.csv: filas=$(wc -l < "$C") ---"; head -2 "$C"; echo "..."; tail -4 "$C"
  awk -F, 'NR>1{d=($3<0?-$3:$3); if(d>m)m=d} END{printf "[arclen] |drift|_max=%.1f mm  cruza_88=%s\n", m*1000, (m>0.088?"SI":"no")}' "$C"
else echo "  (no se genero arclen_hysteresis.csv)"; fi
