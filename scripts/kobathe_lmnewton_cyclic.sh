#!/usr/bin/env bash
# CICLICO cuasi-estatico por NEWTON REGULARIZADO (Levenberg-Marquardt).
# Ruta modular RegularizedNewtonContinuation<Backend, LevenbergMarquardt>:
#   (K + mu*I) du = -R con mu adaptativo acotado a mu_max_frac*||diag(K)||.
#   mu>0 cruza el punto limite (~88mm) donde Newton puro se atasca; mu->0 en
#   la descarga bien-condicionada da Newton LOCAL warm-started -> se queda en
#   la rama fisica continua, sin el "bump" de reversa de la relajacion dinamica.
# NO usa KOBATHE_DYNAMIC (esta es la continuacion cuasi-estatica en solve_outcome).
# Escribe <O>/cyc/lmnewton_hysteresis.csv (step,p,drift_m,base_shear_MN,
#   base_shear_coupled_MN,iters,mu).
set -u
export PATH="/c/msys64/ucrt64/bin:$PATH"
cd "$(dirname "$0")/.."
O="${1:-data/output/kobathe_lmnewton_cyclic_$(date +%Y%m%d)}"
EXE=./build-release/fall_n_reduced_rc_column_continuum_reference_benchmark.exe
THREADS=${THREADS:-3}
mkdir -p "$O/logs"

# Newton regularizado (gate) + tuning validado (ATOL 1e-6, mu_max=1e-1*||diagK||).
export KOBATHE_LMNEWTON=1
export KOBATHE_LM_ATOL="${LM_ATOL:-1e-6}"
export KOBATHE_LM_MUMAXFRAC="${LM_MUMAXFRAC:-1e-1}"
export KOBATHE_LM_MAXIT="${LM_MAXIT:-120}"
export KOBATHE_LM_MU0="${LM_MU0:-1e-2}"
export KOBATHE_LM_STAG="${LM_STAG:-12}"
[ -n "${LM_GROW:-}" ] && export KOBATHE_LM_GROW="$LM_GROW"
[ -n "${LM_DROP:-}" ] && export KOBATHE_LM_DROP="$LM_DROP"
# Predictor secante: ON por defecto (mata los spikes de reversa; ver informe).
# Sub-division de reversa: OFF por defecto (REVSUBDIV=1). Se probó (REVSUBDIV=4)
#  y EMPEORA el lazo -> confirma que el spike de reversa es multi-equilibrios
#  del ablandamiento, no sobre-paso: pasos menores convergen MAS firme a la
#  rama espuria de fuerza alta. Se deja env-gated como experimento reproducible.
export KOBATHE_LM_PREDICT="${LM_PREDICT:-1}"
export KOBATHE_LM_REVSUBDIV="${LM_REVSUBDIV:-1}"
export KOBATHE_LM_REVSUBN="${LM_REVSUBN:-3}"

# Solver lineal directo (LU) dentro del lazo LM (ver componente).
export PETSC_OPTIONS="${PETSC_OPTIONS:-}"

base=(--analysis cyclic --amplitudes-mm "${AMPS:-100}" --steps-per-segment "${SPS:-8}"
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

# Volcado VTK opcional (env-gated, default OFF): WRITE_VTK=1 escribe malla, puntos
# de Gauss, fisuras y tubos de barra por paso en <O>/cyc/vtk/ (ParaView, .pvd).
# VTK_STRIDE controla la densidad temporal (1 = cada paso => evolucion continua).
[ -n "${WRITE_VTK:-}" ] && base+=(--write-vtk --vtk-stride "${VTK_STRIDE:-1}")

echo "[lm-cyc] arranca $(date +%H:%M:%S)  amps=${AMPS:-100} sps=${SPS:-8} atol=$KOBATHE_LM_ATOL mumaxfrac=$KOBATHE_LM_MUMAXFRAC vtk=${WRITE_VTK:-0}/${VTK_STRIDE:-1}"
OMP_NUM_THREADS=$THREADS "$EXE" --output-dir "$O/cyc" "${base[@]}" \
  > "$O/logs/cyc.log" 2>&1
rc=$?
echo "[lm-cyc] fin rc=$rc $(date +%H:%M:%S)"
echo "--- ultimas lineas del log (incluye [lm] DONE) ---"
tail -20 "$O/logs/cyc.log"
C="$O/cyc/lmnewton_hysteresis.csv"
if [ -f "$C" ]; then
  echo "--- lmnewton_hysteresis.csv ---"; head -2 "$C"; echo "..."; tail -3 "$C"
  awk -F, 'NR>1{d=($3<0?-$3:$3); if(d>m)m=d} END{printf "[lm-cyc] |drift|_max=%.4f m (%.1f mm)  cruza_88=%s\n", m, m*1000, (m>0.088?"SI":"no")}' "$C"
else
  echo "  (no se genero lmnewton_hysteresis.csv)"
fi
