#!/usr/bin/env bash
# CA-TUNER del solver LM por REPLAY de ventana (Fases P3/I de la extension de
# seleccion de rama). Tres fases:
#   A) barrido EXTERNO del gen material latcheado KOBATHE_CLOSURE_STIFF_FRAC:
#      se lee UNA vez en un static const del material (KoBatheConcrete3D.hh),
#      asi que cada valor exige un proceso nuevo del driver;
#   B) por cada valor, Algoritmo Cultural con semilla fija sobre la ventana de
#      replay (KOBATHE_CA=1 + KOBATHE_LM_REPLAY_WINDOW="k0:m"): corre el
#      protocolo hasta k0-1, toma checkpoint (modelo+solver) y maximiza el
#      fitness re-jugando la ventana por candidato; persiste ca_history.csv y
#      ca_best_genome.json y termina (el replay NO continua el protocolo);
#   C) corrida FULL con el mejor genoma global (mayor fitness entre los valores
#      de A) y comparacion vs el baseline G0.
# La ventana por defecto (20:6) bracketea la PRIMERA reversa del protocolo
# 50/100/150/200 mm con SPS=8 (reversa en el paso 20, pico +50 mm).
# Uso:
#   [CLOSURE_FRACS='0.2 0.5 1.0'] [WINDOW=20:6] [CA_POP=8] [CA_GEN=10]
#   [CA_SEED=20260715] [AMPS='50,100,150,200'] [SPS=8] [NX=2 NY=2 NZ=4] \
#     bash scripts/kobathe_ca_tuner.sh data/output/p3_ca
set -u
export PATH="/c/msys64/ucrt64/bin:$PATH"
cd "$(dirname "$0")/.."
O="${1:-data/output/kobathe_ca_tuner_$(date +%Y%m%d)}"
EXE=./build-release/fall_n_reduced_rc_column_continuum_reference_benchmark.exe
THREADS=${THREADS:-3}
G0="${G0:-data/output/g0_baseline/cyc/lmnewton_hysteresis.csv}"
mkdir -p "$O/logs"

# Config LM base (identica a kobathe_lmnewton_cyclic.sh; el CA parte de aqui).
export KOBATHE_LMNEWTON=1
export KOBATHE_LM_ATOL="${LM_ATOL:-1e-6}"
export KOBATHE_LM_MUMAXFRAC="${LM_MUMAXFRAC:-1e-1}"
export KOBATHE_LM_MAXIT="${LM_MAXIT:-120}"
export KOBATHE_LM_MU0="${LM_MU0:-1e-2}"
export KOBATHE_LM_STAG="${LM_STAG:-12}"
export KOBATHE_LM_PREDICT="${LM_PREDICT:-1}"
export KOBATHE_LM_REVSUBDIV="${LM_REVSUBDIV:-1}"
export KOBATHE_LM_REVSUBN="${LM_REVSUBN:-3}"
export PETSC_OPTIONS="${PETSC_OPTIONS:-}"

base=(--analysis cyclic --amplitudes-mm "${AMPS:-50,100,150,200}" --steps-per-segment "${SPS:-8}"
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

WINDOW="${WINDOW:-20:6}"
CLOSURE_FRACS="${CLOSURE_FRACS:-0.2 0.5 1.0}"
CA_POP=${CA_POP:-8}
CA_GEN=${CA_GEN:-10}
CA_SEED=${CA_SEED:-20260715}
CA_TOPFRAC=${CA_TOPFRAC:-0.25}

# ── Fases A+B: CA por valor del gen material latcheado ───────────────────────
best_fit=""; best_dir=""; best_frac=""
for frac in $CLOSURE_FRACS; do
  D="$O/ca_frac_${frac}"
  mkdir -p "$D"
  echo "[ca-tuner] A/B frac=$frac window=$WINDOW pop=$CA_POP gen=$CA_GEN seed=$CA_SEED $(date +%H:%M:%S)"
  KOBATHE_CLOSURE_STIFF_FRAC=$frac \
  KOBATHE_CA=1 KOBATHE_LM_REPLAY_WINDOW="$WINDOW" \
  KOBATHE_CA_POP=$CA_POP KOBATHE_CA_GEN=$CA_GEN KOBATHE_CA_SEED=$CA_SEED \
  KOBATHE_CA_TOPFRAC=$CA_TOPFRAC \
  OMP_NUM_THREADS=$THREADS "$EXE" --output-dir "$D" "${base[@]}" \
    > "$O/logs/ca_frac_${frac}.log" 2>&1
  rc=$?
  J="$D/ca_best_genome.json"
  if [ $rc -ne 0 ] || [ ! -f "$J" ]; then
    echo "[ca-tuner]   frac=$frac FALLO rc=$rc (sin ca_best_genome.json; ver log)"
    continue
  fi
  fit=$(sed -n 's/.*"fitness": \([^,]*\),*/\1/p' "$J")
  echo "[ca-tuner]   frac=$frac fitness=$fit"
  if [ -z "$best_fit" ] || awk -v a="$fit" -v b="$best_fit" 'BEGIN{exit !(a>b)}'; then
    best_fit="$fit"; best_dir="$D"; best_frac="$frac"
  fi
done
if [ -z "$best_dir" ]; then
  echo "[ca-tuner] ninguna corrida A/B produjo genoma; abortando"
  exit 1
fi
echo "[ca-tuner] MEJOR: frac=$best_frac fitness=$best_fit ($best_dir)"

# ── Fase C: corrida FULL con el mejor genoma ─────────────────────────────────
J="$best_dir/ca_best_genome.json"
getj() { sed -n "s/.*\"$1\": \([^,\"]*\),*.*/\1/p" "$J"; }
export KOBATHE_LM_MU0="$(getj KOBATHE_LM_MU0)"
export KOBATHE_LM_GROW="$(getj KOBATHE_LM_GROW)"
export KOBATHE_LM_DROP="$(getj KOBATHE_LM_DROP)"
export KOBATHE_LM_STAG="$(getj KOBATHE_LM_STAG)"
export KOBATHE_LM_PREDICT_MAXSCALE="$(getj KOBATHE_LM_PREDICT_MAXSCALE)"
export KOBATHE_LM_ACCEPT_FLOOR="$(getj KOBATHE_LM_ACCEPT_FLOOR)"
export KOBATHE_LM_REVSUBDIV="$(getj KOBATHE_LM_REVSUBDIV)"
export KOBATHE_LM_REVSUBN="$(getj KOBATHE_LM_REVSUBN)"
echo "[ca-tuner] C: FULL genoma MU0=$KOBATHE_LM_MU0 GROW=$KOBATHE_LM_GROW DROP=$KOBATHE_LM_DROP STAG=$KOBATHE_LM_STAG MAXSCALE=$KOBATHE_LM_PREDICT_MAXSCALE FLOOR=$KOBATHE_LM_ACCEPT_FLOOR SUBDIV=$KOBATHE_LM_REVSUBDIV SUBN=$KOBATHE_LM_REVSUBN frac=$best_frac"
KOBATHE_CLOSURE_STIFF_FRAC=$best_frac \
OMP_NUM_THREADS=$THREADS "$EXE" --output-dir "$O/full" "${base[@]}" \
  > "$O/logs/full.log" 2>&1
echo "[ca-tuner] C rc=$? $(date +%H:%M:%S)"

# ── Resumen vs G0 (conv es la columna 9 de lmnewton_hysteresis.csv) ──────────
C="$O/full/lmnewton_hysteresis.csv"
summarize() { # $1 = etiqueta, $2 = csv
  awk -F, -v tag="$1" 'NR>1{n++; if($9==0)nc++; v=($4<0?-$4:$4); if(v>vm)vm=v}
    END{printf "[ca-tuner] %s: pasos=%d regularizados(noconv)=%d |V|max=%.4f MN\n", tag, n, nc, vm}' "$2"
}
if [ -f "$C" ]; then summarize FULL "$C"; else echo "[ca-tuner] (fase C sin CSV)"; fi
if [ -f "$G0" ]; then summarize "G0  " "$G0"; else echo "[ca-tuner] (sin baseline G0 en $G0)"; fi
