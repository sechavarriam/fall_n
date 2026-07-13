# Ko-Bathe 3D — curva cíclica limpia por continuación LM regularizada (modular)

## 1. El problema
La columna corta RC con concreto Ko-Bathe 3D no producía un lazo histerético limpio:
- **Anomalía de reversa:** en la descarga la fuerza *aumentaba* en lugar de disminuir
  (violación de la física: sin inercia excesiva la reversa tiene que descargar).
- **Atasco de carga a ~88 mm:** el Newton estándar (`nl.step_to`) se estancaba justo
  en el plateau de fluencia y no alcanzaba ±100/200 mm.

## 2. Diagnóstico (validado con datos)
- La anomalía de reversa es un problema de **multi-equilibrios del ablandamiento**:
  a una misma deriva existen dos equilibrios estáticos convergidos, y la relajación
  dinámica (control por desplazamiento) aterrizaba en la rama de **fuerza alta**.
- El atasco a 88 mm es una **falla de convergencia** (tangente casi-singular en el
  punto límite de carga), **NO un snapback** — la relajación dinámica encuentra
  equilibrios a todas las derivas. → El **arc-length es la herramienta equivocada**
  (se probó: no converge). La herramienta correcta es un **Newton regularizado**.

## 3. Solución: continuación Newton regularizado (Levenberg-Marquardt)
Resuelve `(K + μI) du = -R` con μ adaptativo:
- **μ > 0** vuelve resoluble la tangente casi-singular → **cruza el punto límite**
  (88 mm) donde Newton puro se atasca.
- **μ → 0** en la descarga bien-condicionada da **Newton LOCAL** warm-started desde
  el estado previo → se queda en la **rama física continua** (sin bump de reversa).
- **μ acotado** a `μ_max = μ_max_frac·‖diag(K)‖`: sin tope μ explota (1e28) en el
  punto límite y congela el paso; con tope da un paso Newton **amortiguado** que sí
  progresa a la solución regularizada.
- Corte por **estancamiento** (acepta el mejor residuo cuando toca el piso ~1e-4/1e-5
  del acoplamiento por penalización).

### Modularización (C++23, decisiones en tiempo de compilación)
`src/analysis/RegularizedNewtonContinuation.hh`:
- `concept RegularizationPolicy` — schedule de μ. Implementaciones:
  `LevenbergMarquardt` (μ0/grow/drop/μ_min) y `PureNewton`.
- `concept ContinuationBackend` — inyecta la física (ensamble R/K, control, commit).
- `template<ContinuationBackend Backend, RegularizationPolicy Reg=LevenbergMarquardt>
   class RegularizedNewtonContinuation` con `advance_to(double p)`.
- `template<typename AnalysisT> class NLAnalysisContinuationBackend` — adaptador
  header-only sobre `NLAnalysis` (sin capa virtual), reusa sus costuras
  (`evaluate_residual_at`, `evaluate_tangent_at`, `apply_incremental_control_parameter`,
  `accept_external_solution_step`, `create_global_vector`, `create_tangent_matrix`).
El driver (`ReducedRCColumnContinuumBaseline.cpp`, rama `KOBATHE_LMNEWTON=1`)
construye `NLAnalysisContinuationBackend backend{nl}` + `RegularizedNewtonContinuation
solver{backend, cfg, reg}` y hace el lazo sobre los targets.

## 4. Predictor secante (mata los spikes de reversa)
En la inversión de deriva el warm-start desde el estado extremo (ablandado) deja a
LM caer en la rama espuria de fuerza alta → **spike de reversa** (±33.7 kN a ±94 mm).
El predictor secante `u_pred = u_k + s(u_k - u_{k-1})`, `s=(p-p_k)/(p_k-p_{k-1})`,
arranca ya sobre la rama de descarga elástica. Con **fallback**: si el guess predicho
sube el residuo sobre el warm-start puro, se restaura `u_k` (nunca empeora).
Resultado AMPS=100: |V|max **33.7 → 31.1 kN**, lazo suave (ver overlay).
Config: `secant_predictor` (concept-friendly, en `RegularizedNewtonConfig`),
gate `KOBATHE_LM_PREDICT=1` (default ON).

## 5. Probado y DESCARTADO: sub-división de reversa
Hipótesis: partir los primeros incrementos post-reversa en sub-pasos suavizaría la
tangente. **Empeoró** (|V|max 31 → 80 kN): pasos menores convergen **más firme** a la
rama espuria. Esto **confirma** que el spike de reversa es multi-equilibrios físicos
del ablandamiento, **no** un sobre-paso numérico. Se deja env-gated
(`KOBATHE_LM_REVSUBDIV`, default 1 = OFF) como experimento reproducible.

## 6. Resultados
- **AMPS=100:** cruza ±88 mm, alcanza ±100 mm, descarga monótona en +100
  (25.4→21.6→20.5→17.5→13.2→8.4 kN, idéntica a la corrida inline de referencia).
- **Protocolo completo 50/100/150/200:** alcanza ±200 mm; envolvente por amplitud
  ~1.3× la referencia OpenSees hi-fi (`[17.5, 22.2, 21.3, 22.0]` kN — sobre-rigidez
  documentada del modelo 3D didáctico), y **reproduce el PLATEAU** de la referencia
  tras 100 mm (no crece monótona); **energía disipada monótona y física** (~33 kJ).
- Sobreviven pocos puntos NO convergidos (multi-equilibrio de reversa a alta
  amplitud), marcados explícitamente en la figura (no presentados como equilibrios).

## 7. Archivos
- `src/analysis/RegularizedNewtonContinuation.hh` (NUEVO — componente modular).
- `src/validation/ReducedRCColumnContinuumBaseline.cpp` (rama LM reescrita para usar
  el componente; predictor y sub-div env-gated; CSV con resid/conv).
- `scripts/kobathe_lmnewton_cyclic.sh` (NUEVO — corrida reproducible).
- `scratchpad/plot_lm_*.py`, `kobathe_lm_*.png` (figuras de monitoreo/finales).

## 8. Env knobs (todos default-safe; build por defecto = comportamiento del paper)
`KOBATHE_LMNEWTON=1` activa la rama. `KOBATHE_LM_{ATOL,RTOL,MAXIT,MU0,GROW,DROP,
MUMIN,MUMAXFRAC,STAG}` = tuning LM. `KOBATHE_LM_PREDICT` (default 1) predictor
secante. `KOBATHE_LM_REVSUBDIV`/`REVSUBN` sub-división de reversa (default OFF).

## 9. Nota de build
Compilar SIEMPRE en PowerShell (`$env:Path = "C:\msys64\ucrt64\bin;"+$env:Path;
cmake --build build-release ...`). En el Bash tool el `cc1plus` nativo no usa el
`TEMP` POSIX y muere sin diagnóstico en TUs grandes. Ver memoria `build-shell-powershell`.
