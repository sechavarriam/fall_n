param(
    [string]$BuildDir = "build",
    [int]$Jobs = [Math]::Max([Environment]::ProcessorCount, 1),
    [switch]$SkipConfigure,
    [switch]$SkipDocs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $repoRoot $BuildDir

function Invoke-Step {
    param([string]$Command, [string]$WorkingDirectory = $repoRoot)
    Write-Host "==> $Command"
    Push-Location $WorkingDirectory
    try {
        Invoke-Expression $Command
    } finally {
        Pop-Location
    }
}

function Invoke-BuildTargets {
    param(
        [string[]]$Targets,
        [string]$Label
    )

    Write-Host "==> Building $Label"
    foreach ($target in $Targets) {
        Invoke-Step ("ninja -C `"$buildPath`" -j$Jobs " + $target)
    }
}

function Resolve-PythonLauncher {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return "py -3"
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return "python"
    }

    throw "Python launcher not found. The OpenSeesPy dry-run contract requires either 'py' or 'python'."
}

if (-not $SkipConfigure) {
    Invoke-Step "cmake -S `"$repoRoot`" -B `"$buildPath`" -G Ninja"
}

$regressionTargets = @(
    "fall_n_multiscale_api_test",
    "fall_n_micro_solve_executor_test",
    "fall_n_evolver_advanced_test",
    "fall_n_tangent_validation_benchmark_test",
    "fall_n_coupling_residual_benchmark_test",
    "fall_n_beam_test",
    "fall_n_beam_axis_quadrature_test",
    "fall_n_reduced_rc_column_structural_matrix_test",
    "fall_n_reduced_rc_column_material_baseline_test",
    "fall_n_reduced_rc_column_section_baseline_test",
    "fall_n_reduced_rc_column_moment_curvature_closure_test",
    "fall_n_reduced_rc_column_moment_curvature_closure_matrix_test",
    "fall_n_reduced_rc_column_node_refinement_study_test",
    "fall_n_reduced_rc_column_cyclic_node_refinement_study_test",
    "fall_n_reduced_rc_column_cyclic_continuation_sensitivity_study_test",
    "fall_n_reduced_rc_column_quadrature_sensitivity_study_test",
    "fall_n_reduced_rc_column_cyclic_quadrature_sensitivity_study_test",
    "fall_n_reduced_rc_column_validation_claim_catalog_test",
    "fall_n_reduced_rc_column_benchmark_trace_catalog_test",
    "fall_n_reduced_rc_column_evidence_closure_catalog_test",
    "fall_n_computational_variational_slice_catalog_test",
    "fall_n_computational_claim_trace_catalog_test",
    "fall_n_computational_validation_readiness_catalog_test",
    "fall_n_validation_campaign_catalog_test",
    "fall_n_steppable_solver_test",
    "fall_n_steppable_dynamic_test"
)

$heavyExampleTargets = @(
    "fall_n_lshaped_multiscale",
    "fall_n_lshaped_multiscale_16",
    "fall_n_table_multiscale",
    "fall_n_table_cyclic_validation"
)

Invoke-BuildTargets -Targets $regressionTargets -Label "multiscale regression targets"
Invoke-BuildTargets -Targets $heavyExampleTargets -Label "heavy multiscale examples"

$executables = @(
    "fall_n_multiscale_api_test.exe",
    "fall_n_micro_solve_executor_test.exe",
    "fall_n_evolver_advanced_test.exe",
    "fall_n_tangent_validation_benchmark_test.exe",
    "fall_n_coupling_residual_benchmark_test.exe",
    "fall_n_beam_test.exe",
    "fall_n_beam_axis_quadrature_test.exe",
    "fall_n_reduced_rc_column_structural_matrix_test.exe",
    "fall_n_reduced_rc_column_material_baseline_test.exe",
    "fall_n_reduced_rc_column_section_baseline_test.exe",
    "fall_n_reduced_rc_column_moment_curvature_closure_test.exe",
    "fall_n_reduced_rc_column_moment_curvature_closure_matrix_test.exe",
    "fall_n_reduced_rc_column_node_refinement_study_test.exe",
    "fall_n_reduced_rc_column_cyclic_node_refinement_study_test.exe",
    "fall_n_reduced_rc_column_cyclic_continuation_sensitivity_study_test.exe",
    "fall_n_reduced_rc_column_quadrature_sensitivity_study_test.exe",
    "fall_n_reduced_rc_column_cyclic_quadrature_sensitivity_study_test.exe",
    "fall_n_reduced_rc_column_validation_claim_catalog_test.exe",
    "fall_n_reduced_rc_column_benchmark_trace_catalog_test.exe",
    "fall_n_reduced_rc_column_evidence_closure_catalog_test.exe",
    "fall_n_rc_section_layout_test.exe",
    "fall_n_computational_variational_slice_catalog_test.exe",
    "fall_n_computational_claim_trace_catalog_test.exe",
    "fall_n_computational_validation_readiness_catalog_test.exe",
    "fall_n_validation_campaign_catalog_test.exe",
    "fall_n_steppable_solver_test.exe",
    "fall_n_steppable_dynamic_test.exe"
)

foreach ($exe in $executables) {
    Invoke-Step ("& `"" + (Join-Path $buildPath $exe) + "`"")
}

$pythonLauncher = Resolve-PythonLauncher
$openseesDryRunDir = Join-Path $repoRoot "data/output/cyclic_validation/opensees_reference_ci_dry_run"
Invoke-Step (
    "$pythonLauncher scripts/opensees_reduced_rc_column_reference.py " +
    "--dry-run --analysis cyclic --beam-element-family disp " +
    "--beam-integration legendre --integration-points 3 " +
    "--mapping-policy cyclic-diagnostic " +
    "--geom-transf linear --axial-compression-mn 0.02 --amplitudes-mm 1.25 " +
    "--steps-per-segment 1 --reversal-substep-factor 2 --max-bisections 4 " +
    "--output-dir `"$openseesDryRunDir`""
)

$openseesManifest = Join-Path $openseesDryRunDir "reference_manifest.json"
if (-not (Test-Path -LiteralPath $openseesManifest -PathType Leaf)) {
    throw "OpenSeesPy dry-run contract did not produce $openseesManifest"
}

$openseesMaterialDryRunDir = Join-Path $repoRoot "data/output/cyclic_validation/opensees_material_reference_ci_dry_run"
Invoke-Step (
    "$pythonLauncher scripts/opensees_reduced_rc_material_reference.py " +
    "--dry-run --material steel --protocol cyclic --steps-per-branch 8 " +
    "--output-dir `"$openseesMaterialDryRunDir`""
)

$openseesMaterialManifest = Join-Path $openseesMaterialDryRunDir "reference_manifest.json"
if (-not (Test-Path -LiteralPath $openseesMaterialManifest -PathType Leaf)) {
    throw "OpenSeesPy material dry-run contract did not produce $openseesMaterialManifest"
}

$openseesMaterialConcrete01DryRunDir = Join-Path $repoRoot "data/output/cyclic_validation/opensees_material_reference_concrete01_ci_dry_run"
Invoke-Step (
    "$pythonLauncher scripts/opensees_reduced_rc_material_reference.py " +
    "--dry-run --material concrete --protocol monotonic --concrete-model Concrete01 --steps-per-branch 8 " +
    "--output-dir `"$openseesMaterialConcrete01DryRunDir`""
)

$openseesMaterialConcrete01Manifest = Join-Path $openseesMaterialConcrete01DryRunDir "reference_manifest.json"
if (-not (Test-Path -LiteralPath $openseesMaterialConcrete01Manifest -PathType Leaf)) {
    throw "OpenSeesPy Concrete01 material dry-run contract did not produce $openseesMaterialConcrete01Manifest"
}

if (-not $SkipDocs) {
    try {
        Invoke-Step "latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex" `
            (Join-Path $repoRoot "doc")
    } catch {
        Write-Warning "latexmk failed; retrying the document build with pdflatex."
        Invoke-Step "pdflatex -interaction=nonstopmode -halt-on-error main.tex" `
            (Join-Path $repoRoot "doc")
        Invoke-Step "pdflatex -interaction=nonstopmode -halt-on-error main.tex" `
            (Join-Path $repoRoot "doc")
    }
}
