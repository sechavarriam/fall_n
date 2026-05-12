param(
    [string]$BuildDir = "build",
    [string]$OutputRoot = "data/output/lshaped_16storey_two_way_xfem_whole_element_smoke",
    [double]$Scale = 1.0,
    [double]$Duration = 10.0,
    [int]$Fe2MaxSites = 1,
    [int]$Fe2StepsAfterActivation = 1,
    [double]$Fe2Phase2Dt = 0.005,
    [int]$Fe2MaxStaggered = 12,
    [double]$Fe2Tolerance = 0.05,
    [double]$Fe2Relaxation = 0.35,
    [int]$Fe2MacroCutbackAttempts = 6,
    [double]$Fe2MacroCutbackFactor = 0.5,
    [int]$Fe2TwoWayConvergenceCutbackAttempts = 6,
    [double]$Fe2TwoWayConvergenceCutbackFactor = 0.5,
    [double]$Fe2TwoWayConvergenceCutbackMinDt = 0.000125,
    [switch]$UseLinearAlarmRestart,
    [string]$LinearAlarmRestartDir = "data/output/lshaped_16storey_physical_scale1_linear_steel_yield_20260509/recorders",
    [double]$LinearAlarmRestartTime = 1.30,
    [int]$LinearAlarmRestartStep = 65,
    [switch]$GravityPreload,
    [switch]$SkipPostprocess
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$exe = Join-Path (Resolve-Path (Join-Path $repoRoot $BuildDir)) "fall_n_lshaped_multiscale_16.exe"
if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe. Build target fall_n_lshaped_multiscale_16 first."
}

$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture
function Format-Real([double]$Value) {
    return $Value.ToString("R", $InvariantCulture)
}

if ([math]::Abs($Scale - 1.0) -gt 1.0e-12 -and $OutputRoot -notmatch "stress[_-]?test") {
    throw "Two-way publication smoke is physical only by default. Use Scale=1.0, or include 'stress_test' in OutputRoot."
}

$out = Join-Path $repoRoot $OutputRoot
New-Item -ItemType Directory -Force -Path $out | Out-Null

$args = @(
    "--local-family", "managed-xfem",
    "--xfem-local-site-mode", "whole_element",
    "--scale", (Format-Real $Scale),
    "--duration", (Format-Real $Duration),
    "--output-root", $out,
    "--fe2-max-sites", "$Fe2MaxSites",
    "--fe2-steps-after-activation", "$Fe2StepsAfterActivation",
    "--fe2-phase2-dt", (Format-Real $Fe2Phase2Dt),
    "--fe2-max-staggered", "$Fe2MaxStaggered",
    "--fe2-tol", (Format-Real $Fe2Tolerance),
    "--fe2-relax", (Format-Real $Fe2Relaxation),
    "--fe2-macro-cutback-attempts", "$Fe2MacroCutbackAttempts",
    "--fe2-macro-cutback-factor", (Format-Real $Fe2MacroCutbackFactor),
    "--fe2-two-way-convergence-cutback-attempts", "$Fe2TwoWayConvergenceCutbackAttempts",
    "--fe2-two-way-convergence-cutback-factor", (Format-Real $Fe2TwoWayConvergenceCutbackFactor),
    "--fe2-two-way-convergence-cutback-min-dt", (Format-Real $Fe2TwoWayConvergenceCutbackMinDt),
    "--fe2-adaptive-site-relax",
    "--fe2-site-relax-attempts", "3",
    "--fe2-site-relax-factor", "0.5",
    "--fe2-site-relax-min-alpha", "0.1",
    "--global-vtk-interval", "0",
    "--local-vtk-interval", "1",
    "--local-vtk-profile", "publication",
    "--local-vtk-crack-opening-threshold", "0.0005",
    "--local-vtk-crack-filter-mode", "both",
    "--local-vtk-gauss-fields", "minimal",
    "--local-vtk-placement-frame", "both",
    "--local-vtk-global-placement"
)

if ($SkipPostprocess) {
    $args += "--skip-postprocess"
}
if ($GravityPreload) {
    $args += "--gravity-preload"
}
if ($UseLinearAlarmRestart) {
    $restartDir = Join-Path $repoRoot $LinearAlarmRestartDir
    $disp = Join-Path $restartDir "linear_first_alarm_displacement.vec"
    $vel = Join-Path $restartDir "linear_first_alarm_velocity.vec"
    if (-not (Test-Path $disp)) {
        throw "Linear alarm restart displacement not found: $disp"
    }
    if (-not (Test-Path $vel)) {
        throw "Linear alarm restart velocity not found: $vel"
    }
    $args += "--restart-displacement"
    $args += (Resolve-Path -LiteralPath $disp).Path
    $args += "--restart-velocity"
    $args += (Resolve-Path -LiteralPath $vel).Path
    $args += "--restart-time"
    $args += (Format-Real $LinearAlarmRestartTime)
    $args += "--restart-step"
    $args += "$LinearAlarmRestartStep"
}

$manifest = @{
    schema = "fall_n_lshaped_16storey_two_way_smoke_v1"
    local_family = "managed-xfem"
    xfem_local_site_mode = "whole_element"
    scale = $Scale
    duration_s = $Duration
    fe2_max_sites = $Fe2MaxSites
    fe2_steps_after_activation = $Fe2StepsAfterActivation
    fe2_phase2_dt = $Fe2Phase2Dt
    fe2_max_staggered = $Fe2MaxStaggered
    fe2_tolerance = $Fe2Tolerance
    fe2_relaxation = $Fe2Relaxation
    fe2_macro_cutback_attempts = $Fe2MacroCutbackAttempts
    fe2_two_way_convergence_cutback_attempts = $Fe2TwoWayConvergenceCutbackAttempts
    fe2_two_way_convergence_cutback_factor = $Fe2TwoWayConvergenceCutbackFactor
    fe2_two_way_convergence_cutback_min_dt = $Fe2TwoWayConvergenceCutbackMinDt
    gravity_preload = [bool]$GravityPreload
    restart_from_linear_alarm = [bool]$UseLinearAlarmRestart
    output_root = "$out"
} | ConvertTo-Json -Depth 4
Set-Content -Path (Join-Path $out "two_way_smoke_manifest.json") -Value $manifest

Write-Host "Running FE2 two-way smoke with one whole-element XFEM local model..."
& $exe @args
if ($LASTEXITCODE -ne 0) {
    throw "FE2 two-way smoke failed with exit code $LASTEXITCODE"
}
Write-Host "FE2 two-way smoke finished: $out"
