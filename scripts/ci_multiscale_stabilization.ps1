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

if (-not $SkipConfigure) {
    Invoke-Step "cmake -S `"$repoRoot`" -B `"$buildPath`" -G Ninja"
}

$regressionTargets = @(
    "fall_n_multiscale_api_test",
    "fall_n_micro_solve_executor_test",
    "fall_n_evolver_advanced_test",
    "fall_n_tangent_validation_benchmark_test",
    "fall_n_beam_test",
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
    "fall_n_beam_test.exe",
    "fall_n_steppable_solver_test.exe",
    "fall_n_steppable_dynamic_test.exe"
)

foreach ($exe in $executables) {
    Invoke-Step ("& `"" + (Join-Path $buildPath $exe) + "`"")
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
