param(
    [string]$BuildDir = "build",
    [string]$OutputRoot = "data/output/cyclic_validation/predefinitive_validation",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$buildPath = Join-Path $repoRoot $BuildDir
$outputPath = Join-Path $repoRoot $OutputRoot
$logPath = Join-Path $outputPath "logs"
$summaryPath = Join-Path $outputPath "summary.csv"

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
New-Item -ItemType Directory -Force -Path $logPath | Out-Null

function Invoke-Step {
    param(
        [string]$Command,
        [string]$WorkingDirectory = $repoRoot
    )

    Write-Host "==> $Command"
    Push-Location $WorkingDirectory
    try {
        Invoke-Expression $Command
    } finally {
        Pop-Location
    }
}

function Build-Targets {
    $targets = @(
        "fall_n_multiscale_api_test",
        "fall_n_evolver_advanced_test",
        "fall_n_tangent_validation_benchmark_test",
        "fall_n_coupling_residual_benchmark_test",
        "fall_n_cyclic_validation_api_test",
        "fall_n_table_cyclic_validation"
    )

    foreach ($target in $targets) {
        Invoke-Step "ninja -C `"$buildPath`" -j1 $target"
    }
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$Command,
        [string]$WorkingDirectory = $repoRoot
    )

    $logFile = Join-Path $logPath ($Name + ".log")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Push-Location $WorkingDirectory
    try {
        & powershell -NoProfile -Command $Command 2>&1 |
            Tee-Object -FilePath $logFile | Out-Host
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
        $sw.Stop()
    }

    [pscustomobject]@{
        Name = $Name
        ExitCode = $exitCode
        DurationSeconds = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
        LogFile = $logFile
    }
}

function Get-LastCsvRow {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return $null
    }

    $rows = Import-Csv -Path $Path
    if ($null -eq $rows -or $rows.Count -eq 0) {
        return $null
    }

    return $rows[-1]
}

if (-not $SkipBuild) {
    Build-Targets
}

$summary = New-Object System.Collections.Generic.List[object]

$testCommands = @(
    @{ Name = "multiscale_api_test"; Command = "& `"$buildPath\fall_n_multiscale_api_test.exe`"" },
    @{ Name = "evolver_advanced_test"; Command = "& `"$buildPath\fall_n_evolver_advanced_test.exe`"" },
    @{ Name = "tangent_validation_benchmark"; Command = "& `"$buildPath\fall_n_tangent_validation_benchmark_test.exe`"" },
    @{ Name = "coupling_residual_benchmark"; Command = "& `"$buildPath\fall_n_coupling_residual_benchmark_test.exe`"" },
    @{ Name = "cyclic_validation_api_test"; Command = "& `"$buildPath\fall_n_cyclic_validation_api_test.exe`"" }
)

foreach ($entry in $testCommands) {
    $result = Invoke-LoggedCommand -Name $entry.Name -Command $entry.Command
    $summary.Add([pscustomobject]@{
        category = "regression"
        id = $entry.Name
        outcome = $(if ($result.ExitCode -eq 0) { "PASS" } else { "FAIL" })
        duration_s = $result.DurationSeconds
        notes = ""
        drift_m = ""
        base_shear_MN = ""
        total_cracks = ""
        max_opening = ""
        fe2_iterations = ""
        failed_sites = ""
    })
}

$case4Root = Join-Path $repoRoot "data/output/cyclic_validation/case4"
$case5Root = Join-Path $repoRoot "data/output/cyclic_validation/case5"

$case4Result = Invoke-LoggedCommand `
    -Name "case4_predefinitive" `
    -Command "& `"$buildPath\fall_n_table_cyclic_validation.exe`" --case 4 --protocol extended50 --fe2-profile crack50 --max-steps 1 --global-output-interval 0 --submodel-output-interval 0"

$case4Row = Get-LastCsvRow (Join-Path $case4Root "recorders/global_history.csv")
$summary.Add([pscustomobject]@{
    category = "cyclic_fe2"
    id = "case4_one_way_step1"
    outcome = $(if ($case4Result.ExitCode -eq 0) { "TRUNCATED_OK" } else { "FAIL" })
    duration_s = $case4Result.DurationSeconds
    notes = "Expected predefinitive frontier: one-way FE2 reaches first protocol point and preserves crack/hysteresis recorders."
    drift_m = $(if ($case4Row) { $case4Row.drift_m } else { "" })
    base_shear_MN = $(if ($case4Row) { $case4Row.base_shear_MN } else { "" })
    total_cracks = $(if ($case4Row) { $case4Row.total_cracks } else { "" })
    max_opening = $(if ($case4Row) { $case4Row.max_opening } else { "" })
    fe2_iterations = $(if ($case4Row) { $case4Row.fe2_iterations } else { "" })
    failed_sites = ""
})

$case5Result = Invoke-LoggedCommand `
    -Name "case5_predefinitive" `
    -Command "& `"$buildPath\fall_n_table_cyclic_validation.exe`" --case 5 --protocol extended50 --fe2-profile crack50 --max-steps 1 --global-output-interval 0 --submodel-output-interval 0"

$case5Solver = Get-LastCsvRow (Join-Path $case5Root "recorders/solver_diagnostics.csv")
$case5Outcome = "FAIL"
$case5Notes = "Unexpected failure."
if ($case5Result.ExitCode -eq 0 -and $case5Solver) {
    if ($case5Solver.termination_reason -eq "MicroSolveFailed") {
        $case5Outcome = "KNOWN_FRONTIER"
        $case5Notes = "Expected current frontier: iterated FE2 aborts on the first cracked point with explicit rollback and failed-site reporting."
    } else {
        $case5Outcome = "UNEXPECTED_OK"
        $case5Notes = "Iterated FE2 progressed beyond the documented frontier; review and promote if reproducible."
    }
}

$summary.Add([pscustomobject]@{
    category = "cyclic_fe2"
    id = "case5_two_way_step1"
    outcome = $case5Outcome
    duration_s = $case5Result.DurationSeconds
    notes = $case5Notes
    drift_m = ""
    base_shear_MN = ""
    total_cracks = ""
    max_opening = ""
    fe2_iterations = $(if ($case5Solver) { $case5Solver.fe2_iterations } else { "" })
    failed_sites = $(if ($case5Solver) { $case5Solver.failed_sites } else { "" })
})

$summary |
    Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Predefinitive physical validation summary written to:"
Write-Host "  $summaryPath"
