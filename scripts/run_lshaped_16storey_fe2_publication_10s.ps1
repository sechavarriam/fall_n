param(
    [string]$BuildDir = "build",
    [ValidateSet("xfem", "kobathe-hex27", "both", "smoke")]
    [string]$Mode = "both",
    [double]$StartTime = 87.65,
    [double]$Duration = 10.0,
    [double]$SmokeDuration = 0.10,
    [double]$Scale = 20.0,
    [int]$GlobalVtkInterval = 1,
    [int]$LocalVtkInterval = 1,
    [double]$CrackOpeningThreshold = 0.0005,
    [ValidateSet("visual", "minimal", "full", "debug")]
    [string]$GaussFields = "minimal",
    [ValidateSet("reference", "current", "both")]
    [string]$PlacementFrame = "both",
    [int]$Fe2MaxSites = 3,
    [string]$OutputRootBase = "data/output/lshaped_16storey_publication_10s",
    [switch]$UseLinearAlarmRestart,
    [double]$LinearAlarmRestartTime = 5.20,
    [switch]$KobatheEnableArcLength,
    [int]$KobatheArcLengthThreshold = 3,
    [int]$KobatheTailRescueAttempts = 0,
    [switch]$SkipPostprocess,
    [switch]$SkipAudit
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

$runDuration = if ($Mode -eq "smoke") { $SmokeDuration } else { $Duration }
$driverDuration = $runDuration
if ($UseLinearAlarmRestart -and $runDuration -le $LinearAlarmRestartTime) {
    $driverDuration = $LinearAlarmRestartTime + $runDuration
}
$families = @()
switch ($Mode) {
    "xfem" { $families = @(@{ Name = "xfem"; Family = "managed-xfem" }) }
    "kobathe-hex27" { $families = @(@{ Name = "kobathe_hex27"; Family = "continuum-kobathe-hex27" }) }
    "both" {
        $families = @(
            @{ Name = "xfem"; Family = "managed-xfem" },
            @{ Name = "kobathe_hex27"; Family = "continuum-kobathe-hex27" }
        )
    }
    "smoke" {
        $families = @(
            @{ Name = "smoke_xfem"; Family = "managed-xfem" },
            @{ Name = "smoke_kobathe_hex27"; Family = "continuum-kobathe-hex27" }
        )
    }
}

foreach ($item in $families) {
    $outputRoot = Join-Path $repoRoot (Join-Path $OutputRootBase $item.Name)
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

    $args = @(
        "--fe2-one-way-only",
        "--local-family", $item.Family,
        "--scale", (Format-Real $Scale),
        "--start-time", (Format-Real $StartTime),
        "--duration", (Format-Real $driverDuration),
        "--output-root", $outputRoot,
        "--global-vtk-interval", "$GlobalVtkInterval",
        "--local-vtk-interval", "$LocalVtkInterval",
        "--local-vtk-profile", "publication",
        "--local-vtk-crack-opening-threshold", (Format-Real $CrackOpeningThreshold),
        "--local-vtk-crack-filter-mode", "both",
        "--local-vtk-gauss-fields", $GaussFields,
        "--local-vtk-placement-frame", $PlacementFrame,
        "--fe2-max-sites", "$Fe2MaxSites"
    )
    if ($SkipPostprocess) {
        $args += "--skip-postprocess"
    }
    if ($UseLinearAlarmRestart) {
        $args += "--restart-from-linear-alarm"
    }
    if ($item.Family -like "continuum-kobathe*" -and
        ($KobatheEnableArcLength -or
         $KobatheArcLengthThreshold -ne 3 -or
         $KobatheTailRescueAttempts -gt 0)) {
        if ($KobatheEnableArcLength) {
            $args += "--kobathe-enable-arc-length"
        }
        $args += "--kobathe-arc-length-threshold"
        $args += "$KobatheArcLengthThreshold"
        if ($KobatheTailRescueAttempts -gt 0) {
            $args += "--kobathe-tail-rescue-attempts"
            $args += "$KobatheTailRescueAttempts"
        }
    }

    Write-Host "Running $($item.Name) FE2 one-way publication case..."
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "$($item.Name) failed with exit code $LASTEXITCODE"
    }

    $manifest = @{
        schema = "fall_n_lshaped_16storey_publication_10s_run_v1"
        mode = $item.Name
        local_family = $item.Family
        scale = $Scale
        start_time_s = $StartTime
        requested_duration_s = $runDuration
        driver_duration_s = $driverDuration
        restart_from_linear_alarm = [bool]$UseLinearAlarmRestart
        linear_alarm_restart_time_s = $LinearAlarmRestartTime
        local_vtk_profile = "publication"
        local_vtk_crack_opening_threshold_m = $CrackOpeningThreshold
        local_vtk_crack_filter_mode = "both"
        local_vtk_gauss_fields = $GaussFields
        local_vtk_placement_frame = $PlacementFrame
        local_vtk_global_placement = $true
        fe2_max_sites = $Fe2MaxSites
        kobathe_enable_arc_length = [bool]$KobatheEnableArcLength
        kobathe_arc_length_threshold = $KobatheArcLengthThreshold
        kobathe_tail_rescue_attempts = $KobatheTailRescueAttempts
        output_root = "$outputRoot"
    } | ConvertTo-Json -Depth 4
    Set-Content -Path (Join-Path $outputRoot "publication_run_manifest.json") -Value $manifest

    if (-not $SkipAudit) {
        $audit = Join-Path $repoRoot "scripts/audit_publication_vtk.py"
        $auditOut = Join-Path $outputRoot "recorders/publication_vtk_audit.json"
        & python $audit --root $outputRoot --output $auditOut --gauss-fields-profile $GaussFields --crack-opening-threshold (Format-Real $CrackOpeningThreshold) --max-files-per-category 8 --check-local-global-endpoints
        if ($LASTEXITCODE -ne 0) {
            throw "$($item.Name) VTK audit failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "Publication FE2 campaign finished."
