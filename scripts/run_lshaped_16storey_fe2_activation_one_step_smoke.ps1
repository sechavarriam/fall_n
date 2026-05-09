param(
    [string]$BuildDir = "build",
    [ValidateSet("xfem", "kobathe-hex27", "both")]
    [string]$Mode = "both",
    [double]$StartTime = 87.65,
    [double]$SearchDuration = 10.0,
    [int]$StepsAfterActivation = 1,
    [int]$GlobalVtkInterval = 1,
    [int]$LocalVtkInterval = 1,
    [double]$CrackOpeningThreshold = 0.0005,
    [ValidateSet("visual", "minimal", "full", "debug")]
    [string]$GaussFields = "minimal",
    [ValidateSet("reference", "current", "both")]
    [string]$PlacementFrame = "both",
    [int]$Fe2MaxSites = 3,
    [switch]$IncludeColumnProbeSites,
    [switch]$IncludeCenterProbeSite,
    [double]$Scale = 20.0,
    [string]$OutputRootBase = "data/output/lshaped_16storey_activation_one_step",
    [switch]$UseLinearAlarmRestart,
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
}

foreach ($item in $families) {
    $outputRoot = Join-Path $repoRoot (Join-Path $OutputRootBase $item.Name)
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

    $args = @(
        "--fe2-one-way-only",
        "--local-family", $item.Family,
        "--scale", (Format-Real $Scale),
        "--start-time", (Format-Real $StartTime),
        "--duration", (Format-Real $SearchDuration),
        "--fe2-steps-after-activation", "$StepsAfterActivation",
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
    if ($IncludeColumnProbeSites) {
        $args += "--fe2-include-column-probe-sites"
    }
    if ($IncludeCenterProbeSite) {
        $args += "--fe2-include-center-probe-site"
    }

    Write-Host "Running activation + $StepsAfterActivation FE2 step smoke: $($item.Name)..."
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "$($item.Name) failed with exit code $LASTEXITCODE"
    }

    $manifest = @{
        schema = "fall_n_lshaped_16storey_activation_one_step_smoke_v1"
        mode = $item.Name
        local_family = $item.Family
        scale = $Scale
        start_time_s = $StartTime
        search_duration_s = $SearchDuration
        steps_after_activation = $StepsAfterActivation
        restart_from_linear_alarm = [bool]$UseLinearAlarmRestart
        local_vtk_profile = "publication"
        local_vtk_crack_opening_threshold_m = $CrackOpeningThreshold
        local_vtk_crack_filter_mode = "both"
        local_vtk_gauss_fields = $GaussFields
        local_vtk_placement_frame = $PlacementFrame
        local_vtk_global_placement = $true
        fe2_max_sites = $Fe2MaxSites
        include_column_probe_sites = [bool]$IncludeColumnProbeSites
        include_center_probe_site = [bool]$IncludeCenterProbeSite
        output_root = "$outputRoot"
    } | ConvertTo-Json -Depth 4
    Set-Content -Path (Join-Path $outputRoot "activation_one_step_manifest.json") -Value $manifest

    if (-not $SkipAudit) {
        $audit = Join-Path $repoRoot "scripts/audit_publication_vtk.py"
        $auditOut = Join-Path $outputRoot "recorders/publication_vtk_audit_activation_one_step.json"
        & python $audit --root $outputRoot --output $auditOut --gauss-fields-profile $GaussFields --crack-opening-threshold (Format-Real $CrackOpeningThreshold) --max-files-per-category 8 --check-local-global-endpoints
        if ($LASTEXITCODE -ne 0) {
            throw "$($item.Name) VTK audit failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "Activation one-step FE2 smoke finished."
