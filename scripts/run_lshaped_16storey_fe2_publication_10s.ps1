param(
    [string]$BuildDir = "build",
    [ValidateSet("xfem", "kobathe-hex27", "both", "smoke")]
    [string]$Mode = "both",
    [double]$StartTime = 87.65,
    [double]$Duration = 10.0,
    [double]$SmokeDuration = 0.10,
    [double]$Scale = 1.0,
    [double]$Fe2Phase2Dt = 0.02,
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
    [int]$ManagedLocalTransitionSteps = 2,
    [int]$ManagedLocalMaxTransitionSteps = 8,
    [int]$ManagedLocalAdaptiveMaxBisections = 10,
    [string]$OutputRootBase = "data/output/lshaped_16storey_publication_10s",
    [switch]$UseLinearAlarmRestart,
    [double]$LinearAlarmRestartTime = 5.20,
    [switch]$GravityPreload,
    [double]$GravityAccel = 9.80665,
    [int]$GravityPreloadSteps = 8,
    [int]$GravityPreloadBisections = 6,
    [double]$KobathePenaltyFactor = 10.0,
    [ValidateSet("small", "tl", "ul", "corotational")]
    [string]$KobatheKinematics = "small",
    [int]$KobatheSnesMaxIt = 60,
    [double]$KobatheSnesAtol = 1.0e-6,
    [double]$KobatheSnesRtol = 1.0e-2,
    [switch]$KobatheEnableArcLength,
    [switch]$KobatheDisableSubsequentAdaptive,
    [switch]$KobatheSkipSubsequentFullStep,
    [switch]$KobatheBondSlip,
    [double]$KobatheBondSlipReference = 0.0005,
    [double]$KobatheBondSlipResidualRatio = 0.2,
    [double]$KobatheAdaptiveInitialFraction = 0.25,
    [double]$KobatheAdaptiveGrowthFactor = 2.0,
    [int]$KobatheAdaptiveEasyIters = 8,
    [int]$KobatheAdaptiveHardIters = 18,
    [double]$KobatheAdaptiveHardShrinkFactor = 0.5,
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

$isPhysicalScale = [math]::Abs($Scale - 1.0) -lt 1.0e-12
if (-not $isPhysicalScale -and $OutputRootBase -notmatch "stress[_-]?test") {
    throw "Non-physical scale=$Scale is reserved for stress tests. Use scale=1.0 for publication runs, or include 'stress_test' in OutputRootBase."
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
        "--fe2-max-sites", "$Fe2MaxSites",
        "--fe2-phase2-dt", (Format-Real $Fe2Phase2Dt),
        "--managed-local-transition-steps", "$ManagedLocalTransitionSteps",
        "--managed-local-max-transition-steps", "$ManagedLocalMaxTransitionSteps",
        "--managed-local-adaptive-max-bisections", "$ManagedLocalAdaptiveMaxBisections"
    )
    if ($IncludeColumnProbeSites) {
        $args += "--fe2-include-column-probe-sites"
    }
    if ($IncludeCenterProbeSite) {
        $args += "--fe2-include-center-probe-site"
    }
    if ($SkipPostprocess) {
        $args += "--skip-postprocess"
    }
    if ($UseLinearAlarmRestart) {
        $args += "--restart-from-linear-alarm"
    }
    if ($GravityPreload) {
        $args += "--gravity-preload"
        $args += "--gravity-accel"
        $args += (Format-Real $GravityAccel)
        $args += "--gravity-preload-steps"
        $args += "$GravityPreloadSteps"
        $args += "--gravity-preload-bisections"
        $args += "$GravityPreloadBisections"
    }
    if ($item.Family -like "continuum-kobathe*" -and
        ($KobatheEnableArcLength -or
         $KobathePenaltyFactor -ne 10.0 -or
         $KobatheKinematics -ne "small" -or
         $KobatheSnesMaxIt -ne 60 -or
         $KobatheSnesAtol -ne 1.0e-6 -or
         $KobatheSnesRtol -ne 1.0e-2 -or
         $KobatheDisableSubsequentAdaptive -or
         $KobatheSkipSubsequentFullStep -or
         $KobatheBondSlip -or
         $KobatheBondSlipReference -ne 0.0005 -or
         $KobatheBondSlipResidualRatio -ne 0.2 -or
         $KobatheAdaptiveInitialFraction -ne 0.25 -or
         $KobatheAdaptiveGrowthFactor -ne 2.0 -or
         $KobatheAdaptiveEasyIters -ne 8 -or
         $KobatheAdaptiveHardIters -ne 18 -or
         $KobatheAdaptiveHardShrinkFactor -ne 0.5 -or
         $KobatheArcLengthThreshold -ne 3 -or
         $KobatheTailRescueAttempts -gt 0)) {
        if ($KobatheEnableArcLength) {
            $args += "--kobathe-enable-arc-length"
        }
        if ($KobatheDisableSubsequentAdaptive) {
            $args += "--kobathe-no-subsequent-adaptive"
        }
        if ($KobatheSkipSubsequentFullStep) {
            $args += "--kobathe-skip-subsequent-full-step"
        }
        if ($KobatheBondSlip) {
            $args += "--kobathe-bond-slip"
        }
        $args += "--kobathe-penalty-factor"
        $args += (Format-Real $KobathePenaltyFactor)
        $args += "--kobathe-kinematics"
        $args += $KobatheKinematics
        $args += "--kobathe-bond-slip-reference"
        $args += (Format-Real $KobatheBondSlipReference)
        $args += "--kobathe-bond-slip-residual-ratio"
        $args += (Format-Real $KobatheBondSlipResidualRatio)
        $args += "--kobathe-snes-max-it"
        $args += "$KobatheSnesMaxIt"
        $args += "--kobathe-snes-atol"
        $args += (Format-Real $KobatheSnesAtol)
        $args += "--kobathe-snes-rtol"
        $args += (Format-Real $KobatheSnesRtol)
        $args += "--kobathe-adaptive-initial-fraction"
        $args += (Format-Real $KobatheAdaptiveInitialFraction)
        $args += "--kobathe-adaptive-growth-factor"
        $args += (Format-Real $KobatheAdaptiveGrowthFactor)
        $args += "--kobathe-adaptive-easy-iters"
        $args += "$KobatheAdaptiveEasyIters"
        $args += "--kobathe-adaptive-hard-iters"
        $args += "$KobatheAdaptiveHardIters"
        $args += "--kobathe-adaptive-hard-shrink-factor"
        $args += (Format-Real $KobatheAdaptiveHardShrinkFactor)
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
        fe2_phase2_dt = $Fe2Phase2Dt
        include_column_probe_sites = [bool]$IncludeColumnProbeSites
        include_center_probe_site = [bool]$IncludeCenterProbeSite
        gravity_preload = [bool]$GravityPreload
        gravity_accel = $GravityAccel
        gravity_preload_steps = $GravityPreloadSteps
        gravity_preload_bisections = $GravityPreloadBisections
        managed_local_transition_steps = $ManagedLocalTransitionSteps
        managed_local_max_transition_steps = $ManagedLocalMaxTransitionSteps
        managed_local_adaptive_max_bisections = $ManagedLocalAdaptiveMaxBisections
        kobathe_penalty_factor = $KobathePenaltyFactor
        kobathe_kinematics = $KobatheKinematics
        kobathe_snes_max_it = $KobatheSnesMaxIt
        kobathe_snes_atol = $KobatheSnesAtol
        kobathe_snes_rtol = $KobatheSnesRtol
        kobathe_enable_arc_length = [bool]$KobatheEnableArcLength
        kobathe_subsequent_adaptive = (-not [bool]$KobatheDisableSubsequentAdaptive)
        kobathe_skip_subsequent_full_step = [bool]$KobatheSkipSubsequentFullStep
        kobathe_bond_slip = [bool]$KobatheBondSlip
        kobathe_bond_slip_reference_m = $KobatheBondSlipReference
        kobathe_bond_slip_residual_ratio = $KobatheBondSlipResidualRatio
        kobathe_adaptive_initial_fraction = $KobatheAdaptiveInitialFraction
        kobathe_adaptive_growth_factor = $KobatheAdaptiveGrowthFactor
        kobathe_adaptive_easy_iters = $KobatheAdaptiveEasyIters
        kobathe_adaptive_hard_iters = $KobatheAdaptiveHardIters
        kobathe_adaptive_hard_shrink_factor = $KobatheAdaptiveHardShrinkFactor
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
