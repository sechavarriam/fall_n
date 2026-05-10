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
    [switch]$Fe2DisablePhase2DtRegrowth,
    [double]$Fe2Phase2DtGrowthFactor = 1.25,
    [int]$Fe2Phase2DtEasySteps = 2,
    [switch]$Fe2AdaptiveSiteRelaxation,
    [int]$Fe2SiteRelaxAttempts = 4,
    [double]$Fe2SiteRelaxGrowth = 1.25,
    [double]$Fe2SiteRelaxFactor = 0.5,
    [double]$Fe2SiteRelaxMinAlpha = 0.05,
    [int]$Fe2MacroCutbackAttempts = 0,
    [double]$Fe2MacroCutbackFactor = 0.5,
    [int]$Fe2OneWayMicroCutbackAttempts = 6,
    [double]$Fe2OneWayMicroCutbackFactor = 0.5,
    [double]$Fe2OneWayMicroCutbackMinDt = 0.000125,
    [double]$Fe2LocalSolveWallBudgetSeconds = 0.0,
    [double]$Fe2LocalSolveBudgetCutbackFactor = 0.5,
    [switch]$Fe2StopOnLocalSolveBudget,
    [int]$Fe2MacroBacktrackAttempts = 0,
    [double]$Fe2MacroBacktrackFactor = 0.5,
    [switch]$DisableManagedLocalAdaptiveTransition,
    [switch]$IncludeColumnProbeSites,
    [switch]$IncludeCenterProbeSite,
    [int]$ManagedLocalTransitionSteps = 2,
    [int]$ManagedLocalMaxTransitionSteps = 8,
    [int]$ManagedLocalAdaptiveMaxBisections = 10,
    [int]$ManagedLocalFailureRescueAttempts = 2,
    [double]$ManagedLocalFailureRescueStepFactor = 2.0,
    [string]$OutputRootBase = "data/output/lshaped_16storey_publication_10s",
    [switch]$UseLinearAlarmRestart,
    [string]$LinearAlarmRestartDir = "data/output/lshaped_16storey_physical_scale1_linear_steel_yield_20260509/recorders",
    [string]$LinearAlarmRestartDisplacement = "",
    [string]$LinearAlarmRestartVelocity = "",
    [double]$LinearAlarmRestartTime = 1.30,
    [int]$LinearAlarmRestartStep = 65,
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
        "--fe2-phase2-dt-growth-factor", (Format-Real $Fe2Phase2DtGrowthFactor),
        "--fe2-phase2-dt-easy-steps", "$Fe2Phase2DtEasySteps",
        "--fe2-site-relax-attempts", "$Fe2SiteRelaxAttempts",
        "--fe2-site-relax-growth", (Format-Real $Fe2SiteRelaxGrowth),
        "--fe2-site-relax-factor", (Format-Real $Fe2SiteRelaxFactor),
        "--fe2-site-relax-min-alpha", (Format-Real $Fe2SiteRelaxMinAlpha),
        "--fe2-macro-cutback-attempts", "$Fe2MacroCutbackAttempts",
        "--fe2-macro-cutback-factor", (Format-Real $Fe2MacroCutbackFactor),
        "--fe2-one-way-micro-cutback-attempts", "$Fe2OneWayMicroCutbackAttempts",
        "--fe2-one-way-micro-cutback-factor", (Format-Real $Fe2OneWayMicroCutbackFactor),
        "--fe2-one-way-micro-cutback-min-dt", (Format-Real $Fe2OneWayMicroCutbackMinDt),
        "--fe2-local-solve-wall-budget-seconds", (Format-Real $Fe2LocalSolveWallBudgetSeconds),
        "--fe2-local-solve-budget-cutback-factor", (Format-Real $Fe2LocalSolveBudgetCutbackFactor),
        "--fe2-macro-backtrack-attempts", "$Fe2MacroBacktrackAttempts",
        "--fe2-macro-backtrack-factor", (Format-Real $Fe2MacroBacktrackFactor),
        "--managed-local-transition-steps", "$ManagedLocalTransitionSteps",
        "--managed-local-max-transition-steps", "$ManagedLocalMaxTransitionSteps",
        "--managed-local-adaptive-max-bisections", "$ManagedLocalAdaptiveMaxBisections",
        "--managed-local-failure-rescue-attempts", "$ManagedLocalFailureRescueAttempts",
        "--managed-local-failure-rescue-step-factor", (Format-Real $ManagedLocalFailureRescueStepFactor)
    )
    if ($Fe2AdaptiveSiteRelaxation) {
        $args += "--fe2-adaptive-site-relax"
    }
    if ($Fe2DisablePhase2DtRegrowth) {
        $args += "--fe2-disable-phase2-dt-regrowth"
    }
    if ($Fe2StopOnLocalSolveBudget) {
        $args += "--fe2-stop-on-local-solve-budget"
    }
    if (-not $DisableManagedLocalAdaptiveTransition) {
        $args += "--adaptive-managed-local-transition"
    }
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
        $restartDir = Join-Path $repoRoot $LinearAlarmRestartDir
        $restartDisp = if ([string]::IsNullOrWhiteSpace($LinearAlarmRestartDisplacement)) {
            Join-Path $restartDir "linear_first_alarm_displacement.vec"
        } else {
            $LinearAlarmRestartDisplacement
        }
        $restartVel = if ([string]::IsNullOrWhiteSpace($LinearAlarmRestartVelocity)) {
            Join-Path $restartDir "linear_first_alarm_velocity.vec"
        } else {
            $LinearAlarmRestartVelocity
        }
        if (-not (Test-Path $restartDisp)) {
            throw "Linear alarm restart displacement not found: $restartDisp"
        }
        if (-not (Test-Path $restartVel)) {
            throw "Linear alarm restart velocity not found: $restartVel"
        }
        $args += "--restart-displacement"
        $args += (Resolve-Path -LiteralPath $restartDisp).Path
        $args += "--restart-velocity"
        $args += (Resolve-Path -LiteralPath $restartVel).Path
        $args += "--restart-time"
        $args += (Format-Real $LinearAlarmRestartTime)
        $args += "--restart-step"
        $args += "$LinearAlarmRestartStep"
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
        linear_alarm_restart_dir = $LinearAlarmRestartDir
        linear_alarm_restart_time_s = $LinearAlarmRestartTime
        linear_alarm_restart_step = $LinearAlarmRestartStep
        local_vtk_profile = "publication"
        local_vtk_crack_opening_threshold_m = $CrackOpeningThreshold
        local_vtk_crack_filter_mode = "both"
        local_vtk_gauss_fields = $GaussFields
        local_vtk_placement_frame = $PlacementFrame
        local_vtk_global_placement = $true
        fe2_max_sites = $Fe2MaxSites
        fe2_phase2_dt = $Fe2Phase2Dt
        fe2_phase2_dt_regrowth = -not [bool]$Fe2DisablePhase2DtRegrowth
        fe2_phase2_dt_growth_factor = $Fe2Phase2DtGrowthFactor
        fe2_phase2_dt_easy_steps = $Fe2Phase2DtEasySteps
        fe2_adaptive_site_relaxation = [bool]$Fe2AdaptiveSiteRelaxation
        fe2_site_relax_attempts = $Fe2SiteRelaxAttempts
        fe2_site_relax_growth = $Fe2SiteRelaxGrowth
        fe2_site_relax_factor = $Fe2SiteRelaxFactor
        fe2_site_relax_min_alpha = $Fe2SiteRelaxMinAlpha
        fe2_macro_cutback_attempts = $Fe2MacroCutbackAttempts
        fe2_macro_cutback_factor = $Fe2MacroCutbackFactor
        fe2_one_way_micro_cutback_attempts = $Fe2OneWayMicroCutbackAttempts
        fe2_one_way_micro_cutback_factor = $Fe2OneWayMicroCutbackFactor
        fe2_one_way_micro_cutback_min_dt = $Fe2OneWayMicroCutbackMinDt
        fe2_local_solve_wall_budget_seconds = $Fe2LocalSolveWallBudgetSeconds
        fe2_local_solve_budget_cutback_factor = $Fe2LocalSolveBudgetCutbackFactor
        fe2_stop_on_local_solve_budget = [bool]$Fe2StopOnLocalSolveBudget
        fe2_macro_backtrack_attempts = $Fe2MacroBacktrackAttempts
        fe2_macro_backtrack_factor = $Fe2MacroBacktrackFactor
        managed_local_adaptive_transition = (-not [bool]$DisableManagedLocalAdaptiveTransition)
        include_column_probe_sites = [bool]$IncludeColumnProbeSites
        include_center_probe_site = [bool]$IncludeCenterProbeSite
        gravity_preload = [bool]$GravityPreload
        gravity_accel = $GravityAccel
        gravity_preload_steps = $GravityPreloadSteps
        gravity_preload_bisections = $GravityPreloadBisections
        managed_local_transition_steps = $ManagedLocalTransitionSteps
        managed_local_max_transition_steps = $ManagedLocalMaxTransitionSteps
        managed_local_adaptive_max_bisections = $ManagedLocalAdaptiveMaxBisections
        managed_local_failure_rescue_attempts = $ManagedLocalFailureRescueAttempts
        managed_local_failure_rescue_step_factor = $ManagedLocalFailureRescueStepFactor
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
