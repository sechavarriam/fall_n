param(
    [Parameter(Mandatory = $true)]
    [int]$PidToWait,

    [string]$RepoRoot = "C:/MyLibs/fall_n",
    [string]$Prefix = "lshaped_16_fe2_two_way_10s_publication",
    [string]$SnapshotDir = "data/output/stage_c_16storey/fe2_two_way_10s_publication_snapshot",
    [string]$PostprocessLog = "data/output/stage_c_16storey/fe2_two_way_10s_publication_postprocess.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path (Split-Path $PostprocessLog -Parent) | Out-Null

function Write-PostLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $Message" | Tee-Object -FilePath $PostprocessLog -Append
}

Write-PostLog "waiting for FE2 process pid=$PidToWait"
while (Get-Process -Id $PidToWait -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}
Write-PostLog "FE2 process completed"

New-Item -ItemType Directory -Force -Path $SnapshotDir | Out-Null
foreach ($name in @("recorders", "local_sites", "evolution")) {
    $src = Join-Path "data/output/lshaped_multiscale_16" $name
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $SnapshotDir $name) -Recurse -Force
        Write-PostLog "archived $name"
    }
}

$plotArgs = @(
    "scripts/plot_lshaped_16storey_fe2_two_way_closure.py",
    "--structural-roof", "data/output/stage_c_16storey/falln_n4_nonlinear_restart_alarm_to_10s_roof_displacement.csv",
    "--structural-force", "data/output/stage_c_16storey/falln_n4_nonlinear_restart_alarm_to_10s_selected_element_0_global_force.csv",
    "--fe2-roof", "data/output/lshaped_multiscale_16/recorders/roof_displacement.csv",
    "--fe2-force", "data/output/lshaped_multiscale_16/recorders/selected_element_0_global_force.csv",
    "--coupling-audit", "data/output/lshaped_multiscale_16/recorders/fe2_two_way_coupling_audit.csv",
    "--crack-evolution", "data/output/lshaped_multiscale_16/recorders/crack_evolution.csv",
    "--time-index", "data/output/lshaped_multiscale_16/recorders/multiscale_time_index.csv",
    "--prefix", $Prefix
)

Write-PostLog "running publication plotter"
& py -3.11 @plotArgs *>> $PostprocessLog
if ($LASTEXITCODE -ne 0) {
    throw "publication plotter failed with exit code $LASTEXITCODE"
}
Write-PostLog "publication postprocess completed"
