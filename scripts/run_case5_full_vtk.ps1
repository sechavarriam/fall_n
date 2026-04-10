param(
    [string]$BuildDir = "build",
    [string]$Protocol = "extended50",
    [string]$Profile = "crack50",
    [int]$GlobalOutputInterval = 1,
    [int]$SubmodelOutputInterval = 1,
    [string]$CaseOutputDir = "data/output/cyclic_validation/case5",
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$exePath = Join-Path $repoRoot (Join-Path $BuildDir "fall_n_table_cyclic_validation.exe")

if (-not (Test-Path $exePath)) {
    throw "Executable not found: $exePath"
}

$caseOutputPath = Join-Path $repoRoot $CaseOutputDir
$runRoot = Join-Path $caseOutputPath "full_run"
$stdoutPath = Join-Path $runRoot "stdout.log"
$stderrPath = Join-Path $runRoot "stderr.log"
$pidPath = Join-Path $runRoot "pid.txt"
$manifestPath = Join-Path $runRoot "manifest.txt"
$completionPath = Join-Path $runRoot "completion.txt"
$workerPath = Join-Path $runRoot "run_case5_full_vtk_worker.ps1"

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

if ((Test-Path $pidPath) -and -not $ForceRestart) {
    $existingPid = (Get-Content $pidPath -ErrorAction Stop | Select-Object -First 1).Trim()
    if ($existingPid) {
        $existingProcess = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
        if ($existingProcess) {
            throw "A Case 5 run is already active with PID $existingPid. Use -ForceRestart to replace the PID file after stopping it."
        }
    }
}

$arguments = @(
    "--case", "5",
    "--protocol", $Protocol,
    "--fe2-profile", $Profile,
    "--global-output-interval", $GlobalOutputInterval.ToString(),
    "--submodel-output-interval", $SubmodelOutputInterval.ToString()
)

$commandLine = @(
    '"' + $exePath + '"'
    ($arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    })
) -join ' '

$manifest = @(
    "timestamp_utc=$((Get-Date).ToUniversalTime().ToString('o'))",
    "repo_root=$repoRoot",
    "exe=$exePath",
    "command=$commandLine",
    "stdout=$stdoutPath",
    "stderr=$stderrPath",
    "completion=$completionPath",
    "protocol=$Protocol",
    "profile=$Profile",
    "global_output_interval=$GlobalOutputInterval",
    "submodel_output_interval=$SubmodelOutputInterval"
) -join [Environment]::NewLine

Set-Content -Path $manifestPath -Value $manifest -Encoding UTF8
if (Test-Path $completionPath) {
    Remove-Item $completionPath -Force
}
foreach ($logPath in @($stdoutPath, $stderrPath)) {
    if (Test-Path $logPath) {
        Remove-Item $logPath -Force
    }
}

$quotedArgs = ($arguments | ForEach-Object {
    "'" + ($_ -replace "'", "''") + "'"
}) -join ", "

$workerScript = @"
`$ErrorActionPreference = 'Stop'
`$stdout = '$($stdoutPath -replace "'", "''")'
`$stderr = '$($stderrPath -replace "'", "''")'
`$completion = '$($completionPath -replace "'", "''")'
`$exe = '$($exePath -replace "'", "''")'
`$arguments = @($quotedArgs)

"started_utc=`$((Get-Date).ToUniversalTime().ToString('o'))" | Set-Content -Path `$completion -Encoding UTF8

try {
    & `$exe @arguments 1>> `$stdout 2>> `$stderr
    `$code = `$LASTEXITCODE
} catch {
    `$code = 1
    ("worker_exception=" + `$_.Exception.Message) | Add-Content -Path `$stderr -Encoding UTF8
}

"finished_utc=`$((Get-Date).ToUniversalTime().ToString('o'))" | Add-Content -Path `$completion -Encoding UTF8
"exit_code=`$code" | Add-Content -Path `$completion -Encoding UTF8
"@

Set-Content -Path $workerPath -Value $workerScript -Encoding UTF8

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $workerPath
    ) `
    -WorkingDirectory $repoRoot `
    -PassThru

Set-Content -Path $pidPath -Value $process.Id -Encoding ASCII

Write-Host "Case 5 full FE2 run launched."
Write-Host "  PID: $($process.Id)"
Write-Host "  stdout: $stdoutPath"
Write-Host "  stderr: $stderrPath"
Write-Host "  completion: $completionPath"
Write-Host "  worker: $workerPath"
Write-Host "  manifest: $manifestPath"
Write-Host "  pid file: $pidPath"
