param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BuildResultSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BuildSupervisorSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BuildExternalSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WitnessResultSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WitnessSupervisorSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$WitnessExternalSHA256
)
$package = 'D:\Documents\my 3dgs\research\experiments\speedy-frozen-kernel-control-v0\evaluation-v1'
$out = Join-Path $package 'actualrun-v1'
if (Test-Path -LiteralPath $out) { throw 'One actual attempt only: existing actualrun-v1 must be preserved' }
if (Test-Path -LiteralPath (Join-Path $package 'EXTERNAL_RUN.json')) { throw 'One actual attempt only: existing EXTERNAL_RUN.json must be preserved' }
$base = '/mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0'
$started = [DateTime]::UtcNow.ToString('o')
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& wsl.exe -d Ubuntu-22.04 --exec /bin/bash "$base/evaluation-v1/run-evaluation.sh" `
    "$base/build-v1/actualrun-v1/BUILD_RESULT.json" $BuildResultSHA256 `
    "$base/build-v1/actualrun-v1/SUPERVISOR.json" $BuildSupervisorSHA256 `
    "$base/build-v1/actualrun-v1/HOST_EXTERNAL_RUN.json" $BuildExternalSHA256 `
    "$base/build-v1/witness-v1/RESULTS.json" $WitnessResultSHA256 `
    "$base/build-v1/witness-v1/SUPERVISOR.json" $WitnessSupervisorSHA256 `
    "$base/build-v1/witness-v1/HOST_EXTERNAL_RUN.json" $WitnessExternalSHA256 `
    2>&1 | Tee-Object -FilePath (Join-Path $package 'EXTERNAL_RUN.log')
$exitCode = $LASTEXITCODE
$timer.Stop()
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
$record = [ordered]@{schema='speedy-frozen-control/host-external-v1'; scope='Frozen original Truck teacher, native versus official Speedy full forward'; started_UTC=$started; finished_UTC=[DateTime]::UtcNow.ToString('o'); exit_code=$exitCode; actual_host_wall_seconds=$timer.Elapsed.TotalSeconds; hard_limit_seconds=240; within_limit=($timer.Elapsed.TotalSeconds -le 240); attempts=1; automatic_retry=$false; shared_main_plus_audit_limit_seconds=340; independent_audit_reserved_seconds=100; build_result_sha256=$BuildResultSHA256; build_supervisor_sha256=$BuildSupervisorSHA256; build_external_sha256=$BuildExternalSHA256; witness_result_sha256=$WitnessResultSHA256; witness_supervisor_sha256=$WitnessSupervisorSHA256; witness_external_sha256=$WitnessExternalSHA256}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $package 'EXTERNAL_RUN.json') -Encoding utf8
if (($exitCode -ne 0) -or ($timer.Elapsed.TotalSeconds -gt 240)) { throw 'Main failed or exceeded true host budget; preserve evidence, no retry' }
