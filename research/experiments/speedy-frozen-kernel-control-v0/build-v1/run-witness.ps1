param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PreparationManifestSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BuildResultSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BuildSupervisorSHA256,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$BuildExternalSHA256
)
$package = 'D:\Documents\my 3dgs\research\experiments\speedy-frozen-kernel-control-v0\build-v1'
$out = Join-Path $package 'witness-v1'
if (Test-Path -LiteralPath $out) { throw 'One witness attempt only: witness-v1 already exists' }
$started = [DateTime]::UtcNow.ToString('o')
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& wsl.exe -d Ubuntu-22.04 --exec /usr/bin/env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/timeout --signal=TERM --kill-after=5s 55s /home/administrator/venvs/3dgs/bin/python -B -u '/mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/tiny_forward_witness.py' --preparation-manifest-sha256 $PreparationManifestSHA256 --build-result-sha256 $BuildResultSHA256 --build-supervisor-sha256 $BuildSupervisorSHA256 --build-external-sha256 $BuildExternalSHA256 2>&1 | Tee-Object -FilePath (Join-Path $package 'WITNESS_HOST_EXTERNAL.log')
$exitCode = $LASTEXITCODE
$timer.Stop()
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
$record = [ordered]@{schema='speedy-private-extension/host-external-v1'; scope='Fixed seed0 2G/4x4 SH3 forward-only witness'; started_UTC=$started; finished_UTC=[DateTime]::UtcNow.ToString('o'); exit_code=$exitCode; actual_host_wall_seconds=$timer.Elapsed.TotalSeconds; hard_limit_seconds=60; within_limit=($timer.Elapsed.TotalSeconds -le 60); attempts=1; automatic_retry=$false; preparation_manifest_sha256=$PreparationManifestSHA256; build_result_sha256=$BuildResultSHA256; build_supervisor_sha256=$BuildSupervisorSHA256; build_external_sha256=$BuildExternalSHA256}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'HOST_EXTERNAL_RUN.json') -Encoding utf8
if (($exitCode -ne 0) -or ($timer.Elapsed.TotalSeconds -gt 60)) { throw 'Witness failed or exceeded external budget; preserve evidence, no retry' }
