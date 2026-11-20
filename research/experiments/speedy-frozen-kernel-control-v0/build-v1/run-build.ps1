param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$PreparationManifestSHA256
)
$package = 'D:\Documents\my 3dgs\research\experiments\speedy-frozen-kernel-control-v0\build-v1'
$out = Join-Path $package 'actualrun-v1'
if (Test-Path -LiteralPath $out) { throw 'One build attempt only: actualrun-v1 already exists' }
$started = [DateTime]::UtcNow.ToString('o')
$timer = [System.Diagnostics.Stopwatch]::StartNew()
& wsl.exe -d Ubuntu-22.04 --exec /usr/bin/env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/timeout --signal=TERM --kill-after=5s 175s /home/administrator/venvs/3dgs/bin/python -B -u '/mnt/d/Documents/my 3dgs/research/experiments/speedy-frozen-kernel-control-v0/build-v1/run_private_build.py' --preparation-manifest-sha256 $PreparationManifestSHA256 2>&1 | Tee-Object -FilePath (Join-Path $package 'BUILD_HOST_EXTERNAL.log')
$exitCode = $LASTEXITCODE
$timer.Stop()
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
$record = [ordered]@{schema='speedy-private-extension/host-external-v1'; scope='One official private author build and import; zero CUDA dispatch'; started_UTC=$started; finished_UTC=[DateTime]::UtcNow.ToString('o'); exit_code=$exitCode; actual_host_wall_seconds=$timer.Elapsed.TotalSeconds; hard_limit_seconds=180; within_limit=($timer.Elapsed.TotalSeconds -le 180); attempts=1; automatic_retry=$false; preparation_manifest_sha256=$PreparationManifestSHA256}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'HOST_EXTERNAL_RUN.json') -Encoding utf8
if (($exitCode -ne 0) -or ($timer.Elapsed.TotalSeconds -gt 180)) { throw 'Build/import failed or exceeded external budget; preserve evidence, no retry' }
