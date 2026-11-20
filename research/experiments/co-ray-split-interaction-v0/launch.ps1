param(
    [Parameter(Mandatory=$true)][string]$Contract,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ContractSha256,
    [Parameter(Mandatory=$true)][string]$Arms,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ArmsSha256,
    [Parameter(Mandatory=$true)][ValidatePattern('^run-[A-Za-z0-9_-]+$')][string]$RunName
)
$ErrorActionPreference = 'Stop'
$stageDirectory = $PSScriptRoot
$runDirectory = Join-Path $stageDirectory $RunName
if (Test-Path -LiteralPath $runDirectory) { throw 'Existing output refused; no retry or overwrite.' }
$contractPath = (Resolve-Path -LiteralPath $Contract).Path
$armsPath = (Resolve-Path -LiteralPath $Arms).Path
if ((Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ContractSha256) { throw 'Contract SHA mismatch.' }
if ((Get-FileHash -LiteralPath $armsPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ArmsSha256) { throw 'Arms SHA mismatch.' }
function Convert-ToWslPath([string]$taskPath) {
    if ($taskPath -notmatch '^([A-Za-z]):\\(.*)$') { throw 'Absolute local drive path required.' }
    return '/mnt/' + $Matches[1].ToLowerInvariant() + '/' + $Matches[2].Replace('\','/')
}
$launcher = Convert-ToWslPath (Join-Path $stageDirectory 'run-diagnostic.sh')
$runWsl = Convert-ToWslPath $runDirectory
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$timedOut = $false
$exitCode = $null
try {
    # PowerShell's argument array avoids shell interpolation of path contents.
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = 'wsl.exe'
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    foreach ($argument in @('-d','Ubuntu-22.04','--','bash',$launcher,(Convert-ToWslPath $contractPath),$ContractSha256,(Convert-ToWslPath $armsPath),$ArmsSha256,$runWsl)) {
        [void]$processInfo.ArgumentList.Add($argument)
    }
    $taskProcess = [System.Diagnostics.Process]::Start($processInfo)
    if (-not $taskProcess.WaitForExit(300000)) {
        $timedOut = $true
        # Inner GNU timeout and the supervisor independently reap the Linux worker.
        $taskProcess.Kill($true)
        $taskProcess.WaitForExit()
    }
    $exitCode = $taskProcess.ExitCode
} finally {
    $watch.Stop()
    if (Test-Path -LiteralPath $runDirectory) {
        $record = [ordered]@{
            schema='co-ray-split-interaction-v0/external-root-v1'; true_external_wall_seconds=$watch.Elapsed.TotalSeconds
            maximum_seconds=300; within_300_seconds=($watch.Elapsed.TotalSeconds -le 300); timed_out=$timedOut
            exit_code=$exitCode; attempts=1; automatic_retry=$false
            launcher_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
            contract_sha256=$ContractSha256; arms_sha256=$ArmsSha256
        }
        $destination = Join-Path $runDirectory 'EXTERNAL_ROOT.json'
        $stream = [System.IO.File]::Open($destination,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write)
        $writer = New-Object System.IO.StreamWriter($stream,(New-Object System.Text.UTF8Encoding($false)))
        try { $writer.WriteLine(($record | ConvertTo-Json -Depth 8)) } finally { $writer.Dispose() }
    }
}
if ($timedOut -or $null -eq $exitCode -or $exitCode -ne 0) { throw 'Diagnostic incomplete; original artifacts retained and no retry performed.' }
