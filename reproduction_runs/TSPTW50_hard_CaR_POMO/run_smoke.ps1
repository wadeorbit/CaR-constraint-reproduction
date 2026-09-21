param(
    [int]$TestEpisodes = 2,
    [int]$TestBatchSize = 2,
    [int]$ValidationImproveSteps = 1,
    [string]$LogPrefix = 'smoke_test'
)

$ErrorActionPreference = 'Stop'

$runDir = Join-Path (Get-Location) 'reproduction_runs\TSPTW50_hard_CaR_POMO'
$stdoutPath = Join-Path $runDir "$LogPrefix.stdout.log"
$stderrPath = Join-Path $runDir "$LogPrefix.stderr.log"
$logPath = Join-Path $runDir "$LogPrefix.log"
$gpuPath = Join-Path $runDir "$LogPrefix.gpu_samples.csv"
foreach ($path in @($stdoutPath, $stderrPath, $logPath, $gpuPath)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite $path" }
}

$pythonExe = 'D:\Miniconda3\envs\car\python.exe'
$pythonArgs = @(
    '-u', 'test.py',
    '--problem', 'TSPTW',
    '--problem_size', '50',
    '--hardness', 'hard',
    '--checkpoint', 'pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt',
    '--test_episodes', $TestEpisodes.ToString(),
    '--test_batch_size', $TestBatchSize.ToString(),
    '--test_pomo_size', '1',
    '--improve_steps', '5',
    '--validation_improve_steps', $ValidationImproveSteps.ToString(),
    '--pomo_size', '50',
    '--pomo_start', 'false',
    '--soft_constrained', 'true',
    '--eval_type', 'softmax',
    '--sample_size', '1',
    '--seed', '2023',
    '--gpu_id', '0'
)

function Get-GpuMemoryUsedMiB {
    try {
        $text = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        # Some PowerShell hosts leave LASTEXITCODE unset after this pipeline even
        # when nvidia-smi returns a valid value, so validate the value itself.
        if ($null -eq $text) { return $null }
        $parsed = 0
        if ([int]::TryParse($text.ToString().Trim(), [ref]$parsed)) { return $parsed }
    } catch {
        # GPU sampling is auxiliary; preserve the evaluation log if it fails.
    }
    return $null
}

'timestamp,gpu_memory_used_mib' | Set-Content -LiteralPath $gpuPath -Encoding UTF8
$baseline = Get-GpuMemoryUsedMiB
$baselineLabel = if ($null -eq $baseline) { 'NA' } else { $baseline.ToString() }
$peak = $baseline
$started = Get-Date
try {
    $process = Start-Process -FilePath $pythonExe -ArgumentList $pythonArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
} catch {
    ('Launch failed: {0}' -f $_.Exception.Message) | Set-Content -LiteralPath $logPath -Encoding UTF8
    throw
}
do {
    $used = Get-GpuMemoryUsedMiB
    $usedLabel = if ($null -eq $used) { 'NA' } else { $used.ToString() }
    if ($null -ne $used -and ($null -eq $peak -or $used -gt $peak)) { $peak = $used }
    ('{0},{1}' -f (Get-Date -Format 'o'), $usedLabel) | Add-Content -LiteralPath $gpuPath -Encoding UTF8
    Start-Sleep -Milliseconds 500
    $process.Refresh()
} while (-not $process.HasExited)
$process.WaitForExit()
$finished = Get-Date
$exitCode = $process.ExitCode
$elapsed = ($finished - $started).TotalSeconds
$peakLabel = if ($null -eq $peak) { 'NA' } else { $peak.ToString() }

$metadata = @(
    ('TSPTW50-hard CaR-POMO {0}' -f $LogPrefix),
    ('Python executable: {0}' -f $pythonExe),
    ('Arguments: {0}' -f ($pythonArgs -join ' ')),
    ('Started: {0}' -f $started.ToString('o')),
    ('Finished: {0}' -f $finished.ToString('o')),
    ('Elapsed seconds: {0:F3}' -f $elapsed),
    ('Exit code: {0}' -f $exitCode),
    ('GPU baseline MiB: {0}' -f $baselineLabel),
    ('GPU sampled peak MiB: {0}' -f $peakLabel),
    '----- STDOUT -----'
)
$metadata | Set-Content -LiteralPath $logPath -Encoding UTF8
Get-Content -LiteralPath $stdoutPath | Add-Content -LiteralPath $logPath -Encoding UTF8
'----- STDERR -----' | Add-Content -LiteralPath $logPath -Encoding UTF8
Get-Content -LiteralPath $stderrPath | Add-Content -LiteralPath $logPath -Encoding UTF8
Get-Content -LiteralPath $logPath
exit $exitCode
