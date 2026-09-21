param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(5, 10)]
    [int]$ValidationImproveSteps,

    [string]$RunLabel,

    [string]$OutputDir,

    [string]$PythonExe = 'D:\Miniconda3\envs\car\python.exe',

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Get-IntegrityRecords {
    param(
        [Parameter(Mandatory = $true)][string[]]$Paths,
        [Parameter(Mandatory = $true)][string]$RepositoryRoot
    )

    foreach ($target in $Paths) {
        $item = Get-Item -LiteralPath $target -ErrorAction Stop
        if ($item.PSIsContainer) {
            throw "Expected a file but found a directory: $target"
        }
        [pscustomobject]@{
            Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $item.FullName).Hash
            RelativePath = [System.IO.Path]::GetRelativePath(
                $RepositoryRoot,
                $item.FullName
            ).Replace('\', '/')
            Length = $item.Length
        }
    }
}

function Write-IntegritySnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$GitHead,
        [Parameter(Mandatory = $true)][object[]]$Records
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("Snapshot: $Label")
    $lines.Add("Timestamp: $((Get-Date).ToString('o'))")
    $lines.Add("Git HEAD: $GitHead")
    $lines.Add('Algorithm: SHA256')
    foreach ($record in $Records) {
        $lines.Add("$($record.Hash)  $($record.RelativePath)  $($record.Length)")
    }
    $lines | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Get-GpuMemoryUsedMiB {
    try {
        $gpuText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        if ($null -eq $gpuText) { return $null }
        $parsed = 0
        if ([int]::TryParse($gpuText.ToString().Trim(), [ref]$parsed)) { return $parsed }
    } catch {
        # GPU sampling is auxiliary; evaluator stdout/stderr remain authoritative.
    }
    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

if ([string]::IsNullOrWhiteSpace($RunLabel)) {
    $RunLabel = "refinement_$ValidationImproveSteps"
}
if ($RunLabel -notmatch '^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?$') {
    throw 'RunLabel must contain only letters, digits, dot, underscore, or hyphen; it must start with a letter or digit and cannot end with a dot.'
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $runDir = Join-Path (Join-Path $PSScriptRoot 'reruns') $RunLabel
} elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $runDir = $OutputDir
} else {
    $runDir = Join-Path $PSScriptRoot $OutputDir
}
$runDir = [System.IO.Path]::GetFullPath($runDir)
if (Test-Path -LiteralPath $runDir) {
    throw "Refusing to overwrite existing output path: $runDir"
}

# Resolve and validate every executable and input before creating the output directory.
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
$pythonExePath = (Resolve-Path -LiteralPath $PythonExe).Path
$gitCommand = Get-Command git -CommandType Application -ErrorAction Stop |
    Select-Object -First 1
$nvidiaSmiCommand = Get-Command nvidia-smi -CommandType Application -ErrorAction Stop |
    Select-Object -First 1

$checkpoint = 'pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt'
$checkpointPath = Join-Path $repoRoot $checkpoint
$datasetPath = Join-Path $repoRoot 'data\TSPTW\tsptw50_hard.pkl'
$referenceSolutionPath = Join-Path $repoRoot 'data\TSPTW\lkh_tsptw50_hard.pkl'
$baselineLog = Join-Path $PSScriptRoot 'full_evaluation.log'
$envsDir = Join-Path $repoRoot 'envs'
$modelsDir = Join-Path $repoRoot 'models'
foreach ($directory in @($envsDir, $modelsDir)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Required source directory not found: $directory"
    }
}

$corePaths = @(
    (Join-Path $repoRoot 'train.py'),
    (Join-Path $repoRoot 'test.py'),
    (Join-Path $repoRoot 'Trainer.py'),
    (Join-Path $repoRoot 'utils.py')
)
$corePaths += Get-ChildItem -LiteralPath $envsDir -File -Filter '*.py' |
    Select-Object -ExpandProperty FullName
$corePaths += Get-ChildItem -LiteralPath $modelsDir -File -Filter '*.py' |
    Select-Object -ExpandProperty FullName
$corePaths = @($corePaths | Sort-Object -Unique)
if ($corePaths.Count -eq 4) {
    throw 'No Python source files were found under envs or models.'
}

$hashTargets = @(
    $datasetPath,
    $referenceSolutionPath,
    $checkpointPath,
    $baselineLog,
    $PSCommandPath
) + $corePaths

$pythonArgs = @(
    '-u', 'test.py',
    '--problem', 'TSPTW',
    '--problem_size', '50',
    '--hardness', 'hard',
    '--checkpoint', $checkpoint,
    '--test_episodes', '10000',
    '--test_batch_size', '32',
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
$argumentText = $pythonArgs -join ' '
$expectedBaselineArguments = 'Arguments: -u test.py --problem TSPTW --problem_size 50 --hardness hard --checkpoint pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt --test_episodes 10000 --test_batch_size 32 --test_pomo_size 1 --improve_steps 5 --validation_improve_steps 20 --pomo_size 50 --pomo_start false --soft_constrained true --eval_type softmax --sample_size 1 --seed 2023 --gpu_id 0'
$baselineArgumentMatches = @(
    Select-String -LiteralPath $baselineLog -Pattern '^Arguments:' -ErrorAction Stop
)
if ($baselineArgumentMatches.Count -ne 1) {
    throw "Expected exactly one argument line in the 20-step baseline; found $($baselineArgumentMatches.Count)."
}
if ($baselineArgumentMatches[0].Line -ne $expectedBaselineArguments) {
    throw 'The recorded 20-step baseline arguments do not match the expected formal configuration.'
}
$expectedCurrentArguments = $expectedBaselineArguments.Replace(
    '--validation_improve_steps 20',
    "--validation_improve_steps $ValidationImproveSteps"
)
if ("Arguments: $argumentText" -ne $expectedCurrentArguments) {
    throw 'Current arguments differ from the 20-step formal configuration in more than the requested validation step value.'
}

$gitHeadOutput = @(& $gitCommand.Source -C $repoRoot rev-parse HEAD 2>&1)
if ($LASTEXITCODE -ne 0 -or $gitHeadOutput.Count -ne 1) {
    throw 'Unable to resolve exactly one Git HEAD before evaluation.'
}
$gitHeadBefore = $gitHeadOutput[0].ToString().Trim()
if ($gitHeadBefore -notmatch '^[0-9a-fA-F]{40,64}$') {
    throw "Unexpected Git HEAD value: $gitHeadBefore"
}
$integrityBefore = @(Get-IntegrityRecords -Paths $hashTargets -RepositoryRoot $repoRoot)

$stdoutPath = Join-Path $runDir "$RunLabel.stdout.log"
$stderrPath = Join-Path $runDir "$RunLabel.stderr.log"
$logPath = Join-Path $runDir "$RunLabel.log"
$gpuPath = Join-Path $runDir "$RunLabel.gpu_samples.csv"
$commandPath = Join-Path $runDir 'command.txt'
$integrityBeforePath = Join-Path $runDir 'integrity_before.txt'
$integrityAfterPath = Join-Path $runDir 'integrity_after.txt'
$plannedOutputs = @(
    $commandPath,
    $integrityBeforePath,
    $stdoutPath,
    $stderrPath,
    $gpuPath,
    $logPath,
    $integrityAfterPath
)

if ($DryRun) {
    Write-Output 'Command:'
    Write-Output ('  & "{0}" {1}' -f $pythonExePath, $argumentText)
    Write-Output 'Inputs:'
    Write-Output "  Python: $pythonExePath"
    Write-Output "  Git: $($gitCommand.Source)"
    Write-Output "  NVIDIA SMI: $($nvidiaSmiCommand.Source)"
    Write-Output "  Git HEAD: $gitHeadBefore"
    foreach ($record in $integrityBefore) {
        Write-Output ("  {0} ({1} bytes, SHA256 {2})" -f $record.RelativePath, $record.Length, $record.Hash)
    }
    Write-Output 'Outputs:'
    Write-Output "  Directory: $runDir"
    foreach ($path in $plannedOutputs) {
        Write-Output "  $path"
    }
    return
}

# No filesystem output is created before all preflight checks above have passed.
$runParent = Split-Path -Parent $runDir
if (-not (Test-Path -LiteralPath $runParent -PathType Container)) {
    [void](New-Item -ItemType Directory -Path $runParent -Force -ErrorAction Stop)
}
[void](New-Item -ItemType Directory -Path $runDir -ErrorAction Stop)

$baselineRelativePath = [System.IO.Path]::GetRelativePath(
    $repoRoot,
    $baselineLog
).Replace('\', '/')
@(
    "Run label: $RunLabel",
    "Python executable: $pythonExePath",
    "Arguments: $argumentText",
    "Reference baseline: $baselineRelativePath",
    "Only requested parameter change: --validation_improve_steps 20 -> $ValidationImproveSteps"
) | Set-Content -LiteralPath $commandPath -Encoding UTF8

$beforeSnapshotParameters = @{
    Path = $integrityBeforePath
    Label = 'before evaluation'
    GitHead = $gitHeadBefore
    Records = $integrityBefore
}
Write-IntegritySnapshot @beforeSnapshotParameters

'timestamp,gpu_memory_used_mib' | Set-Content -LiteralPath $gpuPath -Encoding UTF8
$baselineMemory = Get-GpuMemoryUsedMiB
$peakMemory = $baselineMemory
$started = Get-Date
$nextProgress = $started.AddSeconds(15)

try {
    $startParameters = @{
        FilePath = $pythonExePath
        ArgumentList = $pythonArgs
        WorkingDirectory = $repoRoot
        PassThru = $true
        WindowStyle = 'Hidden'
        RedirectStandardOutput = $stdoutPath
        RedirectStandardError = $stderrPath
    }
    $process = Start-Process @startParameters
} catch {
    ("Launch failed: {0}" -f $_.Exception.Message) |
        Set-Content -LiteralPath $logPath -Encoding UTF8
    throw
}

do {
    $used = Get-GpuMemoryUsedMiB
    $usedLabel = if ($null -eq $used) { 'NA' } else { $used.ToString() }
    if ($null -ne $used -and ($null -eq $peakMemory -or $used -gt $peakMemory)) {
        $peakMemory = $used
    }
    ('{0},{1}' -f (Get-Date -Format 'o'), $usedLabel) |
        Add-Content -LiteralPath $gpuPath -Encoding UTF8

    if ((Get-Date) -ge $nextProgress) {
        $latestProgress = Get-Content -LiteralPath $stdoutPath -ErrorAction SilentlyContinue |
            Select-String -Pattern '^episode\s+' |
            Select-Object -Last 1
        if ($null -ne $latestProgress) {
            Write-Output ("[{0}] {1}" -f $RunLabel, $latestProgress.Line)
        } else {
            Write-Output ("[{0}] evaluator is starting" -f $RunLabel)
        }
        $nextProgress = (Get-Date).AddSeconds(15)
    }

    Start-Sleep -Milliseconds 500
    $process.Refresh()
} while (-not $process.HasExited)

$process.WaitForExit()
$finished = Get-Date
$exitCode = $process.ExitCode
$elapsed = ($finished - $started).TotalSeconds
$baselineLabel = if ($null -eq $baselineMemory) { 'NA' } else { $baselineMemory.ToString() }
$peakLabel = if ($null -eq $peakMemory) { 'NA' } else { $peakMemory.ToString() }

$gitHeadAfterOutput = @(& $gitCommand.Source -C $repoRoot rev-parse HEAD 2>&1)
if ($LASTEXITCODE -ne 0 -or $gitHeadAfterOutput.Count -ne 1) {
    throw 'Unable to resolve exactly one Git HEAD after evaluation.'
}
$gitHeadAfter = $gitHeadAfterOutput[0].ToString().Trim()
$integrityAfter = @(Get-IntegrityRecords -Paths $hashTargets -RepositoryRoot $repoRoot)
$afterSnapshotParameters = @{
    Path = $integrityAfterPath
    Label = 'after evaluation'
    GitHead = $gitHeadAfter
    Records = $integrityAfter
}
Write-IntegritySnapshot @afterSnapshotParameters

$metadata = @(
    "TSPTW50-hard CaR-POMO $RunLabel",
    "Python executable: $pythonExePath",
    "Arguments: $argumentText",
    "Started: $($started.ToString('o'))",
    "Finished: $($finished.ToString('o'))",
    ('Elapsed seconds: {0:F3}' -f $elapsed),
    "Exit code: $exitCode",
    "GPU baseline MiB: $baselineLabel",
    "GPU sampled peak MiB: $peakLabel",
    '----- STDOUT -----'
)
$metadata | Set-Content -LiteralPath $logPath -Encoding UTF8
Get-Content -LiteralPath $stdoutPath |
    Add-Content -LiteralPath $logPath -Encoding UTF8
'----- STDERR -----' | Add-Content -LiteralPath $logPath -Encoding UTF8
Get-Content -LiteralPath $stderrPath |
    Add-Content -LiteralPath $logPath -Encoding UTF8

Write-Output ("[{0}] finished: exit={1}, wall={2:F3}s, gpu_peak={3} MiB" -f $RunLabel, $exitCode, $elapsed, $peakLabel)
Get-Content -LiteralPath $stdoutPath -Tail 28
exit $exitCode
