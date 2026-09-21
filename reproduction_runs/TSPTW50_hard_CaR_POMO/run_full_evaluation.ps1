# Formal 10,000-instance evaluation wrapper. Executed after user approval.
$ErrorActionPreference = 'Stop'
try {
    & (Join-Path $PSScriptRoot 'run_smoke.ps1') `
        -TestEpisodes 10000 `
        -TestBatchSize 32 `
        -ValidationImproveSteps 20 `
        -LogPrefix 'full_evaluation'
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.ToString())
    exit 1
}
