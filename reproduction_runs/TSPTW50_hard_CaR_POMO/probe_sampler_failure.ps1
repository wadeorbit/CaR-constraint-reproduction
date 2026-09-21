# Exercise the two-instance evaluator while simulating an unavailable GPU monitor.
function global:nvidia-smi {
    throw 'Simulated nvidia-smi failure for log-capture verification'
}

& (Join-Path $PSScriptRoot 'run_smoke.ps1') -LogPrefix 'sampler_failure_probe'
exit $LASTEXITCODE
