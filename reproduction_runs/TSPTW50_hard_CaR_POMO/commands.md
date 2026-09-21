# Executed commands and formal evaluation record

All experiment-execution commands below ran from `D:\AI-Research\CaR-Reproduction` in PowerShell. Python was always invoked through `D:\Miniconda3\envs\car\python.exe`. The first `git status --short` launch hit a Windows `CreateProcessWithLogonW` error; the identical retry succeeded. During the original experiment-execution phase, no dependency installation, source edit, Git commit, or push command was run. Local archival commits were created only after the evaluations and their artifacts had passed review. Documentation and helper scripts were written with `apply_patch`; evaluation logs were written by the helper script.

## Read-only audit commands executed

```powershell
git status --short
git diff
git status --short
git rev-parse HEAD
rg --files -g AGENTS.md -g README.md -g test.py -g Trainer.py -g TSPTWEnv.py
Get-Item -LiteralPath 'data\TSPTW\tsptw50_hard.pkl','pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt' -ErrorAction SilentlyContinue | Select-Object FullName,Length
rg -n "test_episodes|test_batch_size|test_pomo_size|validation_improve_steps|eval_type|pomo_start|soft_constrained|parse_args|checkpoint|dataset|hard" test.py README.md Trainer.py envs/TSPTWEnv.py
Get-Content -LiteralPath test.py -TotalCount 260
Get-Content -LiteralPath test.py | Select-Object -Skip 395 -First 75
rg -n "disable_preset_args|eval_only|test_opt_path|aug_factor|test_pomo_size|sample_size" test.py README.md Trainer.py
Get-Content -LiteralPath README.md | Select-Object -Skip 115 -First 60
Get-Content -LiteralPath README.md | Select-Object -Skip 88 -First 35
Get-Content -LiteralPath Trainer.py | Select-Object -Skip 425 -First 55
Get-Content -LiteralPath envs\TSPTWEnv.py | Select-Object -Skip 570 -First 45
Get-Content -LiteralPath Trainer.py | Select-Object -Skip 1315 -First 115
git show -s --format=%H%n%cI%n%s HEAD
Get-FileHash -Algorithm SHA256 -LiteralPath 'data\TSPTW\tsptw50_hard.pkl','pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt' | Select-Object Path,Hash
Get-FileHash -Algorithm SHA256 -LiteralPath 'data\TSPTW\tsptw50_hard.pkl','pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt' | Format-List
rg -n -A 25 -B 3 "def get_default_test_dataset_name|def get_data_dir_for_problem|def get_opt_sol_path" utils.py
Get-Item -LiteralPath 'data\TSPTW\lkh_tsptw50_hard.pkl' -ErrorAction SilentlyContinue | Select-Object FullName,Length
Get-Content -LiteralPath envs\TSPTWEnv.py | Select-Object -Skip 258 -First 48
Get-Content -LiteralPath Trainer.py | Select-Object -Skip 125 -First 62
Get-Content -LiteralPath Trainer.py -TotalCount 125
rg -n -A 60 "class SINGLEModel|def __init__" models\SINGLEModel.py
Test-Path -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO'
```

## Environment and data commands executed

```powershell
& 'D:\Miniconda3\envs\car\python.exe' -c "import sys; print(sys.executable); print(sys.version)"
& 'D:\Miniconda3\envs\car\python.exe' -c "import torch; print('torch='+torch.__version__); print('cuda='+str(torch.version.cuda)); print('available='+str(torch.cuda.is_available())); print('gpu='+torch.cuda.get_device_name(0)); print('capability='+str(torch.cuda.get_device_capability(0))); print('architectures='+','.join(torch.cuda.get_arch_list())); p=torch.cuda.get_device_properties(0); print('total_vram_bytes='+str(p.total_memory))"
& 'D:\Miniconda3\envs\car\python.exe' -m pip check
Get-ChildItem -LiteralPath 'data\TSPTW' -File | Measure-Object -Property Length -Sum
& 'D:\Miniconda3\envs\car\python.exe' -c "import pickle, pathlib; p=pathlib.Path(r'data\TSPTW\tsptw50_hard.pkl'); f=p.open('rb'); d=pickle.load(f); f.close(); print('dataset_count='+str(len(d))); print('first_item_field_count='+str(len(d[0]))); print('first_item_field_lengths='+str([len(x) for x in d[0]]))"
Get-Item -LiteralPath 'data\TSPTW\tsptw50_hard.pkl','data\TSPTW\lkh_tsptw50_hard.pkl','pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt' | Select-Object FullName,Length
Get-PSDrive -Name D | Select-Object Name,Used,Free
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
& 'D:\Miniconda3\envs\car\python.exe' -c "import torch; p=r'pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt'; c=torch.load(p,map_location='cpu',weights_only=False); s=c['model_state_dict'] if 'model_state_dict' in c else c; print('checkpoint_fields='+str(list(c.keys())[:12])); print('weight_keys='+str(len(s))); print('first_weights='+str([(k,tuple(v.shape)) for k,v in list(s.items())[:5]])); print('has_module_prefix='+str(any(k.startswith('module.') for k in s)))"
& 'D:\Miniconda3\envs\car\python.exe' -m pip freeze
```

## Experiment commands executed

```powershell
New-Item -ItemType Directory -Path 'reproduction_runs\TSPTW50_hard_CaR_POMO' -ErrorAction Stop
Get-Item -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1' | Select-Object FullName,Length
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1'
& 'D:\Miniconda3\envs\car\python.exe' 'reproduction_runs\TSPTW50_hard_CaR_POMO\verify_checkpoint.py'
& 'D:\Miniconda3\envs\car\python.exe' -m reproduction_runs.TSPTW50_hard_CaR_POMO.verify_checkpoint
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1' -ValidationImproveSteps 20 -LogPrefix 'calibration_20step'
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\calibration_20step.gpu_samples.csv' | Select-Object -First 20
git status --short
git diff -- train.py
Get-ChildItem -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO' -File | Select-Object Name,Length
Import-Csv -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\results.csv' | Select-Object phase,episodes,batch_size,validation_improve_steps,improvement_aug_score,exit_code
$parseErrors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_full_evaluation.ps1'), [ref]$null, [ref]$parseErrors) | Out-Null; $parseErrors
& 'D:\Miniconda3\envs\car\python.exe' 'reproduction_runs\TSPTW50_hard_CaR_POMO\verify_checkpoint.py'
$parseErrors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1'), [ref]$null, [ref]$parseErrors) | Out-Null; $parseErrors
Test-Path -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log'
git diff --name-only
```

The first direct `verify_checkpoint.py` invocation failed because a script launched from a subdirectory did not have the repository root on `sys.path`. The helper was fixed to add that root; both the `-m` invocation and the final direct invocation succeeded and strictly loaded all 134 weights. The failed check did **not** run the evaluator. The first `run_smoke.ps1` call used its default 2 episodes, batch 2, and 1 validation improvement step. The second used the same 2 episodes and batch 2 with 20 steps. The runner logs the exact Python arguments, wall time, exit code, and GPU samples. It was parameterized after the first run; its default behavior was preserved.

## Prelaunch reliability review commands executed

```powershell
git status --short
Get-ChildItem -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO' -File | Select-Object Name,Length,LastWriteTime
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object { $_.CommandLine -match 'CaR-Reproduction|TSPTW|test.py' } | Select-Object ProcessId,CommandLine
Test-Path -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log'
Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object Id,Path,StartTime
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1'
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\probe_sampler_failure.ps1'
$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1'), [ref]$null, [ref]$errors) | Out-Null; $errors
$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_full_evaluation.ps1'), [ref]$null, [ref]$errors) | Out-Null; $errors
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\sampler_failure_probe.gpu_samples.csv' | Select-Object -First 5
Test-Path -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log'
```

`Get-CimInstance` returned access denied, so the process check fell back to `Get-Process`, which found no Python process. The full evaluation log was absent both before and after the review. The two-instance sampler failure probe used the same car Python executable and evaluation configuration as the one-step smoke test; its deliberate monitor failure produced `NA` samples while the evaluator and complete log finished normally.

## 100-instance benchmark commands executed

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1' -TestEpisodes 100 -TestBatchSize 32 -ValidationImproveSteps 20 -LogPrefix 'benchmark_100'
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\benchmark_100.log' -Raw
Import-Csv -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\benchmark_100.gpu_samples.csv' | Group-Object gpu_memory_used_mib
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1' -TestEpisodes 100 -TestBatchSize 32 -ValidationImproveSteps 20 -LogPrefix 'benchmark_100_memory'
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\benchmark_100_memory.log' -Raw
Import-Csv -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\benchmark_100_memory.gpu_samples.csv' | Measure-Object -Property gpu_memory_used_mib -Maximum -Minimum
```

The first command's evaluator completed normally, but every auxiliary GPU sample was `NA`. Inspection showed that a valid `nvidia-smi` value was rejected when PowerShell left `$LASTEXITCODE` unset. The helper's memory probe was minimally changed to parse the returned value; the second command then provided 25/25 valid samples. Both runs produced identical model metrics. No model, data, algorithm, or metric source was edited.

## 10,000-instance formal evaluation commands executed

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_full_evaluation.ps1'
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.stdout.log' -Tail 20
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log' -TotalCount 55
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log' -Tail 55
Import-Csv -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.gpu_samples.csv' | Measure-Object -Property gpu_memory_used_mib -Maximum -Minimum
& 'D:\Miniconda3\envs\car\python.exe' 'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py'
```

The runner continuously redirected evaluator stdout and stderr to their respective files and sampled GPU memory every 0.5 seconds. After process exit it assembled `full_evaluation.log`. The parser validated the command and printed configuration, final episode count, metrics, exit code, empty stderr, absence of common fatal markers, and agreement between the GPU CSV and log before updating `results.csv` and `parsed_evaluation_summary.json`.

## Final integrity commands

```powershell
& 'D:\Miniconda3\envs\car\python.exe' 'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py'
& 'D:\Miniconda3\envs\car\python.exe' -c "import sys, torch; print('python='+sys.executable); print('torch='+torch.__version__); print('cuda_available='+str(torch.cuda.is_available())); print('cuda='+str(torch.version.cuda)); print('gpu='+torch.cuda.get_device_name(0)); print('capability='+str(torch.cuda.get_device_capability(0)))"
Get-FileHash -Algorithm SHA256 -LiteralPath 'data\TSPTW\tsptw50_hard.pkl','pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt'
git diff --no-ext-diff --binary --output=reproduction_runs/TSPTW50_hard_CaR_POMO/source_changes.diff
git status --short
git diff -- train.py
Get-PSDrive -Name D | Select-Object Name,Used,Free
```

## Five-step and ten-step refinement phase

The 20-step command, effective printed configuration, target-directory absence, Git state, GPU availability, disk space, and pre-run hashes were checked read-only before launch. The dedicated runner was parsed by PowerShell before use and compares its argument list against `full_evaluation.log` before creating output.

```powershell
Get-Content -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log' -TotalCount 42
Select-String -LiteralPath 'reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log' -Pattern '^Arguments:','Test Episodes:','Validation Improve Steps:','POMO Start:','Eval Type:'
Get-FileHash -Algorithm SHA256 -LiteralPath 'data\TSPTW\tsptw50_hard.pkl','pretrained\TSPTW\CaR-POMO_50_hard\checkpoint.pt','reproduction_runs\TSPTW50_hard_CaR_POMO\full_evaluation.log'
git status --short
git diff --name-status -- train.py test.py Trainer.py utils.py envs models data pretrained
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1'),[ref]$tokens,[ref]$errors) | Out-Null; $errors
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 5
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 10
& 'D:\Miniconda3\envs\car\python.exe' -c "import ast, pathlib; p=pathlib.Path(r'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py'); ast.parse(p.read_text(encoding='utf-8')); print('Python parse: PASS')"
& 'D:\Miniconda3\envs\car\python.exe' 'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py'
```

After each run, read-only checks verified the printed command and configuration, final metric lines, empty stderr, 313 progress rows ending at 10,000, all-valid GPU samples, matching before/after integrity snapshots, unchanged 20-step log hash, and no core source diff. Process-session polling only read the continuously growing stdout file; it did not invoke another evaluator.

## Reproducible reruns after archival

The absolute Python path below is a local-machine record. On another machine, pass its environment-specific interpreter with `-PythonExe`. The refinement runner writes to `reruns/<RunLabel>/` by default, performs all input and integrity preflight checks before creating that directory, and refuses to overwrite an existing path.

Dry runs used to validate both formal configurations without launching `test.py`:

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 5 -RunLabel 'dryrun_refinement_5' -PythonExe 'D:\Miniconda3\envs\car\python.exe' -DryRun
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 10 -RunLabel 'dryrun_refinement_10' -PythonExe 'D:\Miniconda3\envs\car\python.exe' -DryRun
```

Future 5-step and 10-step reruns must use unique labels or a new explicit output directory:

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 5 -RunLabel 'refinement_5_rerun_YYYYMMDD_HHMMSS' -PythonExe 'D:\Miniconda3\envs\car\python.exe'
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 10 -RunLabel 'refinement_10_rerun_YYYYMMDD_HHMMSS' -PythonExe 'D:\Miniconda3\envs\car\python.exe'
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' -ValidationImproveSteps 5 -RunLabel 'portable_refinement_5' -PythonExe '<path-to-python>' -OutputDir '<new-output-directory>'
```

The summarizer accepts explicit input and output locations. It prefers raw stdout/stderr when present and otherwise reads their sections from the committed combined logs. Generated summaries should go to a new output directory when the committed baseline must remain untouched.

```powershell
& 'D:\Miniconda3\envs\car\python.exe' 'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py' --input-dir 'reproduction_runs\TSPTW50_hard_CaR_POMO' --output-dir 'reproduction_runs\TSPTW50_hard_CaR_POMO\reruns\summary_YYYYMMDD_HHMMSS'
```

The 20-step wrapper uses the existing `run_smoke.ps1`. A rerun must supply a new `LogPrefix`; `full_evaluation` is the immutable committed baseline and must not be reused.

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1' -TestEpisodes 10000 -TestBatchSize 32 -ValidationImproveSteps 20 -LogPrefix 'full_evaluation_rerun_YYYYMMDD_HHMMSS'
```

Wrapper-only verification commands, which do not start model evaluation:

```powershell
$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path 'reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1'),[ref]$tokens,[ref]$errors) | Out-Null; if ($errors.Count -ne 0) { $errors | Format-List | Out-String | Write-Error } else { 'PowerShell parse: PASS' }
& 'D:\Miniconda3\envs\car\python.exe' -c "import ast, pathlib; p=pathlib.Path(r'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py'); ast.parse(p.read_text(encoding='utf-8')); print('Python parse: PASS')"
```
