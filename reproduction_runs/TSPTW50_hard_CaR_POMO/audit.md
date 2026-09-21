# TSPTW50-hard CaR-POMO audit

## Initial repository state (before any experiment files were added)

`git rev-parse HEAD`: `56b25c50209b17c5d0002d56e65720c1e2da6a6b` (`Update arxiv`, 2026-09-13T20:10:23+08:00).

Initial `git status --short`:

```text
 M train.py
?? __pycache__/
?? envs/__pycache__/
?? models/__pycache__/
?? results/
```

Initial `git diff`:

```diff
diff --git a/train.py b/train.py
index 60c3af4..91c03d1 100644
--- a/train.py
+++ b/train.py
@@ -353,7 +353,7 @@ if __name__ == "__main__":
     parser.add_argument('--gpu_id', type=str, default="0")
     parser.add_argument('--world_size', type=int, default=1)
     parser.add_argument("--multiple_gpu", type=str2bool, default=False)
-    parser.add_argument('--occ_gpu', type=float, default=0., help="occupy (X)% GPU memory in advance, please use sparingly.")
+    parser.add_argument('--occ_gpu', type=float, default=0., help="occupy (X)%% GPU memory in advance, please use sparingly.")
     parser.add_argument('--tb_logger', type=str2bool, default=True)
     parser.add_argument('--wandb_logger', type=str2bool, default=True)
     parser.add_argument('--clean_cache', type=str2bool, default=False)
```

The pre-existing edit and untracked directories were left untouched. No algorithm, data, metric, or original source file was changed in this stage. No `source_changes.diff` was needed.

## Files and data

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `data/TSPTW/tsptw50_hard.pkl` | 24,693,418 | `E8FB50D07692785D0E62A5F80BEBCD602C01051E81640718BAE783C395F33882` |
| `pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt` | 6,599,556 | `A85BE08B0CDC2C7EAE08DB838D4F295B7A3F493773293DB7163659F1AA5B4A80` |
| `data/TSPTW/lkh_tsptw50_hard.pkl` | 1,234,572 | not calculated |

The target dataset contains **10,000** instances. The first instance has four fields, each with 50 nodes. `data/TSPTW` contains 16 regular files totalling 200,440,753 bytes. The default dataset and LKH reference paths both resolve to the files above.

## Source behavior

- `README.md` gives the TSPTW50-hard CaR-POMO checkpoint command and recommends `validation_improve_steps=20` for evaluation. Its note on `--disable_preset_args` conflicts with the current code.
- `test.py:267` declares `--disable_preset_args` with `action='store_false', default=True`; `test.py:421` calls `set_problem_defaults` only when this value is false. Thus **passing this flag enables the preset**, and omitting it leaves explicit CLI values intact.
- The TSPTW preset in `test.py:155-170` forces `test_episodes=10000`, `test_batch_size=3334` for size 50, `pomo_start=False`, `soft_constrained=True`, and `eval_type=softmax`. Passing the flag with a 2-instance command would silently undo its sample and batch limits.
- Without the flag, parser defaults are 1,000 episodes, batch 1,000, `pomo_start=True`, `soft_constrained=False`, `eval_type=argmax`, `test_pomo_size=1`, and `validation_improve_steps=20`. The bare README command therefore does not reproduce the preset's effective configuration.
- `test.py:191-200` prints the effective values after parsing. Both completed 2-instance runs printed `Test Episodes: 2 | Batch Size: 2`, `POMO Start: False`, `Test POMO Size: 1`, `Soft Constrained: True`, and `Eval Type: softmax`.
- `Trainer.py:428-467` uses the effective test count and batch size and selects the default TSPTW dataset through `utils.py:145-158`. If `pomo_start=False`, it sets the evaluation environment's POMO size to `test_pomo_size=1`.
- `Trainer.py:1333-1345` loads instances in order and uses eightfold augmentation. `envs/TSPTWEnv.py:584-601` reads the pickle and slices by offset and count. `Trainer.py:1394-1406` loads `lkh_tsptw50_hard.pkl` for Gap. For a custom reference path, this code uses `--val_opt_path`; `--test_opt_path` is parsed but not used there.
- `Trainer.py:133-160` has a potentially silent `strict=False` checkpoint fallback. A separate read-only reconstruction of the smoke model configuration loaded the checkpoint with `strict=True`: **134 checkpoint keys matched 134 model keys**, and checkpoint problem is `TSPTW`. No fallback is needed for this configuration.
- `test.py` defaults to seed 2024, while [paper Appendix E.9](https://arxiv.org/pdf/2602.16012) states inference seed 2023. Both runs here explicitly used 2023.

## Small evaluation status

The one-step smoke and 20-step two-instance calibration both exited with code 0, had empty stderr and no Traceback, and used the designated CUDA GPU. Logs and metrics are in `smoke_test.log`, `calibration_20step.log`, and `results.csv`. A third two-instance diagnostic deliberately made the auxiliary `nvidia-smi` sampler fail. The evaluator still exited with code 0, full stdout/stderr and metrics were preserved in `sampler_failure_probe.log`, and GPU samples were recorded as `NA`. This verifies that a monitor outage will not interrupt the evaluation log.

## Official reference

[Paper Table 2](https://arxiv.org/pdf/2602.16012) reports the following TSPTW-50 Hard CaR-POMO values on 10,000 instances, with eightfold augmentation and a single RTX 4090. The paper's TSPTW inference uses no multi-start and one sampled route per augmented instance.

| Improvement steps | Objective | Gap | Infeasible rate | RTX 4090 time |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 25.619 | 0.034% | 0.02% | 15 s |
| 10 | 25.615 | 0.020% | 0.01% | 27 s |
| 20 | 25.614 | 0.014% | 0.01% | 51 s |

The [author repository](https://github.com/jieyibi/CaR-constraint) supplies the pretrained model and dataset used in this stage.

## 100-instance timing benchmark (2026-09-20)

The authoritative benchmark used the formal model configuration with 100 episodes and batch size 32. The evaluator printed `Test Episodes: 100 | Batch Size: 32`, loaded `tsptw50_hard.pkl`, used CUDA device 0, and exited with code 0. Wall time was 14.289 seconds (0.142890 seconds per instance); evaluator time was 6.96 seconds (0.069600 seconds per instance). The sampled whole-GPU peak was 331 MiB from a 12 MiB baseline. Improvement AUG objective was 25.7621, Gap was 0.0292%, solution-level infeasibility was 0.125%, and instance-level infeasibility was 0.000%.

The first 100-instance attempt also completed successfully and produced identical model metrics, but its auxiliary GPU samples were all `NA`: PowerShell left `$LASTEXITCODE` unset after a valid `nvidia-smi` pipeline. The reproduction helper was changed to validate the returned memory value directly. This change affected monitoring only; no repository model, environment, dataset, algorithm, or metric source was changed. The same evaluation was rerun so that the requested peak-memory field was valid.

Linear extrapolation from the authoritative wall time predicted 1,428.9 seconds (23 minutes 48.9 seconds) for 10,000 instances. This was below the two-hour threshold, the peak used only 331/8,151 MiB, and the run showed no evaluator error. The user-authorized automatic launch conditions were therefore met.

## 10,000-instance formal evaluation (2026-09-20)

The full run processed all 10,000 instances in 313 contiguous batches (312 batches of 32 and a final batch of 16). It exited with code 0; stderr was empty; no Traceback, CUDA OOM, NaN, Infinity, RuntimeError, AssertionError, or missing/duplicate progress marker was found.

| Item | Recorded value |
| --- | ---: |
| Wall time | 498.447 s |
| Evaluator time | 491.02 s |
| Wall time per instance | 0.0498447 s |
| Evaluator time per instance | 0.049102 s |
| GPU baseline | 12 MiB |
| Sampled GPU peak | 331 MiB |
| Valid GPU samples | 906/906 |
| Improvement AUG objective | 25.6142 |
| Improvement AUG Gap | 0.0151% |
| Improvement solution-level infeasibility | 0.061% |
| Improvement instance-level infeasibility | 0.000% |

The log's effective configuration is TSPTW size 50 Hard, official CaR-POMO checkpoint, 10,000 episodes, batch 32, 20 validation improvement steps, eightfold augmentation, `test_pomo_size=1`, `pomo_start=false`, `soft_constrained=true`, `eval_type=softmax`, `sample_size=1`, seed 2023, and GPU 0. The target dataset and checkpoint hashes were rechecked after the run and remained unchanged.

The paper's Infsb metric is instance-level: it evaluates the best solution per original instance after construction and refinement. `Trainer.py` first takes `any` over the eight augmented streams for each original instance before computing this value. The solution-level percentage is retained as a useful diagnostic but is not used for the main paper comparison.
