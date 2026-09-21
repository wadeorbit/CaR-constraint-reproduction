# 10,000 样本正式评测记录

## 已执行命令

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_full_evaluation.ps1'
```

该包装脚本实际调用：

```text
--problem TSPTW --problem_size 50 --hardness hard
--checkpoint pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt
--test_episodes 10000 --test_batch_size 32 --test_pomo_size 1
--improve_steps 5 --validation_improve_steps 20
--pomo_size 50 --pomo_start false --soft_constrained true
--eval_type softmax --sample_size 1 --seed 2023 --gpu_id 0
```

未传入 `--disable_preset_args`，避免当前 `test.py` 的反向布尔逻辑覆盖显式参数。`Trainer.py` 对每个实例使用 8 倍增强，并加载仓库中的 TSPTW50-hard 数据和 LKH 参考解。

## 启动依据

100 样本基准的墙钟线性估计为 1,428.9 秒（约 23 分 49 秒），低于 2 小时；显存采样峰值 331 MiB，运行无异常。按照用户授权，满足条件后直接启动正式评测。

## 实际结果

- 开始：2026-09-20T21:50:10.5854464+08:00
- 结束：2026-09-20T21:58:29.0326520+08:00
- 墙钟：498.447 秒
- 程序内部评测时间：491.02 秒
- GPU 采样峰值：331 MiB；基线 12 MiB
- 进度：10,000/10,000，共 313 批，末批 16 个实例
- 退出码：0
- stderr：空
- Improvement AUG：objective 25.6142，Gap 0.0151%，instance-level Infsb 0.000%

完整日志持续写入 `full_evaluation.stdout.log` 和 `full_evaluation.stderr.log`；结束后汇总为 `full_evaluation.log`。GPU 采样在 `full_evaluation.gpu_samples.csv`，结构化结果在 `results.csv` 和 `parsed_evaluation_summary.json`。
