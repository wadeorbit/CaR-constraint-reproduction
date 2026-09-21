# TSPTW50-hard CaR-POMO：10 步 refinement

## 配置

- Python：`D:\Miniconda3\envs\car\python.exe`
- 数据：`data/TSPTW/tsptw50_hard.pkl`
- checkpoint：`pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt`
- episodes / batch：10,000 / 32
- augmentation：8
- `test_pomo_size=1`，`pomo_start=false`
- `soft_constrained=true`，`eval_type=softmax`，`sample_size=1`
- `validation_improve_steps=10`，seed 2023，GPU 0

运行器逐字核对 20 步基线命令后，仅将 `--validation_improve_steps 20` 替换为 10。

## 结果

| 指标 | 数值 |
| --- | ---: |
| Improvement AUG Objective | 25.6154 |
| Improvement AUG Gap | 0.0195% |
| Improvement solution-level Infsb | 0.076% |
| Improvement instance-level Infsb | 0.010% |
| Evaluation time | 329.83 s |
| Wall time | 337.866 s |
| GPU baseline / sampled peak | 55 / 428 MiB |
| GPU sampled peak delta | 373 MiB |
| GPU valid samples | 613/613 |
| Exit code | 0 |

论文 Table 2 为 Objective 25.615、Gap 0.020%、instance-level Infsb 0.01%、Time 27 s。本机 Objective、Gap、Infsb 的绝对差分别为 0.0004、0.0005 pp、0.000 pp；相对误差分别为 0.001562%、2.500%、0%。

313 个批次完整处理 10,000/10,000 个实例，stderr 为空。运行前后 15 项数据、checkpoint、20 步基线日志和核心源码哈希完全一致。
