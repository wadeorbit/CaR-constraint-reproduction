# 100 样本正式配置计时基准

## 实际生效配置

```text
Python: D:\Miniconda3\envs\car\python.exe
Problem: TSPTW
Problem size: 50
Hardness: hard
Checkpoint: pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt
Dataset: data/TSPTW/tsptw50_hard.pkl
Test episodes: 100
Test batch size: 32
Validation improve steps: 20
Augmentation: 8
Test POMO size: 1
POMO start: false
Soft constrained: true
Eval type: softmax
Sample size: 1
Seed: 2023
GPU id: 0
CUDA active: true
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

命令没有传入 `--disable_preset_args`，因此上述显式参数没有被仓库预设覆盖。程序打印确认 `Test Episodes: 100 | Batch Size: 32`。

## 计时与显存

| 项目 | 结果 |
| --- | ---: |
| 墙钟总耗时 | 14.289 s |
| 墙钟平均单样本耗时 | 0.142890 s |
| 程序内部评测耗时 | 6.96 s |
| 程序内部平均单样本耗时 | 0.069600 s |
| 运行前 GPU 显存 | 12 MiB |
| `nvidia-smi` 采样峰值 | 331 MiB |
| 峰值相对基线增量 | 319 MiB |
| GPU 采样有效点 | 25/25 |
| 退出码 | 0 |

## 指标

| 阶段 | No-Aug objective | AUG objective | AUG gap | Solution-level Infsb | Instance-level Infsb |
| --- | ---: | ---: | ---: | ---: | ---: |
| Construction | 25.8195 | 25.8209 | 0.5381% | 18.000% | 8.000% |
| Improvement, 20 步 | 25.7680 | 25.7621 | 0.0292% | 0.125% | 0.000% |

## 10,000 样本启动判断

按 100 样本墙钟时间线性外推：`14.289 / 100 × 10,000 = 1,428.9 s`，即约 **23 分 48.9 秒**。该估计低于用户规定的 2 小时阈值。采样峰值 331 MiB 仅占 8,151 MiB 总显存约 4.06%，stderr 为空且运行无异常，因此满足直接启动 10,000 样本评测的三个条件。

第一次相同模型配置的 100 样本运行也以退出码 0 完成且指标完全相同，但辅助 GPU 采样器因 PowerShell 的 `$LASTEXITCODE` 未设置而记录为 `NA`。只修正采样器的返回值判断后，以同一模型配置重新运行，得到本文件采用的 14.289 秒和 331 MiB。该修正没有接触模型、数据、算法或指标代码。
