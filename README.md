# CaR 约束路由求解器：独立复现记录

> [!IMPORTANT]
> 本仓库是论文 **Towards Efficient Constraint Handling in Neural Solvers for Routing Problems** 的独立复现记录，**不是论文作者的官方仓库，也不代表原作者立场**。模型、数据、算法代码和论文结论的原始来源均属于作者团队；本仓库主要补充可追溯的本机评测日志、配置、校验信息和结果对比。

## 论文与官方资源

- ICLR 2026 论文页：[Towards Efficient Constraint Handling in Neural Solvers for Routing Problems](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1413fbda783a5bad7947c2f7396ba64b-Abstract-Conference.html)
- 正式论文 PDF：[ICLR 2026 Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/1413fbda783a5bad7947c2f7396ba64b-Paper-Conference.pdf)
- arXiv：[arXiv:2602.16012](https://arxiv.org/abs/2602.16012)
- OpenReview：[raDFGuQxvD](https://openreview.net/forum?id=raDFGuQxvD)
- 作者官方代码：[jieyibi/CaR-constraint](https://github.com/jieyibi/CaR-constraint)

作者提供的标准引用：

```bibtex
@inproceedings{
  bi2026towards,
  title={Towards Efficient Constraint Handling in Neural Solvers for Routing Problems},
  author={Bi, Jieyi and Cao, Zhiguang and Zhou, Jianan and Song, Wen and Wu, Yaoxin and Zhang, Jie and Ma, Yining and Wu, Cathy},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## 当前复现范围

本阶段已经完成：

- 问题：TSPTW（Traveling Salesman Problem with Time Windows）
- 规模与难度：TSPTW-50 Hard
- 模型：作者提供的 CaR-POMO 预训练 checkpoint
- 数据：作者提供的 `tsptw50_hard.pkl`
- Refinement：5、10、20 步
- 每组测试实例：10,000
- 数据增强：8 倍
- 单 POMO 起点，`eval_type=softmax`
- `seed=2023`，GPU 0

三组正式评测的共同参数如下。训练侧的 `improve_steps=5` 保持 checkpoint 官方配置；实验只改变评测侧的 `validation_improve_steps`。

| 参数 | 实际值 |
| --- | --- |
| `test_episodes` | 10,000 |
| `test_batch_size` | 32 |
| `augmentation_enable` / `aug_factor` | `true` / 8 |
| `test_pomo_size` / `pomo_start` | 1 / `false` |
| `soft_constrained` / `eval_type` | `true` / `softmax` |
| `sample_size` / `seed` / `gpu_id` | 1 / 2023 / 0 |
| `improve_steps` | 5（固定） |
| `validation_improve_steps` | 5、10、20（分别评测） |

> [!WARNING]
> **目前尚未完成整篇论文的全部实验。** 当前结论只适用于 TSPTW-50 Hard、CaR-POMO 和上述三组 refinement 配置，不能外推到论文中的其他问题、规模、模型或消融实验。

## 论文结果与本机结果

论文数据来自 Table 2。本机 Objective 使用最终 Improvement AUG Objective；Infsb 使用与论文一致的 **instance-level** 不可行率。完整误差计算见 [comparison.md](reproduction_runs/TSPTW50_hard_CaR_POMO/comparison.md)。

| Refinement | Objective 论文 | Objective 本机 | Gap 论文 | Gap 本机 | Instance Infsb 论文 | Instance Infsb 本机 | 论文时间（RTX 4090） | 本机 evaluation | 本机 wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 25.619 | 25.6188 | 0.034% | 0.0335% | 0.02% | 0.020% | 15 s | 222.48 s | 247.666 s |
| 10 | 25.615 | 25.6154 | 0.020% | 0.0195% | 0.01% | 0.010% | 27 s | 329.83 s | 337.866 s |
| 20 | 25.614 | 25.6142 | 0.014% | 0.0151% | 0.01% | 0.000% | 51 s | 491.02 s | 498.447 s |

精度对比：

- 三个本机 Objective 按论文的三位小数显示分别为 **25.619、25.615、25.614**，均与论文一致。
- Gap 的绝对差分别为 0.0005、0.0005 和 0.0011 个百分点。
- 5 步和 10 步的实例级不可行率与论文显示值一致。
- 20 步本机为 0.000%，论文为 0.01%，对应 10,000 个实例中约 1 个失败实例的差异。

因此，在论文采用的显示精度与聚合口径下，**Objective 按三位小数一致，Gap 和实例级不可行率基本一致**。

原始结果与日志：

- [results.csv](reproduction_runs/TSPTW50_hard_CaR_POMO/results.csv)
- [5/10/20 步对比 CSV](reproduction_runs/TSPTW50_hard_CaR_POMO/refinement_comparison.csv)
- [5 步完整日志](reproduction_runs/TSPTW50_hard_CaR_POMO/refinement_5/refinement_5.log)
- [10 步完整日志](reproduction_runs/TSPTW50_hard_CaR_POMO/refinement_10/refinement_10.log)
- [20 步完整日志](reproduction_runs/TSPTW50_hard_CaR_POMO/full_evaluation.log)

## 运行时间说明

论文时间在 NVIDIA RTX 4090 上报告，本机使用 NVIDIA GeForce RTX 5060 Laptop GPU。两类 GPU 的算力、显存带宽、功耗限制和散热条件不同，本机还使用了 Windows、batch size 32 及当前软件栈，因此时间不能视为同硬件性能复现。

| Refinement | 论文 RTX 4090 | RTX 5060 Laptop 本机 | 本机/论文 |
| ---: | ---: | ---: | ---: |
| 5 | 15 s | 222.48 s | 14.832× |
| 10 | 27 s | 329.83 s | 12.216× |
| 20 | 51 s | 491.02 s | 9.628× |

本机三次运行的显存采样峰值分别为 374、428 和 331 MiB，均有充足余量。时间差异主要反映硬件和运行环境差异，不改变上面的精度复现结论。

## 本机环境

| 项目 | 版本或配置 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.12.14 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute capability | 12.0 / sm_120 |
| NVIDIA driver | 591.91 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| scikit-learn | 1.5.2 |
| pandas | 2.2.3 |

完整依赖、硬件记录和文件哈希见 [environment.txt](reproduction_runs/TSPTW50_hard_CaR_POMO/environment.txt)。其中的 Windows 绝对路径仅是本机实验记录；可移植命令通过 `-PythonExe` 显式传入解释器。

关键输入哈希：

```text
tsptw50_hard.pkl:
E8FB50D07692785D0E62A5F80BEBCD602C01051E81640718BAE783C395F33882

CaR-POMO_50_hard/checkpoint.pt:
A85BE08B0CDC2C7EAE08DB838D4F295B7A3F493773293DB7163659F1AA5B4A80
```

## 目录结构

```text
.
├── data/                         # 上游作者仓库提供的数据
├── pretrained/                   # 上游作者仓库提供的预训练模型
├── envs/                         # 路由问题环境
├── models/                       # 神经求解器模型
├── test.py                       # 官方评测入口
├── Trainer.py                    # 官方训练与评测逻辑
└── reproduction_runs/
    └── TSPTW50_hard_CaR_POMO/
        ├── audit.md
        ├── commands.md
        ├── environment.txt
        ├── results.csv
        ├── comparison.md
        ├── refinement_comparison.csv
        ├── parsed_evaluation_summary.json
        ├── full_evaluation.log
        ├── full_evaluation.gpu_samples.csv
        ├── refinement_5/
        │   ├── refinement_5.log
        │   ├── refinement_5.gpu_samples.csv
        │   ├── summary.md
        │   └── integrity_*.txt
        ├── refinement_10/
        │   ├── refinement_10.log
        │   ├── refinement_10.gpu_samples.csv
        │   ├── summary.md
        │   └── integrity_*.txt
        ├── run_refinement_evaluation.ps1
        ├── run_smoke.ps1
        └── summarize_evaluation.py
```

## 复现命令

以下命令从仓库根目录运行。先把解释器设置为目标环境中的 Python；本机实验使用的具体路径记录在 `environment.txt`。

```powershell
$PythonExe = '<path-to-python>'
```

### 1. 只检查配置，不启动模型

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' `
  -ValidationImproveSteps 5 `
  -RunLabel 'dryrun_refinement_5' `
  -PythonExe $PythonExe `
  -DryRun

& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' `
  -ValidationImproveSteps 10 `
  -RunLabel 'dryrun_refinement_10' `
  -PythonExe $PythonExe `
  -DryRun
```

`DryRun` 会打印正式参数、输入文件、哈希和计划输出目录，不会启动 `test.py`，也不会创建结果目录。

### 2. 运行 5 步或 10 步独立重跑

必须使用新的 `RunLabel` 或 `OutputDir`。脚本默认写入 `reproduction_runs/TSPTW50_hard_CaR_POMO/reruns/<RunLabel>/`，目标目录存在时会拒绝覆盖。

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' `
  -ValidationImproveSteps 5 `
  -RunLabel 'refinement_5_rerun_YYYYMMDD_HHMMSS' `
  -PythonExe $PythonExe

& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_refinement_evaluation.ps1' `
  -ValidationImproveSteps 10 `
  -RunLabel 'refinement_10_rerun_YYYYMMDD_HHMMSS' `
  -PythonExe $PythonExe
```

包装脚本会把命令与 20 步正式基线逐项比较，确保除了 `validation_improve_steps` 外，其余正式配置不变。

### 3. 20 步重跑

20 步正式基线已经提交。重跑必须提供新的 `LogPrefix`，不能使用 `full_evaluation`，避免覆盖正式基线：

```powershell
& '.\reproduction_runs\TSPTW50_hard_CaR_POMO\run_smoke.ps1' `
  -TestEpisodes 10000 `
  -TestBatchSize 32 `
  -ValidationImproveSteps 20 `
  -LogPrefix 'full_evaluation_rerun_YYYYMMDD_HHMMSS'
```

`run_smoke.ps1` 是原始 20 步包装器，当前仍使用 `environment.txt` 中记录的本机 Python 路径，没有 `-PythonExe` 参数。跨机器重跑时需要先确认脚本中的解释器路径指向等价环境；5/10 步包装器已经通过 `-PythonExe` 参数化。

### 4. 从日志生成独立汇总

```powershell
& $PythonExe 'reproduction_runs\TSPTW50_hard_CaR_POMO\summarize_evaluation.py' `
  --input-dir 'reproduction_runs\TSPTW50_hard_CaR_POMO' `
  --output-dir 'reproduction_runs\TSPTW50_hard_CaR_POMO\reruns\summary_YYYYMMDD_HHMMSS'
```

汇总脚本优先读取原始 stdout/stderr；如果这些重复文件不存在，会从已提交的组合日志读取对应区段。GPU CSV 缺失时会给出警告，并继续使用组合日志中的显存峰值。

所有实际执行过的命令见 [commands.md](reproduction_runs/TSPTW50_hard_CaR_POMO/commands.md)。

## 可追溯性与仓库卫生

- 5、10 步运行前后的数据、checkpoint、20 步基线日志和核心源码哈希一致。
- 正式日志均正常退出，没有 Traceback、CUDA OOM 或 NaN。
- Python 缓存、根目录 `results/`，以及 refinement 新重跑产生的重复 stdout/stderr、临时命令文件和派生 `result.json` 已由 `.gitignore` 排除。
- 当前复现提交没有新增或修改 `data/`、`pretrained/`、核心算法源码或模型权重。
- 凭据扫描未发现 API key、访问令牌、密码、私钥、用户主目录或账号信息。

需要说明：本仓库基于作者官方代码树，所以上游本身已经跟踪公开的 `data/` 数据和 `pretrained/` checkpoint；ICLR 论文页也将这些资源列为官方发布内容。本复现工作没有再次生成、替换或新增这些资源。

## 后续计划

- 复现 TSPTW-100 Hard 的 CaR-POMO 结果。
- 复现 TSPTW 的 CaR-PIP 结果。
- 扩展到 TSPDL、CVRP、VRPBLTW 和 SOP。
- 在资源允许时补充论文中的更多规模、难度、基线和消融实验。

这些项目均尚未完成；完成后会继续保留配置、日志、哈希和论文误差对比。

## 致谢与许可证

感谢论文作者公开代码、数据和预训练模型。本仓库继承上游项目的 [MIT License](LICENSE)；使用官方代码和资源时，请同时引用原论文并遵守上游许可证。
