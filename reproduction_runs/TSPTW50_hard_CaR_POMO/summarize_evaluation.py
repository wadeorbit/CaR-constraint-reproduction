"""Validate completed CaR-POMO logs and update the traceable results table."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
RESULTS_PATH = RUN_DIR / "results.csv"
SUMMARY_PATH = RUN_DIR / "parsed_evaluation_summary.json"
CHECKPOINT = "pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt"
PYTHON_EXE = r"D:\Miniconda3\envs\car\python.exe"
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

BASE_FIELDS = [
    "phase", "problem", "size", "hardness", "checkpoint", "seed",
    "episodes", "batch_size", "augmentation", "sample_size",
    "test_pomo_size", "validation_improve_steps",
    "construction_no_aug_score", "construction_aug_score",
    "construction_aug_gap_pct", "construction_solution_infeasible_pct",
    "construction_instance_infeasible_pct", "improvement_no_aug_score",
    "improvement_aug_score", "improvement_aug_gap_pct",
    "improvement_solution_infeasible_pct",
    "improvement_instance_infeasible_pct", "evaluation_seconds",
    "wall_seconds", "sampled_gpu_peak_mib", "exit_code",
]
EXTRA_FIELDS = [
    "evaluation_seconds_per_instance", "wall_seconds_per_instance",
    "gpu_baseline_mib", "sampled_gpu_peak_delta_mib",
]
FIELDS = BASE_FIELDS + EXTRA_FIELDS


def one(pattern: str, text: str, label: str, flags: int = re.MULTILINE) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags))
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one {label}; found {len(matches)}")
    return matches[0]


def value(pattern: str, text: str, label: str) -> str:
    return one(pattern, text, label).group(1)


def clean_float(number: str) -> float:
    parsed = float(number)
    if not math.isfinite(parsed):
        raise AssertionError(f"Non-finite value: {number}")
    return 0.0 if parsed == 0.0 else parsed


def parse_metric(text: str, section: str) -> dict[str, float]:
    pattern = (
        rf"^Rank 0 >> Val Score on tsptw50_hard\.pkl: \[{section}\] "
        rf"NO_AUG_Score: (?P<no>{NUM}), NO_AUG_Gap: (?P<nogap>{NUM})% --> "
        rf"AUG_Score: (?P<aug>{NUM}), AUG_Gap: (?P<gap>{NUM})%; "
        rf"Infeasible rate: (?P<sol>{NUM})% \(solution-level\), "
        rf"(?P<ins>{NUM})% \(instance-level\)\s*$"
    )
    match = one(pattern, text, f"{section} metric")
    return {key: clean_float(raw) for key, raw in match.groupdict().items()}


def parse_run(prefix: str, phase: str, expected_episodes: int) -> dict[str, object]:
    log_path = RUN_DIR / f"{prefix}.log"
    stdout_path = RUN_DIR / f"{prefix}.stdout.log"
    stderr_path = RUN_DIR / f"{prefix}.stderr.log"
    gpu_path = RUN_DIR / f"{prefix}.gpu_samples.csv"
    for path in (log_path, stdout_path, stderr_path, gpu_path):
        if not path.is_file():
            raise AssertionError(f"Missing artifact: {path}")

    text = log_path.read_text(encoding="utf-8-sig")
    stdout = stdout_path.read_text(encoding="utf-8-sig")
    stderr = stderr_path.read_text(encoding="utf-8-sig")
    if stderr.strip():
        raise AssertionError(f"{stderr_path.name} is not empty")
    for marker in ("Traceback", "CUDA out of memory"):
        if marker in text:
            raise AssertionError(f"Failure marker in {log_path.name}: {marker}")
    if re.search(r"(?i)(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", text):
        raise AssertionError(f"NaN/Inf token in {log_path.name}")

    python_exe = value(r"^Python executable:\s*(.+?)\s*$", text, "Python executable")
    if python_exe != PYTHON_EXE:
        raise AssertionError(f"Unexpected Python executable: {python_exe}")
    args = value(r"^Arguments:\s*(.+?)\s*$", text, "argument line")
    expected_args = {
        "--problem": "TSPTW", "--problem_size": "50", "--hardness": "hard",
        "--checkpoint": CHECKPOINT, "--test_episodes": str(expected_episodes),
        "--test_batch_size": "32", "--test_pomo_size": "1",
        "--improve_steps": "5", "--validation_improve_steps": "20",
        "--pomo_size": "50", "--pomo_start": "false",
        "--soft_constrained": "true", "--eval_type": "softmax",
        "--sample_size": "1", "--seed": "2023", "--gpu_id": "0",
    }
    tokens = args.split()
    if tokens[:2] != ["-u", "test.py"]:
        raise AssertionError(f"Unexpected command prefix: {tokens[:2]}")
    parsed_args = dict(zip(tokens[2::2], tokens[3::2], strict=True))
    if parsed_args != expected_args:
        raise AssertionError(f"Effective argument mismatch: {parsed_args}")
    if "--disable_preset_args" in args:
        raise AssertionError("Reverse-logic --disable_preset_args must not be present")

    required_patterns = [
        r"^>> USE_CUDA: True, CUDA_DEVICE_NUM: 0\s*$",
        r"^\s*Problem:\s+TSPTW \(Size: 50, Hardness: hard\)\s*$",
        rf"^\s*Checkpoint:\s+{re.escape(CHECKPOINT)}\s*$",
        rf"^\s*Test Episodes:\s+{expected_episodes} \| Batch Size: 32\s*$",
        r"^\s*Validation Improve Steps:\s+20\s*$",
        r"^\s*Soft Constrained:\s+True\s*$",
        r"^\s*Improvement Method:\s+kopt \| N2S Decoder: False\s*$",
        r"^\s*POMO Start:\s+False \| Test POMO Size: 1\s*$",
        r"^\s*Unified Encoder:\s+True \| Unified Decoder: False\s*$",
        r"^\s*Eval Type:\s+softmax\s*$",
        r"^\s*Sample Size:\s+1\s*$",
        r"^>> TEST dataset tsptw50_hard\.pkl\s*$",
        rf"^>> TEST {expected_episodes} instances\.\s*$",
        rf"^episode\s+{expected_episodes}/{expected_episodes},.*$",
    ]
    for idx, pattern in enumerate(required_patterns):
        one(pattern, stdout, f"configuration gate {idx}")

    wall = clean_float(value(rf"^Elapsed seconds:\s*({NUM})\s*$", text, "wall time"))
    evaluation = clean_float(
        value(rf"^>> Evaluation finished within\s+({NUM})s\s*$", text, "evaluation time")
    )
    exit_code = int(value(r"^Exit code:\s*(-?\d+)\s*$", text, "exit code"))
    if exit_code != 0:
        raise AssertionError(f"Nonzero exit code: {exit_code}")
    baseline = int(value(r"^GPU baseline MiB:\s*(\d+)\s*$", text, "GPU baseline"))
    peak = int(value(r"^GPU sampled peak MiB:\s*(\d+)\s*$", text, "GPU peak"))
    if not 0 <= baseline <= peak < 8151:
        raise AssertionError(f"Unsafe or invalid GPU memory values: {baseline}, {peak}")

    with gpu_path.open("r", encoding="utf-8-sig", newline="") as handle:
        gpu_rows = list(csv.DictReader(handle))
    if not gpu_rows or any(row["gpu_memory_used_mib"] == "NA" for row in gpu_rows):
        raise AssertionError(f"Missing GPU samples in {gpu_path.name}")
    gpu_values = [int(row["gpu_memory_used_mib"]) for row in gpu_rows]
    if max(gpu_values) != peak:
        raise AssertionError(f"GPU peak mismatch: log={peak}, csv={max(gpu_values)}")

    construction = parse_metric(text, "Construction")
    improvement = parse_metric(text, "Improvement")
    return {
        "phase": phase, "problem": "TSPTW", "size": 50, "hardness": "hard",
        "checkpoint": CHECKPOINT, "seed": 2023, "episodes": expected_episodes,
        "batch_size": 32, "augmentation": 8, "sample_size": 1,
        "test_pomo_size": 1, "validation_improve_steps": 20,
        "construction_no_aug_score": construction["no"],
        "construction_aug_score": construction["aug"],
        "construction_aug_gap_pct": construction["gap"],
        "construction_solution_infeasible_pct": construction["sol"],
        "construction_instance_infeasible_pct": construction["ins"],
        "improvement_no_aug_score": improvement["no"],
        "improvement_aug_score": improvement["aug"],
        "improvement_aug_gap_pct": improvement["gap"],
        "improvement_solution_infeasible_pct": improvement["sol"],
        "improvement_instance_infeasible_pct": improvement["ins"],
        "evaluation_seconds": evaluation, "wall_seconds": wall,
        "sampled_gpu_peak_mib": peak, "exit_code": exit_code,
        "evaluation_seconds_per_instance": evaluation / expected_episodes,
        "wall_seconds_per_instance": wall / expected_episodes,
        "gpu_baseline_mib": baseline,
        "sampled_gpu_peak_delta_mib": peak - baseline,
        "gpu_sample_count": len(gpu_values),
    }


def update_results(new_rows: list[dict[str, object]]) -> None:
    with RESULTS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        existing = list(csv.DictReader(handle))
    for row in existing:
        for key, raw in row.items():
            if raw in {"-0", "-0.0", "-0.00", "-0.000", "-0.0000"}:
                row[key] = raw[1:]
    by_phase = {row["phase"]: row for row in existing}
    for row in new_rows:
        by_phase[str(row["phase"])] = {field: row.get(field, "") for field in FIELDS}
    ordered_phases = [row["phase"] for row in existing]
    ordered_phases += [str(row["phase"]) for row in new_rows if str(row["phase"]) not in ordered_phases]
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for phase in ordered_phases:
            writer.writerow({field: by_phase[phase].get(field, "") for field in FIELDS})


benchmark = parse_run("benchmark_100_memory", "benchmark_100", 100)
full = parse_run("full_evaluation", "full_evaluation", 10_000)
update_results([benchmark, full])

paper = {
    "objective": 25.614,
    "gap_pct": 0.014,
    "instance_infeasible_pct": 0.01,
    "evaluation_seconds": 51.0,
}
local = {
    "objective": full["improvement_aug_score"],
    "gap_pct": full["improvement_aug_gap_pct"],
    "instance_infeasible_pct": full["improvement_instance_infeasible_pct"],
    "evaluation_seconds": full["evaluation_seconds"],
}
comparisons = {}
for metric in ("objective", "gap_pct", "instance_infeasible_pct", "evaluation_seconds"):
    signed = float(local[metric]) - paper[metric]
    comparisons[metric] = {
        "signed_difference": signed,
        "absolute_difference": abs(signed),
        "relative_error_pct": abs(signed) / abs(paper[metric]) * 100,
    }

diagnostics = {
    "local_solution_infeasible_pct": full["improvement_solution_infeasible_pct"],
    "solution_infeasible_comparable_to_paper_infsb": False,
    "paper_infsb_aggregation": "instance-level best solution",
}

summary = {
    "validation": "passed",
    "benchmark_100": benchmark,
    "full_evaluation": full,
    "paper_reference": paper,
    "comparisons": comparisons,
    "diagnostics": diagnostics,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
