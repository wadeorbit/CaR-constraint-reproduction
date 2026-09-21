"""Validate completed CaR-POMO logs and update the traceable results table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from pathlib import Path


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = "pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt"
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STDOUT_MARKER = "----- STDOUT -----"
STDERR_MARKER = "----- STDERR -----"

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


def extract_combined_streams(text: str, log_path: Path) -> tuple[str, str]:
    """Extract the captured stdout and stderr sections from a combined log."""
    if text.count(STDOUT_MARKER) != 1 or text.count(STDERR_MARKER) != 1:
        raise AssertionError(
            f"Expected exactly one stdout/stderr section in {log_path}"
        )
    _, stdout_and_stderr = text.split(STDOUT_MARKER, 1)
    stdout, stderr = stdout_and_stderr.split(STDERR_MARKER, 1)
    return stdout.lstrip("\r\n"), stderr.lstrip("\r\n")


def read_captured_streams(
    stdout_path: Path,
    stderr_path: Path,
    combined_text: str,
    log_path: Path,
) -> tuple[str, str]:
    """Prefer raw streams and fall back to the committed combined log."""
    missing = [
        path.name for path in (stdout_path, stderr_path) if not path.is_file()
    ]
    fallback_stdout = fallback_stderr = ""
    if missing:
        fallback_stdout, fallback_stderr = extract_combined_streams(
            combined_text, log_path
        )
        warnings.warn(
            f"Missing raw stream artifact(s) {', '.join(missing)}; "
            f"using captured sections from {log_path}",
            RuntimeWarning,
            stacklevel=2,
        )
    stdout = (
        stdout_path.read_text(encoding="utf-8-sig")
        if stdout_path.is_file()
        else fallback_stdout
    )
    stderr = (
        stderr_path.read_text(encoding="utf-8-sig")
        if stderr_path.is_file()
        else fallback_stderr
    )
    return stdout, stderr


def read_gpu_sample_count(gpu_path: Path, expected_peak: int, log_path: Path) -> int | None:
    """Validate an available GPU CSV; retain log metadata when it is absent."""
    if not gpu_path.is_file():
        warnings.warn(
            f"Missing GPU sample CSV {gpu_path}; using GPU baseline/peak from "
            f"{log_path} and leaving gpu_sample_count unavailable",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    with gpu_path.open("r", encoding="utf-8-sig", newline="") as handle:
        gpu_rows = list(csv.DictReader(handle))
    if not gpu_rows or any(row.get("gpu_memory_used_mib") == "NA" for row in gpu_rows):
        raise AssertionError(f"Missing GPU samples in {gpu_path.name}")
    try:
        gpu_values = [int(row["gpu_memory_used_mib"]) for row in gpu_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError(f"Invalid GPU sample CSV: {gpu_path}") from exc
    if max(gpu_values) != expected_peak:
        raise AssertionError(
            f"GPU peak mismatch: log={expected_peak}, csv={max(gpu_values)}"
        )
    return len(gpu_values)


def validate_integrity_snapshots(artifact_dir: Path) -> None:
    """Validate optional before/after snapshots when either one is available."""
    before_path = artifact_dir / "integrity_before.txt"
    after_path = artifact_dir / "integrity_after.txt"
    if not before_path.exists() and not after_path.exists():
        return
    for path in (before_path, after_path):
        if not path.is_file():
            raise AssertionError(f"Missing integrity snapshot: {path}")
    hash_pattern = re.compile(r"^[0-9A-F]{64}  .+  \d+$")
    before_hashes = sorted(
        line for line in before_path.read_text(encoding="utf-8-sig").splitlines()
        if hash_pattern.fullmatch(line)
    )
    after_hashes = sorted(
        line for line in after_path.read_text(encoding="utf-8-sig").splitlines()
        if hash_pattern.fullmatch(line)
    )
    if not before_hashes or before_hashes != after_hashes:
        raise AssertionError(f"Integrity snapshot mismatch in {artifact_dir.name}")


def parse_run(
    input_root: Path,
    prefix: str,
    phase: str,
    expected_episodes: int,
    expected_steps: int = 20,
    relative_dir: str | None = None,
) -> dict[str, object]:
    artifact_dir = input_root if relative_dir is None else input_root / relative_dir
    log_path = artifact_dir / f"{prefix}.log"
    stdout_path = artifact_dir / f"{prefix}.stdout.log"
    stderr_path = artifact_dir / f"{prefix}.stderr.log"
    gpu_path = artifact_dir / f"{prefix}.gpu_samples.csv"
    if not log_path.is_file():
        raise AssertionError(f"Missing combined log: {log_path}")

    text = log_path.read_text(encoding="utf-8-sig")
    stdout, stderr = read_captured_streams(
        stdout_path, stderr_path, text, log_path
    )
    if stderr.strip():
        raise AssertionError(f"Captured stderr for {phase} is not empty")
    for marker in ("Traceback", "CUDA out of memory"):
        if marker in stdout or marker in stderr:
            raise AssertionError(f"Failure marker in {phase}: {marker}")
    if re.search(r"(?i)(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", stdout + stderr):
        raise AssertionError(f"NaN/Inf token in {phase}")

    python_exe = value(r"^Python executable:\s*(.+?)\s*$", text, "Python executable")
    executable_name = re.split(r"[\\/]", python_exe)[-1]
    if not re.fullmatch(r"(?i)python(?:\d+(?:\.\d+)*)?(?:\.exe)?", executable_name):
        raise AssertionError(f"Unexpected Python executable: {python_exe}")
    args = value(r"^Arguments:\s*(.+?)\s*$", text, "argument line")
    expected_args = {
        "--problem": "TSPTW", "--problem_size": "50", "--hardness": "hard",
        "--checkpoint": CHECKPOINT, "--test_episodes": str(expected_episodes),
        "--test_batch_size": "32", "--test_pomo_size": "1",
        "--improve_steps": "5", "--validation_improve_steps": str(expected_steps),
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
        rf"^\s*Validation Improve Steps:\s+{expected_steps}\s*$",
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
        value(rf"^>> Evaluation finished within\s+({NUM})s\s*$", stdout, "evaluation time")
    )
    exit_code = int(value(r"^Exit code:\s*(-?\d+)\s*$", text, "exit code"))
    if exit_code != 0:
        raise AssertionError(f"Nonzero exit code: {exit_code}")
    baseline = int(value(r"^GPU baseline MiB:\s*(\d+)\s*$", text, "GPU baseline"))
    peak = int(value(r"^GPU sampled peak MiB:\s*(\d+)\s*$", text, "GPU peak"))
    if not 0 <= baseline <= peak:
        raise AssertionError(f"Unsafe or invalid GPU memory values: {baseline}, {peak}")
    gpu_sample_count = read_gpu_sample_count(gpu_path, peak, log_path)

    if relative_dir is not None:
        validate_integrity_snapshots(artifact_dir)

    construction = parse_metric(stdout, "Construction")
    improvement = parse_metric(stdout, "Improvement")
    return {
        "phase": phase, "problem": "TSPTW", "size": 50, "hardness": "hard",
        "checkpoint": CHECKPOINT, "seed": 2023, "episodes": expected_episodes,
        "batch_size": 32, "augmentation": 8, "sample_size": 1,
        "test_pomo_size": 1, "validation_improve_steps": expected_steps,
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
        "gpu_sample_count": gpu_sample_count,
    }


def update_results(
    input_root: Path,
    output_path: Path,
    new_rows: list[dict[str, object]],
) -> None:
    input_path = input_root / "results.csv"
    source_path = output_path if output_path.is_file() else input_path
    if source_path.is_file():
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    else:
        existing = []
    for row in existing:
        for key, raw in row.items():
            if raw in {"-0", "-0.0", "-0.00", "-0.000", "-0.0000"}:
                row[key] = raw[1:]
    by_phase = {row["phase"]: row for row in existing}
    for row in new_rows:
        by_phase[str(row["phase"])] = {field: row.get(field, "") for field in FIELDS}
    available_phases = list(by_phase)
    preferred_order = [
        "smoke", "calibration_20step", "sampler_failure_probe", "benchmark_100",
        "refinement_5", "refinement_10", "full_evaluation",
    ]
    ordered_phases = [phase for phase in preferred_order if phase in by_phase]
    ordered_phases += [phase for phase in available_phases if phase not in ordered_phases]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for phase in ordered_phases:
            writer.writerow({field: by_phase[phase].get(field, "") for field in FIELDS})


def build_comparisons(
    runs_by_steps: dict[int, dict[str, object]],
    paper_by_steps: dict[int, dict[str, float]],
) -> tuple[
    dict[int, dict[str, dict[str, float]]],
    dict[int, dict[str, object]],
]:
    comparisons_by_steps: dict[int, dict[str, dict[str, float]]] = {}
    diagnostics_by_steps: dict[int, dict[str, object]] = {}
    for steps, run in runs_by_steps.items():
        paper = paper_by_steps[steps]
        local = {
            "objective": run["improvement_aug_score"],
            "gap_pct": run["improvement_aug_gap_pct"],
            "instance_infeasible_pct": run["improvement_instance_infeasible_pct"],
            "evaluation_seconds": run["evaluation_seconds"],
        }
        comparisons: dict[str, dict[str, float]] = {}
        for metric in (
            "objective", "gap_pct", "instance_infeasible_pct", "evaluation_seconds",
        ):
            signed = round(float(local[metric]) - paper[metric], 12)
            absolute = round(abs(signed), 12)
            comparisons[metric] = {
                "signed_difference": signed,
                "absolute_difference": absolute,
                "relative_error_pct": round(
                    absolute / abs(paper[metric]) * 100, 12
                ),
            }
        comparisons_by_steps[steps] = comparisons
        diagnostics_by_steps[steps] = {
            "local_solution_infeasible_pct": run[
                "improvement_solution_infeasible_pct"
            ],
            "solution_infeasible_comparable_to_paper_infsb": False,
            "paper_infsb_aggregation": "instance-level best solution",
        }
    return comparisons_by_steps, diagnostics_by_steps


def write_refinement_comparison(
    output_path: Path,
    runs_by_steps: dict[int, dict[str, object]],
    paper_by_steps: dict[int, dict[str, float]],
    comparisons_by_steps: dict[int, dict[str, dict[str, float]]],
) -> None:
    fields = [
        "validation_improve_steps",
        "paper_objective", "local_objective", "objective_absolute_difference",
        "objective_relative_error_pct",
        "paper_gap_pct", "local_gap_pct", "gap_absolute_difference_pp",
        "gap_relative_error_pct",
        "paper_instance_infeasible_pct", "local_instance_infeasible_pct",
        "infeasible_absolute_difference_pp", "infeasible_relative_error_pct",
        "paper_time_seconds", "local_evaluation_seconds",
        "time_absolute_difference_seconds", "time_relative_error_pct",
        "local_wall_seconds", "local_sampled_gpu_peak_mib", "exit_code",
    ]
    rows: list[dict[str, object]] = []
    for steps in (5, 10, 20):
        run = runs_by_steps[steps]
        paper = paper_by_steps[steps]
        comparison = comparisons_by_steps[steps]
        rows.append({
            "validation_improve_steps": steps,
            "paper_objective": paper["objective"],
            "local_objective": run["improvement_aug_score"],
            "objective_absolute_difference": comparison["objective"][
                "absolute_difference"
            ],
            "objective_relative_error_pct": comparison["objective"][
                "relative_error_pct"
            ],
            "paper_gap_pct": paper["gap_pct"],
            "local_gap_pct": run["improvement_aug_gap_pct"],
            "gap_absolute_difference_pp": comparison["gap_pct"][
                "absolute_difference"
            ],
            "gap_relative_error_pct": comparison["gap_pct"]["relative_error_pct"],
            "paper_instance_infeasible_pct": paper["instance_infeasible_pct"],
            "local_instance_infeasible_pct": run[
                "improvement_instance_infeasible_pct"
            ],
            "infeasible_absolute_difference_pp": comparison[
                "instance_infeasible_pct"
            ]["absolute_difference"],
            "infeasible_relative_error_pct": comparison[
                "instance_infeasible_pct"
            ]["relative_error_pct"],
            "paper_time_seconds": paper["evaluation_seconds"],
            "local_evaluation_seconds": run["evaluation_seconds"],
            "time_absolute_difference_seconds": comparison["evaluation_seconds"][
                "absolute_difference"
            ],
            "time_relative_error_pct": comparison["evaluation_seconds"][
                "relative_error_pct"
            ],
            "local_wall_seconds": run["wall_seconds"],
            "local_sampled_gpu_peak_mib": run["sampled_gpu_peak_mib"],
            "exit_code": run["exit_code"],
        })
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the recorded TSPTW50-hard evaluations."
    )
    parser.add_argument(
        "--input-root", "--input-dir", "--run-dir",
        dest="input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Directory containing full_evaluation.log, benchmark artifacts, and "
            "the refinement_5/refinement_10 directories "
            f"(default: {DEFAULT_INPUT_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV/JSON files (default: input root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    input_root = args.input_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_root
    )
    if not input_root.is_dir():
        raise AssertionError(f"Input root is not a directory: {input_root}")

    benchmark = parse_run(
        input_root, "benchmark_100_memory", "benchmark_100", 100
    )
    refinement_5 = parse_run(
        input_root,
        "refinement_5", "refinement_5", 10_000,
        expected_steps=5,
        relative_dir="refinement_5",
    )
    refinement_10 = parse_run(
        input_root,
        "refinement_10", "refinement_10", 10_000,
        expected_steps=10,
        relative_dir="refinement_10",
    )
    full = parse_run(
        input_root, "full_evaluation", "full_evaluation", 10_000
    )

    paper_by_steps = {
        5: {
            "objective": 25.619, "gap_pct": 0.034,
            "instance_infeasible_pct": 0.02, "evaluation_seconds": 15.0,
        },
        10: {
            "objective": 25.615, "gap_pct": 0.020,
            "instance_infeasible_pct": 0.01, "evaluation_seconds": 27.0,
        },
        20: {
            "objective": 25.614, "gap_pct": 0.014,
            "instance_infeasible_pct": 0.01, "evaluation_seconds": 51.0,
        },
    }
    runs_by_steps = {5: refinement_5, 10: refinement_10, 20: full}
    comparisons_by_steps, diagnostics_by_steps = build_comparisons(
        runs_by_steps, paper_by_steps
    )
    summary = {
        "validation": "passed",
        "benchmark_100": benchmark,
        "refinement_5": refinement_5,
        "refinement_10": refinement_10,
        "full_evaluation": full,
        "paper_reference_by_steps": paper_by_steps,
        "comparisons_by_steps": comparisons_by_steps,
        "diagnostics_by_steps": diagnostics_by_steps,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    update_results(
        input_root, output_dir / "results.csv",
        [benchmark, refinement_5, refinement_10, full],
    )
    write_refinement_comparison(
        output_dir / "refinement_comparison.csv",
        runs_by_steps,
        paper_by_steps,
        comparisons_by_steps,
    )
    (output_dir / "parsed_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for steps in (5, 10):
        result_document = {
            "validation": "passed",
            "run": runs_by_steps[steps],
            "paper_reference": paper_by_steps[steps],
            "comparison": comparisons_by_steps[steps],
            "diagnostics": diagnostics_by_steps[steps],
        }
        result_dir = output_dir / f"refinement_{steps}"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(
            json.dumps(result_document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
