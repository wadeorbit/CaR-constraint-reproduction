"""Read-only strict checkpoint compatibility check for the smoke-test model."""

import argparse
import ast
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import test
from models.SINGLEModel import SINGLEModel


def parser_from_source(source_path: Path) -> argparse.ArgumentParser:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    main_block = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    parser_statements = []
    for statement in main_block.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "args"
            for target in statement.targets
        ):
            break
        parser_statements.append(statement)
    module = ast.fix_missing_locations(ast.Module(body=parser_statements, type_ignores=[]))
    namespace = {"argparse": argparse, "str2bool": test.str2bool}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["parser"]


parser = parser_from_source(Path("test.py"))
args = parser.parse_args(
    [
        "--problem", "TSPTW",
        "--problem_size", "50",
        "--hardness", "hard",
        "--checkpoint", "pretrained/TSPTW/CaR-POMO_50_hard/checkpoint.pt",
        "--test_episodes", "2",
        "--test_batch_size", "2",
        "--test_pomo_size", "1",
        "--validation_improve_steps", "1",
        "--pomo_size", "50",
        "--pomo_start", "false",
        "--soft_constrained", "true",
        "--eval_type", "softmax",
        "--sample_size", "1",
        "--seed", "2023",
        "--gpu_id", "0",
    ]
)
assert args.disable_preset_args is True
_, model_params, _, _, _ = test.args2dict(args)
model = SINGLEModel(**model_params)
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]
model.load_state_dict(state_dict, strict=True)
print(f"strict_load_ok=True")
print(f"checkpoint_weight_keys={len(state_dict)}")
print(f"model_weight_keys={len(model.state_dict())}")
print(f"checkpoint_problem={checkpoint.get('problem')}")
