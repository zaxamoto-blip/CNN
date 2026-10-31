"""Repeat all nine trainings in a temporary directory and compare exact states.

Original reference outputs are never overwritten. This is stronger than merely
reloading saved checkpoints. It is expected to match exactly in the reference
CPU environment; different PyTorch/CPU versions need not be bitwise identical.
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import experiment as ex


def main():
    reference = ex.ROOT
    ex.setup()
    with tempfile.TemporaryDirectory(prefix="retraining_", dir=reference) as folder:
        ex.ROOT = Path(folder)
        for name in ("config.json", "PREREGISTRATION.md", "preregistration_record.json"):
            shutil.copy2(reference / name, ex.ROOT / name)
        for name in ("data", "results", "checkpoints", "figures"):
            (ex.ROOT / name).mkdir()
        normal, masks, labels, splits = ex.get_data()
        renderer = ex.Renderer()
        checks = []
        for seed in ex.CONFIG["seeds"]:
            for condition in ex.CONFIG["conditions"]:
                ex.train_one(condition, seed, normal, masks, labels, splits, renderer)
                filename = f"{condition}_seed{seed}.pt"
                ref = torch.load(reference / "checkpoints" / filename, weights_only=False, map_location="cpu")
                new = torch.load(ex.ROOT / "checkpoints" / filename, weights_only=False, map_location="cpu")
                errors = {k: float(torch.max(torch.abs(v-new["state_dict"][k]))) for k,v in ref["state_dict"].items()}
                assert all(v == 0 for v in errors.values()), (condition, seed, errors)
                checks.append({"condition": condition, "seed": seed, "max_parameter_difference": max(errors.values())})
                print("EXACT RETRAINING MATCH", condition, seed, flush=True)
        views = ex.evaluation_views(normal, masks, labels, splits["test"], renderer)
        ex.evaluate_all(views)
        assert json.loads((reference/"results/per_run_metrics.json").read_text()) == json.loads((ex.ROOT/"results/per_run_metrics.json").read_text())
        prediction_checks = 0
        for filename in (reference/"results").glob("predictions_*.npz"):
            a = np.load(filename)
            b = np.load(ex.ROOT/"results"/filename.name)
            assert all(np.array_equal(a[k],b[k]) for k in a.files), filename.name
            prediction_checks += 1
        result = {"checked_utc": ex.utc_now(), "full_retraining_runs": len(checks),
                  "all_weights_identical": True, "all_metrics_identical": True,
                  "all_logits_and_predictions_identical": True,
                  "prediction_files_compared": prediction_checks, "checks": checks,
                  "scope": "same pinned reference environment, fixed CPU threads and data split"}
        ex.save_json(reference/"results/full_retraining_check.json", result)
        print("PASS: all nine independent re-trainings exactly reproduce weights, logits, and metrics",flush=True)


if __name__ == "__main__":
    main()
