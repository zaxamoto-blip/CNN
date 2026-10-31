"""Controlled shape/texture study. Train first; inspect test predictions afterward.

CPU reference run: Python 3.12, PyTorch 2.6.0+cpu. No model/data download needed.
See PREREGISTRATION.md for hypotheses, controls, and limits of interpretation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup() -> None:
    torch.set_num_threads(CONFIG["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.deterministic = True
    for folder in ("data", "results", "figures", "checkpoints"):
        (ROOT / folder).mkdir(exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(24 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Renderer:
    def __init__(self):
        size = CONFIG["image_size"]
        yy, xx = np.mgrid[:size, :size].astype(np.float64)
        self.xx = (xx + 0.5) / size
        self.yy = (yy + 0.5) / size
        self.projections = []
        for freq in CONFIG["texture_cycles_per_image"]:
            for degrees in CONFIG["texture_angles_degrees"]:
                angle = np.deg2rad(degrees)
                self.projections.append(
                    2 * np.pi * freq * (self.xx * np.cos(angle) + self.yy * np.sin(angle))
                )
        self.projections = np.asarray(self.projections)
        self.disk = ((self.xx - 0.5) ** 2 + (self.yy - 0.5) ** 2) < 0.33 ** 2

    def textured(self, masks, texture_labels, phases):
        masks = np.asarray(masks, dtype=bool)
        result = np.zeros(masks.shape, dtype=np.float32)
        low, high = CONFIG["texture_range"]
        # The tie-break noise is independent of digit and texture label. It only
        # orders equal sine values; it does not change the prescribed histogram.
        tie = np.random.default_rng(63031).uniform(0, 1e-9, (32, 32))
        for i, (mask, texture, phase) in enumerate(zip(masks, texture_labels, phases)):
            count = int(mask.sum())
            assert count > 1
            wave = np.sin(self.projections[int(texture)] + float(phase)) + tie
            rank = np.argsort(np.argsort(wave[mask], kind="stable"), kind="stable")
            result[i][mask] = np.linspace(low, high, count, dtype=np.float32)[rank]
        return result


def get_data():
    data = load_digits()
    images = data.images.astype(np.float32) / 16.0
    labels = data.target.astype(np.int64)
    indices = np.arange(len(labels))
    trainval, test = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=CONFIG["split_seed"]
    )
    train, val = train_test_split(
        trainval, test_size=0.25, stratify=labels[trainval], random_state=CONFIG["split_seed"]
    )
    splits = {"train": train, "validation": val, "test": test}
    # Splitting happens before derivatives. Verify original indices and hashes.
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in data.images]
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(splits[a]) & set(splits[b])
        assert not {hashes[i] for i in splits[a]} & {hashes[i] for i in splits[b]}, "Duplicate leakage"
    normal = F.interpolate(
        torch.from_numpy(images[:, None]), size=(32, 32), mode="bilinear", align_corners=False
    ).numpy()[:, 0]
    masks = normal > CONFIG["mask_threshold"]
    split_info = {
        k: {"count": len(v), "class_counts": np.bincount(labels[v], minlength=10).tolist(),
            "indices": v.tolist()} for k, v in splits.items()
    }
    save_json(ROOT / "data/split.json", split_info)
    np.savez_compressed(ROOT / "data/source_digits.npz", images=data.images.astype(np.uint8), labels=labels)
    save_json(ROOT / "data/audit.json", {
        "total_original_images": len(images), "unique_original_hashes": len(set(hashes)),
        "cross_split_duplicate_count": 0, "split_before_transform": True,
        "writer_disjoint": False,
        "source_url": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        "loader_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "license": "CC BY 4.0", "authors": "E. Alpaydin and C. Kaynak (1998)",
        "doi": "10.24432/C50P49", "note": "1,797-image sklearn copy of UCI's original test subset; re-split for this study."
    })
    return normal, masks, labels, splits


def epoch_inputs(normal, masks, labels, condition, seed, epoch, renderer):
    rng = np.random.default_rng(seed * 100_000 + epoch)
    use_normal = rng.random(len(labels)) < CONFIG["normal_fraction"]
    random_texture = rng.integers(0, 10, len(labels))
    phases = rng.uniform(0, 2 * np.pi, len(labels))
    if condition == "plain":
        x = masks.astype(np.float32) * 0.6
    else:
        t = labels if condition == "correlated" else random_texture
        x = renderer.textured(masks, t, phases)
    x[use_normal] = normal[use_normal]
    return torch.from_numpy(x[:, None])


@torch.inference_mode()
def predict(model, images, return_logits=False):
    model.eval()
    x = torch.as_tensor(images, dtype=torch.float32)
    if x.ndim == 3:
        x = x[:, None]
    logits = torch.cat([model(x[i:i + 256]).cpu() for i in range(0, len(x), 256)]).numpy()
    return logits if return_logits else logits.argmax(axis=1)


def train_one(condition, seed, normal, masks, labels, splits, renderer):
    checkpoint = ROOT / f"checkpoints/{condition}_seed{seed}.pt"
    if checkpoint.exists():
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert stored["config"] == CONFIG, "Checkpoint/config mismatch"
        print(f"REUSE {condition} seed={seed}", flush=True)
        return stored["training_record"]
    seed_everything(seed)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    train, val = splits["train"], splits["validation"]
    targets = torch.from_numpy(labels[train])
    history = []
    started = utc_now()
    start_clock = time.perf_counter()
    initial_hash = hashlib.sha256(b"".join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        x = epoch_inputs(normal[train], masks[train], labels[train], condition, seed, epoch, renderer)
        order = np.random.default_rng(seed * 10_000 + epoch).permutation(len(train))
        total_loss = total_correct = 0.0
        for start in range(0, len(order), CONFIG["batch_size"]):
            batch = order[start:start + CONFIG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output = model(x[batch])
            loss = F.cross_entropy(output, targets[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_correct += (output.argmax(1) == targets[batch]).sum().item()
        if epoch == 1 or epoch % 5 == 0 or epoch == CONFIG["epochs"]:
            val_logits = predict(model, normal[val], return_logits=True)
            val_acc = float((val_logits.argmax(1) == labels[val]).mean())
            val_loss = float(F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(labels[val])))
            history.append({"epoch": epoch, "train_loss": total_loss / len(train),
                            "train_accuracy": total_correct / len(train),
                            "normal_validation_accuracy": val_acc, "normal_validation_loss": val_loss})
            print(f"{condition:10} seed={seed:2} epoch={epoch:2} train={total_correct/len(train):.3f} "
                  f"val_normal={val_acc:.3f} elapsed={time.perf_counter()-start_clock:.1f}s", flush=True)
    record = {"condition": condition, "seed": seed, "started_utc": started,
              "finished_utc": utc_now(), "seconds": time.perf_counter() - start_clock,
              "initial_state_sha256": initial_hash, "epochs": CONFIG["epochs"],
              "parameter_count": sum(p.numel() for p in model.parameters()), "history": history,
              "checkpoint_selection": "fixed_final_epoch; validation monitoring only"}
    torch.save({"state_dict": model.state_dict(), "config": CONFIG, "training_record": record}, checkpoint)
    save_json(ROOT / f"results/training_{condition}_seed{seed}.json", record)
    return record


def evaluation_views(normal, masks, labels, test, renderer):
    n = len(test)
    y = labels[test]
    rng = np.random.default_rng(CONFIG["evaluation_seed"])
    # One phase for each source-image/texture pair; identical in aligned/conflict.
    phases = rng.uniform(0, 2 * np.pi, (n, 10))
    all_t = np.tile(np.arange(10), n)
    all_images = renderer.textured(np.repeat(masks[test], 10, axis=0), all_t, phases.ravel())
    match = all_t == np.repeat(y, 10)
    conflict = ~match
    views = {
        "normal": (normal[test], y, np.full(n, -1)),
        "plain": (masks[test].astype(np.float32) * 0.6, y, np.full(n, -1)),
        "aligned": (all_images[match], y, y.copy()),
        "conflict": (all_images[conflict], np.repeat(y, 10)[conflict], all_t[conflict]),
        "texture_only_disk": (
            renderer.textured(np.repeat(renderer.disk[None], n * 10, axis=0), all_t, phases.ravel()),
            np.repeat(y, 10), all_t
        ),
    }
    # Exact histogram equality across every replacement texture for each shape.
    sorted_pixels = np.sort(all_images.reshape(n, 10, -1), axis=-1)
    histogram_error = float(np.abs(sorted_pixels - sorted_pixels[:, :1]).max())
    assert histogram_error == 0.0
    save_json(ROOT / "results/render_audit.json", {
        "paired_texture_max_sorted_pixel_difference": histogram_error,
        "identical_foreground_masks": True, "conflicts_per_original_test_image": 9,
        "test_original_count": n, "conflict_count": int(conflict.sum()),
        "normal_images_untextured_grayscale": True,
        "texture_id_definition": {str(k): {"cycles_per_image": CONFIG["texture_cycles_per_image"][k // 5],
          "angle_degrees": CONFIG["texture_angles_degrees"][k % 5]} for k in range(10)}
    })
    np.savez_compressed(ROOT / "data/test_views.npz", **{k: v[0] for k, v in views.items()},
                        digit_labels=y, test_indices=test, texture_ids=views["conflict"][2], phases=phases)
    return views


def evaluate_all(views):
    records = []
    raw = {}
    for condition in CONFIG["conditions"]:
        for seed in CONFIG["seeds"]:
            model = SmallCNN()
            stored = torch.load(ROOT / f"checkpoints/{condition}_seed{seed}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(stored["state_dict"])
            raw[(condition, seed)] = {}
            for name, (x, shape, texture) in views.items():
                logits = predict(model, x, return_logits=True)
                pred = logits.argmax(1)
                record = {"condition": condition, "seed": seed, "view": name, "n": len(pred),
                          "shape_accuracy": float(np.mean(pred == shape)),
                          "texture_agreement": float(np.mean(pred == texture)) if np.all(texture >= 0) else None}
                if name == "conflict":
                    shape_match = pred == shape
                    texture_match = pred == texture
                    s, t = int(shape_match.sum()), int(texture_match.sum())
                    record.update(shape_count=s, texture_count=t, other_count=len(pred)-s-t,
                                  other_rate=float(1 - (s + t) / len(pred)),
                                  coverage=float((s + t) / len(pred)),
                                  conditional_shape_bias=float(s / (s+t)) if s+t else None)
                records.append(record)
                raw[(condition, seed)][name] = {"pred": pred, "shape": shape, "texture": texture}
                np.savez_compressed(ROOT / f"results/predictions_{condition}_seed{seed}_{name}.npz",
                                    logits=logits, prediction=pred, shape_label=shape, texture_label=texture)
            print("EVALUATED", condition, seed, flush=True)
    save_json(ROOT / "results/per_run_metrics.json", records)
    columns = list(dict.fromkeys(k for row in records for k in row))
    with (ROOT / "results/per_run_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    for condition in CONFIG["conditions"]:
        for view in views:
            rows = [r for r in records if r["condition"] == condition and r["view"] == view]
            result = {"condition": condition, "view": view, "n": rows[0]["n"]}
            for key in ("shape_accuracy", "texture_agreement", "other_rate", "coverage", "conditional_shape_bias"):
                if key not in rows[0] or rows[0][key] is None:
                    continue
                values = np.asarray([r[key] for r in rows], dtype=float)
                result[key + "_mean"] = float(values.mean())
                result[key + "_sd"] = float(values.std(ddof=1))
            summaries.append(result)
    save_json(ROOT / "results/summary.json", summaries)
    # Pair all 9 conflicts belonging to the same original test image; first
    # average across trained seeds. Bootstrap original images, never variants.
    by_condition = {}
    for condition in ("correlated", "randomized"):
        sm = np.stack([raw[(condition, seed)]["conflict"]["pred"] == raw[(condition, seed)]["conflict"]["shape"]
                       for seed in CONFIG["seeds"]]).reshape(len(CONFIG["seeds"]), -1, 9)
        by_condition[condition] = sm.mean(axis=(0, 2))
    per_image_delta = by_condition["randomized"] - by_condition["correlated"]
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    bootstrap = np.asarray([per_image_delta[rng.integers(0, len(per_image_delta), len(per_image_delta))].mean()
                            for _ in range(CONFIG["bootstrap_replicates"])])
    seed_differences = []
    for seed in CONFIG["seeds"]:
        r = next(r for r in records if r["condition"] == "randomized" and r["seed"] == seed and r["view"] == "conflict")
        c = next(r for r in records if r["condition"] == "correlated" and r["seed"] == seed and r["view"] == "conflict")
        seed_differences.append(r["shape_accuracy"] - c["shape_accuracy"])
    effect = {"contrast": "randomized_minus_correlated_conflict_shape_accuracy",
              "mean_difference": float(per_image_delta.mean()),
              "paired_seed_differences": seed_differences,
              "paired_seed_difference_sd": float(np.std(seed_differences, ddof=1)),
              "test_image_cluster_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "bootstrap_replicates": CONFIG["bootstrap_replicates"],
              "interpretation": "CI conditional on fixed trained models and split; does not include writer/split/training uncertainty."}
    save_json(ROOT / "results/paired_effect.json", effect)
    print(json.dumps(summaries, indent=2), flush=True)
    print("PAIRED_EFFECT", effect, flush=True)


def verify_reference():
    """Re-evaluate every saved model and require identical predictions/logits."""
    normal, masks, labels, splits = get_data()
    views = evaluation_views(normal, masks, labels, splits["test"], Renderer())
    checks = []
    for condition in CONFIG["conditions"]:
        for seed in CONFIG["seeds"]:
            model = SmallCNN()
            stored = torch.load(ROOT / f"checkpoints/{condition}_seed{seed}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(stored["state_dict"])
            for name, (x, _, _) in views.items():
                reference = np.load(ROOT / f"results/predictions_{condition}_seed{seed}_{name}.npz")
                logits = predict(model, x, return_logits=True)
                assert np.array_equal(logits.argmax(1), reference["prediction"])
                error = float(np.max(np.abs(logits-reference["logits"])))
                assert error == 0, (condition, seed, name, error)
                checks.append({"condition": condition, "seed": seed, "view": name, "max_logit_difference": error})
    save_json(ROOT / "results/reproduction_check.json", {"checked_utc": utc_now(), "checks": checks,
              "all_predictions_identical": True, "all_logits_identical": True,
              "scope": "fresh model load and deterministic rendering/evaluation; not a second full training of every run"})
    print(f"PASS: exact logits and predictions for all {len(checks)} model/view combinations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    setup()
    if args.verify:
        verify_reference()
        return
    reg = json.loads((ROOT / "preregistration_record.json").read_text())
    assert sha256(ROOT / "PREREGISTRATION.md") == reg["protocol_sha256"]
    assert sha256(ROOT / "config.json") == reg["config_sha256"]
    environment = {"started_utc": utc_now(), "python": sys.version, "torch": torch.__version__,
                   "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(),
                   "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
                   "device": "cpu", "config": CONFIG, "protocol_sha256": reg["protocol_sha256"],
                   "experiment_source_sha256": sha256(Path(__file__))}
    save_json(ROOT / "results/environment.json", environment)
    normal, masks, labels, splits = get_data()
    renderer = Renderer()
    training_records = []
    # No test evaluation occurs until every preregistered model has been trained.
    for seed in CONFIG["seeds"]:
        for condition in CONFIG["conditions"]:
            training_records.append(train_one(condition, seed, normal, masks, labels, splits, renderer))
    for seed in CONFIG["seeds"]:
        initial = [r["initial_state_sha256"] for r in training_records if r["seed"] == seed]
        assert len(set(initial)) == 1, "Initializations not paired"
    save_json(ROOT / "results/training_records.json", training_records)
    if not args.train_only:
        views = evaluation_views(normal, masks, labels, splits["test"], renderer)
        evaluate_all(views)
    environment["finished_utc"] = utc_now()
    save_json(ROOT / "results/environment.json", environment)


if __name__ == "__main__":
    main()
