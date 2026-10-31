"""Post-primary, explicitly exploratory texture-learnability control."""
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

import experiment as ex


def main():
    ex.setup()
    declaration = ex.ROOT / "EXPLORATORY_ADDENDUM.md"
    marker = ex.ROOT / "exploratory_record.json"
    if not marker.exists():
        ex.save_json(marker, {"recorded_before_extra_training_utc": ex.utc_now(),
                             "addendum_sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
                             "primary_results_already_observed": True})
    renderer = ex.Renderer()
    n = 1077
    masks = np.repeat(renderer.disk[None], n, axis=0)
    targets = np.arange(n) % 10
    y = torch.from_numpy(targets)
    views = np.load(ex.ROOT / "data/test_views.npz")
    test_x = views["texture_only_disk"]
    test_y = np.tile(np.arange(10), len(test_x) // 10)
    output = []
    for seed in ex.CONFIG["seeds"]:
        ex.seed_everything(seed)
        model = ex.SmallCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=ex.CONFIG["learning_rate"],
                                     weight_decay=ex.CONFIG["weight_decay"])
        started = ex.utc_now()
        clock = time.perf_counter()
        for epoch in range(1, ex.CONFIG["epochs"] + 1):
            model.train()
            rng = np.random.default_rng(33_000_000 + seed * 100_000 + epoch)
            phase = rng.uniform(0, 2*np.pi, n)
            x = torch.from_numpy(renderer.textured(masks, targets, phase)[:, None])
            order = rng.permutation(n)
            for a in range(0, n, ex.CONFIG["batch_size"]):
                idx = order[a:a+ex.CONFIG["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[idx]), y[idx])
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print("texture_positive_control", seed, epoch, float(loss), flush=True)
        logits = ex.predict(model, test_x, return_logits=True)
        pred = logits.argmax(1)
        result = {"seed": seed, "n": len(test_y), "accuracy": float((pred==test_y).mean()),
                  "started_utc": started, "finished_utc": ex.utc_now(), "seconds": time.perf_counter()-clock,
                  "status": "exploratory added after primary results"}
        torch.save({"state_dict": model.state_dict(), "config": ex.CONFIG, "result": result},
                   ex.ROOT/f"checkpoints/texture_positive_control_seed{seed}.pt")
        np.savez_compressed(ex.ROOT/f"results/positive_control_seed{seed}.npz",
                            logits=logits,prediction=pred,target=test_y)
        output.append(result)
        print(result,flush=True)
    ex.save_json(ex.ROOT/"results/positive_control.json", {
        "status": "exploratory added after primary results",
        "accuracy_mean": float(np.mean([r["accuracy"] for r in output])),
        "accuracy_sd": float(np.std([r["accuracy"] for r in output],ddof=1)), "runs": output})


if __name__ == "__main__":
    main()
