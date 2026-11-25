"""Small helpers for the explicitly reduced Lab 3 deadline experiment."""
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import Booster, XGBClassifier
from xgboost.callback import TrainingCallback


def proportional_sample(labels, size, seed=42):
    """Fixed proportional sample, reserving at least one row per original class."""
    labels = np.asarray(labels)
    if size >= len(labels):
        return np.arange(len(labels))
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if size < len(classes):
        raise ValueError("Sample size must cover every original class.")
    quotas = (size - len(classes)) * counts / counts.sum()
    allocation = np.minimum(counts, 1 + np.floor(quotas).astype(int))
    order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
    remaining = size - allocation.sum()
    while remaining:
        for i in order:
            if allocation[i] < counts[i]:
                allocation[i] += 1
                remaining -= 1
                if not remaining:
                    break
    rng = np.random.default_rng(seed)
    rows = [rng.choice(np.flatnonzero(inverse == i), n, replace=False)
            for i, n in enumerate(allocation)]
    return np.sort(np.concatenate(rows))


def experiment_path(run_directory, protocol, train_ids, validation_ids):
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
    for ids in [train_ids, validation_ids]:
        for row_id in ids:
            key.update(str(row_id).encode())
            key.update(b"\n")
    directory = Path(run_directory) / "deadline" / key.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return directory


def atomic_joblib(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


class BudgetStop(TrainingCallback):
    """Checkpoint every fully evaluated round; stop at patience or wall budget.

    The time limit is checked BETWEEN rounds, not a hard process timeout.
    Data preparation or one slow round can exceed it.
    """
    def __init__(self, directory, seconds, patience=15):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seconds = seconds
        self.patience = patience
        self.started = time.perf_counter()
        self.history = None
        self.best = 0
        self.best_loss = np.inf
        self.reason = "round_cap"

    def after_iteration(self, model, epoch, evals_log):
        self.history = {name: {metric: list(values) for metric, values in metrics.items()}
                        for name, metrics in evals_log.items()}
        losses = self.history["validation_1"]["mlogloss" if "mlogloss" in self.history["validation_1"] else "logloss"]
        if losses[-1] < self.best_loss:
            self.best_loss = losses[-1]
            self.best = epoch
        model.set_attr(best_iteration=str(self.best), best_score=str(self.best_loss))
        elapsed = time.perf_counter() - self.started
        if epoch - self.best >= self.patience:
            self.reason = "early_stopping"
        elif elapsed >= self.seconds:
            self.reason = "wall_budget"
        record = {"history": self.history, "best_iteration": self.best,
                  "termination_reason": self.reason, "elapsed_seconds": elapsed}
        temporary = self.directory / "checkpoint.tmp"
        temporary.write_bytes(model.save_raw(raw_format="ubj"))
        temporary.replace(self.directory / "checkpoint.ubj")
        temporary = self.directory / "history.tmp"
        temporary.write_text(json.dumps(record))
        temporary.replace(self.directory / "history.json")
        if epoch == 0 or (epoch + 1) % 10 == 0 or self.reason != "round_cap":
            print(f"Round {epoch + 1}: validation loss={losses[-1]:.6f}; elapsed={elapsed / 60:.1f} min", flush=True)
        return self.reason != "round_cap"

    def after_training(self, model):
        return model[:self.best + 1]


def fit_xgb(params, train_x, train_y, validation_x, validation_y, directory,
            max_rounds=100, seconds=600, patience=15, workers=2):
    """Fit or load one budgeted configuration; do not share original full-data caches."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "fit.joblib"
    if saved.exists():
        print(f"Loading {directory.name}", flush=True)
        return joblib.load(saved)
    classes = np.unique(train_y)
    multiclass = len(classes) > 2
    metric = "mlogloss" if multiclass else "logloss"
    settings = dict(params, objective="multi:softprob" if multiclass else "binary:logistic",
                    eval_metric=metric, n_estimators=max_rounds, n_jobs=workers,
                    tree_method="hist", subsample=1.0, colsample_bytree=1.0,
                    colsample_bylevel=1.0, colsample_bynode=1.0, random_state=42)
    callback = BudgetStop(directory, seconds, patience)
    candidate = XGBClassifier(**settings, callbacks=[callback])
    start = time.perf_counter()
    if (directory / "checkpoint.ubj").exists() and (directory / "history.json").exists():
        print("Retaining the previously checkpointed partial fit; not claiming convergence.", flush=True)
        record = json.loads((directory / "history.json").read_text())
        native = Booster(model_file=directory / "checkpoint.ubj")
        completed_rounds = min(len(record["history"]["validation_0"][metric]),
                               len(record["history"]["validation_1"][metric]), native.num_boosted_rounds())
        losses = record["history"]["validation_1"][metric][:completed_rounds]
        best = int(np.argmin(losses))
        candidate._Booster = native[:best + 1]
        candidate._Booster.set_attr(best_iteration=str(best), best_score=str(losses[best]))
        candidate.n_classes_ = len(classes)
        history = record["history"]
        reason = "recovered_checkpoint"
        elapsed = record["elapsed_seconds"]
    else:
        print(f"Fitting {directory.name}: {len(train_y):,} train / {len(validation_y):,} stopping rows", flush=True)
        candidate.fit(train_x, train_y, eval_set=[(train_x, train_y), (validation_x, validation_y)], verbose=False)
        history = callback.history
        best = callback.best
        reason = callback.reason
        elapsed = time.perf_counter() - start
        completed_rounds = len(history["validation_1"][metric])
    candidate.set_params(callbacks=None, n_estimators=best + 1)
    candidate.evals_result_ = history
    candidate.get_booster().set_attr(best_iteration=str(best), best_score=str(history["validation_1"][metric][best]))
    curve = pd.DataFrame({"round": np.arange(1, completed_rounds + 1),
                          "training_log_loss": history["validation_0"][metric][:completed_rounds],
                          "validation_log_loss": history["validation_1"][metric][:completed_rounds]})
    result = {"model": candidate, "curve": curve, "fit_seconds": elapsed,
              "termination_reason": reason, "selected_rounds": best + 1,
              "rounds_run": completed_rounds, "training_rows": len(train_y),
              "stopping_rows": len(validation_y)}
    curve.to_csv(directory / "curve.csv", index=False)
    atomic_joblib(result, saved)
    return result
