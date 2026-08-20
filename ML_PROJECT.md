# PROJECT.md — Gargoyle ML Pipeline (Detail)

This document covers the internals of the `ml/` project — the training pipeline, the contract with Go, and what "done" actually means for a model before it ships.

---

## 1. Responsibilities of this project

Exactly three jobs. Nothing here runs in production, and nothing here talks to the network at request time.

1. **Turn labeled traffic into features** — per the shared feature spec
2. **Train and evaluate a classifier** — until it beats a simple baseline with defensible precision/recall
3. **Export to ONNX** — produce the one artifact the Go core will ever consume

---

## 2. The contract with the Go core

This is the single most important section in this document — it's the one place a mismatch fails silently rather than throwing an error.

**The feature vector must match exactly between training and inference:** same features, same order, same scaling/encoding. Both sides reference one shared spec rather than trusting two separate implementations to stay in sync from memory.

`ml/features/feature_spec.md` defines the contract, for example:
```
index 0: requests_last_60s          (float, count)
index 1: avg_interval_ms            (float, mean time between requests)
index 2: interval_stddev_ms         (float, variance in timing — low = bot-like)
index 3: distinct_endpoints_last_5m (int)
index 4: failed_auth_count_last_5m  (int)
index 5: header_anomaly_score       (float, 0-1, from Go's rule layer)
```

Python computes these numbers from historical/simulated data during training. Go's `MLScorer` (see the main repo's PROJECT.md §6) must compute the *exact same six numbers, in the exact same order*, from a live request before calling the ONNX session. Change a feature on either side, update the spec file and both implementations together, and bump a version marker (§8).

---

## 3. Folder layout

```
ml/
  data/
    raw/                 captured traffic logs from the simulator (CSV/JSON)
    processed/            cleaned, feature-extracted datasets ready for training
  features/
    extract.py            turns raw traffic logs into feature vectors
    feature_spec.md      the shared contract — source of truth for both Python and Go
  training/
    train.py               trains the classifier, evaluates it, saves metrics
    model_config.py     hyperparameters, model choice
  export/
    to_onnx.py             converts the trained sklearn model to ONNX
  evaluation/
    evaluate.py            precision/recall/F1 on a held-out test set, confusion matrix
  models/
    abuse_model.onnx     the final artifact consumed by the Go core
    abuse_model.pkl        the raw sklearn model, kept for retraining/debugging
  requirements.txt
```

---

## 4. Data source

Training data comes from the Go project's `simulator/` — a tool that generates both normal and clearly-labeled attack traffic against a running Gargoyle instance. Its output (raw request logs, each row labeled `normal` or an attack type) lands in `ml/data/raw/`.

**This is the one real dependency this project has on the Go side:** you cannot train a meaningful model without labeled data, and the labeled data comes from the simulator. Until it exists, this project works entirely against hand-written synthetic data (see TIMELINE.md Phase P2).

---

## 5. Training pipeline

```
1. Load raw labeled traffic (ml/data/raw/)
2. Extract features per feature_spec.md (ml/features/extract.py)
3. Split train/test
4. Train classifier (ml/training/train.py)
5. Evaluate (ml/evaluation/evaluate.py) — precision/recall/F1, confusion matrix
6. If acceptable, export to ONNX (ml/export/to_onnx.py)
7. Copy the .onnx file into the Go repo (gateway/internal/abuse/model/, or wherever Go's loader expects it)
```

Steps 1–5 are cheap to iterate on — expect to retrain several times as you tune features and thresholds. Step 6 only happens once evaluation numbers are actually good enough (§6).

---

## 6. What "good enough" looks like

Don't chase a perfect score. For a resume project, a model that's clearly better than the rule-based layer alone, with numbers you can explain, is the real goal.

- **Precision matters more than recall for blocking** — a false positive blocks a real user, worse than letting one abusive request slip through
- Track precision/recall separately, not just accuracy — with imbalanced data (most traffic is normal), accuracy alone is misleading
- Keep a simple baseline (e.g. "flag anything the rule layer alone would catch") to compare against — the goal is being able to say "the model catches X% more abuse than rules alone, at Y precision"

---

## 7. Validating the Go contract before shipping

Before copying a new `.onnx` file into the Go repo, run a parity check: feed the same handful of requests through Python's `extract.py` and Go's feature extraction code, and confirm they produce identical vectors. This is the step that catches a silent feature-order mismatch before it ships — the failure mode otherwise is a model that loads fine and runs fine, just scores everything wrong, with no error anywhere.

---

## 8. Versioning

Keep `feature_spec.md` versioned alongside the model file. If you retrain with new or reordered features, bump a version marker so Go's loader can, at minimum, log a warning if it's given a model built against a spec it doesn't recognize — silent version drift is exactly the failure mode §2 and §7 exist to prevent.

---

## 9. What ships where

- `ml/` never gets deployed as a running service — it's a development-time project only
- The only artifact that leaves this folder is `abuse_model.onnx`
- Everything else here (`data/`, `training/`, `evaluation/`, `models/*.pkl`) is for your own iteration and is never touched by the Go core
