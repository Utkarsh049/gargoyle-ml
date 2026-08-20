# PYTHON.md — Gargoyle ML Pipeline

This document covers the Python side of Gargoyle only — the `ml/` folder in the repo. It's a separate, self-contained project from the Go core: its only deliverable to the rest of Gargoyle is a trained model file (`abuse_model.onnx`) and the feature spec that Go's inference code must match exactly.

On the Go side, this file is consumed by `MLScorer`, one implementation of the `AbuseScorer` interface (see PROJECT.md §6). If this file is never produced, or fails to load, Gargoyle logs a warning and runs on rule-based scoring alone — nothing in the Go core breaks or depends on this project completing.

You can build this alongside the Go core rather than after it — see the phase mapping at the bottom for exactly where the two tracks depend on each other and where they don't.

---

## 1. Purpose

Take labeled traffic data (normal vs abusive) and train a lightweight classifier that scores incoming requests for abuse likelihood. The output of this project is not a running service — it's a portable model file the Go core loads once at startup and runs entirely on its own.

**Explicitly not this project's job:**
- Serving predictions over a network at request time (Go does inference in-process via ONNX — see PROJECT.md §6)
- Generating the traffic used to train on (that's `simulator/` — a separate small tool)
- Anything user-facing

---

## 2. The contract with the Go core

This is the single most important thing to get right, and it's worth stating explicitly because it's the one place a mismatch fails silently.

**The feature vector must match exactly between training and inference** — same features, same order, same scaling/encoding. Both sides should reference one shared spec rather than trusting two separate implementations to stay in sync from memory.

`ml/feature_spec.md` (or a shared JSON schema, your call) should define, for example:
```
index 0: requests_last_60s          (float, count)
index 1: avg_interval_ms            (float, mean time between requests)
index 2: interval_stddev_ms         (float, variance in timing — low = bot-like)
index 3: distinct_endpoints_last_5m (int)
index 4: failed_auth_count_last_5m  (int)
index 5: header_anomaly_score       (float, 0-1, from Go's rule layer)
```
Python computes these features from historical/simulated data during training. Go's `internal/abuse/model` package must compute the *exact same six numbers, in the exact same order*, from a live request before calling the ONNX session. Change a feature on one side, update the spec file and both implementations together.

---

## 3. Folder layout

```
ml/
  data/
    raw/                 captured traffic logs from the simulator (CSV/JSON)
    processed/            cleaned, feature-extracted datasets ready for training
  features/
    extract.py            turns raw traffic logs into the feature vectors defined in feature_spec.md
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

## 4. Tech stack

| Purpose | Tool |
|---|---|
| Data handling | pandas |
| Model | scikit-learn (start with LogisticRegression or RandomForestClassifier — no need for anything heavier at this scale) |
| ONNX export | skl2onnx |
| Evaluation | scikit-learn metrics (precision, recall, F1, ROC-AUC) |
| Notebook exploration (optional) | Jupyter, for eyeballing feature distributions before committing to the pipeline script |

Deliberately no deep learning here — a classical model is the right scope for tabular, hand-engineered features like these, and it's easier to explain and defend in an interview than "I used a neural net because it's AI."

---

## 5. Data source

Training data comes from `simulator/` (see PROJECT.md / TIMELINE.md Phase 7 in the Go timeline) — a tool that generates both normal and clearly-labeled attack traffic against a running Gargoyle instance. Its output (raw request logs, each row labeled `normal` or one of the attack types) lands in `ml/data/raw/`.

**This is the one real dependency between the two tracks:** you cannot train a meaningful model without labeled data, and the labeled data comes from the simulator. Rule-based heuristics (Go Phase 6) and the simulator (Go Phase 7) should exist before serious training work starts — see the phase mapping below.

---

## 6. Training pipeline

```
1. Load raw labeled traffic (ml/data/raw/)
2. Extract features per feature_spec.md (ml/features/extract.py)
3. Split train/test
4. Train classifier (ml/training/train.py)
5. Evaluate (ml/evaluation/evaluate.py) — precision/recall/F1, confusion matrix
6. If acceptable, export to ONNX (ml/export/to_onnx.py)
7. Drop the .onnx file into gateway/internal/abuse/model/ (or wherever Go loads it from)
```

Steps 1–5 can be iterated on repeatedly and cheaply — this is normal ML workflow, expect to retrain several times as you tune features and thresholds. Step 6 only happens once you're satisfied with evaluation numbers.

---

## 7. What "good enough" looks like

Don't chase a perfect score — for a resume project, a model that's clearly better than the rule-based layer alone, with defensible precision/recall numbers you can explain, is the actual goal. Concretely:

- **Precision matters more than recall for blocking** — a false positive blocks a real user, which is worse than letting one abusive request slip through
- Track precision/recall separately, not just accuracy — with imbalanced data (most traffic is normal), accuracy alone is misleading
- Keep a simple baseline (e.g. "flag anything the rule layer alone would catch") to compare the model against — you want to be able to say "the model catches X% more abuse than rules alone, at Y precision" in an interview

---

## 8. Phases (parallel to the Go timeline)

These map onto the Go core's TIMELINE.md phases so you know what can start early and what has to wait.

**Phase P1 — Feature spec draft**
Write `feature_spec.md` early, even before Go's rule layer exists — you can sketch the intended features based on what you know you'll want (timing, sequencing, header anomalies). Refine it once Go Phase 6 is real.
*Can start: immediately, in parallel with Go Phase 1–5.*

**Phase P2 — Synthetic data generation (standalone)**
Before the real simulator (Go Phase 7) exists, you can hand-write a small synthetic dataset in Python matching the feature spec, just to get the training pipeline itself working end-to-end.
*Can start: immediately, doesn't block on Go at all.*

**Phase P3 — Baseline training pipeline**
Build `train.py` and `evaluate.py` against the synthetic data from P2. Get the full pipeline (load → features → train → evaluate) running, even with fake data and a mediocre model.
*Can start: right after P2.*

**Phase P4 — Real data ingestion**
Once Go Phase 7 (simulator) produces real labeled traffic, swap the synthetic data source for the real one. Re-run training, compare results.
*Depends on: Go Phase 7 being done.*

**Phase P5 — ONNX export + feature parity check**
Export the trained model. Write a small cross-check script/test that feeds the same request through both the Python feature extractor and Go's feature extractor and confirms identical output — this is the step that catches the "silent mismatch" risk from §2 before it ships.
*Depends on: Go's `internal/abuse/model` package existing (Go Phase 8 start).*

**Phase P6 — Integration validation**
Load the exported model into the running Go core (Go Phase 8) and confirm end-to-end: simulated attack traffic gets scored correctly by the real inference path, not just in a Python notebook.
*Depends on: Go Phase 8.*

---

## 9. What ships where

- `ml/` never gets deployed as a running service — it's a development-time project
- The only artifact that leaves this folder is `abuse_model.onnx`, copied into the Go core's repo path at build/deploy time
- Keep `feature_spec.md` versioned alongside the model — if you retrain with new features, bump a version number so Go's loader can (at minimum) log a warning if it's given a model built against a spec it doesn't recognize
