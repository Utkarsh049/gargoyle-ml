# TIMELINE.md — Gargoyle ML Pipeline Build Plan

Six phases, named P1–P6 to keep them distinct from the Go core's numbered phases in the main repo's TIMELINE.md. Each lists what to build, what it depends on (if anything), and what "done" looks like.

---

## Phase P1 — Feature spec draft `[COMPLETED]`

**Status:** Completed (Spec Version `1.0.0`)

**Deliverables:**
- [`features/feature_spec.md`](features/feature_spec.md) — Canonical spec defining 6 ordered features (`float_input` shape `[-1, 6]`), types, bounds, cold-start defaults, and mathematical definitions.
- [`features/spec.py`](features/spec.py) — Programmatic dataclass model, boundary validation, cold-start generator, and dictionary export.
- [`tests/test_spec.py`](tests/test_spec.py) — Unit test suite validating ordering, indexing, boundary checks, and error handling (10/10 tests passing).
- [`requirements.txt`](requirements.txt) & [`.gitignore`](.gitignore) — Pinned Python dependencies and project configuration.

**Why first:** Everything downstream depends on this being defined. Both the Python training pipeline and Go's runtime `MLScorer` rely on this shared contract.

**Depends on:** Nothing.

**Done when:** You have a numbered, ordered list of features with types, bounds, and edge-case definitions. (Verified)

---

## Phase P2 — Synthetic data generation `[COMPLETED]`

**Status:** Completed

**Deliverables:**
- [`data/generate_synthetic.py`](data/generate_synthetic.py) — CLI & module for generating structurally compliant synthetic feature datasets and raw request logs across 5 traffic profiles (`normal`, `rate_burst`, `brute_force`, `endpoint_scan`, `header_bot`).
- [`data/processed/synthetic_features.csv`](data/processed/synthetic_features.csv) — 1,500 labeled feature vector rows (Spec v1.0.0, 6 features + `label` + `is_abusive`).
- [`data/raw/synthetic_traffic.csv`](data/raw/synthetic_traffic.csv) — 1,000 raw simulated HTTP request logs.
- [`tests/test_synthetic.py`](tests/test_synthetic.py) — Unit tests for synthetic generation, label diversity, CSV serialization, and feature spec compliance (5/5 tests passing).

**Why here:** Lets you build and test the entire training and evaluation pipeline before real labeled data from Go's simulator exists.

**Depends on:** P1.

**Done when:** You have a CSV/dataframe of synthetic rows, correctly labeled, matching the feature spec's shape. (Verified)

---

## Phase P3 — Baseline training pipeline `[COMPLETED]`

**Status:** Completed

**Deliverables:**
- [`features/extract.py`](features/extract.py) — Sliding-window feature extraction engine (60s rate/timing variance, 5m distinct paths/auth failures, path normalization) and dataset loaders.
- [`training/model_config.py`](training/model_config.py) — Hyperparameter dataclasses for `LogisticRegression` and `RandomForestClassifier` with JSON serialization.
- [`training/train.py`](training/train.py) — End-to-end training pipeline with stratified train/test splitting, probability estimation, threshold sweeps, baseline comparison, and artifact persistence (`models/abuse_model.pkl`, `models/train_metrics.json`).
- [`evaluation/evaluate.py`](evaluation/evaluate.py) — Evaluation suite computing Precision, Recall, F1, Specificity, ROC-AUC, Confusion Matrix, and Go rule baseline comparison report.
- [`tests/test_extract.py`](tests/test_extract.py), [`tests/test_evaluation.py`](tests/test_evaluation.py), [`tests/test_training.py`](tests/test_training.py) — Comprehensive unit test suites (28/28 tests passing).

**Why here:** Establishes the complete training and evaluation pipeline end-to-end. Once real simulator data arrives (P4), swapping data sources requires zero structural changes.

**Depends on:** P2.

**Done when:** Running `train.py` produces a trained model (`abuse_model.pkl`) and `evaluate.py` prints precision/recall/F1 and comparison metrics. (Verified)

---

## Phase P4 — Real data ingestion `[COMPLETED]`

**Status:** Completed

**Deliverables:**
- [`data/raw/ground_truth_5k.csv`](data/raw/ground_truth_5k.csv) — 5,000 real HTTP simulator request logs across 4 batches (`normal`: 2,000, `endpoint_sweep`: 1,000, `credential_stuffing`: 1,000, `rate_probe`: 1,000).
- [`features/extract.py`](features/extract.py) — Ingestion pipeline supporting ISO timestamps, JSON header anomaly scoring, and endpoint normalization.
- [`data/processed/real_features.csv`](data/processed/real_features.csv) — 5,000 feature-extracted vectors (Spec v1.0.0).
- Retrained **Logistic Regression** and **Random Forest** models on real simulator data.

**Performance Benchmark vs. Go Rule Baseline (Test Set, N=1,000):**
- **ML Model Precision:** `1.0000` (Random Forest) / `0.9983` (Logistic Regression) vs. Rule Baseline `0.6082`
- **ML Model Specificity (TNR):** `1.0000` (0 False Positives) vs. Rule Baseline `0.0400` (rules misclassify 96% of sustained normal traffic)
- **ML Model F1-Score:** `1.0000` (RF) / `0.9992` (LR) vs. Rule Baseline `0.7544`

**Why here:** Validates that the ML model learns genuine abuse patterns (timing variance, auth failure clusters, endpoint scans) and provides a defensible improvement over static rules alone.

**Depends on:** Go core simulator output (`ground_truth_5k.csv`).

**Done when:** Training runs against real simulator output, and evaluation metrics are meaningfully better than the trivial rule baseline. (Verified)

---

## Phase P5 — ONNX export + feature parity check

**Build:** `export/to_onnx.py`, producing `abuse_model.onnx`. Then a small cross-check script that feeds identical sample requests through both Python's `extract.py` and Go's feature extraction code, confirming they output the same numbers in the same order.

**Why here:** This is the step that catches the "silent mismatch" risk described in PROJECT.md §2 and §7 — before the file ever reaches the Go repo, not after.

**Depends on:** Go's `MLScorer`/feature-extraction code existing on the Go side (main repo Phase 8 start) so there's something to check parity against.

**Done when:** The exported `.onnx` file loads without error in a standalone test, and the parity check confirms identical feature vectors on both sides for the same sample input.

---

## Phase P6 — Integration validation

**Build:** Nothing new in Python — this phase is about confirming the exported model works correctly once it's actually inside the running Go core (main repo Phase 8).

**Why here:** Final check that the whole pipeline — training, export, and Go-side loading/inference — produces sensible results on real traffic, not just in isolation.

**Depends on:** Main repo Phase 8 (Go's `MLScorer` fully wired in).

**Done when:** Running the simulator's attack traffic against the live Gargoyle instance produces `abuse_score` values that clearly separate normal from malicious requests, visible in the dashboard/logs.

---

## Notes on scope and parallelism

- **P1–P3 have zero dependency on the Go core.** Start them the same week you start Go Phase 1 — there's no reason to wait.
- **P4 is the one hard sync point** — it needs real labeled data, which only the Go core's simulator (Phase 7) can produce. Everything before P4 can be built and tested against synthetic data in the meantime.
- **P5–P6 need Go-side code to exist** (Phase 8's `MLScorer`) to validate against — these are naturally the last two phases, and there's no way to meaningfully pull them earlier.
- If time runs short, **P1–P4 alone still produce a real, evaluable model and a legitimate ML story for a resume** — P5–P6 (the Go integration) can slip without invalidating the training work already done; the main repo's Gargoyle still runs fine on rules alone in the meantime (see main repo TIMELINE.md, Phase 8 notes).
