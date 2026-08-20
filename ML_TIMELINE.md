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

## Phase P2 — Synthetic data generation

**Build:** A small Python script that hand-generates fake traffic rows matching the feature spec — some labeled `normal`, some labeled with an attack type.

**Why here:** Lets you build and test the entire pipeline before real labeled data exists. Doesn't need to be realistic, just structurally correct.

**Depends on:** P1.

**Done when:** You have a CSV/dataframe of a few hundred synthetic rows, correctly labeled, matching the feature spec's shape.

---

## Phase P3 — Baseline training pipeline

**Build:** `train.py` and `evaluate.py`, run end-to-end against the synthetic data from P2. Get the full loop working: load → extract features → train → evaluate → print metrics.

**Why here:** The goal isn't a good model yet — it's a working *pipeline*. Once this runs cleanly on fake data, swapping in real data later (P4) is a small change, not a rebuild.

**Depends on:** P2.

**Done when:** Running `train.py` produces a trained model and `evaluate.py` prints precision/recall/F1, even if the numbers are mediocre (expected, on fake data).

---

## Phase P4 — Real data ingestion

**Build:** Swap the synthetic data source for real output from the Go core's `simulator/` (main repo Phase 7). Re-run training, compare results against the P3 baseline.

**Why here:** This is the first point where the model is actually learning something meaningful, rather than validating plumbing.

**Depends on:** The Go core's simulator (main repo Phase 7) being done and producing labeled traffic logs.

**Done when:** Training runs against real simulator output, and evaluation metrics are meaningfully better than a trivial baseline (e.g. "flag everything the rules already catch").

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
