# Gargoyle ML — Abuse Detection Model

This is the Python half of Gargoyle: a training pipeline that produces `abuse_model.onnx`, a portable model file the Go core optionally loads to add ML-based abuse scoring on top of its rule-based detection.

It's a companion project to `gargoyle` (the Go gateway), buildable and testable on its own — see `../README.md` and `../PROJECT.md` in the main repo for how the two connect.

---

## The problem this solves

Rule-based abuse detection (timing patterns, endpoint sweeps, header anomalies — all built in Go) catches a lot, but hand-written rules have a ceiling: they only catch patterns someone thought to write a rule for. A trained classifier can pick up on subtler combinations of signals that no single rule captures — the goal here isn't to replace the rules, it's to add a second, complementary signal on top of them.

---

## What this project actually produces

One artifact: `abuse_model.onnx`. Not a running service, not an API, not anything that stays alive in production. Python's job ends the moment this file is exported — the Go core loads it once at startup and does all inference itself, in-process, with zero live Python dependency at runtime.

---

## How it connects to Go — in short

```
Python trains model  -->  exports abuse_model.onnx  -->  file copied into Go repo
                                                                    |
                                                                    v
                                          Go loads it once at startup (if present)
                                                                    |
                                                                    v
                                    Go extracts features itself, runs inference in-process
```

**The one thing that must never drift:** the feature vector Python trains on and the feature vector Go computes at request time must match exactly — same features, same order. Both sides read from one shared spec (`feature_spec.md`) rather than trusting two implementations to stay in sync from memory. See PROJECT.md §2 for the full contract.

---

## Is this required for Gargoyle to work?

No. The Go core runs completely fine on rule-based detection alone. This project is an optional enhancement layer — genuinely useful to build (it's the real "AI" component of Gargoyle, and a legitimate ML pipeline you can defend in an interview), but not a blocker for anything else in the system.

---

## Tech stack

| Purpose | Tool |
|---|---|
| Data handling | pandas |
| Model | scikit-learn — LogisticRegression or RandomForestClassifier |
| ONNX export | skl2onnx |
| Evaluation | scikit-learn metrics (precision, recall, F1, ROC-AUC) |
| Exploration (optional) | Jupyter |

No deep learning — a classical model is the right scope for hand-engineered tabular features like these, and it's easier to defend in an interview than reaching for a neural net because it sounds more "AI."

---

## Project structure

```
ml/
  data/
    raw/                 captured traffic logs from the simulator
    processed/            cleaned, feature-extracted datasets
  features/
    extract.py            raw traffic -> feature vectors
    feature_spec.md      the shared contract with Go — source of truth
  training/
    train.py               trains + evaluates the classifier
    model_config.py     hyperparameters, model choice
  export/
    to_onnx.py             sklearn model -> ONNX
  evaluation/
    evaluate.py            precision/recall/F1, confusion matrix
  models/
    abuse_model.onnx     the artifact the Go core consumes
    abuse_model.pkl        raw sklearn model, kept for retraining/debugging
  requirements.txt
```

---

