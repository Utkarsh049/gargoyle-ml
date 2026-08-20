# Gargoyle ML — Abuse Detection Model Pipeline

An offline machine learning training, evaluation, and ONNX export pipeline for the **Gargoyle API Gateway**. 

Gargoyle ML produces `abuse_model.onnx`, a lightweight, portable model artifact that the Go gateway core loads at startup to perform **sub-millisecond, in-process abuse likelihood scoring** on incoming HTTP traffic.

---

## 1. Project Overview & Philosophy

Modern API gateways commonly rely on static, rule-based heuristics (e.g., token bucket rate limiters, static failure counters, regex header matching) to block abusive traffic. While effective against naive flooding, static thresholds suffer from critical limitations:

1. **High False Positive Rates on Legitimate Users ("The Power User Problem"):** Active users and Single-Page Applications (SPAs) frequently trigger rapid bursts of API calls during normal navigation (e.g., loading assets, search-as-you-type, multi-tab browsing). Static rate rules blindly flag these legitimate sessions as DoS attacks, returning `429 Too Many Requests`.
2. **Vulnerability to Low-and-Slow Attacks:** Attackers who throttle request rates just below static limits (e.g., 30–50 requests/min) easily evade traditional rate limiters.
3. **Inability to Model Multivariate Correlations:** Static rules evaluate signals in isolation. They cannot express nuanced conditions such as: *"A high request rate is safe IF inter-request timing variance is high (human-like) AND authentication errors are zero."*

### The Gargoyle Solution
Gargoyle ML complements the gateway's rule engine by learning non-linear behavioral correlations across time-windowed traffic features. 

```
+-----------------------------------------------------------------------------+
|                                OFFLINE (Python)                             |
|                                                                             |
|  Raw Traffic Logs ---> Feature Extractor ---> Classifier Training           |
|  (Simulator Data)      (Sliding Windows)      (Random Forest / Logistic Reg)|
|                                                        |                    |
|                                                        v                    |
|                                                ONNX Model Export            |
|                                              (models/abuse_model.onnx)      |
+--------------------------------------------------------|--------------------+
                                                         |
                                    Copy artifact into Go gateway
                                                         |
+--------------------------------------------------------v--------------------+
|                                PRODUCTION (Go Core)                         |
|                                                                             |
|  Incoming HTTP Request ---> In-Memory Feature Vector ---> ONNX Runtime      |
|                                (Exact 6-Feature Spec)     (In-Process)      |
|                                                                |            |
|                                                                v            |
|                                                      Abuse Score [0.0, 1.0] |
+-----------------------------------------------------------------------------+
```

### Key Architectural Guarantees
* **Zero Runtime Python Dependency:** Python is used strictly offline for data processing, training, and model export. The Go gateway executes inference natively in-process via ONNX Runtime with zero inter-process overhead.
* **Strict Cross-Language Feature Parity:** Both Python and Go adhere to a single canonical specification ([`features/feature_spec.md`](features/feature_spec.md)) and are validated against shared test fixtures ([`fixtures/parity_fixtures.json`](fixtures/parity_fixtures.json)) to ensure **0% feature drift**.
* **Graceful Degradation:** If the ONNX model is absent, corrupted, or disabled, the Go gateway logs a warning and automatically falls back to its deterministic rule engine without dropping traffic.

---

## 2. Feature Specification (Spec v1.0.0)

Both the Python offline pipeline and Go runtime compute an identical **6-dimensional continuous feature vector** representing client behavior across sliding time windows:

| Index | Feature Name | Type | Value Range | Cold-Start Default | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | `requests_last_60s` | `float32` | $[1.0, \infty)$ | `1.0` | Total request count from client in trailing 60s sliding window |
| **1** | `avg_interval_ms` | `float32` | $[0.0, 60000.0]$ | `0.0` | Mean inter-request interval (ms) in trailing 60s window |
| **2** | `interval_stddev_ms` | `float32` | $[0.0, 60000.0]$ | `0.0` | **Timing variance** (Low = programmatic bot; High = human browsing) |
| **3** | `distinct_endpoints_last_5m` | `float32` | $[1.0, \infty)$ | `1.0` | Count of unique normalized URI paths accessed in trailing 5m |
| **4** | `failed_auth_count_last_5m` | `float32` | $[0.0, \infty)$ | `0.0` | Count of HTTP 401/403 status codes in trailing 5m |
| **5** | `header_anomaly_score` | `float32` | $[0.0, 1.0]$ | `0.0` | Heuristic header penalty score computed from browser headers |

---

## 3. Directory Layout

```
gargoyle-ml/
├── data/
│   ├── raw/                      # Raw simulator and synthetic traffic CSV logs
│   ├── processed/                # Extracted feature vector datasets
│   └── generate_synthetic.py     # Standalone synthetic dataset and log generator
├── features/
│   ├── feature_spec.md           # Canonical spec contract shared between Python & Go
│   ├── spec.py                   # Dataclass model and runtime feature validation
│   └── extract.py                # Sliding-window feature extraction engine
├── training/
│   ├── model_config.py           # Hyperparameter configuration and serialization
│   └── train.py                  # End-to-end model training, threshold sweeps & metrics
├── evaluation/
│   └── evaluate.py               # Standalone evaluation & comparison against rule baseline
├── export/
│   ├── to_onnx.py                # ONNX exporter with skl2onnx conversion & verification
│   └── parity_check.py           # Cross-language fixture parity checker
├── fixtures/
│   └── parity_fixtures.json      # Canonical test cases shared with Go test suite
├── models/
│   ├── abuse_model.onnx          # Exported production ONNX model artifact
│   └── train_metrics.json        # Serialized evaluation metrics and threshold sweeps
├── tests/                        # Comprehensive unit test suites (35 tests)
│   ├── test_spec.py
│   ├── test_synthetic.py
│   ├── test_extract.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_onnx_export.py
│   └── test_parity.py
├── requirements.txt              # Pinned Python dependencies
├── OUTPUT.md                     # Empirical benchmark report and case studies
└── README.md
```

---

## 4. Getting Started & Installation

### Prerequisites
* Python 3.10+ (tested through 3.14)
* Virtual environment tool (`venv`)

### Setup Environment
```bash
# Clone repository and enter directory
git clone https://github.com/Utkarsh049/gargoyle-ml.git
cd gargoyle-ml

# Create and activate virtual environment
python3 -m venv .venv

# Bash / Zsh:
source .venv/bin/activate

# Fish:
source .venv/bin/activate.fish

# Install dependencies
pip install -r requirements.txt
```

---

## 5. End-to-End Pipeline Execution

### Step 1: Generate Synthetic Data (Optional)
Generate labeled synthetic HTTP traffic across 5 behavioral profiles (`normal`, `rate_burst`, `brute_force`, `endpoint_scan`, `header_bot`):
```bash
python data/generate_synthetic.py --records 2000 --output-features data/processed/synthetic_features.csv
```

### Step 2: Feature Extraction
Extract sliding-window features from raw gateway traffic logs (e.g., simulator logs):
```bash
python features/extract.py --input data/raw/ground_truth_5k.csv --output data/processed/real_features.csv
```

### Step 3: Model Training & Evaluation
Train the classifier (Random Forest or Logistic Regression), perform stratified train/test evaluation, sweep decision thresholds, and serialize model artifacts:
```bash
# Train Random Forest (Default)
python training/train.py --data data/processed/real_features.csv --model-type random_forest

# Train Logistic Regression
python training/train.py --data data/processed/real_features.csv --model-type logistic_regression
```

### Step 4: Standalone Benchmark Evaluation
Evaluate model performance against the Go rule-based heuristic baseline:
```bash
python evaluation/evaluate.py --data data/processed/real_features.csv --model models/abuse_model.pkl --threshold 0.5
```

### Step 5: Export to ONNX
Convert the trained scikit-learn model to portable ONNX format and verify runtime inference:
```bash
python export/to_onnx.py --model models/abuse_model.pkl --output models/abuse_model.onnx
```

### Step 6: Parity Validation
Verify that feature extraction and ONNX model predictions produce identical outputs matching the canonical test fixtures:
```bash
python export/parity_check.py --fixtures fixtures/parity_fixtures.json --model models/abuse_model.onnx
```

---

## 6. Performance Summary vs. Static Rule Baseline

Evaluating on 5,000 requests from the Go gateway simulator (`ground_truth_5k.csv`):

| Metric | Static Rule Baseline | ML Model (Random Forest) | Impact |
| :--- | :---: | :---: | :--- |
| **Precision (No FP)** | **61.06%** | **100.00%** | **+38.94%** (Zero false alarms on paying users) |
| **False Positives (Blocked Users)** | **1,900 / 2,000 (95.0%)** | **0 / 2,000 (0.0%)** | **Eliminates false customer lockouts** |
| **Recall (Attack Detection)** | **99.30%** | **100.00%** | Catches 100% of brute force, scans, and floods |
| **Low-and-Slow Detection** | **0.0%** ($N \le 50$) | **100.0%** ($N \le 50$) | Catches throttled attacks via timing variance |
| **F1-Score** | 75.66% | **100.00%** | Balanced detection and precision |
| **ROC-AUC** | N/A | **1.0000** | Perfect class separability |

> For complete benchmark tables, confusion matrices, and detailed case studies, see **[`OUTPUT.md`](OUTPUT.md)**.

---

## 7. Integration with the Go Gateway Core (Phase 6)

Once `abuse_model.onnx` is exported:

1. **Deployment:** Copy `models/abuse_model.onnx` into the Go gateway project (`internal/abuse/model/abuse_model.onnx`).
2. **Inference Execution:** Go's `MLScorer` initializes an in-process ONNX Runtime session (`onnxruntime-go`) at startup.
3. **Request Flow:** For each HTTP request, Go extracts the 6 feature values in the exact spec order, executes `session.Run()`, and obtains the abuse probability ($P(\text{Abuse}) \in [0.0, 1.0]$).
4. **Decision Boundary:** Requests with $P(\text{Abuse}) \ge 0.5$ (configurable) are flagged or throttled by the gateway pipeline.

---

## 8. Test Suite

Run the full automated test suite (unit tests, validation, ONNX export, and parity checks):
```bash
pytest
```

```
tests/test_evaluation.py .....                                           [ 14%]
tests/test_extract.py .......                                            [ 34%]
tests/test_onnx_export.py ....                                           [ 45%]
tests/test_parity.py .                                                   [ 48%]
tests/test_spec.py ..........                                            [ 77%]
tests/test_synthetic.py .....                                            [ 91%]
tests/test_training.py ...                                               [100%]

============================== 35 passed in 1.87s ==============================
```

---

## 9. Technology Stack

* **Language:** Python 3.10+ (Static typing validated via Pyright / Pyrefly)
* **Data Processing:** NumPy, Pandas
* **Model Training:** Scikit-Learn (`RandomForestClassifier`, `LogisticRegression`)
* **Model Serialization:** ONNX (`skl2onnx`, `onnx`, `onnxruntime`)
* **Testing:** Pytest, Unittest

