# Gargoyle Feature Specification

- **Spec Version:** `1.0.0`
- **Target ONNX Tensor Name:** `float_input`
- **Target ONNX Tensor Shape:** `[batch_size, 6]`
- **Target ONNX Element Type:** `float32` (IEEE 754 single-precision floating point)
- **Status:** **Active / Canonical Contract**

---

## 1. Overview & Architectural Contract

This document is the **single source of truth** for feature definitions shared between:
1. **Python Training Pipeline** (`ml/features/extract.py`)
2. **Go Gateway Runtime Inference** (`gateway/internal/abuse/model` / `MLScorer`)

### The Invariant
The feature vector extracted by Python during training and the feature vector computed by Go during live request evaluation **must match identically in feature order, data type, unit scaling, and edge-case handling**.

Any alteration to feature definitions, ordering, or count requires:
1. Bumping the **Spec Version** (`1.0.0` -> `1.1.0` or `2.0.0`).
2. Updating both Python extraction and Go extraction simultaneously.
3. Running the parity check validation suite before exporting the new `.onnx` model.

---

## 2. Feature Vector Definition

The feature vector is a 1-D array of 6 continuous `float32` values.

| Index | Feature Name | Go Type | Python Type | Range / Bounds | Cold Start Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `requests_last_60s` | `float32` | `float32` | `[1.0, +inf)` | `1.0` | Total request count from this client identifier in the trailing 60s sliding window (including current request). |
| **1** | `avg_interval_ms` | `float32` | `float32` | `[0.0, 60000.0]` | `0.0` | Mean time interval (in milliseconds) between consecutive requests in the trailing 60s window. |
| **2** | `interval_stddev_ms` | `float32` | `float32` | `[0.0, 60000.0]` | `0.0` | Sample standard deviation of inter-request intervals in the trailing 60s window. (Low = bot-like regularity). |
| **3** | `distinct_endpoints_last_5m` | `float32` (cast from `int`) | `float32` | `[1.0, +inf)` | `1.0` | Count of unique normalized URI paths accessed by this client in the trailing 5-minute window. |
| **4** | `failed_auth_count_last_5m` | `float32` (cast from `int`) | `float32` | `[0.0, +inf)` | `0.0` | Count of HTTP 401/403 responses observed for this client in the trailing 5-minute window. |
| **5** | `header_anomaly_score` | `float32` | `float32` | `[0.0, 1.0]` | `0.0` | Heuristic score (0.0 = clean, 1.0 = highly suspicious) computed by Go's rule layer for the current request. |

---

## 3. Mathematical Definitions & Edge Case Handling

### Index 0: `requests_last_60s`
* **Definition:** Count of request timestamps $t_i$ satisfying $t_{\text{now}} - 60\text{s} \le t_i \le t_{\text{now}}$.
* **Edge Cases:**
  * First request from client: `1.0`.
  * Cannot be negative or zero (current request is always counted).

### Index 1: `avg_interval_ms`
* **Definition:** For $N$ requests with timestamps $t_1, t_2, \dots, t_N$ in the 60s window ($N \ge 2$):
  $$\Delta t_i = t_i - t_{i-1} \quad (\text{in ms})$$
  $$\mu = \frac{1}{N - 1} \sum_{i=2}^{N} \Delta t_i = \frac{t_N - t_1}{N - 1}$$
* **Edge Cases:**
  * $N < 2$ (0 or 1 request): `0.0`.
  * Sub-millisecond intervals: Clamped to minimum `0.0`.

### Index 2: `interval_stddev_ms`
* **Definition:** Sample standard deviation of intervals for $N \ge 3$ (or $N \ge 2$ with Bessel correction):
  $$s = \sqrt{\frac{1}{M - 1} \sum_{i=1}^{M} (\Delta t_i - \mu)^2} \quad \text{where } M = N - 1$$
* **Edge Cases:**
  * $N < 3$ (fewer than 2 intervals): `0.0`.
  * Perfectly periodic traffic (e.g. cron job or bot sending every 100ms): $\Delta t_i = 100 \implies s = 0.0$.
  * Human traffic: Higher variance ($\ge 200\text{ms}$).

### Index 3: `distinct_endpoints_last_5m`
* **Definition:** $|\{ \text{normalize\_path}(p_i) \mid t_i \ge t_{\text{now}} - 300\text{s} \}|$
* **Normalization Rule:** Strip query parameters, trailing slashes, and collapse dynamic IDs (e.g. `/api/v1/users/123` -> `/api/v1/users/:id`).
* **Edge Cases:**
  * First request: `1.0`.

### Index 4: `failed_auth_count_last_5m`
* **Definition:** $\sum_{i} \mathbf{1}_{(\text{status}_i \in \{401, 403\})}$ for $t_i \ge t_{\text{now}} - 300\text{s}$.
* **Edge Cases:**
  * No authentication failures: `0.0`.

### Index 5: `header_anomaly_score`
* **Definition:** Aggregated heuristic penalty score in range $[0.0, 1.0]$ evaluated on current request headers:
  * Missing `User-Agent`: $+0.4$
  * Known malicious/scanner `User-Agent` (e.g. `sqlmap`, `nikto`, `curl/7.x` on browser endpoints): $+0.5$
  * Missing standard browser headers (`Accept`, `Accept-Language`, `Host`): $+0.2$
  * Total clamped to $[0.0, 1.0]$.
* **Edge Cases:**
  * Clean request with valid headers: `0.0`.

---

## 4. ONNX Model I/O Contract

```json
{
  "inputs": [
    {
      "name": "float_input",
      "type": "tensor(float)",
      "shape": [-1, 6]
    }
  ],
  "outputs": [
    {
      "name": "label",
      "type": "tensor(int64)",
      "shape": [-1]
    },
    {
      "name": "probabilities",
      "type": "tensor(float)",
      "shape": [-1, 2]
    }
  ]
}
```

* **Output interpretation in Go:**
  * `probabilities[0][1]` represents $P(\text{Abuse} \mid \vec{x})$, a `float32` in range `[0.0, 1.0]`.
  * `label[0]` is `1` if $P(\text{Abuse}) \ge \text{threshold}$ (default: `0.5`), else `0`.

---

## 5. Parity Verification Checklist

Before shipping any exported `abuse_model.onnx` to the Go gateway repository:
- [ ] Spec version matches between `feature_spec.md` and Go constant `FeatureSpecVersion`.
- [ ] Number of features is exactly `6`.
- [ ] Feature order matches the 0–5 index sequence above.
- [ ] Input tensor name is `float_input`.
- [ ] Test fixtures pass the cross-check parity test (`tests/test_parity.py`).
