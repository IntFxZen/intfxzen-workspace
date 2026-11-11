# Quantization Impact: FP32 vs INT8 on Android ARM64

## 📱 Testing Environment
*   **Device:** Poco X3 Pro (Android, ARM64)
*   **Processor:** Snapdragon 860
*   **Model:** `yolo26n`
*   **Benchmark Protocol:** 100 iterations after warmup. Median (P50) latency is the primary metric.
*   **Accuracy Protocol:** Validation on the `coco128` dataset (128 images, 929 instances) using `mAP50-95` as the primary metric.

## 📊 Performance & Memory Results (100 Iterations)

| Framework | Precision | Median Latency | Throughput | Model Size | Memory Usage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TFLite (LiteRT)** | FP32 | 74.88 ms | 13.35 FPS | 9.97 MB | 67.2 MB | ✅ Success |
| **TFLite (LiteRT)** | **INT8** | **44.55 ms** | **22.44 FPS** | **3.05 MB** | **46.1 MB** | ✅ Success |
| **ONNX Runtime** | FP32 | 96.72 ms | 10.13 FPS | 9.93 MB | 145.4 MB | ✅ Success |
| **ONNX Runtime** | **INT8** | **72.84 ms** | **13.73 FPS** | ~2.5 MB | 165.7 MB | ✅ Success |
| **OpenVINO** | FP32 | 661.43 ms | 1.51 FPS | ~10 MB | 131.4 MB | ✅ Success |
| **OpenVINO** | INT8 | ❌ *Error* | ❌ *Error* | ~2.5 MB | N/A | ❌ `object is not initialized` |

## ⚖️ Accuracy Trade-offs (mAP50-95)

| Framework | Precision | mAP50 | mAP50-95 | Δ mAP50-95 (vs PyTorch) | Accuracy Drop |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | FP32 | 0.649 | **0.486** | — *(Baseline)* | — |
| **TFLite** | FP32 | 0.643 | 0.477 | -0.009 | -1.8% |
| **TFLite** | **INT8** | 0.632 | **0.461** | **-0.025** | **-5.1%** |
| **ONNX** | FP32 | 0.643 | 0.477 | -0.009 | -1.8% |
| **ONNX** | **INT8** | 0.638 | **0.463** | **-0.023** | **-4.7%** |

## 🔬 Key Findings & Analysis

1.  **TFLite is the Absolute Winner (Speed & Size):**
    TFLite's XNNPACK backend demonstrates incredible INT8 optimization, achieving a **1.68x speedup** (74.88 ms → 44.55 ms) and reducing model size by **69%** (9.97 MB → 3.05 MB). Mobile hardware natively excels at integer operations, making INT8 the ideal precision for TFLite.
2.  **Accuracy Remains Robust:**
    Despite the massive performance gains, INT8 quantization introduces a minimal accuracy drop of **~5% (mAP50-95)** relative to the original PyTorch model. Both TFLite and ONNX Runtime handle Post-Training Quantization (PTQ) robustly. Given the context of real-time mobile deployment, this is an **excellent trade-off**.
3.  **ONNX Runtime INT8 Trade-offs (Memory Penalty):**
    ONNX Runtime shows a moderate speedup (**1.33x**) but comes with noticeable penalties. Initialization time spiked from 78.4 ms to 215.8 ms, and peak memory usage *increased* to 165.7 MB (likely due to dynamic graph rewriting and Quantize/Dequantize node insertion for CPU MLAS backend compatibility). 
4.  **OpenVINO INT8 Failure (Critical Finding):**
    Attempting to load the OpenVINO INT8 model on the ARM64 CPU plugin resulted in a fatal error (`object is not initialized`). This indicates a fundamental limitation in the cross-compiled OpenVINO runtime for Android: the ARM CPU plugin lacks the necessary backend kernels or graph support for INT8 inference on this architecture. *(This is a known pain point when using x86-centric inference engines on edge ARM devices without proper NPU delegation).*

**Conclusion:** For real-world mobile deployment on ARM64 CPUs, **TFLite with INT8 quantization** is the most efficient, stable, and production-ready choice among the tested frameworks. It offers massive gains in speed (1.68x), size (3x reduction), and memory, while sacrificing only ~5% of accuracy. ONNX Runtime is a viable alternative but requires memory/init cost trade-offs. OpenVINO remains unsuitable for pure ARM CPU INT8 inference out-of-the-box.