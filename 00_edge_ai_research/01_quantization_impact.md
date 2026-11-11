# Quantization Impact: FP32 vs INT8 on Android ARM64

## 📱 Testing Environment
*   **Device:** Poco X3 Pro (Android, ARM64)
*   **Processor:** Snapdragon 860
*   **Model:** `yolo26n`
*   **Protocol:** 100 iterations after warmup. Median (P50) latency is the primary metric.

## 📊 FP32 vs INT8 Results

| Framework | Precision | Median Latency | Throughput | Model Size | Memory Usage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TFLite (LiteRT)** | FP32 | 74.88 ms | 13.35 FPS | 9.97 MB | 67.2 MB | ✅ Success |
| **TFLite (LiteRT)** | **INT8** | **44.55 ms** | **22.44 FPS** | **3.05 MB** | **46.1 MB** | ✅ Success |
| **ONNX Runtime** | FP32 | 96.72 ms | 10.13 FPS | 9.93 MB | 145.4 MB | ✅ Success |
| **ONNX Runtime** | **INT8** | **72.84 ms** | **13.73 FPS** | ~2.5 MB | 165.7 MB | ✅ Success |
| **OpenVINO** | FP32 | 661.43 ms | 1.51 FPS | ~10 MB | 131.4 MB | ✅ Success |
| **OpenVINO** | INT8 | ❌ *Error* | ❌ *Error* | ~2.5 MB | N/A | ❌ `object is not initialized` |

## 🔬 Key Findings & Analysis

1.  **TFLite is the Absolute Winner:**
    TFLite's XNNPACK backend demonstrates incredible INT8 optimization, achieving a **1.68x speedup** (74.88 ms → 44.55 ms) and reducing model size by **69%** (9.97 MB → 3.05 MB). Mobile hardware natively excels at integer operations, making INT8 the ideal precision for TFLite.
2.  **ONNX Runtime INT8 Trade-offs:**
    ONNX Runtime shows a moderate speedup (**1.33x**) but comes with noticeable penalties. Initialization time spiked from 78.4 ms to 215.8 ms, and peak memory usage *increased* to 165.7 MB (likely due to dynamic graph rewriting and Quantize/Dequantize node insertion for CPU MLAS backend compatibility). 
3.  **OpenVINO INT8 Failure (Critical Finding):**
    Attempting to load the OpenVINO INT8 model on the ARM64 CPU plugin resulted in a fatal error (`object is not initialized`). This indicates a fundamental limitation in the cross-compiled OpenVINO runtime for Android: the ARM CPU plugin lacks the necessary backend kernels or graph support for INT8 inference on this architecture. *(This is a known pain point when using x86-centric inference engines on edge ARM devices without proper NPU delegation).*

**Conclusion:** For real-world mobile deployment on ARM64 CPUs, TFLite with INT8 quantization is the only viable, production-ready choice among the tested frameworks, offering massive gains in speed, size, and memory. ONNX Runtime is a viable alternative but requires memory/init cost trade-offs. OpenVINO remains unsuitable for pure ARM CPU INT8 inference out-of-the-box.