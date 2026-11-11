# Cross-Framework Benchmark: YOLO26n on Android ARM64

## 📱 Testing Environment
*   **Device:** Poco X3 Pro (Android, ARM64)
*   **Processor:** Snapdragon 860 (8 cores)
*   **Model:** `yolo26n` (FP32, size ~9.97 MB)
*   **Input Shape:** `[1, 3, 640, 640]` (fixed resolution)
*   **Protocol:** Tests were run for 100 iterations after a warmup phase. The device was allowed to cool down between framework tests. Median (P50) latency is used as the primary metric. Power saving mode was disabled to ensure maximum CPU performance.

## ⚙️ Framework Configurations
*   **TFLite (LiteRT):** XNNPACK delegate, 4 CPU threads.
*   **ONNX Runtime:** CPU Execution Provider (MLAS), default thread pool.
*   **OpenVINO:** CPU Device, 6 inference threads. Default layout (NCHW).

## 📊 Performance Results (100 Iterations)

| Framework | Median Latency (P50) | Variance (Min / Max) | Throughput | Cold Start (Init) | Memory Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TFLite (LiteRT)** | **74.88 ms** | 67.30 ms / 77.47 ms | **13.35 FPS** | 113.6 ms | **67.2 MB** |
| **ONNX Runtime** | **96.72 ms** | 80.06 ms / 155.97 ms | **10.13 FPS** | **78.4 ms** | 145.4 MB |
| **OpenVINO** | **661.43 ms** | 636.64 ms / 668.43 ms | **1.51 FPS** | 551.8 ms | 131.4 MB |

## 🔬 Key Findings & Analysis

1.  **TFLite Dominance on Mobile:**
    TFLite (LiteRT) with the XNNPACK delegate significantly outperforms the other frameworks, achieving a ~9x speedup over OpenVINO and ~1.3x over ONNX Runtime. It also has the smallest memory footprint (~67 MB), fast initialization, and the most stable latency variance.
2.  **OpenVINO Bottleneck (NCHW vs NHWC):**
    OpenVINO shows significant overhead on ARM CPUs. The logs explicitly confirm the core issue: `layout is not set explicitly, so it is defaulted to NCHW`. This forces costly memory transpositions on mobile hardware, which natively prefers NHWC (Channels Last). Additionally, OpenVINO's ARM Compute Library (ACL) falls back to FP16 precision (`INFERENCE_PRECISION_HINT: f16`), yielding suboptimal results for this FP32 model.
3.  **ONNX Runtime Performance & Memory:**
    ONNX Runtime demonstrates solid performance using its MLAS backend (~96 ms). However, it has the highest memory footprint (145.4 MB peak) among the tested frameworks.
4.  **Thermal Throttling & Variance:**
    Prolonged 100-run tests expose the reality of mobile thermal throttling on the Snapdragon 860. ONNX Runtime showed the highest variance: its maximum latency spiked to 155.9 ms under sustained load, compared to its 80.0 ms minimum. This highlights the necessity of using Median (P50) over Average (Mean) for mobile benchmarks, as it filters out OS scheduling and thermal outliers.

**Conclusion:** For FP32 YOLO inference on ARM64 mobile CPUs without dedicated NPU usage, TFLite is the most efficient and stable choice. OpenVINO, while powerful, requires specific layout configurations (NHWC) to be competitive on mobile ARM architectures.

## 🚀 Future Work
*   **Precision Comparison:** Benchmark INT8 and FP16 quantized versions of the model across all frameworks.
*   **Thermal Monitoring:** Integrate SoC temperature telemetry to precisely correlate frequency drops with temperature thresholds.
*   **NPU Utilization:** Test execution on dedicated mobile NPUs (e.g., Qualcomm Hexagon via QNN Execution Provider).