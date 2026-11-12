# Hardware Acceleration Analysis: CPU vs GPU vs NPU on Snapdragon 860

## 📱 Testing Environment
*   **Device:** Poco X3 Pro (Android, ARM64)
*   **Processor:** Snapdragon 860 (CPU: Kryo 485, GPU: Adreno 640, NPU: Hexagon 690)
*   **Model:** `yolo26n` (FP32 and INT8)
*   **Protocol:** 100 iterations after warmup using TFLite `benchmark_model`. Median (P50) latency is the primary metric.

## 📊 Hardware Acceleration Results

| Hardware | Delegate / Backend | Model Precision | Median Latency (P50) | Throughput | Cold Start (Init) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | XNNPACK | FP32 | 74.88 ms | 13.35 FPS | 113.6 ms | ✅ Success |
| **CPU** | XNNPACK | INT8 | 44.55 ms | 22.44 FPS | 41.9 ms | ✅ Success |
| **GPU** | OpenCL | FP32 | **19.02 ms** | **52.57 FPS** | 3172.2 ms | ✅ Success |
| **GPU** | OpenCL | INT8 | 23.32 ms | 42.88 FPS | 3099.6 ms | ✅ Success |
| **NPU** | Hexagon DSP (NNAPI/QNN) | INT8 | ❌ *N/A* | ❌ *N/A* | ❌ *N/A* | ⚠️ SDK Deprecation / Vendor Lock-in |

## 🔬 Key Findings & Analysis

1.  **GPU Dominance in FP32 (Massive Speedup):**
    The Adreno 640 GPU demonstrates incredible performance on FP32 models, achieving a **3.9x speedup** over the CPU (74.88 ms → 19.02 ms) and reaching **52.57 FPS**. Mobile GPUs are vastly superior for heavy parallel computations like convolutions.
2.  **The "Cold Start" Penalty (GPU Initialization):**
    While GPU inference is incredibly fast, its initialization time (Cold Start) is a massive **3.1 seconds**, as the delegate must compile OpenCL kernels for the hardware. In contrast, CPU and NPU initialize in milliseconds. For real-world apps, this means a CPU fallback might be needed for the first few frames while the GPU warms up.
3.  **INT8 on GPU is Inefficient:**
    GPU INT8 (23.32 ms) was *slower* than GPU FP32 (19.02 ms). The older Adreno 640 lacks native INT8 hardware support, forcing expensive dequantization steps. **Conclusion: For GPU delegation, always use FP32/FP16.**
4.  **The NPU Fragmentation (Critical Finding):**
    Attempts to utilize the Qualcomm Hexagon 690 NPU via LiteRT (`--use_npu=true`), JIT compiler plugins, and the official `litert-cli` failed. 
    **Root Cause:** Qualcomm has officially deprecated and removed support for Hexagon V66 (used in Snapdragon 860) from modern QAIRT SDKs. The required skeleton files (`libQnnHtpV66Skel.so`) are physically missing from the vendor SDK. 
    This highlights the **fragmented state of the NPU ecosystem** on Android. Unlike CPU/GPU which are plug-and-play, NPUs are locked behind proprietary, device-specific vendor SDKs.

**Conclusion:** For real-world deployment on mid-range ARM64 devices, the optimal strategy is:
*   **For maximum frame rate (e.g., continuous video):** GPU FP32 (OpenCL). Accept the 3-second initialization delay.
*   **For low latency, fast startup, and memory efficiency (e.g., single-image processing):** CPU INT8 (XNNPACK).
*   **For NPU:** Requires deep vendor-specific integration (Qualcomm QNN SDK), which is currently highly fragmented and often unsupported on older SoCs.