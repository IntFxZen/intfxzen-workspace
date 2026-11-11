### 📊 OpenVINO Performance Comparison (YOLO26n)

| Metric / Model | 🍏 macOS (M1) | 🤖 Android (Snapdragon 860) | 🐧 Linux (AWS Graviton4) |
| :--- | :--- | :--- | :--- |
| **Device (CPU)** | Apple M1 | Snapdragon 860 | ARM Neoverse-V2 (64 cores) |
| **Architecture** | ARM64 | ARM64 | ARM64 (SVE) |
| **YOLO26n (FP32)** | | | |
| *Latency (Median)* | **72.77 ms** | **633.73 ms** | **78.85 ms** |
| *Throughput (FPS)* | **68.45 FPS** | **3.14 FPS** | **12.68 FPS** |
| | | | |
| **YOLO26n (INT8)** | | | |
| *Latency (Median)* | **110.62 ms** | ❌ *Error* | **66.22 ms** |
| *Throughput (FPS)* | **50.93 FPS** | ❌ *Error* | **15.09 FPS** |
| *INT8 Status* | Slower than FP32 (no hardware instructions) | `object is not initialized` | Minor boost (stream tuning required) |

***

> *"Server note: in `-hint latency` mode, the 64-core Graviton4 is limited to 1 stream, so 63 cores sit idle, and synchronization overhead reduces the overall FPS. This is an architectural feature of Latency mode — it is optimized for minimum latency of a single request, not for maximum throughput. In Throughput mode, the server delivers 13.49 FPS (FP32) and 16.16 FPS (INT8), but the standard benchmark_app on ARM does not allow all 64 cores to be utilized without manual plugin tuning."*

**2. Additional study: Native Linux (Fedora) vs WSL2**
*(Measurements with threads strictly fixed at `6` via `-nthreads 6`). Both environments were tested on the same hardware: Intel Core i7-13700H.*

| Environment | Model | Latency (Median) | Throughput | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Fedora (Native)** | FP32 | 13.94 ms | **71.18 FPS** | No overhead |
| **WSL2 (Ubuntu)** | FP32 | **13.22 ms** | 63.15 FPS | Difference within margin of error |
| **Fedora (Native)** | INT8 | **6.69 ms** | **147.64 FPS** | Maximum core utilization |
| **WSL2 (Ubuntu)** | INT8 | 8.76 ms | 108.02 FPS | Hyper-V overhead with INT8 |

**Virtualization conclusion:**
WSL2 is excellent for local development and code debugging (FP32 losses are minimal). However, with heavy INT8 computations, Hyper-V overhead becomes noticeable (~27% FPS drop), since the virtual machine passes through CPU-specific instructions less efficiently.

***