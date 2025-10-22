## Kronos Inference Optimization Benchmarks

Synthetic workload: `batch_size=2`, `context_length=64`, `prediction_length=16`, `max_context=128`, `clip=5.0`, `sample_count=1`. Relative MAE is measured against the eager FP32 baseline using identical RNG seeds for deterministic multinomial sampling.

### Benchmark run (device=cuda, torch=2.8.0+cu129, seed=42, batch=2, context=64, pred=16)

| Strategy | Latency (ms) | Speedup | Mean |Δ| | Max |Δ| | Rel MAE % | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_fp32 | 75.302 | 1.00x | 0.000e+00 | 0.000e+00 | 0.00% | ok | Reference configuration. |
| inference_mode | 74.689 | 1.01x | 0.000e+00 | 0.000e+00 | 0.00% | ok |  |
| flash_sdp | 69.238 | 1.09x | 1.364e-05 | 4.828e-05 | 0.01% | ok | Forces FlashAttention-style kernels via torch.backends.cuda.sdp_kernel. |
| bf16_autocast | 81.596 | 0.92x | 9.543e-04 | 3.494e-03 | 0.42% | ok | BF16 autocast for Ampere+ GPUs. |
| bf16_compile_reduce_overhead | 44.114 | 1.71x | 9.543e-04 | 3.494e-03 | 0.42% | ok | Compiles decode_s1/decode_s2 graph; excludes compile time from latency. |
| bf16_compile_max_autotune | 42.868 | 1.76x | 9.543e-04 | 3.494e-03 | 0.42% | ok | Aggressive autotuning for kernels. |


### Benchmark run (device=cuda, torch=2.8.0+cu129, seed=42, batch=2, context=64, pred=16)

| Strategy | Latency (ms) | Speedup | Mean |Δ| | Max |Δ| | Rel MAE % | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_fp32 | 358.467 | 1.00x | 0.000e+00 | 0.000e+00 | 0.00% | ok | Reference configuration. |
| inference_mode | 363.390 | 0.99x | 0.000e+00 | 0.000e+00 | 0.00% | ok |  |
| flash_sdp | 349.792 | 1.02x | 1.364e-05 | 4.828e-05 | 0.01% | ok | Forces FlashAttention-style kernels via torch.backends.cuda.sdp_kernel. |
| bf16_autocast | 357.805 | 1.00x | 9.543e-04 | 3.494e-03 | 0.42% | ok | BF16 autocast for Ampere+ GPUs. |
| fp32_compile_max_autotune_flash | 43.610 | 8.22x | 1.364e-05 | 4.828e-05 | 0.01% | ok | Compiled FP32 path; retains full precision while enabling FlashAttention. |
| bf16_compile_reduce_overhead | 71.791 | 4.99x | 9.543e-04 | 3.494e-03 | 0.42% | ok | Compiles decode_s1/decode_s2 graph; excludes compile time from latency. |
| bf16_compile_max_autotune | 67.827 | 5.28x | 9.543e-04 | 3.494e-03 | 0.42% | ok | Aggressive autotuning for kernels. |
