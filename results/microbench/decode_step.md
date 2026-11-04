# Decode step time: fused C++ elementwise kernels on/off

CPU: Intel(R) Xeon(R) Processor, 4 threads, fp32, 300-token prompts, median of 60 steps (ms).

| model | batch | plain | fused | change |
|---|---:|---:|---:|---:|
| smollm2-135m | 1 | 20.0 | 16.4 | -18% |
| smollm2-135m | 4 | 25.0 | 20.6 | -18% |
| smollm2-135m | 8 | 30.8 | 26.3 | -15% |
| qwen3-0.6b | 1 | 58.4 | 55.2 | -5% |
| qwen3-0.6b | 4 | 74.5 | 67.5 | -9% |
| qwen3-0.6b | 8 | 96.9 | 90.4 | -7% |
