# Quantized linear kernels by number of rows (CPU)

Intel(R) Xeon(R) Processor @ 2.80GHz, torch 2.15.0+cpu, 4 threads, git 8741366-dirty.
Layer shape 1024 -> 6144 (Qwen3-0.6B gate/up), ms per call, weights cycled from DRAM.

| rows | fp32 oneDNN | int8 aten kernel | int8 llmserve GEMV | int8 (dispatch) | int4 aten kernel | int4 (llmserve) | W8A8 _int_mm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.17 | 3.59 | 0.39 | 0.71 | 7.42 | 0.51 | 0.86 |
| 4 | 0.95 | 10.72 | 1.01 | 0.89 | 33.29 | 2.96 | 1.30 |
| 16 | 1.53 | 39.86 | 3.22 | 1.23 | 125.77 | 8.12 | 1.06 |
| 64 | 2.37 | 150.86 | 15.67 | 2.26 | 430.88 | 50.13 | 2.06 |
| 256 | 8.85 | 599.08 | 594.57 | 4.31 | 1677.58 | 57.73 | 7.07 |
| 1024 | 30.82 | 2213.04 | 2484.66 | 31.08 | 6289.02 | 65.22 | 28.49 |
