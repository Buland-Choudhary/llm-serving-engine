# Quantized linear kernels by number of rows (CPU)

Intel(R) Xeon(R) Processor @ 2.80GHz, torch 2.15.0+cpu, 4 threads, git 40e975f.
Layer shape 1024 -> 6144 (Qwen3-0.6B gate/up), ms per call, weights cycled from DRAM.

| rows | fp32 oneDNN | int8 aten kernel | int8 llmserve GEMV | int8 (dispatch) | int4 fused kernel | W8A8 _int_mm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.01 | 3.33 | 0.40 | 0.40 | 7.25 | 0.87 |
| 4 | 0.92 | 8.20 | 0.84 | 0.88 | 39.10 | 0.89 |
| 16 | 1.31 | 37.94 | 3.37 | 3.61 | 117.21 | 1.07 |
| 64 | 2.53 | 152.79 | 13.76 | 21.51 | 47.68 | 1.60 |
| 256 | 7.16 | 602.68 | 25.47 | 31.37 | 63.87 | 5.61 |
| 1024 | 37.68 | 2247.45 | 58.95 | 58.32 | 71.37 | 30.13 |
