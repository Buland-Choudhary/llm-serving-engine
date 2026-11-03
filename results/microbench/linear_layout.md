# Linear-layer backends on CPU (cold weights)

Intel(R) Xeon(R) Processor @ 2.80GHz, torch 2.15.0+cpu, 4 threads, fp32, git 78f4c03-dirty. Milliseconds per call, averaged while cycling through >= 256 MB
of distinct weights so they stream from DRAM as in a real forward pass.

| layer | rows (M) | F.linear [out,in] | x @ W_t [in,out] | oneDNN packed | best vs F.linear |
|---|---:|---:|---:|---:|---:|
| smollm2-135m gate_up | 1 | 0.330 | 0.430 | 0.326 | 1.01x |
| smollm2-135m gate_up | 4 | 0.618 | 0.525 | 0.366 | 1.69x |
| smollm2-135m gate_up | 16 | 1.142 | 0.641 | 0.531 | 2.15x |
| smollm2-135m gate_up | 64 | 1.609 | 1.404 | 0.947 | 1.70x |
| smollm2-135m gate_up | 256 | 2.718 | 2.991 | 2.973 | 1.00x |
| smollm2-135m down | 1 | 0.160 | 0.148 | 0.243 | 1.08x |
| smollm2-135m down | 4 | 0.269 | 0.329 | 0.219 | 1.23x |
| smollm2-135m down | 16 | 0.600 | 0.292 | 0.314 | 2.06x |
| smollm2-135m down | 64 | 0.690 | 0.625 | 0.564 | 1.22x |
| smollm2-135m down | 256 | 1.457 | 1.943 | 1.470 | 1.00x |
| qwen3-0.6b qkv | 1 | 0.610 | 0.660 | 0.643 | 1.00x |
| qwen3-0.6b qkv | 4 | 1.006 | 0.990 | 0.605 | 1.66x |
| qwen3-0.6b qkv | 16 | 2.853 | 1.685 | 1.084 | 2.63x |
| qwen3-0.6b qkv | 64 | 4.562 | 3.194 | 1.655 | 2.76x |
| qwen3-0.6b qkv | 256 | 6.055 | 6.587 | 5.808 | 1.04x |
| qwen3-0.6b gate_up | 1 | 0.831 | 0.899 | 0.960 | 1.00x |
| qwen3-0.6b gate_up | 4 | 1.564 | 1.641 | 0.988 | 1.58x |
| qwen3-0.6b gate_up | 16 | 4.603 | 2.402 | 1.300 | 3.54x |
| qwen3-0.6b gate_up | 64 | 5.824 | 3.642 | 2.663 | 2.19x |
| qwen3-0.6b gate_up | 256 | 9.435 | 11.700 | 8.922 | 1.06x |
| qwen3-0.6b down | 1 | 0.589 | 0.465 | 0.554 | 1.27x |
| qwen3-0.6b down | 4 | 0.806 | 1.060 | 0.587 | 1.37x |
| qwen3-0.6b down | 16 | 2.427 | 1.818 | 0.831 | 2.92x |
| qwen3-0.6b down | 64 | 3.341 | 2.327 | 1.918 | 1.74x |
| qwen3-0.6b down | 256 | 7.131 | 7.459 | 5.574 | 1.28x |
