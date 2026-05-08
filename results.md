Checkpoint found, skipping training.
Quantizing model...
Quantization complete.
Loading FP32 model...
Running quantized inference...
Quantized accuracy vs FP32: 72.00%
Systolic array output matches numpy: ✓
Operation | Cycles | % of total

---

linear:blocks.0.attn.qkv.weight | 8832 | 10.29%
attn_qk:layer0:head0 | 736 | 0.86%
attn_qk:layer0:head1 | 736 | 0.86%
attn_qk:layer0:head2 | 736 | 0.86%
attn_qk:layer0:head3 | 736 | 0.86%
attn_pv:layer0:head0 | 736 | 0.86%
attn_pv:layer0:head1 | 736 | 0.86%
attn_pv:layer0:head2 | 736 | 0.86%
attn_pv:layer0:head3 | 736 | 0.86%
linear:blocks.0.attn.proj.weight | 2944 | 3.43%
linear:blocks.0.mlp.0.weight | 11776 | 13.71%
linear:blocks.0.mlp.2.weight | 11776 | 13.71%
linear:blocks.1.attn.qkv.weight | 8832 | 10.29%
attn_qk:layer1:head0 | 736 | 0.86%
attn_qk:layer1:head1 | 736 | 0.86%
attn_qk:layer1:head2 | 736 | 0.86%
attn_qk:layer1:head3 | 736 | 0.86%
attn_pv:layer1:head0 | 736 | 0.86%
attn_pv:layer1:head1 | 736 | 0.86%
attn_pv:layer1:head2 | 736 | 0.86%
attn_pv:layer1:head3 | 736 | 0.86%
linear:blocks.1.attn.proj.weight | 2944 | 3.43%
linear:blocks.1.mlp.0.weight | 11776 | 13.71%
linear:blocks.1.mlp.2.weight | 11776 | 13.71%
linear:head.weight | 3440 | 4.01%

---

Total cycles: 85872
Note: Python simulation wall-clock is slower than CPU; use cycle-based latency.
Estimated latency at 12MHz: 7.16 ms
Running CPU baseline (PyTorch, CPU only)...
Running simulated accelerator benchmark...
| Metric | CPU (M1 Max) | FPGA Sim (12MHz) | FPGA Sim (100MHz) |
|---------------------|---------------|------------------|-------------------|
| Latency (ms) | 0.42 | 7.16 | 0.86 |
| Throughput (inf/s) | 2365.79 | 139.74 | 1164.52 |
| Cycle count | N/A | 85872 | 85872 |
| Accuracy | baseline | 72.00% | 72.00% |
Plots saved to accelerator/benchmark/plots
Pipeline complete.
