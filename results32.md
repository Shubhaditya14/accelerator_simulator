Checkpoint found, skipping training.
Quantizing model...
Quantization complete.
Loading FP32 model...
Running quantized inference...
Quantized accuracy vs FP32: 72.00%
Systolic array output matches numpy: ✓
Operation | Cycles | % of total
----------------------------------------
linear:blocks.0.attn.qkv.weight |     2256 |   9.38%
attn_qk:layer0:head0      |      312 |   1.30%
attn_qk:layer0:head1      |      312 |   1.30%
attn_qk:layer0:head2      |      312 |   1.30%
attn_qk:layer0:head3      |      312 |   1.30%
attn_pv:layer0:head0      |      312 |   1.30%
attn_pv:layer0:head1      |      312 |   1.30%
attn_pv:layer0:head2      |      312 |   1.30%
attn_pv:layer0:head3      |      312 |   1.30%
linear:blocks.0.attn.proj.weight |      752 |   3.13%
linear:blocks.0.mlp.0.weight |     3008 |  12.51%
linear:blocks.0.mlp.2.weight |     3008 |  12.51%
linear:blocks.1.attn.qkv.weight |     2256 |   9.38%
attn_qk:layer1:head0      |      312 |   1.30%
attn_qk:layer1:head1      |      312 |   1.30%
attn_qk:layer1:head2      |      312 |   1.30%
attn_qk:layer1:head3      |      312 |   1.30%
attn_pv:layer1:head0      |      312 |   1.30%
attn_pv:layer1:head1      |      312 |   1.30%
attn_pv:layer1:head2      |      312 |   1.30%
attn_pv:layer1:head3      |      312 |   1.30%
linear:blocks.1.attn.proj.weight |      752 |   3.13%
linear:blocks.1.mlp.0.weight |     3008 |  12.51%
linear:blocks.1.mlp.2.weight |     3008 |  12.51%
linear:head.weight        |     1004 |   4.18%
----------------------------------------
Total cycles: 24044
Note: Python simulation wall-clock is slower than CPU; use cycle-based latency.
Estimated latency at 12MHz: 2.00 ms
Running CPU baseline (PyTorch, CPU only)...
Running simulated accelerator benchmark...
| Metric              | CPU (M1 Max)  | FPGA Sim (12MHz) | FPGA Sim (100MHz) |
|---------------------|---------------|------------------|-------------------|
| Latency (ms)        | 0.42        | 2.00             | 0.24              |
| Throughput (inf/s)  | 2397.03        | 499.09             | 4159.04              |
| Cycle count         | N/A           | 24044            | 24044             |
| Accuracy            | baseline      | 72.00%             | 72.00%              |
Plots saved to accelerator/benchmark/plots
Pipeline complete.
