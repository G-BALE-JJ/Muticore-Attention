# Historical Attention Architectures

These designs are retained for reproducing earlier experiments. The active
architecture and public runner use 8 QK/SFU workers plus 8 PV workers; see
`../baseline/attention_cluster_8qk_8pv/README.md`.

- `attention_sequential_64/`: sequential 64-array worker design and measurements.
- `baseline_attention_cluster_4qk_12pv/`: 4 QK/SFU + 12 PV comparison.
- `reuse_window_flash_attention/`: generic-GEMM reuse-window reference.

The 4:12 and reuse-window scripts run through the shared SST runner using
explicit archived-mode environment settings. Their cycle numbers are historical
comparisons, not the current acceptance target.
