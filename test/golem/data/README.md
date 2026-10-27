# Data Directory

This directory stores host-side tensor data files for matmul pipeline runs.

Recommended layout:

- `data/a.bin` / `data/b.bin`: input tensors (int32 little-endian)
- `data/c_out.csv` or `data/c_out.bin`: unpacked output tensor

Example:

```bash
python3 gen_sample_tensors.py --m 16 --n 4 --k 16 --a-out data/a.bin --b-out data/b.bin

./run_noc_dma_pipeline.sh \
  --tensor-a data/a.bin \
  --tensor-b data/b.bin \
  --dump-c data/c_out.csv
```

Notes:

- `.bin` uses row-major int32 little-endian.
- `.csv` and `.npy` are also accepted for input tensors.
- Keep generated temporary experiment files under `data/` to avoid polluting the repo root.
