# Attention / GQA / prefill full-timing acceptance

All three public entries accept `--sim-mode full-timing`. Array GEMM skips numerical MACs; SFU, embedding, transport, resource and completion checks remain active. Functional mode remains the default. Numerical GEMM checks and prefill argmax are explicitly skipped in timing mode.

| Case | Functional cycles | Timing cycles | Delta cycles |
|---|---:|---:|---:|
| Attention formal | 12852 | 12852 | 0 |
| GQA formal | 37928 | 37928 | 0 |
| prefill smoke | 469442.625 | 469442.625 | 0.0 |
| prefill formal | 6084167.75 | 6084167.75 | 0.0 |

Attention/GQA use the existing QK/SFU/PV completion boundary. Prefill measures first token-ID DMA send through the final valid-token, full-vocabulary logits output DMA ACK. Host preparation, validation and argmax are excluded.

The formal prefill functional reference is the prior accepted run with identical guest and simulator/memory-library digests. Both modes were rerun for the small prefill regression. Each timing case confirms `functional=0` on all sixteen workers; numerical results are marked `SKIPPED`.

Run from the repository root:

```bash
python3 test/run.py Attention --run -- --sim-mode full-timing
python3 test/run.py GQA --run -- --sim-mode full-timing
python3 test/run.py prefill --run -- --sim-mode full-timing
```

See `result.json` for artifact paths, measurement scopes, library/source hashes and validation notes.
