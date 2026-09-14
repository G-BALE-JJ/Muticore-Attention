# Attention Cluster

This directory contains the current worker-local Attention cluster contract,
the final R6 performance report, focused C++ contracts, and compact verification
artifacts. Large simulator run directories are generated locally and ignored.

## Architecture

- 20 cores: four control-only managers and 16 workers.
- 64 physical 64x64 arrays and 64 output CUs per worker.
- Two operand banks per array and 2 MiB modeled FP32 operand capacity per worker.
- Default ownership: arrays 0-15 for QK and 16-63 for PV.
- D128 uses paired D64 paths; Br16/Bc32 is the measured default.
- Timed direct score/P storage, resident O, bounded retries, generation tags,
  cancellation fences, and a 16-lane two-cycle FP32 O FMA.

The cluster path is enabled explicitly with `--attention-cluster`; the legacy
path remains available for feature-off regression.

## Verified Results

| Case | Cycles | Result |
|---|---:|---|
| Q256/K128 | 30,607 | numerical/lifecycle PASS |
| Q256/K1024 | 308,481 | numerical/lifecycle PASS |
| E3 Q1024/K1024 | 330,125 | numerical/lifecycle PASS |
| E4 Q2048/K2048 | 1,007,628 | numerical/lifecycle PASS |
| Q256/K128 WCP pressure | 31,657 | numerical/lifecycle PASS |

RTX 5060 FP32 Scope A is 97,568/345,984 normalized cycles for E3/E4, so the
current SST result remains 3.384x/2.912x slower. This is a verified architecture
baseline, not a GPU-gate success.

## Files

- `R1_R6_OPTIMIZATION_RESULTS.md`: architecture, resource profile, sweeps,
  performance comparison, review, and verification record.
- `test_attention_cluster_contract.cpp`: finite resource, generation,
  cancellation, bank lease, score/P, and resident-O contracts.
- `results/r6_final_*/*.json`: compact numerical, lifecycle, and metrics evidence.
- `results/r6_final_manifest.sha256`: hashes for all retained final JSON files.

Validate the retained evidence from this directory:

```bash
(cd results && sha256sum -c r6_final_manifest.sha256)
```

The next simulator task is R7-D: make every score/SFU/PV operand-bank consumer
explicit in the generation-tagged lease contract before retrying QK producer
overlap. The next larger hardware candidate is a bounded four-worker K/V
distribution/reassembly engine.
