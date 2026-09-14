# Attention Cluster

This directory contains the current worker-local Attention cluster contract,
the R6 and R8-R11 performance reports, focused C++ contracts, and compact verification
artifacts. Large simulator run directories are generated locally and ignored.

## Architecture

- 20 cores: four control-only managers and 16 workers.
- 64 physical 64x64 arrays and 64 output CUs per worker.
- Two operand banks per array and 2 MiB modeled FP32 operand capacity per worker.
- Default ownership: arrays 0-15 for QK and 16-63 for PV.
- D128 uses paired D64 paths; Br16/Bc32 is the measured default.
- Timed direct score/P storage, resident O, bounded retries, generation tags,
  cancellation fences, and a 16-lane two-cycle FP32 O FMA.
- Two-slot manager K/V distribution with tagged four-worker coalescing, timed
  64 B/cycle destination writes, acknowledgements, and generation cancellation.
- Demand-paced manager lookahead uses a released distributor slot to stage one
  sequential K/V tile; it adds neither worker buffers nor manager slots.
- One-entry P-row lookahead overlaps the next SFU FIFO read with current-row PV
  array input programming while retaining tag and context checks.
- Independent QK/PV matrix lookahead uses K-ready and full K/V-ready events to
  fill inactive operand banks before the next physical tile consumes them.
- A timed 512 B/cycle PV row drain feeding eight parallel 16-lane resident-O
  banks, with a feature-off segmented fallback.

The cluster path is enabled explicitly with `--attention-cluster`; the legacy
path remains available for feature-off regression.

## Verified Results

| Case | Cycles | Result |
|---|---:|---|
| Q256/K128 | 16,112 | numerical/lifecycle PASS |
| Q256/K1024 | 119,054 | numerical/lifecycle PASS |
| E3 Q1024/K1024 | 127,589 | numerical/lifecycle PASS |
| E4 Q2048/K2048 | 430,131 | numerical/lifecycle PASS |
| Q256/K128 WCP depth 16 | 16,112 | numerical/lifecycle PASS |
| Q256/K128 MPI 2 | 16,112 | numerical/lifecycle/MPI PASS |

RTX 5060 FP32 Scope A is 97,568/345,984 normalized cycles for E3/E4, so the
current SST result remains 1.308x/1.243x slower. This is a verified architecture
baseline, not a GPU-gate success.

## Files

- `R1_R6_OPTIMIZATION_RESULTS.md`: architecture, resource profile, sweeps,
  performance comparison, review, and verification record.
- `R8_KV_DISTRIBUTION_RESULTS.md`: current architecture, A/B results, traffic
  conservation, and updated bottleneck.
- `R9_PV_O_ROW_FUSION_RESULTS.md`: PV-to-O row fusion, A/B results, resource
  conservation, and the prior bottleneck.
- `R10_MANAGER_KV_P_OVERLAP_RESULTS.md`: early manager K/V delivery, P-row
  overlap, phase cycles, traffic conservation, and the prior bottleneck.
- `R11_QK_PV_MATRIX_LOOKAHEAD_RESULTS.md`: K-ready delivery, QK/PV inactive-bank
  preprogramming, A/B results, and the current bottleneck.
- `test_attention_cluster_contract.cpp`: finite resource, generation,
  cancellation, bank lease, score/P, and resident-O contracts.
- `results/r6_final_*/*.json`: compact numerical, lifecycle, and metrics evidence.
- `results/r6_final_manifest.sha256`: hashes for all retained final JSON files.

Validate the retained evidence from this directory:

```bash
(cd results && sha256sum -c r6_final_manifest.sha256)
```

The next simulator task is to reduce the E4 cross-group delivery long tail.
