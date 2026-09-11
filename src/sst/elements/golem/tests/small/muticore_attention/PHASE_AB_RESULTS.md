# Attention Phase A/B Results

## Status

- Phase A, SST timing contract: complete.
- Phase B, SST critical-path and work breakdown: complete.
- Phase B, GPU FP32 stage samples: complete and imported from the RTX 5060 host.

The raw RTX result in `baseline/gpu_attention_rtx5060.json` contains 200 Scope A
samples, 200 samples for every stage and stage-chain latency, and 1000 empty-Event
samples. The report accepted it as measured evidence after recomputing medians and
checking the same-stream timing contract.

The top-level CUDA Event and default-stream fields apply to both clean Scope A and
the stage-chain run. Scope A synchronizes its end Event once per sample to confirm
completion; synchronization is outside the Event elapsed interval. The stage chain
separately records its consecutive-Event/one-final-synchronize method because its
internal boundaries are used for stage attribution.

## Timing Contract

The frozen runs use an SST timebase of `1e12 ticks/s`. Reported normalized cycles
are calculated at `1 GHz`; they are nanosecond-equivalent latency units, not native
cycles of either device.

The current model records these clocks independently:

| Domain | Frequency |
|---|---:|
| Vanadis CPU, RoCC, SFU, Local GM | 2.3 GHz |
| MVM array | 2.3 GHz |
| Memory controller | 2.3 GHz |
| Archive platform/NodeOS side | 2.0 GHz |
| Comparison normalization | 1.0 GHz |

Phase A deliberately records this historical mixed-clock configuration without
changing it, so the E3/E4 frozen baseline remains valid.

## SST End-To-End Results

| Profile | Raw SST ticks | Latency | Normalized cycles | Frontier coverage |
|---|---:|---:|---:|---:|
| E3 | 3,561,011,377 | 3.561011377 ms | 3,561,012 | 100% |
| E4 | 14,169,967,351 | 14.169967351 ms | 14,169,968 | 100% |

The system frontier partitions descriptor accept through tensor completion into
non-overlapping intervals with zero unattributed ticks. Its final-QK interval still
contains earlier online QK/Softmax/PV tiles, so it is a lifecycle view rather than
an operator-cost view.

## Slowest-Worker Work Breakdown

| Profile | Input movement | QK | Scale + Softmax | PV | Coverage of completion |
|---|---:|---:|---:|---:|---:|
| E3 | 0.176364 ms (4.96%) | 0.510464 ms (14.35%) | 0.031367 ms (0.88%) | 2.838528 ms (79.81%) | 99.88% |
| E4 | 0.637605 ms (4.50%) | 2.041856 ms (14.42%) | 0.125447 ms (0.89%) | 11.358208 ms (80.20%) | 99.95% |

These are non-overlapping accumulated phases on core 19, the slowest worker in
both runs. They must not be summed across the 16 parallel workers.

The dominant detailed phase is PV matrix programming:

| Profile | PV matrix programming | Share of slowest-worker work |
|---|---:|---:|
| E3 | 2.379776 ms | 66.91% |
| E4 | 9.519104 ms | 67.21% |

QK matrix programming contributes about 7.43%/7.46%, and PV input programming
about 5.53%/5.55%. The nearly identical E3/E4 proportions show that the current
bottleneck scales structurally with sequence length rather than appearing as a
small-profile fixed overhead.

## GPU Results

The measured FP32 Scope A results are:

| Profile | RTX 5060 FP32 | SST/GPU latency ratio |
|---|---:|---:|
| E3 | 0.097568 ms | 36.50x |
| E4 | 0.345984 ms | 40.96x |

The same-stream consecutive-Event comparison is:

| Profile | Stage | SST worker work | GPU stage | SST/GPU gap |
|---|---|---:|---:|---:|
| E3 | QK | 0.510464 ms | 0.041472 ms | 12.31x |
| E3 | Scale + Softmax | 0.031367 ms | 0.019136 ms | 1.64x |
| E3 | PV | 2.838528 ms | 0.036640 ms | 77.47x |
| E4 | QK | 2.041856 ms | 0.159744 ms | 12.78x |
| E4 | Scale + Softmax | 0.125447 ms | 0.095840 ms | 1.31x |
| E4 | PV | 11.358208 ms | 0.098304 ms | 115.54x |

The GPU stage-chain medians are `0.097888 ms` for E3 and `0.352416 ms` for E4.
The empty CUDA Event median is `0.002400 ms`, so E3 is about 40.7 times above the
timer floor. The generated comparison reports `gpu_stage_status: measured`.

## Decision For Phase C

The first idealized experiment must target matrix programming, especially PV.
The recommended order is:

1. ideal PV matrix programming;
2. ideal all matrix programming, including QK;
3. ideal PV input/restore/output movement;
4. ideal HBM/Local GM input movement;
5. ideal SFU/Softmax;
6. ideal control and combined lower bound.

Softmax and MPI are not current architecture-latency priorities. A real dataflow
redesign should be selected only after the ideal PV programming bound is known.

## Reproduction

The original GPU measurement can be repeated on the RTX machine:

```bash
python3 gpu_attention_stage_benchmark.py \
  --profiles e3 e4 --warmup 50 --iterations 200 \
  --output gpu_attention_rtx5060_stages.json
```

Then generate the combined report in this repository:

```bash
python3 report_attention_gpu_comparison.py \
  --sst-e3 baseline/e3/lifecycle.json \
  --sst-e4 baseline/e4/lifecycle.json \
  --gpu baseline/gpu_attention_rtx5060.json \
  --result-json baseline/attention_gpu_comparison.json
```
