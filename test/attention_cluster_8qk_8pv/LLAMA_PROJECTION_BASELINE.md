# Llama 32:8/D64 projection and attention baseline

This baseline runs `Hq=32`, `Hkv=8`, `D=64`, `hidden=2048`, GQA group size 4,
and causal prefill at S=1024 and S=2048. Inputs, gamma, and weights are
deterministic synthetic FP16 tensors (NumPy seed 1742). The measured chain
starts at the first RMSNorm issue and ends at the final Attention output DMA
and PV acknowledgment. It includes RMSNorm, Q/K/V projection, RoPE, and
Attention. Wo, residual, and MLP are outside this measurement.

Projection multiplies and accumulates in FP16. Attention uses FP32 internal
accumulation and FP16 storage. These results validate this simulator's
arithmetic contract; they do not measure a pretrained model's accuracy.

| S | RMSNorm cycles | Projection cycles | Attention cycles | End-to-end cycles | Projection share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 187,834 | 4,623,334 | 249,682 | 5,062,292 | 91.3% |
| 2048 | 374,612 | 9,058,312 | 943,175 | 10,379,121 | 87.3% |

Stage handoffs account for the remaining cycles. The optimistic configured
resource floors are 1,873,088 and 4,008,320 cycles (ratios 2.70x and 2.59x).
They exclude memory movement, array programming, control, and handoff; they
are diagnostic model bounds rather than achievable performance targets.

## Reproduce

From the repository root:

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --jobs 8
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --artifact-root results/llama_projection_reproduction
```

Use a fresh artifact root. `--sequences 1024` runs just S=1024. The runner
archives logs, HBM images, numerical checks, MPI placement, resolved config,
Git revision and dirty state, executable hashes, and input hashes. A successful
run emits `summary.json`; it requires evidence from all four projection jobs
and all eight initial RoPE table loads. `--summarize-only` rebuilds the summary
without rerunning simulation. Large artifacts remain ignored by Git; the
compact frozen results are in `llama_projection_baseline.json` beside this file.

S=1024 was run independently twice with the final simulator binary. Both
runs completed in **5,062,292 cycles**. Input-file hashes, stage spans,
projection phase counters, and numerical errors match. Concurrent host runs
can change wall time; the baseline compares simulated cycles.

## Timing audit

The baseline uses four MPI ranks, four managers, eight QK/SFU workers, and eight
PV workers. Local GM is 2 MiB per core, with one read port, one write port,
256 bytes/cycle, one cycle base latency, 4096 byte maximum requests, and a
256 request queue. CPU and array clocks are 1 GHz. The configured array has
one MAC per output CU per cycle and two pipeline cycles: a 64-column MVM
therefore takes 66 array cycles. Both physical operand banks are retained.

| Path | Timing treatment |
| --- | --- |
| HBM input and weight DMA | Existing Ramulator2, NoC, DMA admission, and timed GM landing |
| RMSNorm input/output | Existing SFU vector timing and asynchronous GM reads/writes |
| Projection input and weight reads | Asynchronous GM requests, split to the request limit; queue rejection retries |
| Projection partial/raw staging | Asynchronous GM reads/writes on the D128 reuse path |
| Projection matrix/input/output | Existing array broadcast, scatter/gather, and configured MVM timing |
| RoPE table installation | HBM DMA followed by asynchronous GM reads; DMA and local read spans logged separately |
| QK operands with RoPE | Asynchronous GM payload reads before SFU RoPE and array programming |
| Attention online softmax/PV/O | Existing SFU lanes, P transport, PV panel reads, array service, and output DMA |

Projection retains a bounded 16-row normalized input snapshot, at most 64 KiB
for hidden=2048, after its timed GM read. Each weight program performs a timed
8192-byte GM read even when the weight tile is already cached locally. Byte
accounting verifies input-group bytes plus weight-program bytes independently
from phase counters. Descriptor/control bookkeeping remains functional;
these cycle results describe the configured simulator model.

The previous direct GM accessors only copied backing storage. They did not
charge SRAM service or port contention. Historical projection numbers and
speedups in the main README precede this correction and must be remeasured
before being used as hardware performance claims. The old projection resource
floor also treated a full 64-column MVM as one cycle; the current report reads
the resolved array MAC rate and pipeline depth.

A controlled D128/hidden256 test confirms bandwidth sensitivity. Reducing GM
bandwidth from 256 to 128 bytes/cycle increases each manager's projection
read wait from 24,576 to 46,080 cycles and write wait from 4,992 to 9,088 cycles,
with identical 5,505,024 read bytes and 1,048,576 write bytes. These tests isolate
the projection timing change before the additional QK/table correction.

## Numerical checks

All four nodes compare normalized X and raw Q/K/V with independently generated
NumPy tensors. RMSNorm uses atol=0.002 and rtol=0.005; Q/K/V uses atol=0.0005
and rtol=0.005. Panel layouts match device-produced raw tensors byte for byte,
and each input HBM weight region matches the weight file byte for byte.
Final O is compared with an independent causal GQA/RoPE NumPy computation
using the golden projected tensors, at atol=0.0002 and rtol=0.002. The report
also requires complete QK tile, SFU element, PV window, and MPI counts.

A negative check that changes one golden Q element by 0.02 is rejected by the
Q/K/V checker. Stage maximum errors and output element counts are retained in
the compact baseline JSON.

| S | Max RMSNorm error | Max Q/K/V error | Max O error | O elements checked |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 0.001953125 | 0.0003662109375 | 0.0001220703125 | 2,097,152 |
| 2048 | 0.001953125 | 0.00048828125 | 0.0001220703125 | 4,194,304 |

## Remaining bottlenecks

The true shape uses the row-major projection path. Each 16-row group traverses
32 input tiles per head; only two tiles can remain programmed in the array
banks. The next row group traverses the same weights again. Local caching
avoids most HBM rereads but does not avoid GM reads and array reprogramming.
S=1024 executes 98,304 weight programs across four managers; S=2048 executes
196,608. Neither shape records array weight reuse.

For S=1024, projection takes 4,623,334 cycles, 91.3% of the 5,062,292-cycle
chain. Manager 0 spends 1,671,168 phase cycles in array launch/compute,
1,105,920 in local-GM reads, and 958,464 in matrix programming. It reads
240 MiB locally: 48 MiB of input groups and 192 MiB of repeatedly programmed
weights, while loading only 12 MiB of weights from HBM. Phase counters include
setup and waiting; they are residence times rather than hardware utilization,
and different managers overlap.

The next optimization should target D64/hidden2048 with bounded weight-first
row blocks, accounting for input reads and FP16 partial restore/store costs.
Input tiles should be gathered at tile granularity to avoid rereading an entire
hidden-dimension row group on every reuse iteration. Array compute remains
necessary work; overlap and weight reuse should be evaluated against the
66-cycle launch model and the corrected SRAM timing.

Attention's causal row bands have unequal work. The later query bands execute
more QK tiles and PV windows, and PV requests can wait behind earlier work.
At S=1024, PV worker busy time ranges from 19,328 to 77,312 cycles; received
PV requests wait an average 885.8 cycles and at most 11,779 cycles to start.
After projection reuse, compare load balancing and bounded dynamic PV dispatch
with the frozen baseline. Table startup and the 51-cycle per-manager
synchronization handoff are smaller targets at this shape.

For S=2048 the projection-to-Attention interval is 2,587 cycles, including
235 cycles to the final projection flag barrier, 51 cycles to descriptor
acceptance, and 2,301 cycles to the first measured QK job start. Different
workers start at different times: the first RoPE table's complete DMA-plus-GM
load ranges from 2,316 to 8,342 cycles across workers, including 1,088 GM read
cycles per worker. These overlapping spans cannot be added to the aggregate
handoff interval. Startup is measured before first QK execution; operand
installation and SFU RoPE work during QK execution belong to the Attention
stage.
