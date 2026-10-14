# Fixed-resource operator fusion

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

This execution plan composes existing timed SFU vector operations, local-GM
accesses and projection-array commands. It does not change array count, operand
banks, output contexts, SRAM capacity, ports, bandwidth, DMA credits or NoC
links. Independent RMSNorm, projection and Attention paths remain available.

## Implemented boundaries

1. **RMSNorm -> QKV projection.** Normalize one 256-token block into the
   existing input-block region of manager GM. The paired-weight traversal
   consumes this same block across all Q/K/V heads. Gamma is loaded once from HBM into
   existing scratch; every batch still pays a timed GM gamma read. Normalization uses the existing SFU reduction,
   epsilon and FP16 output rounding. Production mode eliminates normalized
   output HBM writeback and the subsequent projection input read.
2. **Projection -> RoPE -> packed Q/K panels.** After the last hidden tile,
   quantize projection output to FP16, execute the existing SFU RoPE operation,
   wait for its completion, and pack FP16 rotated panels. The existing scratch
   holds one 256-position coefficient block and one array-width result. QK
   consumes the panels without a second rotation. V retains its transpose
   layout. All new local transfers use the existing timed GM API, split at its
   request limit and retry queue backpressure.
3. **Consumer layout only.** Optional production mode omits raw Q/K/V and
   fused-normalization audit writebacks. Packed Q/K/V and final Attention O
   remain in HBM. Audit mode retains raw tensors for independent intermediate
   validation. This is optional materialization, not a tensor-precision change.

The existing QK -> online softmax -> P FIFO -> PV dataflow is retained.

## Explicitly retained boundaries

The current Attention descriptor accepts a complete HBM tensor job; it does
not describe resident Q panels or per-block readiness. Therefore this change
retains the four-manager projection-completion synchronization and Q/K/V panel
HBM transport. It does not add dedicated ready messages, direct-Q FIFOs,
manager/worker bypass links or four array output contexts. Block-level early
Attention and direct-Q transfer remain outside this fixed-interface plan.
Wo, residual and MLP remain outside the workload.

## Storage ownership

For Hq32/Hkv8/D64/hidden2048, each manager retains its existing 2 MiB GM and
1 MiB WCP C-buffer. With scratch base GM+0x8000, the relevant absolute offsets
within GM are:

| Existing GM region | Begin | End, exclusive | Maximum payload |
| --- | ---: | ---: | ---: |
| RoPE coefficient block | 0x28000 | 0x30000 | 32 KiB |
| RoPE result/packing scratch | 0x38000 | 0x3a000 | 8 KiB |
| RMSNorm batch input | 0x48000 | 0x4c000 | 16 KiB |
| Gamma | 0x58000 | 0x59000 | 4 KiB |
| Weight cache | 0x68000 | 0xe8000 | 512 KiB |
| Existing GM partial region | 0xe8000 | 0xf0000 | 32 KiB |
| Normalized input block | 0xf0000 | 0x1f0000 | 1 MiB |
| Existing transfer-snapshot budget | 0x1f0000 | 0x1f8000 | 32 KiB |

The final 64 GM bytes remain reserved for DMA status. Fusion scratch is within
previously reserved scratch below the weight cache, not newly allocated SRAM.
The host vectors represent bounded payloads of those regions and existing
array/SFU operations. The one-output-context array architecture is unchanged.
C-buffer partial prefetch remains a separate opt-in baseline optimization,
using 32 KiB of the existing WCP buffer; Attention begins after it is released.

This initial fusion plan requires the existing D64 paired-weight/shared-input
traversal. Unsupported fusion shapes or incompatible row-first/shared-input
options fail explicitly. Unfused fallback paths remain available.

## Controls and validation

- `GOLEM_FUSION_NORM_PROJECTION=1`: normalize directly into the projection block.
- `GOLEM_FUSION_PROJECTION_ROPE=1`: produce rotated Q/K panels.
- `GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1`: omit intermediate audit materialization.

All three default to zero. They are forwarded explicitly to both the host
controller and simulated RISC-V process. The report verifies requested modes
against measured execution, four managers' work/traffic counts and completion.
The projection software descriptor is extended to carry coefficient addresses
and execution-plan flags; the existing command opcode and physical hardware
interfaces are retained. Rebuild both the guest and host library.

Audit layout checks compare normalized/raw projections with the NumPy reference
and rotated panels with the FP16 rotation of the actual raw outputs. Production
checks compare packed panels and final O with the numerical reference. The
reference comparator additionally checks unchanged regions and O byte for byte,
and compares rotated panels byte for byte with rotation of control-run raw Q/K.
Reference inputs can be reused only after shape/PASS and recorded SHA256 checks;
HBM state and simulator outputs are regenerated for every case.

```bash
source scripts/env_local_install.sh
scripts/build_and_install_local.sh --no-autogen --jobs 8
GOLEM_PROJECTION_CBUFFER_PARTIAL=1 \
  python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_fusion_control
GOLEM_PROJECTION_CBUFFER_PARTIAL=1 GOLEM_FUSION_NORM_PROJECTION=1 \
  GOLEM_FUSION_PROJECTION_ROPE=1 GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1 \
  python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_fusion_production \
  --reference-root results/fresh_fusion_control
```

## Cycle interpretation

Projection's reported span includes fused RMSNorm/RoPE when enabled. The
RMSNorm span can include projection work between successive input blocks;
`active_cycles_by_manager` reports normalization batch residence separately.
Do not add these overlapping spans. E2E starts at the first measured RMSNorm
issue in unfused mode and projection execution-plan start in fused mode, and
ends at the existing final Attention/PV completion boundary.

An early manager may start Attention after observing all producer flags before
the last manager finishes its own polling. Reporting checks per-manager
barrier-before-descriptor order; it does not impose a nonexistent global
barrier between all descriptor accepts and the first worker start.

SST accelerator cycles use the same 1 GHz profile and 66-cycle full-width
array compute. Host input generation and simulator wall time are not latency.
The optimistic resource floor retains the existing RMSNorm/array/exp-work
scope, excluding DMA, NoC, control and RoPE vector work. SFU vector work and
fusion traffic are recorded separately.

To run all five configurations with matched inputs and automatic cycle/traffic
analysis, use a fresh output root:

```bash
python3 baseline/attention_cluster_8qk_8pv/run_operator_fusion.py \
  --artifact-root results/fresh_fusion_matrix \
  --input-reference-root results/fresh_fusion_control
```

On a host with sufficient physical cores, `--cpu-sets 4-7 8-11` assigns two
independent serial queues to disjoint CPU sets. Every case has private HBM and
statistics directories. `analysis.json` and `analysis.csv` contain E2E, savings,
stage spans, vector work, fusion traffic and numerical error.

The legacy unfused RMSNorm job reads/writes its SFU operands through synchronous
GM copies after/before timed DMA. Fusion explicitly charges timed local operand
reads and result writes. The comparison is against that existing executable
path, not a claim that all legacy operand-copy timing has been repaired.

## Measured results

All ten cases passed with matched inputs: FP16, causal GQA, Hq32/Hkv8/D64,
hidden2048, seed1742. Every configuration uses the same C-buffer partial
prefetch setting. These are accelerator E2E cycles, not complete decoder-layer
cycles. The archived machine-readable results are in
[`operator_fusion_results.json`](operator_fusion_results.json); full logs, HBM
outputs and reports are under `results/operator_fusion_final`.

| Execution plan | S1024 cycles | Reduction | S2048 cycles | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Control | 1,485,688 | — | 3,414,999 | — |
| RMSNorm + projection, audit | 1,442,054 | 2.94% | 3,326,136 | 2.60% |
| RMSNorm + projection, production | **1,418,196** | **4.54%** | **3,278,429** | **4.00%** |
| RMSNorm + projection + RoPE, audit | 1,473,539 | 0.82% | 3,396,037 | 0.56% |
| RMSNorm + projection + RoPE, production | 1,449,587 | 2.43% | 3,347,531 | 1.98% |

The recommended opt-in plan for this workload is
`GOLEM_FUSION_NORM_PROJECTION=1`, `GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1`,
`GOLEM_FUSION_PROJECTION_ROPE=0`. Defaults remain unchanged. RoPE fusion is
available for comparison but is not the fastest plan.

Moving RoPE to projection reduces rotated elements from 4,718,592 to 2,621,440
at S1024, and from 9,502,720 to 5,242,880 at S2048. However, worker RoPE originally
largely overlaps the Attention pipeline. The fused producer must finish its
RoPE before releasing the complete Attention job. Compared with norm-only
production, full production changes the critical path as follows:

| Critical-path segment | S1024 delta cycles | S2048 delta cycles |
| --- | ---: | ---: |
| Producer, including normalization | +38,493 | +77,425 |
| Producer-to-Attention handoff | -1,349 | -2,935 |
| Attention | -5,753 | -5,388 |
| E2E | **+31,391** | **+69,102** |

Each manager adds 20,960 / 41,920 RoPE vector cycles and 3.75 / 7.5 MiB of local
GM read/write traffic. Less total arithmetic therefore does not imply lower
latency with the retained whole-job synchronization. No hardware bypass or
extra output context was added to conceal this cost.

### Verification and timing evidence

- All eight fusion cases produced final O byte-identical to control. Audit
  comparisons checked 32 regions; production comparisons checked 16 retained
  regions, including every O region. Independent NumPy O checks passed with
  maximum absolute error 0.0001220703125 at both sequence lengths.
- Pre-rotated QK still pays timed GM operand reads. The report checks each
  worker and the exact causal tile byte count, including 71,303,168 bytes at
  S1024. SFU counters confirm workers perform no duplicate RoPE in this mode.
- The control/norm cases were run before a correction that enables timed QK
  reads specifically for pre-rotated panels. Their paths are unchanged by that
  correction. A control repeat with the final library reproduced exactly
  1,485,688 cycles and identical outputs. Per-case binary SHA256 values are
  retained rather than presenting these runs as one identical binary.
- One audit-panel check initially failed on a near-zero FP16 value. The exact
  panel oracle now matches the existing SFU conversion, which flushes values
  below 2^-24 before storage. The simulator arithmetic, numerical tolerances
  and independent NumPy O oracle were unchanged. Existing failure logs remain
  available; the corrected postprocessing checks passed.
- The final Python regression suite passed 130 tests. Shell syntax and diff
  checks passed. An unfused D128 E2E regression also passed (161,993 cycles),
  confirming that the original non-D64 execution path remains usable.

Fusion remains an execution-plan optimization with D64 eligibility checks.
The tests establish correctness for the stated workloads; they do not imply
that every operator combination or shape benefits from fusion.

A subsequent fixed-resource input-prefetch scheduling change improves the best
production plan by another 8.45% / 7.30%. See
[`PROJECTION_SCHEDULE.md`](PROJECTION_SCHEDULE.md) for the exclusive cycle
breakdown and matched-library E2E evidence. The fusion-only results above remain
the original archived comparison.
