# Native LLM operator and timing-entry updates

The preceding Attention/GQA/prefill full-timing work was committed and pushed
first as `c36ec21`. This follow-up unifies the four public mode selectors,
repairs the same-build timing-control runner, expands paired regressions, and
connects opt-in native matrix workloads to the production TE / NoC / HBM path.

## Contracts

All accelerator measurements use cycles. Each functional/timing pair uses the
same installed compute and memory libraries and identical shapes, inputs and
resources. Timing skips array numerical MAC and numerical materialization;
transfer, compute-completion, dependency and final write-ACK events still run.
Timing numerical verification is SKIPPED.

The default full decoder retains its selected FP16 arithmetic and guest ABI.
Native operator experiments cover QKV, WO, gate, up, down and LM-head. They
explicitly select INT4, INT8, FP16, BF16, TF32, FP32 or either FP8 mode, with
INT32/FP32 accumulation/output. They retain four managers, sixteen workers,
64 arrays per worker and four data HBM nodes. Independent instances partition
these existing workers/nodes and receive distinct inputs. Placement supports
group-affine/striped panels and optional B replication.

Native tests use exact representable -2..2 inputs across all modes. External
native integer or FP16/FP32 tensors are supported for a single instance;
FP8/BF16/TF32 currently have a restricted validation encoder rather than a
general model quantizer. Native dimensions are padded to 64 and padded work
is timed. LM-head's one logical row is padded to 64 rows. Its cycles therefore
do not measure the selected decoder's optimized last-token LM-head. This
operator entry does not perform full-decoder eight-precision inference.

## Bugs exposed by expanded validation

The asymmetric GQA case Sq=512, Skv=1024, Hq=4, Hkv=2, D=128 initially failed
functional verification for heads 2/3. Packed Q-panel addressing incorrectly
used the KV rows per HBM node for its Q-head stride. Q addressing now uses
the independent Q band length in both the RoCC header and TE panel mapping;
K addressing continues to use the KV band length. The regression matrix also
checks the reverse asymmetry, Sq=1024 and Skv=512.

A formal GQA attempt reached 37928 cycles and completed all sixteen jobs,
but MPI console forwarding split one QK summary line across another rank's
trace. The lifecycle parser correctly rejected incomplete evidence. The
pipeline now captures independent rank streams, waits for MPI to close them,
and merges them before generating reports. Raw console output is preserved.
The remaining prefill in that superseded gate was stopped; that gate is not
counted as passing evidence.

Early native probes also rejected mismatched planned/physical HBM capacity
and incompatible inherited Attention lookahead defaults. The native entry now
sets its HBM capacity explicitly and uses the GEMM-compatible KV buffer flags.

## Default functional gates

All four formal functional gates passed on the final main-directory build:

| Stage | Cycles | Delta from selected baseline |
| --- | ---: | ---: |
| GEMM | 3500 | 0 |
| Attention | 12852 | 0 |
| GQA | 37928 | 0 |
| Full prefill | 6084167.75 | 0 |

The prefill boundary is the first token-ID DMA send through the final valid
last-token full-vocabulary logits write ACK; host argmax is excluded. Its
original selected guest hash is retained. Numerical correctness, lifecycle,
GM resource accounting, final norm, embedding and LM-head checks passed.
The public library pin now points to this validated build.

## Completed paired validation

The ten general functional/timing pairs passed with identical cycles and
completed-work signatures. They cover K=64/192, D64/D128, causal/RoPE,
GQA groups 2/4, both Q/KV rectangular directions, one/two decoder layers,
logical lengths 1/65/1025 (including padding to 2048), and MPI 1/2/4.

Eighteen native operator pairs passed on the final main-directory libraries:
all eight precisions, all six matrix operators, stripe and B-replica controls,
two/four independent instances (including MPI4), and a formal H2048 QKV with
logical shape M128/N3072/K2048. Every functional output was checked, every
timing numerical check was skipped, and cycles/output counts agreed within
every pair. The formal QKV pair measured 66934 cycles. External FP16 operands
passed with maximum absolute error 2.682209014892578e-7; full signed-range
INT8 operands passed exactly.

One same-input, same-build FP16 QKV placement comparison (M1024/N256/K128):

| Placement | Cycles | Delta from group-affine |
| --- | ---: | ---: |
| Group-affine | 4989 | 0 |
| Reuse-striped | 4474 | -515 |
| Group-affine with B replication | 4498 | -491 |

These are measurements of this operator shape under the native numerical
contract. They do not establish a full-decoder speedup and do not change the
selected default schedule. B replication uses additional space within the
same existing HBM nodes.

The repaired timing-fast-path control passed at 3500 cycles, with all six
execution/terminal/memory/DMA/NoC/HBM summary hashes equal in the same build.
An explicit GEMM `--verify-c` with full-timing was also confirmed to fail.

## Reproduction

```bash
python3 test/run_timing_regression.py --artifact-root /tmp/llm-timing --cpu-set 8-11
python3 test/run_native_llm_regression.py --artifact-root /tmp/llm-native --cpu-set 4-7
python3 test/maintenance/mechanism_alignment/validate_phase.py native_workloads_final
python3 test/maintenance/mechanism_alignment/validate_timing.py native_pair \
  --compare test/reports/alignment_06_timing_fast_path/result.json --cpuset 12
```

Compact machine-readable evidence accompanies the final validation record.
Full raw artifacts remain in the recorded artifact directories.
