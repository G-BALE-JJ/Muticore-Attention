# Reuse-window FlashAttention baseline

This baseline keeps the production `attention_sequential_64` path unchanged. It
models the alternative dataflow requested for the reference generic-GEMM WCP:

```text
QK output tile grid (64 x 64 tiles)

             N / key tiles
          0    1    2    3
       +===================+
M = 0  || C0 | C1 | C2 | C3 ||
M = 1  || C4 | C5 | C6 | C7 ||   QK reuse window = 2 x 4
       +===================+

QK: [128,128] x [128,256] -> [128,256]
    A/Q reuse-N = 4, B/K reuse-M = 2, reduction K tiles = 2

online softmax consumes the bounded [128,256] score window

PV: [128,256] x [256,128] -> [128,128]
    output reuse window = 2 x 2, reduction K tiles = 4
```

The analytical baseline below keeps only a bounded score/probability window.
GQA maps query head `hq` to K/V head `hq // (Hq/Hkv)`; shared K/V is counted
once in physical HBM traffic.

The standalone analytical schedule uses one complete Query-head GEMM as its
unit of submission:

```text
submit Q0 x K0^T as one GEMM(1024,1024,128)
  -> reference scheduler distributes its 32 reuse windows over all 16 workers
  -> bounded window softmax/PV states merge into O0

submit Q1 x K0^T as one GEMM(1024,1024,128)
  -> reuse the retained K0/V0
  -> produce O1
```

The same rule applies to larger GQA groups: Query heads execute sequentially,
while each individual GEMM exposes its full output-window parallelism to all
workers. This avoids using concurrency between unrelated Q matrices to hide a
poorly parallelized GEMM.

Each completed score window independently produces a FlashAttention summary
`(m_j, l_j, O_j)`. Summaries for the same Query rows are reduced as:

```text
m = max(m_a, m_b)
l = exp(m_a-m)*l_a + exp(m_b-m)*l_b
O = exp(m_a-m)*O_a + exp(m_b-m)*O_b
```

The reduction is associative apart from normal floating-point rounding, so the
generic GEMM scheduler may execute output windows in parallel or complete them
out of order.

## Run

The default is the target case `Hq=4,Hkv=2,Sq=Skv=1024,Dh=128`:

```bash
baseline/reuse_window_flash_attention/run.sh
```

Artifacts are written to
`baseline/reuse_window_flash_attention/artifacts/latest/`:

- `result.json`: numerical result, traffic proof, and lower bounds;
- `schedule.json`: every QK and PV generic-GEMM reuse window;
- `output.bin`: query-major `[Sq,Hq,Dh]` FP32 output.

Use `--no-numerical` for a quick schedule/traffic-only run, or pass different
supported GQA dimensions:

```bash
baseline/reuse_window_flash_attention/run.sh \
  --num-query-heads 8 --num-kv-heads 2 --no-numerical
```

Dimension constraints for the current fixed `2x4` window are:

```text
Hq % Hkv = 0, with Hq/Hkv in {1,2,4}
Sq  % 128 = 0
Skv % 256 = 0
Dh  % 64  = 0
```

Larger `Dh` may require more WCP local slots. For example, `Dh=256` needs 48:

```bash
baseline/reuse_window_flash_attention/run.sh \
  --head-dim 256 --local-slots 48 --no-numerical
```

## Reference WCP contract

For the default QK window, the generated contract is:

```text
GOLEM_REQUEST_SCHEDULER_ENABLE=1
GOLEM_A_REUSE_N_TILES=4
GOLEM_B_REUSE_M_TILES=2
GOLEM_DMA_WINDOW_K_TILES=2
GOLEM_DMA_SLOT_COUNT=24
GOLEM_GM_C_BUFFER_BYTES>=131072
GOLEM_OUTPUT_MODE=fusion
```

The slot count is `3 buffers * K=2 * max(reuseM=2,reuseN=4) = 24`, using
the reference engine's active plus two-prefetch-window contract. The existing
1 MiB WCP C-buffer is larger than the 128 KiB QK window requirement.

This directory is a functional and scheduling baseline. `engine_contract` in
`result.json` describes that analytical execution.

## SST bridge and cycle measurement

The bridge runs the same target shape through the SST model:

```bash
GOLEM_ARTIFACT_ROOT=/tmp/reuse_window_attention \
  baseline/reuse_window_flash_attention/run_sst_qk_bridge.sh
```

For each KV head, the four managers collectively execute the two associated
Query heads as one logical `GEMM(2048,1024,128)`. Each manager owns a
`GEMM(512,1024,128)` band, and each of its four workers owns 128 Query rows.
The WCP QK window is `2x4`; softmax uses `64x64` tiles; each 256-key group feeds
a WCP PV `GEMM(64,128,256)` with a `1x2` effective output window. Final output
is written as query-major `[Sq,Hq,Dh]`.

The measured result is written to `sst_qk_bridge_result.json`; numerical output
verification is written to `fused_attention_result.json`. A successful run
requires both reports to pass.

The current SST bridge does not write score or probability matrices to HBM,
but it collects one worker's full `[128,Skv]` QK score band in simulator state
before launching softmax/PV. Thus it validates the reuse-window GEMMs and the
end-to-end cycle path, but it is not yet a strict bounded-SRAM FlashAttention
pipeline. Eliminating that staging requires interleaving a QK window's WCP
completion with its softmax and PV work before the next QK window is issued.
