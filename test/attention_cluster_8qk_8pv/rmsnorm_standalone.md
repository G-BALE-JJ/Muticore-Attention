# FP16 RMSNorm device workload

The guest writes a 128-byte `SFUJobDesc` to local GM and issues the existing
SFU job/wait RoCC instructions. The descriptor names FP16 X and gamma in HBM,
an HBM output address, a local staging address, row count, width, and FP32
epsilon in the low 32 bits of `reserved1`. On core 0, the SFU reads X and
gamma over DMA, executes the shared vector pipeline with FP32 square-sum and
inverse RMS, then writes FP16 output to HBM after the modeled vector latency.
The host only prepares the HBM image and checks the final output.

| Rows | Width | Elements | Job cycles, DMA through output | Vector cycles | Max abs error |
|---:|---:|---:|---:|---:|---:|
| 16 | 128 | 2,048 | 1,439 | 963 | 0.000484424 |
| 2 | 4,096 | 8,192 | 2,748 | 1,651 | 0.000487214 |
| 64 | 128 | 8,192 | 4,784 | 3,843 | 0.000484424 |

All three runs used full-timing SST with one rank and passed HBM output
verification. The first row of X is zero to cover the epsilon path. Job
cycles start at SFU acceptance and end at HBM write acknowledgement, so they
include DMA and vector processing but exclude guest setup. The vector cycle
count is the configured SFU latency model, not measured silicon throughput.
The current descriptor is one bounded batch: input, gamma, and output must
fit together in the 64 KiB local staging budget. Larger batches require
multiple jobs or row streaming. RMSNorm is independent of the Attention
guest until a pre-projection hidden-state workload is added.

Reproduce with `bash src/sst/elements/golem/tests/small/muticore_attention/run_rmsnorm.sh --sim-mode full-timing`.
Set `GOLEM_SFU_RMSNORM_ROWS` and `GOLEM_SFU_RMSNORM_COLS` to vary the shape.
