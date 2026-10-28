# Native TU model and fixed guest runtime

Items 3 and 4 of the reference-project comparison are implemented in
`9a543c9` and `c43350a`. The final machine-readable acceptance evidence is
[result.json](result.json).

`TileUnitArrayModel` now owns the numerical banks, reductions and timed events.
The native `TileUnitArray` uses this model directly; historical MVM classes
are thin adapters. The shared `ComputeArray` transport/API remains. The moved
implementation preserves numerical order, buffers, array counts and callbacks.

The native operator entry now defaults to one fixed static RISC-V launcher.
Host-generated version-1 commands describe precision, geometry, worker ownership
and HBM placement. An optional read-only mapping occupies a reserved 1 MiB of
existing Node0 capacity. The guest copies its command into existing GM metadata,
submits it to the production TE and waits on a per-core token. Workload changes
reuse the guest binary; they do not require recompiling a shape-specific guest
or feeding execution parameters into each TE.

Submission validates bounded lengths, checksums, geometry, placement, regions
and core ownership. One token holds the single submission slot until consumed.
Wait completes after the TE's final output write ACK. Physical core selection
uses `sched_getcpu()`: Vanadis process argument order differs from physical core
order, and setting affinity does not migrate a running process. Guest diagnostics
are checked in their archived stdout files.

## Validation boundaries

All accelerator latency evidence is in cycles. Native measurements run from the
first guest descriptor timestamp through the last worker completion after final
C ACK, inclusive +1. Default prefill remains first token-ID DMA send through final
valid-token full-vocabulary logits ACK; host argmax is excluded. Native launch
has a different control path from the legacy launcher, so its cycles are not a
TU speedup comparison. ABI self-tests add guest instructions and are separate
from performance pairs.

The final evidence records all four full functional gates, ten general
functional/timing pairs, eighteen fixed-guest native pairs, same-build numerical
materialization controls, negative ABI tests, external input tests and main
workspace installation checks. All 36 native launches reuse the same guest hash, with 36 make reuse
confirmations and zero compiler invocations. Candidate libraries are promoted byte-for-byte
into the main installation; the evidence records all three library hashes and
the identical guest hashes. The full candidate build's copied C++ sources are
also compared with the committed source snapshot. The formal gate began before
the implementation commit; its recorded base commit is retained alongside the
validated implementation and source hashes.

Initial diagnostic runs were rejected and excluded from acceptance: one used
the wrong pipeline executable variable, and another assumed argument/core order.
The first main-workspace smoke also exposed the globally ignored new Makefile;
`c43350a` explicitly tracks it. Fresh main smoke and host checks validate the
packaged checkout. The full prefill simulation completed successfully, then its
verifier stopped because this isolated worktree lacked the historical `results`
symlink. Restoring that fixture allowed complete verification and campaign
summarization of the saved outputs. Initial failure receipts and explicit
postprocessing recovery commands are retained; SST was not rerun.

## Scope

The versioned format is this project's native ABI, not donor TileBin wire
compatibility. This delivery supplies a runtime and fixed launcher; it does not
add a TileLang compiler or a multi-operator execution graph. Tokens are per-core;
a future dependent-operator runtime needs an operator-wide visibility/barrier
contract. Version 1 supports GEMM on the documented fixed machine.

Eight precisions apply to explicit native matrix-operator experiments with
INT32/FP32 outputs. The full decoder retains its accepted FP16 contract. Timing
mode keeps simulated computation, transfers and final ACKs, skips array numerical
MAC/materialization, and does not produce real next tokens. See
[the runtime documentation](../../../software/native/README.md) for commands,
ABI limits and the `--launcher legacy` compatibility option.
