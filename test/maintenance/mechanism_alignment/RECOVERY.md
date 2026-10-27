# Session recovery — 2026-10-27

Recovered from session 01a17c0f-53e6-7273-bf35-4e5577edee1e.
Before takeover all original service/task PIDs were absent and /proc contained
no candidate simulation, validation or follow-up process. Original phase 01
report remains RUNNING because the process ended before finalization; it is
interrupted evidence, not PASS. Preserve it and use a fresh full four-stage run.

Candidate HEAD at takeover: f8449da. Clean worktree. Phase 01 readiness
guard is committed; phases 02–06 have not been applied. Recovered phase 02/03
scripts in /tmp are drafts and must be reviewed before use.

Continue PLAN.md sequentially; no feature is applied until the preceding
four-stage gate passes. Preserve default arithmetic, hardware, guest ABI and
public selected baseline; no TileLang components.

## Recovered validation

The fresh `alignment_01_partial_c_resume_20261027` run completed all four
functional stages. Its initial report finalization failed because
`summarize_stages.py` loaded the public library pin unconditionally even when
the runner used an explicit private candidate configuration. The summary now
accepts that configuration and verifies that every public selection field
except the rebuilt library digest remains identical. The selected guest,
hardware, scheduling flags, cycle expectations and numerical references were
not changed.

Only report finalization was repeated against those completed artifacts.
`gate.before_summary_fix.json` preserves the original failure; `gate.json`
records the recovery and PASS. The original interrupted phase 01 remains
separate. Subsequent gates record both the compute library and memory backend
library digests and require the prefill provenance to match the installed
libraries at summary time.

Recovery supervisors write `test/reports/recovered_workflow.json` and
`recovered_workflow_04_06.json`. A failure stops the sequence. Raw result
directories and local orchestration logs are intentionally distinct from
compact committed gate evidence.

## Packed active-K repair

The first phase 04 gate was deliberately interrupted after a targeted test
confirmed an active-K mismatch. Configuring packed K=64 and then launching a
shorter active-K MV scheduled completion for the shorter reduction but used
the configured global K for arithmetic. The inherited legacy active-K API
permits each TU to select its own reduction length.

Packed arithmetic now reads the per-TU launch context, while a non-active
launch uses the configured precision K. The regression fills the complete
physical operand capacity and checks both configured-K and explicit active-K
launches, including different active K on two TUs. This makes unused nonzero
lanes detect accidental over-computation. The interrupted `alignment_04_precision`
is not acceptance evidence; the repaired feature uses a fresh
`alignment_04_precision_active_k` full four-stage gate.
