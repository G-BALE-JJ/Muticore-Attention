# Four follow-up updates

Initial three-stage timing work committed and pushed as c36ec21.
Feature branch: feat/llm-native-workloads in exp-native-workloads.

1. Canonical --sim-mode across four public entries; only remove the default GEMM numerical check for timing. Keep explicit invalid numerical flags rejected by the underlying runner. Record observed backend mode.
2. Repair timing-control artifact paths and log lookup; configurable host CPU set. Run the real same-build control, not only syntax tests.
3. Nine paired functional/timing cases covering K depths, D64/D128, causal/RoPE, GQA groups 2/4, one/two layers, padding across 1024, MPI 1/2/4. Retain final output ACK/resource checks.
4. Add explicit native LLM matrix-operator experiments (qkv, wo, gate/up/down, LM head) on the existing guest, production TE, scheduler, NoC and HBM. Use per-TE opt-in PrecisionSpec and WorkloadPlacement parameters with canonical packed layouts and independent inputs. Keep default decoder arithmetic/ABI unchanged. Outputs are FP32/INT32 under the native contract; this does not claim eight-precision full-decoder next-token correctness. Accept external native INT4/INT8 integers or FP16/FP32 activations/weights; other precision validation inputs are exact representable -2..2. Test all eight modes and two independent workloads, striped mapping/B replication, and functional/timing event cycles.

Hardware stays fixed: four managers, sixteen workers, 64 arrays per worker, four data HBM nodes; no added buffers. New latency evidence is in cycles. Build candidate locally and run the four formal functional gates before finalizing the public selection and pushing. Compare same-build native control/placement configurations without claiming unmeasured performance wins.
