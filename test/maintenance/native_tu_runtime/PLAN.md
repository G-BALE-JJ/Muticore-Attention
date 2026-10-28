# Native TU and fixed descriptor runtime acceptance

Requested work: items 3 and 4 from the reference-project comparison.
Base: 4db8d6f. Donor architecture snapshot: cc1b454.

1. Move the existing numerical bank/state/event implementation into a native
   TileUnitArrayModel, leaving thin legacy MVM adapters. Keep numerical order,
   hardware buffers and callbacks identical; test both SST registrations.
2. Add a bounded, checksummed, versioned native command ABI and fixed RISC-V
   launcher. Carry precision, geometry, logical/physical workers and HBM maps
   in the descriptor; bypass the former per-TE parameter bridge for this path.
3. Port only the donor's optional read-only artifact mapping and physical page
   reservation mechanism into Vanadis. Reserve existing Node0 capacity; add
   no HBM nodes or tensor buffers. Keep default mappings disabled.
4. Exercise submit/wait tokens, corruption rejection and single-slot behavior
   with actual guest instructions. Use same binary across shapes/precisions/
   instances; all eighteen native functional/timing pairs must pass.
5. Run four full functional gates, ten general timing pairs and the same-build
   timing materialization control. Preserve selected guest and default cycles:
   GEMM 3500, Attention 12852, GQA 37928, prefill 6084167.75 cycles.

All new accelerator latency evidence uses cycles. Native descriptor launch
has a different guest/control path from legacy launch: compare identical
fixed-launcher pairs and do not attribute launch changes to TU speedups.
Do not claim a full eight-precision decoder or TileBin compatibility.
