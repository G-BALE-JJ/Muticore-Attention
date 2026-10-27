# Project reporting conventions

- Report all new accelerator latency measurements, targets, comparisons, tables, and progress updates in **cycles**. Do not convert them to ms or us.
- Simulator tick timestamps may be retained as raw evidence, but identify the tick-to-cycle conversion explicitly in analysis outputs.
- Host preparation, simulation wall time, and validation wall time are separate from accelerator latency; label these explicitly if reported.
- Preserve the original units in archived historical evidence. New reports must use cycles and state their measurement boundaries.
- Keep fixed hardware resources and numerical contracts when evaluating software scheduling/dataflow changes. Adopt candidates using same-build paired end-to-end evidence, not isolated stage improvements.
