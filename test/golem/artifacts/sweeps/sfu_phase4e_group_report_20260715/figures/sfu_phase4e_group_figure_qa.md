# Phase 4E Group Figure QA

## Figure contract

- Core conclusion: VN0, VN1, and VN2 produce identical explicit-NoC outcomes, while reduction-network latency and xbar pressure increase with worker count.
- Evidence layout: panel a tests end-to-end VN equivalence against modeled NoC; dominant panel b shows reduction latency growth; panel c reports NoC pressure and all validation gates.
- Archetype: asymmetric quantitative grid for a 13.333 x 7.5 inch English group-meeting slide.
- Backend: Python 3 from the approved project-external virtual environment, Matplotlib 3.10.9 using the noninteractive Agg backend.

## Sources and generation

The four immutable source roots were:

- `sfu_phase4e_modeled_control_matrix_20260715`
- `sfu_phase4e_explicit_vn0_matrix_20260715`
- `sfu_phase4e_explicit_vn1_matrix_20260715`
- `sfu_phase4e_explicit_vn2_matrix_20260715`

Generation command:

```bash
MPLCONFIGDIR=/data4/jjgong/.cache/matplotlib-phase4e \
/data4/jjgong/.venvs/sfu-phase4e-figure/bin/python \
  src/sst/elements/golem/tests/artifacts/sweeps/sfu_phase4e_group_report_20260715/figures/plot_sfu_phase4e_group_figure.py \
  --sweeps-root src/sst/elements/golem/tests/artifacts/sweeps \
  --output-dir src/sst/elements/golem/tests/artifacts/sweeps/sfu_phase4e_group_report_20260715/figures
```

The matrix contains 12 canonical unique `PASS/PASS` rows: four series (modeled-NoC plus explicit VN0, VN1, and VN2) at three worker counts (4, 8, and 16). Later cached manifest rows were not selected.

## Numerical validation

- Golden verification: all 12 output tensors were independently checked over 8192 values; every point reported 0 mismatches.
- Explicit transport events: 256, 512, and 1024 at 4, 8, and 16 workers, respectively, for each of VN0, VN1, and VN2.
- Explicit VN equality: runtime, aggregate-average latency, maximum latency, inbox high-water, xbar stalls, and transport events are equal among VN0/VN1/VN2 at each worker count.
- DMA gates: timeout retry, timeout exhaustion, and write-timeout retry are all zero for all 12 rows.
- Queue/reject/stale gates: queued, rejected, and stale totals are all zero for all 12 rows.
- Inbox high-water: 4 for every explicit-NoC point.
- Maximum explicit-vs-modeled runtime difference: exactly `0.060613050701470689%`, below the strict `0.061%` gate.
- Aggregate-average latency endpoints: `9440.03125` cycles at 4 workers and `15582.99609375` cycles at 16 workers, displayed as `9,440` and `15,583 cycles`.
- Exact aggregate-average latency growth: `65.073564706154968%`; the data-derived callout displays `+65%` after integer rounding.

Every plotted value is one deterministic simulation outcome. No repeated-run distribution, confidence interval, error bar, hypothesis test, or other inferential statistic was used.

## Deterministic regeneration

Two complete real-data regenerations produced the same SHA-256 values:

| File | First pass | Second pass |
| --- | --- | --- |
| Source CSV | `91772b337c7f061936e3b1e1da5a74ce5c971e405578aae1cd0a1830f2a3a01e` | `91772b337c7f061936e3b1e1da5a74ce5c971e405578aae1cd0a1830f2a3a01e` |
| SVG | `b54d9954437e6459e27941750d3461e4650b612cfb89a1beaa3b63e7357d57d1` | `b54d9954437e6459e27941750d3461e4650b612cfb89a1beaa3b63e7357d57d1` |
| PNG | `068d0daf1d5761ac7342d936b86587a89f34861079e21e94557f8dbdb940f6ce` | `068d0daf1d5761ac7342d936b86587a89f34861079e21e94557f8dbdb940f6ce` |
| PDF | `a18218e550a6cdeba79f1cf869006e4adc686f199bd08f69a66ee2f10b8ca9b5` | `a18218e550a6cdeba79f1cf869006e4adc686f199bd08f69a66ee2f10b8ca9b5` |

Matplotlib's default random SVG IDs and current-time SVG/PDF metadata initially prevented stable vector hashes. The renderer uses a fixed SVG hash salt and omits volatile vector dates. After SVG export, one deterministic text-only normalization removes horizontal whitespace at line endings; it does not parse or rewrite XML structure, path content, or live text.

## Raster and vector QA

- `file`: SVG Scalable Vector Graphics, one-page PDF 1.4, and 8-bit RGBA non-interlaced PNG.
- PNG: exactly 3999 x 2250 pixels. ImageMagick reports 118.11 pixels/cm, equivalent to 300 dpi; Pillow 12.3.0 independently confirms the dimensions and both dpi axes within 0.1 of 300.
- PDF: page size 959.976 x 540 points, matching the fixed 13.333 x 7.5 inch canvas.
- PDF fonts: embedded and subsetted CID TrueType `DejaVuSans-Bold` and `DejaVuSans`, with Unicode mappings; text is not path-only.
- SVG text: 67 live `<text>` nodes with `font-family: 'DejaVu Sans'` are present for titles, labels, ticks, annotations, and validation text.
- SVG normalization: line-ending horizontal whitespace is stripped after Matplotlib save. The normalized output has zero trailing-whitespace lines, parses as valid XML, retains live text, renders identically, and permits whole-feature-range `git diff --check` without an SVG exception.

## Test evidence

Final focused suite command:

```bash
MPLCONFIGDIR=/data4/jjgong/.cache/matplotlib-phase4e \
/data4/jjgong/.venvs/sfu-phase4e-figure/bin/python -m unittest discover \
  -s src/sst/elements/golem/tests/small/mvm_noc_softmax_sfu \
  -p 'test_*.py' -v
```

Result: `Ran 223 tests in 22.873s` and `OK`.

The focused figure tests explicitly assert a three-panel 13.333 x 7.5 inch canvas, all claim-bearing VN equality gates, mutation rejection before export, system-Python parse-only operation without Matplotlib, unique log/completion evidence, data-derived live SVG annotations, byte-identical repeated SVG/PDF/PNG rendering, normalized valid XML with no trailing whitespace, and a 3999 x 2250 PNG with 300 dpi metadata.

## Visual inspection

The final PNG was inspected at full-frame and original 3999 x 2250 resolution.

- No title, annotation, legend, tick label, endpoint label, callout, or validation-block overlap was found.
- Text remains readable at normal slide size.
- Panel b is visibly dominant.
- VN0/VN1/VN2 remain distinguishable through cyan circle, blue square, and navy triangle markers despite equal values.
- Orange is used only for the maximum-latency series/label and the `+65%` latency-growth callout.
- No roadmap, GEMM+softmax fusion, or other future-work text appears.

Visual QA passed without a layout or style adjustment.
