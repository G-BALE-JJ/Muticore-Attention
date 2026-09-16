# Attention and GQA terminology

This file is the naming contract for the active Attention implementation.
Code, command-line interfaces, reports, and new documentation use the canonical
names below. Deprecated names remain accepted only where compatibility with old
commands, environments, or frozen baseline JSON is required.

| Concept | Symbol | Canonical code/config name | Deprecated compatibility name |
|---|---|---|---|
| batch size | `B` | `batch_size` | none; currently fixed to 1 |
| Query sequence length | `Sq` | `query_length` | `queries` |
| shared K/V sequence length | `Skv` | `kv_length` | `keys` |
| Query-head count | `Hq` | `num_query_heads` | `query_heads`, `heads` |
| K/V-head count | `Hkv` | `num_kv_heads` | `kv_heads`, `heads` |
| per-head dimension | `Dh` | `head_dim` | `D` in older prose |
| Query work tile | - | `query_tile`, `query_tile_rows` | `query_block` |
| shared K/V work tile | - | `kv_tile`, `kv_tile_rows` | `key_tile`, `key_block` |
| QK contraction portion along `Dh` | - | `qk_reduction_slice` | `QK panel`, `reduction half` |
| PV output portion along `Dh` | - | `pv_output_slice` | `PV panel` |
| partial output carried across K/V tiles | - | `running_output_accumulator` | `old O` |
| score-row C-buffer transfer | - | `qk_score_row_burst` | `qk_panel_row_burst` |
| dispatch/completion network | - | `control_transport`, `control_vn` | `reduction transport`, `reduction_vn` |

The mathematical stages are:

```text
S = Q K^T / sqrt(Dh)
P = softmax(S)
O = P V
```

Tensor layouts are `Q[B,Sq,Hq,Dh]`, `K/V[B,Skv,Hkv,Dh]`, and
`O[B,Sq,Hq,Dh]`. The current files omit the fixed `B=1` axis. Storage may be
head-major internally, but the final output is written directly in
query-major `[Sq,Hq,Dh]` order; there is no separate concatenation transfer.

`matrix programming` is retained intentionally. In this CIM model it means
loading an operand into array storage before MAC execution; it is a hardware
operation, not an additional Attention mathematical stage.

The 128-byte descriptor is `GolemAttentionDescV2`. Its `group_query_rows`
field is the flattened Query-row count for one GQA K/V-sharing group, not the
per-head `Sq`. `kv_length` is `Skv`.

Frozen historical result files keep their original schema (`queries`, `keys`,
and similar fields) so old measurements remain reproducible. New public CLI
and generated result fields use the canonical names.
