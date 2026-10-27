"""Shared standalone Softmax HBM layout contract."""


def normalize_softmax_hbm_layout(layout: str) -> str:
    value = (layout or "single_node").strip().lower()
    aliases = {"single": "single_node", "striped": "band_striped"}
    value = aliases.get(value, value)
    if value not in {"single_node", "band_striped"}:
        raise ValueError(f"unsupported Softmax HBM layout: {layout}")
    return value


def softmax_row_location(row: int, rows_per_band: int, data_nodes, layout: str):
    if row < 0 or rows_per_band <= 0 or not data_nodes:
        raise ValueError("row, rows_per_band, and data_nodes must define a valid layout")
    layout = normalize_softmax_hbm_layout(layout)
    if layout == "single_node":
        return data_nodes[0], row
    band = row // rows_per_band
    row_in_band = row % rows_per_band
    node = data_nodes[band % len(data_nodes)]
    local_band = band // len(data_nodes)
    return node, local_band * rows_per_band + row_in_band
