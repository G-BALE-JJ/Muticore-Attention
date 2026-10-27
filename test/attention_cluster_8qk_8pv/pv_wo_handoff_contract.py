#!/usr/bin/env python3
"""Build and validate the bounded PV-to-Wo GM landing map for the S=1024 path."""
import argparse
import json


PAYLOAD_BYTES = 8192
GM_STRIDE_BYTES = 0x200000
WO_HANDOFF_OFFSET = 0x1a0000
WO_HANDOFF_BYTES = 64 * 2048 * 2


def build_contract(sequence=1024, hidden=2048, head_dim=64, managers=4):
    if (sequence, hidden, head_dim, managers) != (1024, 2048, 64, 4):
        raise ValueError("PV-Wo handoff contract currently covers S=1024, H=2048, Dh=64, 4 managers")
    if sequence // managers != 256 or hidden % head_dim:
        raise ValueError("unsupported query or head partition")

    rows_per_worker = 64
    heads = hidden // head_dim
    bytes_per_head = rows_per_worker * head_dim * 2
    if bytes_per_head != PAYLOAD_BYTES:
        raise ValueError("each head block must fit exactly one bounded payload")
    payloads_per_head = 1
    if WO_HANDOFF_BYTES != rows_per_worker * hidden * 2:
        raise ValueError("reserved Wo handoff GM does not fit the assembled row tile")

    transfers = []
    covered = set()
    for manager in range(managers):
        for lane in range(4):
            worker = 4 + manager + 4 * lane
            for head in range(heads):
                kv_head, query_head_in_group = divmod(head, 4)
                group_query_row_begin = query_head_in_group * (sequence // managers) + lane * rows_per_worker
                for part in range(payloads_per_head):
                    offset = head * bytes_per_head + part * PAYLOAD_BYTES
                    key = (manager, lane, head, part)
                    if key in covered:
                        raise ValueError("duplicate PV output segment")
                    covered.add(key)
                    transfers.append({
                        "manager": manager,
                        "lane": lane,
                        "worker_core": worker,
                        "head": head,
                        "kv_head": kv_head,
                        "group_query_row_begin": group_query_row_begin,
                        "part": part,
                        "bytes": PAYLOAD_BYTES,
                        "gm_offset": offset,
                    })

    expected = managers * 4 * heads * payloads_per_head
    if len(transfers) != expected:
        raise ValueError("incomplete worker payload coverage")
    for manager in range(managers):
        for lane in range(4):
            worker_transfers = [t for t in transfers if t["manager"] == manager and t["lane"] == lane]
            offsets = sorted(t["gm_offset"] for t in worker_transfers)
            if len(offsets) != heads * payloads_per_head:
                raise ValueError("incomplete Wo worker tile")
            if offsets != list(range(0, WO_HANDOFF_BYTES, PAYLOAD_BYTES)):
                raise ValueError("GM landing has a gap or overlap")

    layer_bytes = len(transfers) * PAYLOAD_BYTES
    return {
        "status": "PASS",
        "shape": {"sequence": sequence, "hidden": hidden, "head_dim": head_dim,
                  "query_heads": 32, "kv_heads": 8, "managers": managers},
        "payload_bytes": PAYLOAD_BYTES,
        "worker_gm_offset": WO_HANDOFF_OFFSET,
        "worker_gm_bytes": WO_HANDOFF_BYTES,
        "gm_stride_bytes": GM_STRIDE_BYTES,
        "worker_gm_capacity_ok": WO_HANDOFF_OFFSET + WO_HANDOFF_BYTES <= GM_STRIDE_BYTES,
        "payloads_per_layer": len(transfers),
        "payload_bytes_per_layer": layer_bytes,
        "local_gm_landing_write_bytes_per_layer": layer_bytes,
        "gm_relocation_read_bytes_per_layer": layer_bytes,
        "gm_relocation_write_bytes_per_layer": layer_bytes,
        "current_pv_o_hbm_write_bytes_per_layer": layer_bytes,
        "current_wo_o_hbm_read_bytes_per_layer": layer_bytes,
        "removable_hbm_bytes_per_layer": 2 * layer_bytes,
        "transfers": transfers,
        "scope": "Routing, coverage and capacity contract only; no SST NoC or local-GM latency is simulated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = build_contract()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(rendered)
    print(json.dumps({key: value for key, value in result.items() if key != "transfers"}, indent=2))


if __name__ == "__main__":
    main()
