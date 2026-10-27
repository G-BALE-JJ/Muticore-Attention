"""Read the same RTL timing table used by the C++ CU model."""
from pathlib import Path
import re


def pe_timing(mode, packets):
    if packets <= 0:
        raise ValueError("PE dot requires at least one packet")
    source = next(p for p in Path(__file__).resolve().parents if (p/"configure.ac").is_file()) / "src/sst/elements/golem/computeunit/peTiming.def"
    entries = re.findall(
        r"^TILEMC_PE_TIMING\(\w+,\s*(\w+),\s*(\d+),\s*(\d+),\s*(\d+)\)$",
        source.read_text(), re.MULTILINE,
    )
    table = {name: (int(ii), int(latency), int(gap)) for name, ii, latency, gap in entries}
    ii, latency, gap = table[mode]
    return dict(packet_ii=ii, single_packet_latency=latency,
                completion_cycles=(packets - 1) * ii + latency,
                start_interval=packets * ii + gap)
