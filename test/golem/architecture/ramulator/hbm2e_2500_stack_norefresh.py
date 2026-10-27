"""Eight-channel HBM2E roofline config; refresh disabled only for microbenchmarking."""

import ramulator


def make_controller():
    dram = ramulator.dram.HBM2(
        org_preset="HBM2_2Gb",
        timing_preset="HBM2_2400Mbps",
        rate=2500,
        tCK_ps=800,
        nBL=2,
        nCL=18,
        nRCDRD=18,
        nRCDWR=15,
        nRP=18,
        nRAS=42,
        nRC=60,
        nWR=20,
        nRTPL=7,
        nCWL=7,
        nCCDS=2,
        nCCDL=4,
        nCCDR=2,
        nRRDS=5,
        nRRDL=5,
        nWTRS=9,
        nWTRL=11,
        nRTW=18,
        nFAW=19,
        nRFC=325,
        nRFCpb=200,
        nRREFD=10,
        nREFI=4875,
        nREFIpb=304,
    )
    return ramulator.controller.HBM12(
        dram=dram,
        scheduler=ramulator.scheduler.FRFCFSRowHit(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.MOP4CLXOR(),
        read_buffer_size=64,
        write_buffer_size=64,
    )


frontend = ramulator.frontend.External(clock_ratio=1)
memory_system = ramulator.memory_system.GenericDRAM(
    clock_ratio=1,
    controllers=[make_controller() for _ in range(8)],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
)
ramulator.Simulation(frontend, memory_system)
