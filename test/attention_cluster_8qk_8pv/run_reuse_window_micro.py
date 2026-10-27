#!/usr/bin/env python3
"""Isolated reuse-window FFN screening using the existing SST/numerical audits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import run_decoder_micro as micro


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--window-k-tiles',type=int,choices=(4,8,16,32),default=32)
    parser.add_argument('--window-prefetch-slots',type=int,choices=(1,4),default=4)
    parser.add_argument('--window-array-groups',type=int,choices=(1,2),default=2)
    parser.add_argument('--window-early-a',type=int,choices=(0,1),default=0)
    parser.add_argument('--window-output-overlap',type=int,choices=(0,1),default=0)
    parser.add_argument('--ffn-block-pipeline',type=int,choices=(0,1),default=0)
    parser.add_argument('--ffn-pipeline-policy',type=int,choices=(0,1,2),default=1)
    parser.add_argument('--ffn-panel-tiles',type=int,choices=(4,8,16),default=16)
    parser.add_argument('--window-late-lease',type=int,choices=(0,1),default=0)
    parser.add_argument('--window-gm-lookahead',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--ffn-gm-lookahead',type=int,choices=(0,1,2,4),default=0)
    parser.add_argument('--ffn-demand-order',action='store_true')
    parser.add_argument('--window-a-first',type=int,choices=(0,1),default=0)
    parser.add_argument('--window-a-prefetch',type=int,choices=(0,1),default=0)
    parser.add_argument('--scatter-wave',type=int,choices=(0,1),default=0)
    parser.add_argument('--b-window-bridge',type=int,choices=(0,1),default=0)
    parser.add_argument('--ffn-sfu-stream',type=int,choices=(0,1),default=0)
    parser.add_argument('--ffn-sfu-chunk',type=int,choices=(1024,2048,4096,8192),default=8192)
    parser.add_argument('--weight-cohort',type=int,choices=(0,2,4),default=0)
    parser.add_argument('--audit-completed-root',type=Path)
    args,remaining=parser.parse_known_args()
    source=micro.ROOT/'src/sst/elements/golem/attention/reuseWindowGemm.h'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    original=micro.configured_environment
    def environment(*positional,**keywords):
        env=original(*positional,**keywords)
        env.update(GOLEM_ALL_GEMM_REUSE_WINDOW='1',
                   GOLEM_REUSE_WINDOW_K_TILES=str(args.window_k_tiles),
                   GOLEM_REUSE_WINDOW_PREFETCH_SLOTS=str(args.window_prefetch_slots),
                   GOLEM_REUSE_WINDOW_ARRAY_GROUPS=str(args.window_array_groups),
                   GOLEM_REUSE_WINDOW_EARLY_A=str(args.window_early_a),
                   GOLEM_REUSE_WINDOW_LATE_LEASE=str(args.window_late_lease),
                   GOLEM_REUSE_WINDOW_GM_LOOKAHEAD=str(args.window_gm_lookahead),
                   GOLEM_DECODER_FFN_GM_LOOKAHEAD=str(args.ffn_gm_lookahead),
                   GOLEM_DECODER_FFN_DEMAND_ORDER=str(int(args.ffn_demand_order)),
                   GOLEM_REUSE_WINDOW_A_FIRST=str(args.window_a_first),
                   GOLEM_REUSE_WINDOW_A_PREFETCH=str(args.window_a_prefetch),
                   GOLEM_REUSE_WINDOW_SCATTER_WAVE=str(args.scatter_wave),
                   GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE=str(args.b_window_bridge),
                   GOLEM_DECODER_FFN_SFU_STREAM=str(args.ffn_sfu_stream),
                   GOLEM_DECODER_FFN_SFU_CHUNK=str(args.ffn_sfu_chunk),
                   GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP=str(args.window_output_overlap),
                   GOLEM_REUSE_WEIGHT_COHORT=str(args.weight_cohort),
                   GOLEM_DECODER_FFN_BLOCK_PIPELINE=str(args.ffn_block_pipeline),
                   GOLEM_DECODER_FFN_PIPELINE_POLICY=str(args.ffn_pipeline_policy),
                   GOLEM_DECODER_FFN_PANEL_TILES=str(args.ffn_panel_tiles))
        if args.weight_cohort and keywords.get('args',positional[1] if len(positional)>1 else None).workers!=16:
            raise ValueError('weight cohort requires all 16 worker jobs')
        return env
    micro.configured_environment=environment
    # Native groups=1 reports no subgroup launches. The window path counts
    # every physical group launch, so validate against its explicit contract.
    def group_audit(case,hidden,workers,legacy_groups,reuse=False):
        flows=[]
        for line in (case/'logs/attention-sst.log').read_text().splitlines():
            if '[DECODER_GROUP_FLOW]' in line:
                flows.append({k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)})
        tiles=3*(hidden//64)*(4*hidden//64)
        groups=2 if args.scatter_wave else args.window_array_groups
        if len(flows)!=workers or {v['core'] for v in flows}!=set(range(4,4+workers)):
            raise ValueError('missing window array group lifecycle')
        for flow in flows:
            if (flow['launches']!=groups*tiles or flow['compute_array_cycles']!=tiles*64*67
                or (groups==1 and flow['both_groups_compute_cycles']!=0)):
                raise ValueError('window physical group launches/compute accounting mismatch')
        return dict(status='PASS',groups=groups,full_row_tile_count_per_worker=tiles,
                    scope='actual window group launches and timed single-context MAC callbacks',flows=flows)
    micro.group_audit=group_audit
    original_partial=micro.down_partial_audit
    micro.down_partial_audit=lambda case,hidden,workers,panel,reuse=False: original_partial(case,hidden,workers,args.ffn_panel_tiles,reuse)
    if args.audit_completed_root:
        root=args.audit_completed_root.resolve();case=root/'layout2';inputs=root/'inputs'
        manifest=json.loads((inputs/'inputs.json').read_text())
        provenance=json.loads((case/'provenance.json').read_text());env=provenance['environment']
        for key,value in (('GOLEM_REUSE_WINDOW_LATE_LEASE',args.window_late_lease),
                          ('GOLEM_DECODER_FFN_DEMAND_ORDER',int(args.ffn_demand_order)),
                          ('GOLEM_REUSE_WINDOW_SCATTER_WAVE',args.scatter_wave),
                          ('GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE',args.b_window_bridge),
                          ('GOLEM_REUSE_WINDOW_GM_LOOKAHEAD',args.window_gm_lookahead),
                          ('GOLEM_DECODER_FFN_GM_LOOKAHEAD',args.ffn_gm_lookahead),
                          ('GOLEM_REUSE_WINDOW_A_FIRST',args.window_a_first),
                          ('GOLEM_REUSE_WINDOW_A_PREFETCH',args.window_a_prefetch),
                          ('GOLEM_DECODER_FFN_SFU_STREAM',args.ffn_sfu_stream),
                          ('GOLEM_DECODER_FFN_SFU_CHUNK',args.ffn_sfu_chunk)):
            if env.get(key,'0' if key=='GOLEM_DECODER_FFN_GM_LOOKAHEAD' else '8192' if key.endswith('CHUNK') else '4' if key.endswith('LOOKAHEAD') else '0')!=str(value):
                raise ValueError(f'requested audit configuration differs from provenance: {key}')
        if (env['GOLEM_REUSE_WINDOW_K_TILES']!=str(args.window_k_tiles)
            or env['GOLEM_REUSE_WINDOW_PREFETCH_SLOTS']!=str(args.window_prefetch_slots)
            or env['GOLEM_REUSE_WINDOW_ARRAY_GROUPS']!=str(args.window_array_groups)
            or env.get('GOLEM_REUSE_WINDOW_EARLY_A','0')!=str(args.window_early_a)
            or env.get('GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP','0')!=str(args.window_output_overlap)
            or env.get('GOLEM_DECODER_FFN_BLOCK_PIPELINE','0')!=str(args.ffn_block_pipeline)
            or env.get('GOLEM_DECODER_FFN_PIPELINE_POLICY','2')!=str(args.ffn_pipeline_policy)
            or env.get('GOLEM_DECODER_FFN_PANEL_TILES','16')!=str(args.ffn_panel_tiles)
            or env.get('GOLEM_REUSE_WEIGHT_COHORT','0')!=str(args.weight_cohort)):
            raise ValueError('audit options differ from recorded simulation configuration')
        workers=int(env['GOLEM_DECODER_MICRO_CONFIG'].split(',')[0]);hidden=manifest['hidden']
        result=dict(status='PASS',variant='layout2',provenance=provenance,
                    shape=dict(hidden=hidden,workers=workers,rows_per_node=256 if workers==16 else 64,
                               with_down=True,with_residual=True),
                    dma_timing=micro.timing(case,manifest,workers,3,True,True),
                    numerics=micro.check_numerics(case,inputs,manifest,workers,True,3,True),
                    resources=micro.verify_resources(case),sfu_work=micro.sfu_work(case),
                    down_partial=micro.down_partial_audit(case,hidden,workers,args.ffn_panel_tiles),
                    array_groups=group_audit(case,hidden,workers,int(env['GOLEM_DECODER_ARRAY_GROUPS'])),
                    audit_note='Rechecked completed simulation; no rerun, no numerical tolerance changes.')
        (case/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result['status'],cycles=result['dma_timing']['cycles'])))
        return
    sys.argv=[sys.argv[0]]+remaining
    try:
        micro.main()
    finally:
        if '--artifact-root' in remaining:
            root=Path(remaining[remaining.index('--artifact-root')+1]).resolve()
            if root.is_dir():
                (root/'reuse_window_configuration.json').write_text(json.dumps(dict(
                    vars(args),source_sha256=source_hash),indent=2)+'\n')

if __name__=='__main__':main()
