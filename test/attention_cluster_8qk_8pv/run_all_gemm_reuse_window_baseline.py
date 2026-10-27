#!/usr/bin/env python3
"""Run and audit the isolated all-GEMM reuse-window prefill baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE_RUNNER = HERE / 'run_decoder_stack.py'
BASELINE_RESULT = (ROOT / 'Muticore-Attention/results/'
                   'prefill_generalization_20261016/formal_l2_s1024/decoder_stack_result.json')
SOURCES = [ROOT / 'src/sst/elements/golem/attention' / name for name in
           ('reuseWindowGemm.h', 'decoderOpJob.h', 'projectionJob.h', 'reuseWeightInterface.h',
            'reuseWeightCohort.h', 'decoderMemoryLayout.h', 'softwareLayerResidency.h',
            'pvWoHandoff.h', 'pvWoInputInterface.h')]
SOURCES += [ROOT/'src/sst/elements/golem'/name for name in ('rocc/roccAnalog.h','globalmemory/globalmemory.h')]


def expected_gm_lookahead(env,operator):
    override=int(env.get('GOLEM_DECODER_FFN_GM_LOOKAHEAD','0'))
    depth=override if override and operator in ('gate','up','down') else int(env.get('GOLEM_REUSE_WINDOW_GM_LOOKAHEAD','4'))
    return min(int(env['GOLEM_REUSE_WINDOW_PREFETCH_SLOTS']),depth)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def gemm_schedule(sequence=1024, hidden=2048, intermediate=8192, layers=2,
                  query_heads=32, kv_heads=8, head_dim=64, vocab=128256):
    operators = []
    for layer in range(1, layers + 1):
        for name, m, n, k, engine in (
            ('qkv', sequence, hidden + 2*kv_heads*head_dim, hidden, 'ReuseWindowGemm'),
            ('qk', sequence, sequence, head_dim, 'existing WCP QK reuse-window'),
            ('pv', sequence, head_dim, sequence, 'existing WCP PV reuse-window'),
            ('wo', sequence, hidden, hidden, 'ReuseWindowGemm'),
            ('gate', sequence, intermediate, hidden, 'ReuseWindowGemm'),
            ('up', sequence, intermediate, hidden, 'ReuseWindowGemm'),
            ('down', sequence, hidden, intermediate, 'ReuseWindowGemm'),
        ):
            operators.append(dict(layer=layer, name=name, M=m, N=n, K=k,
                                  heads=query_heads if name in ('qk','pv') else 1,
                                  engine=engine, reuse_window=True))
    operators.append(dict(name='lm_head', M=1, N=vocab, K=hidden,
                          engine='four-array N-parallel input reuse window', reuse_window=True))
    return dict(operators=operators, tile=[64,64,64],
                decoder_window=dict(reuse_m_tiles=1, reuse_n_tiles=4, k_tiles=4),
                exclusions=['embedding is a token lookup', 'Norm, RoPE, SiLU, Multiply and Add are vector operations'])


def make_isolated_runner(destination: Path, k_tiles=32, prefetch_slots=4, array_groups=1,
                         early_a=0,output_overlap=0,qkv_pack=0,gamma_reuse=0,weight_cohort=0,ffn_block_pipeline=0,ffn_panel_tiles=16,ffn_pipeline_policy=2, late_lease=0, sfu_stream=0, sfu_chunk=8192, norm_handoff=0, epoch_flags=0, gm_lookahead=4, a_first=0, a_prefetch=0, scatter_wave=0, b_window_bridge=0) -> Path:
    source = BASELINE_RUNNER.read_text()
    needle = "    env=environment(case,(1,1,1),'none',m['sequence'],a.shape)\n"
    switches = dict(GOLEM_ATTENTION_REUSE_WINDOW_QK_BRIDGE='1',
                    GOLEM_ATTENTION_WORKER_CLUSTER_BRIDGE='1',
                    GOLEM_ATTENTION_WORKER_CLUSTER_QK_WORKERS_PER_MANAGER='2',
                    GOLEM_ALL_GEMM_REUSE_WINDOW='1', GOLEM_OUTPUT_MODE='fusion',
                    GOLEM_REUSE_WINDOW_K_TILES=str(k_tiles),
                    GOLEM_REUSE_WINDOW_PREFETCH_SLOTS=str(prefetch_slots),
                    GOLEM_REUSE_WINDOW_ARRAY_GROUPS=str(array_groups),
                    GOLEM_REUSE_WINDOW_EARLY_A=str(early_a),
                    GOLEM_REUSE_WINDOW_LATE_LEASE=str(late_lease),
                    GOLEM_REUSE_WINDOW_GM_LOOKAHEAD=str(gm_lookahead),
                    GOLEM_REUSE_WINDOW_A_FIRST=str(a_first),
                    GOLEM_REUSE_WINDOW_A_PREFETCH=str(a_prefetch),
                    GOLEM_REUSE_WINDOW_SCATTER_WAVE=str(scatter_wave),
                    GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE=str(b_window_bridge),
                    GOLEM_DECODER_FFN_SFU_STREAM=str(sfu_stream),
                    GOLEM_DECODER_FFN_SFU_CHUNK=str(sfu_chunk),
                    GOLEM_DECODER_NORM_FFN_HANDOFF=str(norm_handoff),
                    GOLEM_DECODER_EPOCH_FLAGS=str(epoch_flags),
                    GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP=str(output_overlap),
                    GOLEM_REUSE_WINDOW_QKV_INPUT_PACK=str(qkv_pack),
                    GOLEM_DECODER_GAMMA_REUSE=str(gamma_reuse),GOLEM_REUSE_WEIGHT_COHORT=str(weight_cohort),
                    GOLEM_DECODER_FFN_BLOCK_PIPELINE=str(ffn_block_pipeline),
                    GOLEM_DECODER_FFN_PIPELINE_POLICY=str(ffn_pipeline_policy),
                    GOLEM_DECODER_FFN_PANEL_TILES=str(ffn_panel_tiles))
    if source.count(needle) != 1:
        raise RuntimeError('decoder runner injection point changed')
    destination.write_text(source.replace(needle, needle + f'    env.update({switches!r})\n'))
    return destination


def coverage(case: Path, result: dict) -> dict:
    counts, launches = Counter(), Counter()
    workers = {}
    transfers = Counter()
    qk = pv = 0
    execution = Counter()
    lm_parallel = set()
    window_flows=[];input_flows=[];bridge_flows=[]
    for path in (case/'logs').glob('attention-sst*.log'):
        with path.open() as stream:
            for line in stream:
                if '[GEMM_A_PIPELINE]' in line:
                    input_flows.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
                if '[GEMM_WINDOW_BRIDGE]' in line:bridge_flows.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
                if '[GEMM_WINDOW_FLOW]' in line:
                    fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
                    fields['operator']={'linear':'wo'}.get(fields['operator'],fields['operator'])
                    window_flows.append(fields)
                if '[GEMM_REUSE_WINDOW]' in line:
                    fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                    name = {'linear':'wo'}.get(fields['operator'],fields['operator'])
                    counts[name] += 1
                    launches[name] += int(fields['launches'])
                    workers.setdefault(name,set()).add(int(fields['core']))
                    for metric in ('hbm_read','gm_read','gm_write','windows', 'c_gm_read',
                                   'c_gm_write', 'c_drains', 'c_restores', 'transfer_compute_overlap'):
                        if metric not in fields:
                            continue
                        transfers[f'{name}.{metric}'] += int(fields[metric])
                if '[DECODER_PREFETCH_FLOW]' in line or '[DECODER_GROUP_FLOW]' in line:
                    fields=dict(re.findall(r'(\w+)=(\d+)',line))
                    for metric in ('requests','hits','launches','both_groups_compute_cycles'):
                        if metric in fields:
                            execution[metric]+=int(fields[metric])
                if '[DECODER_LM_FLOW]' in line:
                    fields=dict(re.findall(r'(\w+)=(\d+)',line))
                    lm_parallel.add(int(fields['n_parallel']))
                    execution['lm_launches']+=int(fields['launches'])
                    execution['lm_array_k_tiles']+=int(fields['array_k_tiles'])
                if '[PROJECTION_WEIGHT_PREFETCH]' in line:
                    fields=dict(re.findall(r'(\w+)=(\d+)',line))
                    execution['qkv_legacy_records']+=1
                    execution['qkv_legacy_prefetch_requests']+=int(fields['requests'])
                    execution['qkv_legacy_inflight']+=int(fields['inflight'])
                qk += '[ATTENTION_REUSE_WINDOW_QK]' in line
                pv += '[ATTENTION_WORKER_CLUSTER_PV]' in line
    shape = result['shape']
    mtiles = (shape['padded_S']+63)//64
    hidden, intermediate, layers = (shape[k] for k in ('hidden','intermediate','layers'))
    manifest = json.loads((case/'decoder_stack_inputs.json').read_text())
    # Completion records are emitted once per worker and per scheduled panel.
    # The panel counts below mirror the runner's fixed 16-tile FFN panel and
    # the 128-output-tile LM-head chunks; they also scale to the small smoke
    # shape used by the host validation.
    workers_by_op = {name: len(workers.get(name, ())) for name in
                     ('qkv', 'wo', 'gate', 'up', 'down', 'lm_head')}
    run_env=json.loads((case/'decoder_stack_provenance.json').read_text())['environment']
    panel_tiles=int(run_env.get('GOLEM_DECODER_FFN_PANEL_TILES','16'))
    hidden_tiles = (hidden + 63) // 64
    intermediate_tiles = (shape['intermediate'] + 63) // 64
    vocab_tiles = (result['lm_head']['vocab'] + 63) // 64
    expected = dict(
        qkv=layers * workers_by_op['qkv'],
        wo=layers * workers_by_op['wo'] * max(1, (hidden_tiles + 15) // 16),
        gate=layers * workers_by_op['gate'] * max(1, (intermediate_tiles + panel_tiles-1) // panel_tiles),
        up=layers * workers_by_op['up'] * max(1, (intermediate_tiles + panel_tiles-1) // panel_tiles),
        down=layers * workers_by_op['down'] * max(1, (intermediate_tiles + panel_tiles-1) // panel_tiles),
        # Empty final vocabulary shards have completion descriptors but no
        # GEMM window. Count actual owners of each existing 128-tile chunk.
        lm_head=sum(min(16,vocab_tiles-first) for first in range(0,vocab_tiles,128)),
    )
    required = set(expected)
    run_env=json.loads((case/'decoder_stack_provenance.json').read_text())['environment']
    flow_counts=Counter(f['operator'] for f in window_flows)
    flow_passed=all(flow_counts[name]==counts[name] for name in ('qkv','wo','gate','up','down'))
    output_overlap_cycles=Counter()
    for f in window_flows:
        expected_output=(run_env.get('GOLEM_REUSE_WINDOW_OUTPUT_OVERLAP','0')=='1'
                         and f['operator'] in ('wo','gate','up','lm_head'))
        flow_passed=flow_passed and (
            int(f.get('gm_lookahead','4'))==expected_gm_lookahead(run_env,f['operator'])
            and f.get('a_first','0')==run_env.get('GOLEM_REUSE_WINDOW_A_FIRST','0')
            and
            f['early_a']==run_env.get('GOLEM_REUSE_WINDOW_EARLY_A','0')
            and int(f['output_overlap'])==int(expected_output)
            and sum(int(v) for k,v in f.items() if k.startswith('phase_'))==int(f['active_cycles']))
        output_overlap_cycles[f['operator']]+=int(f['output_compute_overlap'])
    k_chunk=int(run_env['GOLEM_REUSE_WINDOW_K_TILES'])
    producer_chunks=(hidden_tiles+k_chunk-1)//k_chunk
    down_chunks=(min(panel_tiles,intermediate_tiles)+k_chunk-1)//k_chunk
    expected_drains=dict(gate=layers*mtiles*intermediate_tiles*producer_chunks,
                         up=layers*mtiles*intermediate_tiles*producer_chunks,
                         down=layers*mtiles*((intermediate_tiles+panel_tiles-1)//panel_tiles)*hidden_tiles*down_chunks)
    drains_passed=all(transfers[f'{name}.c_drains']==count for name,count in expected_drains.items())
    passed = (all(counts[k] == v for k,v in expected.items()) and qk>0 and pv>0
              and all(workers.get(name,set())==set(range(4,20)) for name in required))
    pipeline_passed=(execution['requests']>0 and execution['hits']==execution['requests']
                     and (execution['lm_launches']>0 or lm_parallel=={1})
                     and lm_parallel=={int(run_env.get('GOLEM_DECODER_LM_N_PARALLEL','4'))}
                     and execution['qkv_legacy_records']==counts['qkv']
                     and execution['qkv_legacy_prefetch_requests']==0
                     and execution['qkv_legacy_inflight']==0)
    if run_env['GOLEM_REUSE_WINDOW_ARRAY_GROUPS']=='2':
        pipeline_passed=pipeline_passed and execution['both_groups_compute_cycles']>0
    passed = passed and pipeline_passed and drains_passed and flow_passed
    input_pipeline_passed=True
    if 'GOLEM_REUSE_WINDOW_A_PREFETCH' in run_env:
        enabled=int(run_env['GOLEM_REUSE_WINDOW_A_PREFETCH'])
        input_pipeline_passed=(len(input_flows)==len(window_flows) and bool(input_flows) and all(
            int(f['enabled'])==enabled and int(f['entries'])==2 and int(f['live_at_finish'])==0
            and (0<int(f['high_water'])<=2 if enabled else int(f['high_water'])==0)
            for f in input_flows))
        if enabled:input_pipeline_passed=input_pipeline_passed and sum(int(f['ready_hits']) for f in input_flows)>0
    passed=passed and input_pipeline_passed
    bridge_passed=True
    if run_env.get('GOLEM_REUSE_WINDOW_B_WINDOW_BRIDGE','0')=='1':
        bridge_passed=(len(bridge_flows)==len(window_flows) and bool(bridge_flows) and all(
            int(f['enabled'])==int((min(panel_tiles,intermediate_tiles) if f['operator']=='down' else hidden_tiles)<=k_chunk)
            and int(f['slots'])==int(run_env['GOLEM_REUSE_WINDOW_PREFETCH_SLOTS'])
            and int(f['capacity_added'])==0 and int(f['live_at_finish'])==0 for f in bridge_flows)
            and sum(int(f['seam_prefetches']) for f in bridge_flows)>0)
    passed=passed and bridge_passed
    return dict(status='PASS' if passed else 'FAIL', completed_window_jobs=dict(counts),
                tile_group_launches=dict(launches), expected_window_job_counts=expected,
                workers={k:sorted(v) for k,v in workers.items()},
                qk_completed_jobs=qk, pv_completed_windows=pv, transfers=dict(transfers),
                execution=dict(execution), lm_n_parallel=sorted(lm_parallel),
                expected_ffn_c_drains=expected_drains,c_traffic_status='PASS' if drains_passed else 'FAIL',
                window_flow_status='PASS' if flow_passed else 'FAIL',window_flow_counts=dict(flow_counts),
                output_compute_overlap_cycles=dict(output_overlap_cycles),
                pipeline_execution_status='PASS' if pipeline_passed else 'FAIL',
                input_pipeline_status='PASS' if input_pipeline_passed else 'FAIL',input_pipeline_flows=input_flows,
                window_bridge_status='PASS' if bridge_passed else 'FAIL',window_bridge_flows=bridge_flows)


def comparison(baseline, result):
    same = baseline['shape']==result['shape'] and baseline['lm_head']['vocab']==result['lm_head']['vocab']
    if not same:
        return dict(comparable=False, reason='validation shape/vocabulary differs from the formal baseline')
    def row(name,b,n):
        return dict(scope=name, baseline_cycles=b, reuse_window_cycles=n,
                    delta_cycles=n-b, change_percent=100*(n-b)/b, speedup=b/n)
    rows=[row('token IDs to final logits DMA ACK',baseline['embedding']['end_to_end_cycles'],result['embedding']['end_to_end_cycles']),
          row('two decoder layers only',baseline['dma_timing']['end_to_end_cycles'],result['dma_timing']['end_to_end_cycles']),
          row('LM head only',baseline['lm_head']['lm_head_cycles'],result['lm_head']['lm_head_cycles'])]
    for layer,b in baseline['dma_timing']['layers'].items():
        n=result['dma_timing']['layers'][layer]
        for metric in ('attention_cycles','post_attention_cycles','layer_cycles'):
            rows.append(row(f'layer {layer} {metric}',b[metric],n[metric]))
    for layer,stages in baseline['lifecycle']['stage_spans'].items():
        for stage,label in (('1','Wo'),('4','fused Gate/Up/Down/residual')):
            rows.append(row(f'layer {layer} {label} span',stages[stage]['span_cycles'],
                            result['lifecycle']['stage_spans'][layer][stage]['span_cycles']))
    b,n=baseline['embedding']['end_to_end_cycles'],result['embedding']['end_to_end_cycles']
    return dict(comparable=True, optimized=n<b, end_to_end=rows[0], scopes=rows,
                note='Stage spans overlap and must not be summed. Positive change_percent means slower.')


def projection_panel_audit(case):
    """Check the materialized last-layer Q/K/V panels, including audit-elided raw outputs."""
    import numpy as np
    from run_decoder_layer import read_bands, linear_reference
    from run_decoder_stack import rms
    manifest=json.loads((case/'decoder_stack_inputs.json').read_text())
    env=json.loads((case/'decoder_stack_provenance.json').read_text())['environment']
    if env.get('GOLEM_FUSION_PROJECTION_ROPE')!='0':
        raise ValueError('panel audit expects QK bridge to own RoPE')
    layer=manifest['layers'];entry=manifest['layout']['layers'][-1]
    hq,hkv=manifest['num_query_heads'],manifest['num_kv_heads']
    rows=manifest['sequence']//4;shared=manifest['layout']['shared']
    from verification_retained_input import last_layer_input
    x,input_proof=last_layer_input(case,manifest)
    expected=linear_reference(rms(x,np.fromfile(case/f'layer{layer}_gamma1.bin',dtype='<f2')),
                              np.load(case/f'layer{layer}_qkv.npy')).reshape(-1,hq+2*hkv,64)
    report={}
    for name,heads,start in (('q',hq,0),('k',hkv,hq),('v',hkv,hq+hkv)):
        pieces=[]
        for node in range(1,5):
            with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
                stream.seek(shared[name+'_panel']);raw=stream.read(heads*rows*128)
            if len(raw)!=heads*rows*128:raise ValueError('short projection panel dump')
            values=np.frombuffer(raw,dtype='<f2')
            if name=='v':values=values.reshape(heads,rows//64,64,64).transpose(0,1,3,2)
            pieces.append(values.reshape(heads,rows,64).transpose(1,0,2))
        actual=np.concatenate(pieces)
        want=expected[:,start:start+heads]
        mismatch=int(np.count_nonzero(actual!=want))
        maximum=float(np.max(np.abs(actual.astype('f4')-want.astype('f4'))))
        report[name]=dict(status='PASS' if mismatch==0 else 'FAIL',mismatches=mismatch,
                          max_abs_error=maximum,elements=int(actual.size))
    return dict(status='PASS' if all(v['status']=='PASS' for v in report.values()) else 'FAIL',
                layer=layer,scope='last-layer inputs to materialized unrotated Q/K/V panels',
                input_reference=input_proof,panels=report)


def summarize(case, baseline_path):
    result=json.loads((case/'decoder_stack_result.json').read_text())
    baseline=json.loads(baseline_path.read_text())
    audit=coverage(case,result)
    panels=projection_panel_audit(case)
    prov=json.loads((case/'decoder_stack_provenance.json').read_text())
    oldprov_path=baseline_path.parent/'decoder_stack_provenance.json'
    oldprov=json.loads(oldprov_path.read_text()) if oldprov_path.exists() else {}
    diff={k:dict(baseline=oldprov.get('environment',{}).get(k), experiment=v)
          for k,v in prov.get('environment',{}).items()
          if oldprov.get('environment',{}).get(k)!=v and not any(s in k for s in ('FILE','ROOT','MANIFEST','EXECUTABLE','MPI_ARGS'))}
    return dict(status='PASS' if result['status']=='PASS' and audit['status']=='PASS' and panels['status']=='PASS' else 'FAIL',
                all_gemm_reuse_window_implemented=True, baseline_preserved=True,
                result_file=str(case/'decoder_stack_result.json'), baseline_result_file=str(baseline_path),
                shape=result['shape'], coverage=audit, projection_panels=panels, comparison=comparison(baseline,result),
                next_token=result['lm_head']['next_token'],
                provenance=dict(guest_matches=prov.get('guest_sha256')==oldprov.get('guest_sha256'),
                                memory_model_library_matches=prov.get('sst_memhierarchy_sha256')==oldprov.get('sst_memhierarchy_sha256'),
                                environment_differences=diff),
                implementation_notes=[
                    'All windows use the existing 64 arrays, single operand context and 2 MiB worker GM. Decoder input windows alias existing live GM input/panel regions; QKV stages its packed input through timed GM scratch.',
                    'Within a four-N window, N is outside the contiguous K accumulation chunk. A is shared in GM; programming A into arrays is charged for every K/N tile. This trades input-program reuse for C locality.',
                    'C stays in the physical array accumulators throughout each K chunk. Between chunks, FP32 C uses bounded timed GM slots; there is no untimed partial-sum vector cache.',
                    'GM input reads overlap matrix programming. Final results cross the array port as FP16; only intermediate K-chunk partials use FP32.',
                    'Bounded independent weight slots prefetch HBM and read GM while the array stream executes. Configurable disjoint M subgroups interleave programming and computation.',
                    f"LM head uses recorded N parallelism {prov['environment'].get('GOLEM_DECODER_LM_N_PARALLEL','4')}; N=1 uses the generic four-N GM reuse window and N>1 shares input across existing single-context arrays.",
                    'Fused FFN panels, FP16 down partials across panels, residual epilogues and guest control flow are preserved.',
                    'Output-side partial ready cycles are fed back into the window scheduler so fusion can overlap without overtaking timed producer writes.',
                    'Pipeline counters report actual requests, hits and overlap; a configuration flag alone does not establish execution.'])


def save_report(root,report):
    (root/'all_gemm_reuse_window_report.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root',type=Path,required=True)
    p.add_argument('--baseline-result',type=Path,default=BASELINE_RESULT)
    p.add_argument('--run',action='store_true')
    p.add_argument('--completed-case',type=Path)
    p.add_argument('--shape',choices=('small','formal'),default='formal')
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--vocab',type=int,default=128256)
    p.add_argument('--cpu-set',default='8-15')
    p.add_argument('--window-k-tiles',type=int,choices=(4,8,16,32),default=32)
    p.add_argument('--window-prefetch-slots',type=int,choices=(1,4),default=4)
    p.add_argument('--window-array-groups',type=int,choices=(1,2),default=1)
    p.add_argument('--window-early-a',type=int,choices=(0,1),default=1)
    p.add_argument('--window-output-overlap',type=int,choices=(0,1),default=1)
    p.add_argument('--qkv-input-pack',type=int,choices=(0,1),default=1)
    p.add_argument('--gamma-reuse',type=int,choices=(0,1),default=1)
    p.add_argument('--ffn-block-pipeline',type=int,choices=(0,1),default=0)
    p.add_argument('--ffn-pipeline-policy',type=int,choices=(0,1,2),default=1)
    p.add_argument('--ffn-panel-tiles',type=int,choices=(4,8,16),default=16)
    p.add_argument('--window-late-lease',type=int,choices=(0,1),default=0)
    p.add_argument('--window-gm-lookahead',type=int,choices=(1,2,4),default=4)
    p.add_argument('--window-a-first',type=int,choices=(0,1),default=0)
    p.add_argument('--window-a-prefetch',type=int,choices=(0,1),default=0)
    p.add_argument('--scatter-wave',type=int,choices=(0,1),default=0,help='one broadcast then two scatter/MAC waves on existing arrays')
    p.add_argument('--b-window-bridge',type=int,choices=(0,1),default=0,help='keep existing four weight slots live across N windows')
    p.add_argument('--ffn-sfu-stream',type=int,choices=(0,1),default=0)
    p.add_argument('--ffn-sfu-chunk',type=int,choices=(1024,2048,4096,8192),default=8192)
    p.add_argument('--epoch-flags',type=int,choices=(0,1),default=0)
    p.add_argument('--norm-ffn-handoff',type=int,choices=(0,1),default=0)
    p.add_argument('--ffn-gm-lookahead',type=int,choices=(0,1,2,4),default=0)
    p.add_argument('--ffn-demand-order',action='store_true')
    p.add_argument('--pv-wo-gm-handoff',action='store_true')
    p.add_argument('--pv-wo-direct-input',action='store_true')
    p.add_argument('--gm-stage-chain',action='store_true',help='keep Wo/H1/Norm2 in existing GM; hardware unchanged')
    p.add_argument('--add-norm-fusion',action='store_true',help='fuse residual Add and RMSNorm2 with bounded SFU result forwarding')
    p.add_argument('--eager-final-norm',action='store_true',help='start final Norm after each worker local FFN completion')
    p.add_argument('--final-norm-lm-stream',action='store_true',help='start LM head after each worker final Norm ACK')
    p.add_argument('--verification-reference',type=Path,help='materialized counterpart for host checks of elided intermediate input')
    p.add_argument('--tail-schedule-experiment',action='store_true',help='opt-in tail guest; usable for stream-off paired control')
    p.add_argument('--final-norm-producer-first',action='store_true',help='prioritize the last-token Norm producer job')
    p.add_argument('--layer-residency',action='store_true',help='retain final residual Y for next Norm1/final Norm')
    p.add_argument('--input-residency',action='store_true',help='retain layer input in existing GM for residual Add')
    p.add_argument('--lm-input-residency',action='store_true',help='retain last-token GM input across LM chunks')
    p.add_argument('--elide-intermediate-output',action='store_true',help='elide non-final layer output HBM mirror while GM stage-chain owns the consumer')
    p.add_argument('--lm-n-parallel',type=int,choices=(1,2,4,8),default=4)
    p.add_argument('--weight-cohort',type=int,choices=(0,2,4),default=0)
    a=p.parse_args()
    if a.pv_wo_gm_handoff and a.shape!='formal':p.error('--pv-wo-gm-handoff requires --shape formal')
    if a.pv_wo_direct_input and not a.pv_wo_gm_handoff:p.error('--pv-wo-direct-input requires --pv-wo-gm-handoff')
    if a.add_norm_fusion and not (a.gm_stage_chain and a.input_residency):p.error('--add-norm-fusion requires --gm-stage-chain --input-residency')
    if a.eager_final_norm and not a.layer_residency:p.error('--eager-final-norm requires --layer-residency')
    if a.final_norm_lm_stream and not a.eager_final_norm:p.error('--final-norm-lm-stream requires --eager-final-norm')
    if a.final_norm_producer_first and not a.final_norm_lm_stream:p.error('--final-norm-producer-first requires --final-norm-lm-stream')
    if a.final_norm_lm_stream:a.tail_schedule_experiment=True
    if a.tail_schedule_experiment and not a.eager_final_norm:p.error('--tail-schedule-experiment requires --eager-final-norm')
    if a.gm_stage_chain and (a.norm_ffn_handoff or a.ffn_block_pipeline or a.weight_cohort):p.error('--gm-stage-chain requires private weights and the original single-window schedule; disable old Norm handoff')
    if a.layer_residency and not a.gm_stage_chain:p.error('--layer-residency requires --gm-stage-chain')
    if a.input_residency and not a.layer_residency:p.error('--input-residency requires --layer-residency')
    if a.elide_intermediate_output and not a.layer_residency:p.error('--elide-intermediate-output requires --layer-residency')
    if a.elide_intermediate_output and not a.input_residency:p.error('--elide-intermediate-output requires --input-residency for the next residual Add')
    root=a.artifact_root.resolve();root.mkdir(parents=True,exist_ok=True)
    hidden=2048 if a.shape=='formal' else 256
    if a.input_residency and a.window_k_tiles<hidden//64:
        p.error('--input-residency requires a full projection K window to preserve the GM payload')
    schedule=gemm_schedule(hidden=hidden,intermediate=4*hidden,layers=a.layers,
                           query_heads=hidden//64,kv_heads=8 if hidden==2048 else 2,vocab=a.vocab)
    schedule['decoder_window'].update(k_tiles=a.window_k_tiles,
                                     prefetch_slots=a.window_prefetch_slots,
                                     array_groups=a.window_array_groups,
                                     early_a=a.window_early_a,output_overlap=a.window_output_overlap,
                                     qkv_input_pack=a.qkv_input_pack,gamma_reuse=a.gamma_reuse,weight_cohort=a.weight_cohort,ffn_block_pipeline=a.ffn_block_pipeline,ffn_panel_tiles=a.ffn_panel_tiles,ffn_pipeline_policy=a.ffn_pipeline_policy,
                                     reuse_level='GM input window; C stationary within K chunk')
    schedule['decoder_window'].update(late_lease=a.window_late_lease,sfu_stream=a.ffn_sfu_stream,sfu_chunk=a.ffn_sfu_chunk,norm_handoff=a.norm_ffn_handoff,lm_parallel=a.lm_n_parallel,epoch_flags=a.epoch_flags,gm_lookahead=a.window_gm_lookahead,a_first=a.window_a_first,a_prefetch=a.window_a_prefetch)
    schedule['decoder_window']['gm_stage_chain']=bool(a.gm_stage_chain)
    schedule['decoder_window']['pv_wo_gm_handoff']=bool(a.pv_wo_gm_handoff)
    schedule['decoder_window']['pv_wo_direct_input']=bool(a.pv_wo_direct_input)
    schedule['decoder_window']['ffn_gm_lookahead']=a.ffn_gm_lookahead
    schedule['decoder_window']['ffn_demand_order']=a.ffn_demand_order
    schedule['decoder_window'].update(layer_residency=bool(a.layer_residency),input_residency=bool(a.input_residency),scatter_wave=a.scatter_wave)
    schedule['decoder_window']['b_window_bridge']=a.b_window_bridge
    schedule['decoder_window']['lm_input_residency']=bool(a.lm_input_residency)
    schedule['decoder_window']['elide_intermediate_output']=bool(a.elide_intermediate_output)
    schedule['decoder_window'].update(add_norm_fusion=bool(a.add_norm_fusion),eager_final_norm=bool(a.eager_final_norm),final_norm_lm_stream=bool(a.final_norm_lm_stream),tail_schedule_experiment=bool(a.tail_schedule_experiment),final_norm_producer_first=bool(a.final_norm_producer_first))
    schedule['operators'][-1]['engine']=('ReuseWindowGemm' if a.lm_n_parallel==1 else f'{a.lm_n_parallel}-array N-parallel input reuse window')
    (root/'all_gemm_reuse_window_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    report=dict(status='SCHEDULE_ONLY',all_gemm_reuse_window_implemented=True)
    if a.run:
        case=root/'reuse_window_probe'
        if case.exists():raise SystemExit(f'refusing to overwrite {case}')
        if a.elide_intermediate_output or a.pv_wo_gm_handoff:
            (root/'verification_reference.json').write_text(json.dumps(
                dict(result=str((a.verification_reference or a.baseline_result).resolve())),indent=2)+'\n')
        report['status']='RUNNING';save_report(root,report)
        snapshot=root/'experiment_source';snapshot.mkdir()
        for path in SOURCES+[Path(__file__).resolve()]:shutil.copy2(path,snapshot/path.name)
        (root/'experiment_provenance.json').write_text(json.dumps(dict(
            source_sha256={str(path.relative_to(ROOT)):sha256(path) for path in SOURCES},
            baseline_runner_sha256=sha256(BASELINE_RUNNER),
            library_sha256=sha256(ROOT/'install/lib/sst-elements-library/libgolem.so')),indent=2)+'\n')
        try:
            with tempfile.TemporaryDirectory(prefix='.all_gemm_reuse_runner_',dir=HERE) as td:
                runner=make_isolated_runner(Path(td)/'run_decoder_stack.py',a.window_k_tiles,
                                            a.window_prefetch_slots,a.window_array_groups,
                                            a.window_early_a,a.window_output_overlap,a.qkv_input_pack,a.gamma_reuse,a.weight_cohort,a.ffn_block_pipeline,a.ffn_panel_tiles,a.ffn_pipeline_policy,a.window_late_lease,a.ffn_sfu_stream,a.ffn_sfu_chunk,a.norm_ffn_handoff,a.epoch_flags,a.window_gm_lookahead,a.window_a_first,a.window_a_prefetch,a.scatter_wave,a.b_window_bridge)
                env=os.environ.copy();env['PYTHONPATH']=str(HERE)+os.pathsep+env.get('PYTHONPATH','')
                cmd=['python3',str(runner),'--artifact-root',str(case),'--shape',a.shape,
                     '--layers',str(a.layers),'--sequence','1024','--weights','sparse','--embedding',
                     '--vocab',str(a.vocab),'--mpi-ranks','8','--cpu-set',a.cpu_set,
                     '--gemm-prefetch','--prefetch-slots','4','--weight-trace',
                     '--ffn-down-fusion','--down-residual-fusion','--ffn-panel-tiles',str(a.ffn_panel_tiles),
                     '--ffn-array-groups','2','--ffn-layout','2','--lm-n-parallel',str(a.lm_n_parallel)]
                cmd+=['--ffn-gm-lookahead',str(a.ffn_gm_lookahead)]
                if a.ffn_demand_order:cmd.append('--ffn-demand-order')
                if a.pv_wo_gm_handoff:cmd.append('--pv-wo-gm-handoff')
                if a.pv_wo_direct_input:cmd.append('--pv-wo-direct-input')
                if a.gm_stage_chain:cmd.append('--gm-stage-chain')
                if a.add_norm_fusion:cmd.append('--add-norm-fusion')
                if a.eager_final_norm:cmd.append('--eager-final-norm')
                if a.final_norm_lm_stream:cmd.append('--final-norm-lm-stream')
                if a.tail_schedule_experiment:cmd.append('--tail-schedule-experiment')
                if a.final_norm_producer_first:cmd.append('--final-norm-producer-first')
                if a.layer_residency:cmd.append('--layer-residency')
                if a.input_residency:cmd.append('--input-residency')
                if a.lm_input_residency:cmd.append('--lm-input-residency')
                if a.elide_intermediate_output:cmd.append('--elide-intermediate-output')
                subprocess.run(cmd,cwd=ROOT,env=env,check=True)
            report=summarize(case,a.baseline_result.resolve())
        except Exception as error:
            report.update(status='FAIL',error=str(error));save_report(root,report);raise
    elif a.completed_case:report=summarize(a.completed_case.resolve(),a.baseline_result.resolve())
    save_report(root,report);print(json.dumps(report,indent=2))
    if report['status']=='FAIL':raise SystemExit(1)

if __name__=='__main__':main()
