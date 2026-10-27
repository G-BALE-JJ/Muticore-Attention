#!/usr/bin/env python3
"""Compare matched decoder runs, with byte-identical inputs and output checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from run_decoder_layer import read_bands


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identical_fp16_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and bool(
        np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def profile(case):
    r = json.loads((case / 'decoder_stack_result.json').read_text())
    m = json.loads((case / 'decoder_stack_inputs.json').read_text())
    jobs, flow, sfu_tokens = [], [], {}
    with next((case / 'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done' in line or '[DECODER_DATAFLOW]' in line:
                fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
                (flow if '[DECODER_DATAFLOW]' in line else jobs).append(fields)
            if 'GOLEM_SFU_HW_PIPELINE' in line:
                match=re.search(r'core=(\d+) unit=(exp|reciprocal) .*?accepted_tokens=(\d+)',line)
                if match:sfu_tokens[f'{match[1]}.{match[2]}']=int(match[3])
    expected_sfu={f'{core}.{unit}' for core in range(4,20) for unit in ('exp','reciprocal')}
    if set(sfu_tokens)!=expected_sfu:
        raise ValueError(f'{case}: missing or unexpected worker EXP/reciprocal statistics')
    critical = {}
    for epoch in (1, 2):
        last = max((j for j in jobs if j['stage'] == (4 if m.get('down_residual_fusion') else 9) and j['epoch'] == epoch), key=lambda j: j['tick'])
        core = last['core']
        critical[str(epoch)] = {'core': core, 'jobs_cycles': {
            str(j['stage']): (j['tick'] - j['start_tick']) / 1000
            for j in jobs if j['core'] == core and j['epoch'] == epoch and j['stage'] <= 9}}
    preloads = {(e['name'], e['node'], e['offset']): digest(e['path']) for e in m['preloads']}
    legacy_requests=sum((j['hbm_write']-(8 if j['stage']==9 else 0))//128
                        for j in jobs if j['stage'] in (1,4,5,8))
    return r, m, preloads, {
        'status': r['status'], 'total_cycles': r['embedding']['end_to_end_cycles'],
        'layer_post_attention_cycles': {k: v['post_attention_cycles'] for k, v in r['dma_timing']['layers'].items()},
        'LM_head_cycles': r['lm_head']['lm_head_cycles'],
        'last_finishing_worker_chains': critical,
        'ffn_panel_flow':r.get('lifecycle',{}).get('ffn_panel_flow',[]),
        'ffn_down_flow':r.get('lifecycle',{}).get('ffn_down_flow',[]),
        'SFU_exp_reciprocal_accepted_tokens':sfu_tokens,
        'linear_output_requests': sum(f['output_requests'] for f in flow if f['stage'] in (1, 4, 5, 8)) if flow else legacy_requests,
        'output_request_evidence': 'DECODER_DATAFLOW counters' if flow else 'legacy 128-byte serial writes inferred from DECODER_OP bytes',
        'max_output_in_flight': max((f['output_high_water'] for f in flow),default=1),
        'prefetch_requests': sum(f.get('prefetch_requests', 0) for f in flow),
        'HBM_read_bytes_decoder': sum(j['hbm_read'] for j in jobs if j['stage'] <= 9),
        'HBM_write_bytes_decoder': sum(j['hbm_write'] for j in jobs if j['stage'] <= 9),
        'next_token': r['lm_head']['next_token']['token_id'] if 'token_id' in r['lm_head']['next_token'] else r['lm_head']['next_token'],
        'wall': json.loads((case / 'simulation_wall_time.json').read_text()),
        'provenance': json.loads((case / 'decoder_stack_provenance.json').read_text()),
        'verification_provenance': json.loads((case/'verification_provenance.json').read_text()) if (case/'verification_provenance.json').exists() else None,
    }


def summarize(baseline, variants):
    br, bm, inputs, base = profile(baseline)
    if base['status'] != 'PASS':
        raise ValueError('baseline did not pass')
    report = {'baseline': str(baseline), 'baseline_profile': base, 'variants': {}}
    for name, case in variants:
        r, m, other_inputs, p = profile(case)
        if r['status'] != 'PASS' or inputs != other_inputs:
            raise ValueError(f'{name}: failed run or different input preloads')
        equal = {}
        for layer in (0, 1):
            equal[f'layer{layer+1}_output'] = identical_fp16_bits(
                read_bands(baseline, bm['layout']['layers'][layer]['output'], bm['hidden']),
                read_bands(case, m['layout']['layers'][layer]['output'], m['hidden']))
        equal['final_norm'] = identical_fp16_bits(
            read_bands(baseline, bm['layout']['final_norm']['output'], bm['hidden']),
            read_bands(case, m['layout']['final_norm']['output'], m['hidden']))
        equal['logits'] = identical_fp16_bits(np.load(baseline / 'last_token_logits.npy'), np.load(case / 'last_token_logits.npy'))
        for key in ('wo', 'h1', 'norm2', 'gate', 'up', 'gated', 'down'):
            if key in ('gate','up') and (bm.get('ffn_panel_fusion') or m.get('ffn_panel_fusion')):
                continue  # Elided producer tensors have no HBM output to compare.
            if key=='gated' and (bm.get('ffn_down_fusion') or m.get('ffn_down_fusion')):continue
            if key=='down' and (bm.get('down_residual_fusion') or m.get('down_residual_fusion')):continue
            width = bm['hidden'] * (4 if key in ('gate', 'up', 'gated') else 1)
            equal[key] = identical_fp16_bits(read_bands(baseline, bm['layout']['shared'][key], width),
                                            read_bands(case, m['layout']['shared'][key], width))
        if not all(equal.values()):
            raise ValueError(f'{name}: outputs differ: {equal}')
        p['elided_output_tensors'] = [key for key in ('gate','up','act') if m.get('ffn_panel_fusion') or (key=='act' and m.get('silu_multiply_fusion'))]
        if m.get('ffn_down_fusion'):p['elided_output_tensors'].append('gated')
        if m.get('down_residual_fusion'):p['elided_output_tensors'].append('down')
        p['identical_SFU_exp_reciprocal_work'] = p['SFU_exp_reciprocal_accepted_tokens'] == base['SFU_exp_reciprocal_accepted_tokens']
        if not p['identical_SFU_exp_reciprocal_work']:
            raise ValueError(f'{name}: EXP/reciprocal work differs from the baseline')
        p['identical_input_preloads'] = True
        p['identical_guest_binary'] = p['provenance']['guest_sha256'] == base['provenance']['guest_sha256']
        p['identical_SST_library'] = p['provenance']['sst_library_sha256'] == base['provenance']['sst_library_sha256']
        p['bitwise_equal_outputs'] = equal
        p['cycle_reduction_percent'] = 100 * (1 - p['total_cycles'] / base['total_cycles'])
        p['output_request_reduction_percent'] = 100 * (1 - p['linear_output_requests'] / base['linear_output_requests'])
        report['variants'][name] = p
    report['status'] = 'PASS'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--variant', action='append', required=True, help='name=case directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    variants = [(v.split('=', 1)[0], Path(v.split('=', 1)[1])) for v in args.variant]
    report = summarize(args.baseline, variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {k: p[k] for k in ('total_cycles', 'cycle_reduction_percent', 'linear_output_requests')}
                      for name, p in report['variants'].items()}, indent=2))


if __name__ == '__main__':
    main()
