#!/usr/bin/env python3
"""Summarize completed fixed-hardware causal experiments with the canonical DMA boundary."""
import argparse
import json
from pathlib import Path

from run_projection_overlap import digest
from analyze_projection_holes import analyze as holes
from analyze_projection_causal import analyze as causal


def summarize(root):
    root=Path(root)
    cases=[]
    model_hashes=None
    for sequence in (1024,2048):
        baseline=json.loads((root/f'baseline_s{sequence}'/'dma_e2e_result.json').read_text())['end_to_end_cycles']
        for variant in ('baseline','budget1','depth8'):
            name=f'{variant}_s{sequence}'+('_v2' if sequence==1024 and variant!='baseline' else '')
            case=root/name
            experiment=json.loads((case/'experiment_result.json').read_text())
            if experiment['status']!='PASS':raise ValueError('experiment did not pass')
            dma=json.loads((case/'dma_e2e_result.json').read_text())
            if dma['status']!='PASS' or dma['contract']['version']!='hbmdma_to_O_ack_v1':
                raise ValueError('canonical DMA report missing')
            prov=json.loads((case/'provenance.json').read_text())
            hashes={k:v for k,v in prov['sha256'].items() if k.endswith(('.h','.cc','.so')) or '/riscv64/' in k}
            if model_hashes is None: model_hashes=hashes
            if hashes!=model_hashes: raise ValueError('model or guest hashes differ between cases')
            h=holes(case)
            (case/'projection_holes.json').write_text(json.dumps(h,indent=2)+'\n')
            c=causal(case)
            (case/'projection_causal_analysis.json').write_text(json.dumps(c,indent=2)+'\n')
            core=str(h['critical_worker'])
            current=dma['end_to_end_cycles']
            cases.append({'case':name,'sequence':sequence,'variant':variant,
                'e2e_cycles':current,'saved_cycles':baseline-current,'saved_percent':100*(baseline-current)/baseline,
                'projection_cycles':h['projection_stage_cycles'],'critical_worker':int(core),
                'critical_worker_holes':h['by_worker'][core],
                'critical_worker_requests':c['by_worker'][core],
                'bitwise_output_verification':experiment['byte_exact'],
                'artifact_root':str(case)})
    return {'status':'PASS','contract':'hbmdma_to_O_ack_v1','cases':cases,'model_guest_hashes':model_hashes,
        'scope':'Same simulator library, guest, physical resources and input references across six runs. Two one-factor experiments: background input read budget 0->1; weight DMA lookahead 4->8. They do not increase SRAM, operand banks, array count, DMA slots, GM ports or bandwidth.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args()
    result=summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for case in result['cases']:
        print(f'{case["case"]}: DMA E2E={case["e2e_cycles"]}, saved={case["saved_cycles"]} ({case["saved_percent"]:.3f}%), Projection={case["projection_cycles"]}')

if __name__=='__main__':main()
