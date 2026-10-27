#!/usr/bin/env python3
"""Controlled diagnostic oracles and the complete fixed-hardware 2**3 matrix."""
import argparse
import itertools
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

from run_projection_overlap import ROOT, HERE, SOURCES, digest, summarize
from run_causal_balance import BEST
from run_llama_projection import GM_PROFILE
from analyze_resource_timeline import analyze as analyze_resources
from analyze_critical_path import analyze as analyze_critical

ORACLES = ('gm_read', 'matrix_program', 'input_scatter', 'weight_arrival',
           'worker_start', 'projection_barrier', 'attention_barrier')
MATRIX = {''.join(name for name, on in zip('IWG', (i, w, g - 1)) if on) or 'B': (i, w, g)
          for i, w, g in itertools.product((0, 1), (0, 1), (1, 2))}
SOURCE_FILES = list(dict.fromkeys(SOURCES + [ROOT / path for path in (
    'src/sst/elements/golem/attention/diagnosticOracle.h',
    'src/sst/elements/golem/attention/projectionJobAbi.h',
    'baseline/attention_cluster_8qk_8pv/run_batch_b.py',
    'baseline/attention_cluster_8qk_8pv/run_projection_overlap.py',
    'baseline/attention_cluster_8qk_8pv/analyze_resource_timeline.py',
    'baseline/attention_cluster_8qk_8pv/analyze_critical_path.py',
    'src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py',
)]))


def cases_for(suite):
    cases = [('baseline', 'none', (0, 0, 1))]
    if suite in ('all', 'oracles'):
        cases += [('oracle_' + name, name, (0, 0, 1)) for name in ORACLES]
    if suite in ('all', 'matrix'):
        cases += [('matrix_' + name, 'none', factors) for name, factors in MATRIX.items() if name != 'B']
    return cases


def environment(case, factors, oracle, sequence, stage, origin=None):
    i, w, g = factors
    # Do not inherit experimental GOLEM switches from the caller's shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GOLEM_') or k in ('GOLEM_ARTIFACT_POLICY','GOLEM_ARTIFACT_MIN_FREE_GIB')}
    env.update(GM_PROFILE)
    env.update(BEST)
    env.update({
        'GOLEM_PROJECTION_ARRAYS': '64', 'GOLEM_PROJECTION_RETAIN_OUTPUT': '1',
        'GOLEM_PROJECTION_PIPELINE': '1', 'GOLEM_PROJECTION_SHARED_INPUT': '1',
        'GOLEM_PROJECTION_WEIGHT_PREFETCH': '1', 'GOLEM_PROJECTION_WEIGHT_PREFETCH_DEPTH': '4',
        'GOLEM_ARRAY_OPERAND_CONTEXT_BANKS': '1', 'GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE': '0',
        'GOLEM_ATTENTION_CAUSAL_BALANCE': '1', 'GOLEM_PROJECTION_ROW_FIRST': '0',
        'GOLEM_PROJECTION_INPUT_PAIR_PREFETCH': str(i), 'GOLEM_PROJECTION_LOCAL_WEIGHT_PREFETCH': str(w),
        'GOLEM_PROJECTION_ARRAY_GROUPS': str(g), 'GOLEM_PROJECTION_OVERLAP_TRACE': '0',
        'GOLEM_CRITICAL_PATH_TRACE': '1', 'GOLEM_RESOURCE_TRACE_DOMAINS': 'gm,array,wcp,projection',
        'GOLEM_DIAGNOSTIC_ORACLE': oracle,
        'GOLEM_MPI_RANK_LOG_ROOT': str(case / 'mpi_rank_output'),
        'GOLEM_MPI_ARGS': f'--bind-to core --cpu-set 4-7 --output-filename {case / "mpi_rank_output"}',
    })
    if oracle == 'worker_start':
        if origin is None:
            raise ValueError('worker_start must use the measured baseline origin')
        env['GOLEM_ORACLE_WORKER_START_CYCLE'] = str(origin)
    return env


def factorial_effects(values):
    if set(values) != set(MATRIX):
        raise ValueError('factorial effects require all eight cases')
    def mean_delta(factor):
        differences = []
        for name, triple in MATRIX.items():
            if triple[factor] != (1 if factor == 2 else 0):
                continue
            high = list(triple)
            high[factor] += 1
            other = next(n for n, t in MATRIX.items() if tuple(high) == t)
            differences.append(values[other] - values[name])
        return sum(differences) / len(differences)
    def iw(g):
        return values['IW' + g] - values['I' + g] - values['W' + g] + values[g or 'B']
    return {'main_effect_cycles': {name: mean_delta(k) for k, name in enumerate('IWG')},
            'interaction_IW_cycles': {'G1': iw(''), 'G2': iw('G')},
            'interaction_IG_cycles': {'W0': values['IG']-values['I']-values['G']+values['B'],
                                      'W1': values['IWG']-values['IW']-values['WG']+values['W']},
            'interaction_WG_cycles': {'I0': values['WG']-values['W']-values['G']+values['B'],
                                      'I1': values['IWG']-values['IW']-values['IG']+values['I']},
            'interaction_IWG_cycles': iw('G') - iw(''),
            'sign': 'negative main effect reduces latency; positive pair interaction weakens additive savings'}


def report(results, expected):
    output = {'status': 'PASS' if len(results) == expected else 'RUNNING',
              'expected_cases': expected, 'completed_cases': len(results), 'cases': results,
              'factorial': {}, 'oracles': {},
              'scope': 'Oracle results are diagnostic interventions, not deployable speedups or a global bound. No complete cross-resource DAG/slack is claimed.'}
    sequences = sorted({c['shape']['query_length'] for c in results})
    for seq in sequences:
        cases = {c['case']: c for c in results if c['shape']['query_length'] == seq}
        baseline = cases.get('baseline')
        if baseline is None:
            continue
        origin = baseline['fixed_origin_cycle']
        output['oracles'][str(seq)] = {}
        for oracle in ORACLES:
            c = cases.get('oracle_' + oracle)
            if c:
                saved = baseline['fixed_origin_e2e_cycles'] - c['fixed_origin_e2e_cycles']
                output['oracles'][str(seq)][oracle] = {
                    'fixed_origin_cycle': origin, 'baseline_cycles': baseline['fixed_origin_e2e_cycles'],
                    'oracle_cycles': c['fixed_origin_e2e_cycles'], 'saved_cycles': saved,
                    'saved_percent': 100 * saved / baseline['fixed_origin_e2e_cycles'],
                    'above_1_percent': saved >= baseline['fixed_origin_e2e_cycles'] * .01}
        matrix = {'B': baseline['fixed_origin_e2e_cycles']}
        matrix.update({n: cases['matrix_' + n]['fixed_origin_e2e_cycles'] for n in MATRIX
                       if 'matrix_' + n in cases})
        if set(matrix) == set(MATRIX):
            output['factorial'][str(seq)] = {'e2e_cycles': matrix, **factorial_effects(matrix)}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--stage', choices=('small', 'formal'), default='formal')
    p.add_argument('--suite', choices=('baseline', 'oracles', 'matrix', 'all'), default='all')
    p.add_argument('--sequences', type=int, nargs='+', default=[1024, 2048])
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--case-names', nargs='+', help='Independent shard cases; requires a verified shared baseline for non-baseline cases')
    p.add_argument('--cpu-set', default='4-7', help='Host MPI affinity only; does not change simulated hardware')
    args = p.parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    hashes = {str(f.relative_to(ROOT)): digest(f) for f in SOURCE_FILES}
    stamp = root / 'source_sha256.json'
    if stamp.exists():
        if not args.resume or json.loads(stamp.read_text()) != hashes:
            raise ValueError('existing batch or changed sources; choose a new artifact root')
    else:
        stamp.write_text(json.dumps(hashes, indent=2) + '\n')
        for f in SOURCE_FILES:
            if f.suffix in ('.h', '.cc', '.py', '.sh'):
                dest = root / 'source_version' / f.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
    chosen = cases_for(args.suite)
    if args.case_names:
        if len(set(args.case_names)) != len(args.case_names) or not set(args.case_names) <= {c[0] for c in chosen}:
            raise ValueError('invalid or duplicate shard case names')
        chosen = [c for c in chosen if c[0] in args.case_names]
    expected = len(chosen) * len(args.sequences)
    results = []
    for sequence in args.sequences:
        origin = None
        if chosen[0][0] != 'baseline':
            if not args.reference_root:
                raise ValueError('non-baseline shard requires a verified baseline')
            ref = args.reference_root.resolve() / f's{sequence}'
            base = json.loads((ref/'experiment_result.json').read_text())
            if base['status'] != 'PASS' or base['oracle'] != 'none' or tuple(base['factors']) != (0,0,1):
                raise ValueError('shard reference must be the all-off baseline')
            origin = base['fixed_origin_cycle']
        for name, oracle, factors in chosen:
            case = root / name / f's{sequence}'
            complete = case / 'experiment_result.json'
            if args.resume and complete.exists():
                result = json.loads(complete.read_text())
                if result['status'] != 'PASS': raise ValueError('cannot resume failed case')
            else:
                case.mkdir(parents=True, exist_ok=False)
                own_baseline = root / 'baseline' / f's{sequence}'
                reference = (own_baseline if name != 'baseline' and (own_baseline/'experiment_result.json').exists()
                    else args.reference_root.resolve() / f's{sequence}' if args.reference_root else None)
                env = environment(case, factors, oracle, sequence, args.stage, origin)
                env['GOLEM_MPI_ARGS'] = f'--bind-to core --cpu-set {args.cpu_set} --output-filename {case / "mpi_rank_output"}'
                if reference:
                    env['GOLEM_PROJECTION_INPUT_REFERENCE'] = str(reference)
                heads = (32, 8) if args.stage == 'formal' else (4, 2)
                command = [str(HERE/'run_sst.sh'), '--projection', '--query-length', str(sequence),
                    '--kv-length', str(sequence), '--num-query-heads', str(heads[0]), '--num-kv-heads', str(heads[1]),
                    '--head-dim', '64', '--causal', '--rope', '--artifact-root', str(case)]
                provenance = {'sha256': hashes, 'command': command,
                              'environment': {k:v for k,v in env.items() if k.startswith('GOLEM_')}}
                (case/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
                print(f'Running {name} S{sequence}', flush=True)
                with (case/'driver.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if hashes != {str(f.relative_to(ROOT)): digest(f) for f in SOURCE_FILES}:
                    raise ValueError('source/binary changed during batch')
                result = summarize(case, reference)
                projection = json.loads((case/'projection_e2e_result.json').read_text())
                origin = projection['chain_start_cycle'] if name == 'baseline' else origin
                result.update({'case': name, 'oracle': oracle, 'factors': factors,
                    'fixed_origin_cycle': origin,
                    'fixed_origin_e2e_cycles': projection['stages']['attention']['end_cycle'] - origin,
                    'reported_chain_start_cycle': projection['chain_start_cycle']})
                resources = analyze_resources(case, allow_filtered=True)
                (case/'resource_analysis.json').write_text(json.dumps(resources,indent=2)+'\n')
                result['resource_intervals'] = resources['interval_count']
                result['projection_worker_overlap'] = resources['projection_worker_overlap']
                result['worker_control'] = resources['worker_control']
                result['output_ack_to_root_cycles'] = resources['output_ack_to_root_cycles']
                log = next((case/'logs').glob('attention-sst*.log')).read_text(errors='replace')
                if oracle in ('gm_read','matrix_program','input_scatter'):
                    resource = {'gm_read':'gm.local_read','matrix_program':'array.matrix_broadcast',
                                'input_scatter':'array.input_scatter'}[oracle]
                    busy_key = {'gm_read':'gm_read_busy_cycles','matrix_program':'matrix_program_busy_cycles',
                                'input_scatter':'input_scatter_busy_cycles'}[oracle]
                    measured = resources['projection_worker_overlap']
                    # GM fusion reads remain timed; matrix/scatter are fully
                    # eligible in Projection. Require observable intervention.
                    if oracle != 'gm_read' and any(v[busy_key] for v in measured.values()):
                        raise ValueError('oracle did not eliminate selected service')
                    result['oracle_service_validation'] = {'resource':resource,
                        'by_worker_busy_cycles':{k:v[busy_key] for k,v in measured.items()}}
                if oracle == 'worker_start':
                    starts = re.findall(r'\[ORACLE_EVENT\] oracle=worker_start core=(\d+) cycle=(\d+) action=projection_start',log)
                    if {int(c) for c,t in starts} != set(range(4,20)) or len(starts)!=16 or any(int(t)!=origin for c,t in starts):
                        raise ValueError('worker_start did not preserve the fixed origin for all workers')
                    result['oracle_start_span_cycles'] = 0
                if oracle == 'projection_barrier':
                    ready = re.findall(r'\[ORACLE_EVENT\] oracle=projection_barrier core=(\d+) cycle=(\d+) action=all_producers_ready mask=65535',log)
                    if {int(c) for c,t in ready} != set(range(4)) or len(ready)!=4:
                        raise ValueError('barrier oracle lacks verified producer-ready notifications')
                if oracle == 'attention_barrier':
                    result['oracle_scheduler_progress_passes'] = sum(int(n) for n in re.findall(
                        r'\[ORACLE_SUMMARY\] oracle=attention_barrier core=\d+ scheduler_progress_passes=(\d+)',log))
                if name == 'baseline':
                    critical = analyze_critical(case)
                    (case/'critical_path_analysis.json').write_text(json.dumps(critical,indent=2)+'\n')
                    if critical['reconstructed_e2e_cycles'] != result['fixed_origin_e2e_cycles']:
                        raise ValueError('baseline reconstruction differs from frozen origin')
                provenance['input_sha256'] = {f.name: digest(f) for f in sorted(case.glob('*.bin'))}
                (case/'baseline_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
                complete.write_text(json.dumps(result,indent=2)+'\n')
            if name == 'baseline':
                origin = result['fixed_origin_cycle']
                if args.reference_root and result['end_to_end_comparison']['saved_cycles'] != 0:
                    raise ValueError('all-off baseline changed; stopping batch')
            results.append(result)
            (root/'summary.json').write_text(json.dumps(report(results,expected),indent=2)+'\n')
            print(f'{name} S{sequence}: PASS E2E={result["fixed_origin_e2e_cycles"]}',flush=True)
    print('Batch B complete',flush=True)


if __name__ == '__main__':
    main()
