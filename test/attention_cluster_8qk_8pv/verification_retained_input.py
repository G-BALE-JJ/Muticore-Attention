"""Host-only input oracle for outputs intentionally absent from HBM.

The complete chain still has its independent mathematical reference. Local
operator checks use the materialized input of a verified, identical-input run;
this avoids treating stale candidate HBM as the actual retained GM payload.
"""
import hashlib
import json
from pathlib import Path
from run_decoder_layer import read_bands


def _inputs(manifest):
    m=json.loads(json.dumps(manifest))
    m.pop('_lifecycle',None)
    m.pop('elide_intermediate_output',None)
    m.pop('pv_wo_gm_handoff',None)
    m.pop('pv_wo_direct_input',None)
    # These software schedules preserve the explicit FP16 operator boundaries.
    # Input/weight hashes, layouts, all other flags and the independent complete
    # mathematical chain remain mandatory; record the differing flags below.
    for name in ('add_norm_fusion','eager_final_norm','final_norm_lm_stream','tail_schedule_experiment','final_norm_producer_first','lm_handoff_protocol','ffn_demand_order'):m.pop(name,None)
    for item in m.get('preloads',[]):item.pop('path',None)
    return m


def last_layer_input(case,manifest):
    hidden=manifest['hidden'];rows=manifest['sequence']//4
    offset=manifest['layout']['layers'][-1]['input']
    if not manifest.get('elide_intermediate_output') or manifest['layers']==1:
        return read_bands(case,offset,hidden,rows=rows),dict(source='device HBM')
    config=case.parent/'verification_reference.json'
    if not config.exists():
        raise ValueError('elided HBM input requires an explicit verified materialized reference')
    reference=Path(json.loads(config.read_text())['result']).resolve()
    result=json.loads(reference.read_text());other=reference.parent
    m=json.loads((other/'decoder_stack_inputs.json').read_text())
    if result.get('status')!='PASS' or m.get('elide_intermediate_output') or _inputs(m)!=_inputs(manifest):
        raise ValueError('retained-input reference must be PASS, materialized and have identical inputs/layout/configuration')
    x=read_bands(other,offset,hidden,rows=rows)
    proof=dict(source='verified materialized counterpart; host validation only',
               reference_result=str(reference),
               reference_result_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
               input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
               elements=int(x.size),candidate_hbm_input_intentionally_absent=True)
    proof['equivalent_software_schedules']={name:dict(candidate=bool(manifest.get(name)),reference=bool(m.get(name)))
        for name in ('add_norm_fusion','eager_final_norm','final_norm_lm_stream','tail_schedule_experiment','final_norm_producer_first','lm_handoff_protocol','ffn_demand_order') if bool(manifest.get(name))!=bool(m.get(name))}
    (case/'verification_retained_input.json').write_text(json.dumps(proof,indent=2)+'\n')
    return x,proof


def retained_attention(case,manifest):
    """Use the same-input materialized run only as a host numerical oracle."""
    config=case.parent/'verification_reference.json'
    if not config.exists():raise ValueError('PV-Wo handoff requires a verified materialized Attention reference for local checks')
    reference=Path(json.loads(config.read_text())['result']).resolve()
    result=json.loads(reference.read_text());other=reference.parent
    m=json.loads((other/'decoder_stack_inputs.json').read_text())
    if result.get('status')!='PASS' or m.get('pv_wo_gm_handoff') or _inputs(m)!=_inputs(manifest):
        raise ValueError('PV-Wo reference must be PASS, materialized O and have identical inputs/layout/configuration')
    x=read_bands(other,m['layout']['shared']['o'],m['hidden'],rows=m['sequence']//4)
    proof=dict(source='verified materialized counterpart; host validation only',
               reference_result=str(reference),reference_result_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
               attention_sha256=hashlib.sha256(x.tobytes()).hexdigest(),elements=int(x.size),
               candidate_hbm_attention_intentionally_absent=True)
    return x,proof
