"""Require timed GM payload/publication ACKs before omitting an HBM Y mirror."""
import json
import re
from analyze_resource_timeline import merged


FIELDS=re.compile(r'(\w+)=(-?\d+)')


def audit(case,jobs,manifest):
    p=case/'decoder_stack_provenance.json'
    env=json.loads(p.read_text()).get('environment',{}) if p.exists() else {}
    enabled=env.get('GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT','0')=='1'
    if enabled!=bool(manifest.get('elide_intermediate_output',False)):
        raise ValueError('output residency manifest/configuration mismatch')
    if enabled and (not manifest.get('layer_residency') or not manifest.get('input_residency') or not manifest.get('gm_stage_chain') or
                    manifest.get('sequence',1024)>1024 or manifest.get('retain_outputs')):
        raise ValueError('output elision requires eligible GM-only intermediate consumers')
    expected={(j['core'],j['epoch']):j for js in jobs.values() for j in js
              if enabled and j.get('op')==9 and j['stage']==4 and j['epoch']<manifest['layers']}
    log=next((case/'logs').glob('attention-sst*.log'));flows=[]
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OUTPUT_RESIDENCY]' in line:
                flows.append({k:int(v) for k,v in FIELDS.findall(line)})
    if len(flows)!=len(expected) or {(f['core'],f['epoch']) for f in flows}!=set(expected):
        raise ValueError('output residency lifecycle missing/duplicate/unexpected')
    size=64*manifest.get('hidden',2048)*2
    for f in flows:
        j=expected[f['core'],f['epoch']]
        if (f['stage']!=4 or f['local_produced']!=size or f['saved_hbm_write']!=size or
                f['published']!=1 or f['capacity_added'] or f['status'] or j['hbm_write']!=8):
            raise ValueError('output residency bytes/publication/completion mismatch')
    states={k:dict(spans=[],bytes=0,last=0,records=[],record_bytes=0,first_record=None) for k in expected}
    if expected:
        with log.with_suffix('.resources.jsonl').open() as stream:
            for line in stream:
                if 'resource=local_write ' not in line:continue
                e={k:int(v) for k,v in FIELDS.findall(line)}
                key=next((k for k,j in expected.items() if e['core']==k[0] and
                          j['start_tick']<=e['start_tick']<=e['done_tick']<=j['tick']),None)
                if key is None:continue
                state=states[key];base=e['core']*0x200000;start=e['address'];stop=start+e['bytes']
                for lo,hi,record in ((base+0xb0000,base+0xb0000+size,False),(base+0x3400,base+0x3440,True)):
                    if start>=hi or stop<=lo:continue
                    if e['ok']!=1 or not lo<=start<stop<=hi:raise ValueError('invalid resident Y timed GM write')
                    if record:
                        state['records'].append((start,stop));state['record_bytes']+=e['bytes']
                        state['first_record']=min(state['first_record'] or e['start_tick'],e['start_tick'])
                    else:
                        state['spans'].append((start,stop));state['bytes']+=e['bytes']
                        state['last']=max(state['last'],e['done_tick'])
        for (core,epoch),state in states.items():
            base=core*0x200000
            if (merged(state['spans'])!=[(base+0xb0000,base+0xb0000+size)] or state['bytes']!=size or
                    merged(state['records'])!=[(base+0x3400,base+0x3440)] or state['record_bytes']!=64 or
                    state['first_record'] is None or state['last']>state['first_record']):
                raise ValueError('resident Y must be complete before publishing GM ownership')
    return dict(status='PASS',enabled=enabled,verified_jobs=len(expected),flows=flows,
                saved_hbm_write_bytes=sum(f['saved_hbm_write'] for f in flows),
                actual_gm_payload_bytes=sum(s['bytes'] for s in states.values()),
                scope='Timed GM Y write ACK coverage precedes ownership publication and job completion; final layer retains HBM output.')
