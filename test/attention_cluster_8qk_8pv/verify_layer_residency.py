"""Audit actual software GM Y publishers and next-layer/final-norm consumers."""
import json
import re
from collections import defaultdict


def verify_projection_payload(case, manifest):
    """Check real local port events inside each full-row Norm1 batch."""
    from analyze_resource_timeline import merged
    if not manifest.get('layer_residency') or manifest['sequence']>1024:
        return dict(status='PASS',batches=0,saved_hbm_read_bytes=0)
    accepts=defaultdict(list);ends=defaultdict(list);norms=defaultdict(list)
    log=next((case/'logs').glob('attention-sst*.log'))
    with log.open() as stream:
        for line in stream:
            if not any(s in line for s in ('[CONTROL_EVENT] stage=projection_accept ', '[PROJECTION_JOB]', '[SFU_RMSNORM]')):continue
            f={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)}
            if 'stage=projection_accept ' in line:accepts[f['core']].append(f)
            elif '[PROJECTION_JOB]' in line:ends[f['core']].append(f)
            else:norms[f['core']].append(f)
    batches=defaultdict(list);hidden=manifest['hidden'];stride=manifest['layout']['node_stride']
    for core in range(4,20):
        a=sorted(accepts[core],key=lambda f:f['epoch']);e=sorted(ends[core],key=lambda f:f['start'])
        if len(a)!=manifest['layers'] or len(e)!=len(a):raise ValueError('projection payload missing job bounds')
        for accept,end in zip(a,e):
            epoch=accept['epoch']
            if epoch==1:continue
            ns=[n for n in norms[core] if accept['tick']<=n['issue_tick']<=n['complete_tick']<=end['end']*1000]
            lane=(core-4)//4;local=core*0x200000+0xb0000
            if sum(n['rows'] for n in ns)!=64 or any(n['cols']!=hidden for n in ns):
                raise ValueError('retained Norm1 full-row batch coverage incomplete')
            for n in ns:
                start=local+(n['tag']-lane*64)*hidden*2;size=n['rows']*hidden*2
                batches[core].append(dict(epoch=epoch,begin=n['issue_tick'],end=n['complete_tick'],
                                          start=start,size=size,reads=[],read_bytes=0,first=None))
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            local='resource=local_read ' in line;dma='resource=dma_read_transport ' in line
            if not local and not dma:continue
            f={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)};core=f['core']
            if core not in batches:continue
            b=next((b for b in batches[core] if b['begin']<=f['start_tick']<=f['done_tick']<=b['end']),None)
            if b is None:continue
            address=f['address'];stop=address+f['bytes'];payload=core*0x200000+0xb0000
            if local and address<payload+64*hidden*2 and payload<stop:
                if f['ok']!=1 or not b['start']<=address<stop<=b['start']+b['size']:
                    raise ValueError('retained Norm1 read wrong row or failed')
                b['reads'].append((address,stop));b['read_bytes']+=f['bytes']
                b['first']=f['start_tick'] if b['first'] is None else min(b['first'],f['start_tick'])
            if dma:
                node=1+(core-4)%4;layer=manifest['layout']['layers'][b['epoch']-1]
                start=node*stride+layer['input'];size=256*hidden*2
                if address<start+size and start<stop:raise ValueError('retained Norm1 still reads HBM activation')
    first={};total=0
    for bs in batches.values():
        for b in bs:
            if merged(b['reads'])!=[(b['start'],b['start']+b['size'])] or b['read_bytes']!=b['size']:
                raise ValueError('retained Norm1 timed GM read missing/duplicate')
            first[b['epoch']]=min(first.get(b['epoch'],b['first']),b['first']);total+=b['size']
    return dict(status='PASS',batches=sum(map(len,batches.values())),saved_hbm_read_bytes=total,
                first_input_ticks=first,scope='actual GM read ACKs in each full-row Norm1 batch; zero activation HBM reads')


def audit(case, jobs, manifest):
    path=case/'decoder_stack_provenance.json'
    if not path.exists():path=case/'provenance.json'
    env=json.loads(path.read_text()).get('environment',{}) if path.exists() else {}
    enabled=env.get('GOLEM_DECODER_LAYER_RESIDENCY','0')=='1'
    if enabled!=bool(manifest.get('layer_residency',False)):
        raise ValueError('layer residency manifest/configuration mismatch')
    eligible=enabled and manifest.get('sequence',1024)<=1024
    expected={(j['core'],j['epoch'],j['stage']):j for js in jobs.values() for j in js
              if eligible and (j['op']==9 or j['stage']==10)}
    flows=[];projection=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[SOFTWARE_LAYER_RESIDENCY]' in line:flows.append({k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)})
            if '[PROJECTION_LAYER_RESIDENCY]' in line:projection.append({k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)})
    if len(flows)!=len(expected) or {(f['core'],f['epoch'],f['stage']) for f in flows}!=set(expected):
        raise ValueError('layer residency lifecycle missing/duplicate/unexpected')
    row=64*manifest.get('hidden',2048)*2
    for f in flows:
        producer=f['stage']==4;j=expected[f['core'],f['epoch'],f['stage']]
        if (f['producer']!=int(producer) or f['bytes']!=(row if producer else 0) or
            f['saved_hbm_read']!=(0 if producer else row) or f['metadata_bytes']!=64 or
            f['capacity_added'] or f['status']):raise ValueError('layer residency bytes/ownership mismatch')
        if not producer and j['hbm_read']!=manifest['hidden']*2:
            raise ValueError('final Norm must read only gamma from HBM')
    want={(core,epoch) for core in range(4,20) for epoch in range(2,manifest.get('layers',2)+1)} if eligible else set()
    if len(projection)!=len(want) or {(f['core'],f['epoch']) for f in projection}!=want:
        raise ValueError('projection residency lifecycle missing/duplicate/unexpected')
    if any(f['saved_hbm_read']!=row or f['metadata_bytes']!=64 or f['capacity_added'] or f['status'] for f in projection):
        raise ValueError('projection residency bytes/resource mismatch')
    return dict(status='PASS',enabled=enabled,eligible=eligible,flows=flows,projection=projection,
                saved_hbm_read_bytes=sum(f['saved_hbm_read'] for f in flows+projection))


def audit_input_residency(case, jobs, manifest):
    """Audit retained projection input and residual Add reads in existing GM."""
    path=case/'decoder_stack_provenance.json'
    if not path.exists():path=case/'provenance.json'
    env=json.loads(path.read_text()).get('environment',{}) if path.exists() else {}
    enabled=env.get('GOLEM_DECODER_INPUT_RESIDENCY','0')=='1'
    if enabled!=bool(manifest.get('input_residency',False)):
        raise ValueError('input residency manifest/configuration mismatch')
    layer_enabled=env.get('GOLEM_DECODER_LAYER_RESIDENCY','0')=='1'
    if enabled and (not layer_enabled or not env.get('GOLEM_DECODER_GM_STAGE_CHAIN','0')=='1'):
        raise ValueError('input residency requires layer residency and stage chain')
    eligible=enabled and manifest.get('sequence',1024)<=1024 and manifest.get('hidden',0)<=2048
    row=64*manifest.get('hidden',2048)*2
    expected={(j['core'],j['epoch'],j['stage']):j for js in jobs.values() for j in js
              if eligible and j['stage']==2 and j['op']==2}
    flows=[];projections=[]
    with next((case/'logs').glob('attention-sst*.log')).open() as stream:
        for line in stream:
            if '[DECODER_INPUT_RESIDENCY]' in line:
                flows.append({k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)})
            elif '[PROJECTION_INPUT_RESIDENCY]' in line:
                projections.append({k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)})
    if not eligible:
        if flows or projections:raise ValueError('input residency logged while disabled/ineligible')
        return dict(status='PASS',enabled=enabled,eligible=False,saved_hbm_read_bytes=0,flows=[],projections=[])
    keys={(f['core'],f['epoch'],f['stage']) for f in flows}
    if keys!=set(expected) or len(flows)!=len(expected):
        raise ValueError('residual Add residency lifecycle missing/duplicate/unexpected')
    for f in flows:
        j=expected[f['core'],f['epoch'],f['stage']]
        gamma=manifest['hidden']*2 if manifest.get('add_norm_fusion') else 0
        if f['saved_hbm_read']!=row or f['metadata_bytes']!=64 or f['status'] or j['hbm_read']!=gamma:
            raise ValueError('residual Add must consume the retained row and issue no HBM reads')
    first={(f['core'],f['epoch']) for f in projections if f['epoch']==1}
    want={(core,1) for core in range(4,20)}
    if first!=want or len(projections)!=len(want):
        raise ValueError('first-layer projection input retention lifecycle missing/duplicate')
    if any(f['retained_bytes']!=row or f['metadata_bytes']!=64 or f['status'] for f in projections):
        raise ValueError('projection input retention byte/metadata mismatch')
    return dict(status='PASS',enabled=True,eligible=True,flows=flows,projections=projections,
                saved_hbm_read_bytes=sum(f['saved_hbm_read'] for f in flows),
                retained_input_bytes=sum(f['retained_bytes'] for f in projections),
                scope='projection retains the already-read input; Add uses a timed local GM read')


def verify_input_payload(case, jobs, manifest):
    """Require real GM reads and preserve unread residual rows until Add."""
    from analyze_resource_timeline import merged
    if not manifest.get('input_residency') or manifest['sequence']>1024:
        return dict(status='PASS',verified_jobs=0)
    log=next((case/'logs').glob('attention-sst*.log'))
    accepts=defaultdict(list);norms=defaultdict(list);ends=defaultdict(list)
    with log.open() as stream:
        for line in stream:
            if not any(s in line for s in ('stage=projection_accept ', '[SFU_RMSNORM]', '[PROJECTION_JOB]')):continue
            f={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)}
            if 'projection_accept' in line:accepts[f['core']].append(f)
            elif '[PROJECTION_JOB]' in line:ends[f['core']].append(f)
            else:norms[f['core']].append(f)
    windows=defaultdict(list);size=64*manifest['hidden']*2
    for core, js in jobs.items():
        a=sorted(accepts[core],key=lambda f:f['epoch']);e=sorted(ends[core],key=lambda f:f['start'])
        if len(a)!=manifest['layers'] or len(e)!=len(a):raise ValueError('input payload projection bounds incomplete')
        for accept,end in zip(a,e):
            epoch=accept['epoch'];add=next(j for j in js if j['epoch']==epoch and j['stage']==2)
            ns=[n for n in norms[core] if accept['tick']<=n['issue_tick']<=n['complete_tick']<=end['end']*1000]
            if not ns or sum(n['rows'] for n in ns)!=64:raise ValueError('input payload Norm1 bounds incomplete')
            windows[core].append(dict(epoch=epoch,norm_end=max(n['complete_tick'] for n in ns),
                begin=add['start_tick'],end=add['tick'],payload=core*0x200000+0xb0000,reads=[],writes=[]))
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            is_read='resource=local_read ' in line;is_write='resource=local_write ' in line
            if not is_read and not is_write:continue
            f={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)};core=f['core']
            for w in windows.get(core,[]):
                a=f['address'];b=a+f['bytes'];base=w['payload']
                if not (a<base+size and base<b and w['norm_end']<=f['submit_tick']<=w['end']):continue
                if not base<=a<b<=base+size or f['ok']!=1:raise ValueError('input payload invalid GM interval')
                if is_write and f['submit_tick']<w['begin']:
                    raise ValueError('retained input overwritten between Norm1 and Add')
                if f['submit_tick']<w['begin']:continue
                if f['done_tick']>w['end']:raise ValueError('input payload ACK after Add completion')
                w['reads' if is_read else 'writes'].append((a,b,f['start_tick'],f['done_tick']))
    for ws in windows.values():
        for w in ws:
            target=[(w['payload'],w['payload']+size)]
            for kind in ('reads','writes'):
                intervals=w[kind]
                if merged([(a,b) for a,b,_,_ in intervals])!=target or sum(b-a for a,b,_,_ in intervals)!=size:
                    raise ValueError('Add timed retained input GM coverage missing/duplicate')
            for a,b,start,_ in w['writes']:
                prior=[(x,y) for x,y,_,done in w['reads'] if done<=start]
                if not any(x<=a<b<=y for x,y in merged(prior)):
                    raise ValueError('Add overwrites an unread residual input chunk')
    return dict(status='PASS',verified_jobs=sum(map(len,windows.values())),
                saved_hbm_read_bytes=size*sum(map(len,windows.values())),
                scope='timed complete payload reads; no cross-stage overwrite; per-chunk read ACK before in-place write')
