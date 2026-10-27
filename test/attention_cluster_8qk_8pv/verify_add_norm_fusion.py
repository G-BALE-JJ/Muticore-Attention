"""Verify actual timed payload and SFU work in the fused residual/Norm job."""
from collections import defaultdict
import re
from analyze_resource_timeline import merged


def audit(case, jobs, manifest):
    if not manifest.get('add_norm_fusion'):
        return dict(status='PASS',enabled=False,verified_jobs=0)
    log=next((case/'logs').glob('attention-sst*.log'))
    windows=defaultdict(list);size=64*manifest['hidden']*2
    for core, group in jobs.items():
        for j in group:
            if j['stage']==2:
                windows[core].append(dict(job=j,sfu=[],reads=[],writes=[]))
    with log.open() as stream:
        for line in stream:
            if '[DECODER_ADD_NORM_SFU]' not in line:continue
            f={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)}
            for w in windows.get(f['core'],[]):
                if w['job']['start_tick']<=f['submit_tick']<=f['done_tick']<=w['job']['tick']:
                    w['sfu'].append(f)
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            read='resource=local_read ' in line;write='resource=local_write ' in line
            if not (read or write):continue
            f={k:int(v) for k,v in re.findall(r'(\w+)=(-?\d+)',line)}
            for w in windows.get(f['core'],[]):
                if not w['job']['start_tick']<=f['submit_tick']<=w['job']['tick']:continue
                base=f['core']*0x200000+(0xb0000 if read else 0x10000)
                a=f['address'];b=a+f['bytes']
                if not (a<base+size and base<b):continue
                if (f['ok']!=1 or not base<=a<b<=base+size or
                        f['done_tick']>w['job']['tick']):
                    raise ValueError('fused Add/Norm payload failed, out of bounds or unacknowledged')
                w['reads' if read else 'writes'].append(f)
    requests=0;count=0
    for core, ws in windows.items():
        for w in ws:
            ops=sorted(w['sfu'],key=lambda f:f['submit_tick'])
            chunks=(64+8192//manifest['hidden']-1)//(8192//manifest['hidden'])
            if (len(ops)!=2*chunks or [f['op'] for f in ops]!=[3,2]*chunks or
                    any(f['elements']>8192 or f['elements']<=0 for f in ops) or
                    sum(f['elements'] for f in ops if f['op']==2)*2!=size or
                    any(a['done_tick']>b['submit_tick'] or a['elements']!=b['elements']
                        for a,b in zip(ops[::2],ops[1::2]))):
                raise ValueError('fused Add/Norm must execute bounded paired Add then RMSNorm SFU work')
            for kind, offset in (('reads',0xb0000),('writes',0x10000)):
                base=core*0x200000+offset;events=w[kind]
                if (merged([(f['address'],f['address']+f['bytes']) for f in events])!=[(base,base+size)] or
                        sum(f['bytes'] for f in events)!=size):
                    raise ValueError('fused Add/Norm GM coverage missing, duplicated or H1 reread')
            # Each packed K-tile write must wait for its source SFU rows.
            for f in w['writes']:
                first=(f['address']-(core*0x200000+0x10000))//128%64
                rows=(f['bytes']+127)//128
                begin=0;producer=None
                for norm in ops[1::2]:
                    n=norm['elements']//manifest['hidden']
                    if begin<=first and first+rows<=begin+n:producer=norm;break
                    begin+=n
                if producer is None or f['submit_tick']<producer['done_tick']:
                    raise ValueError('packed Norm GM write precedes its SFU result')
            requests+=len(w['writes']);count+=1
    return dict(status='PASS',enabled=True,verified_jobs=count,
                saved_gm_read_bytes=count*size,norm_write_requests=requests,
                payload='one residual GM read, complete packed Norm writes, bounded Add/Norm SFU pairs')
