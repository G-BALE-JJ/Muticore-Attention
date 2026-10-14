"""Verify timed HBM token gather and the barrier into the first Decoder layer."""
from collections import Counter
import re
import numpy as np
from run_decoder_layer import read_bands


def verify_embedding(case,m,log,begin,lm_head):
    from run_decoder_stack import fields,compare
    from analyze_resource_timeline import FIELDS,merged
    layout=m['layout'];emb=layout['embedding'];stride=layout['node_stride'];hidden=m['hidden'];vocab=layout['lm_head']['vocab']
    tokens=np.fromfile(case/'input_token_ids.bin',dtype='<u4')
    if len(tokens)!=1024 or np.any(tokens>=vocab):raise ValueError('invalid embedding token IDs')
    table=np.load(case/'embedding_weight.npy',mmap_mode='r')
    if table.shape!=(vocab,hidden):raise ValueError('embedding table shape mismatch')
    want=table[tokens];got=read_bands(case,emb['output'],hidden)
    numerical=compare(got,want,0,'embedding gather')
    compare(np.fromfile(case/'projection_x.bin',dtype='<f2').reshape(1024,hidden),want,0,'embedding input oracle')
    flags={node*stride+emb['done']+lane*8 for node in range(1,5) for lane in range(4)}
    ops=[];resets=[]
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done ' in line:
                e=fields(line)
                if e['stage']==12:ops.append(e)
            if '[DECODER_SYNC] event=flag_write_ack ' in line:
                e=fields(line)
                if e['address'] in flags:resets.append(e)
    if len(ops)!=16 or {e['core'] for e in ops}!=set(range(4,20)) or any(e['status'] or e['op']!=6 or e['epoch']!=1 for e in ops):
        raise ValueError('embedding worker completion mismatch')
    start=min(e['start_tick'] for e in ops)
    if len(resets)!=16 or {e['address'] for e in resets}!=flags or any(e['value'] or e['status'] or e['tick']>=start for e in resets):
        raise ValueError('embedding flags not reset/ACKed before release')
    states={e['core']:{'op':e,'tokens':[],'weights':[],'outputs':[],'flags':[],'last_output':0} for e in ops}
    first_tokens=None;first_decoder=None
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if 'resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line:continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            if 'resource=dma_read_transport ' in line and e['start_tick']>=begin[1]:
                node=e['address']//stride;offset=e['address']%stride
                if 1<=node<=4 and emb['output']<=offset<emb['output']+256*hidden*2:
                    first_decoder=e['start_tick'] if first_decoder is None else min(first_decoder,e['start_tick'])
            if e['core'] not in states:continue
            state=states[e['core']];op=state['op']
            if not op['start_tick']<=e['start_tick']<=op['tick']:continue
            if e['ok']!=1:raise ValueError('embedding DMA failure')
            node=1+(e['core']-4)%4;lane=(e['core']-4)//4
            address=e['address'];stop=address+e['bytes']
            if 'resource=dma_read_transport ' in line:
                if address==node*stride+emb['tokens']+lane*64*4 and e['bytes']==256:
                    state['tokens'].append((address,stop))
                    first_tokens=e['start_tick'] if first_tokens is None else min(first_tokens,e['start_tick'])
                else:state['weights'].append((address,stop))
            else:
                if address==node*stride+emb['done']+lane*8 and e['bytes']==8:state['flags'].append(e)
                else:state['outputs'].append((address,stop));state['last_output']=max(state['last_output'],e['done_tick'])
    for core,state in states.items():
        node=1+(core-4)%4;lane=(core-4)//4;owned=tokens[(node-1)*256+lane*64:(node-1)*256+(lane+1)*64]
        wanted=Counter(((1+int(token)%4)*stride+emb['weight']+(int(token)//4)*hidden*2,
                        (1+int(token)%4)*stride+emb['weight']+(int(token)//4+1)*hidden*2) for token in owned)
        base=node*stride+emb['output']+lane*64*hidden*2
        if len(state['tokens'])!=1 or Counter(state['weights'])!=wanted or merged(state['outputs'])!=[(base,base+64*hidden*2)] or sum(b-a for a,b in state['outputs'])!=64*hidden*2:
            raise ValueError('embedding token/weight/output DMA coverage mismatch')
        if state['op']['hbm_read']!=256+64*hidden*2 or state['op']['hbm_write']!=64*hidden*2+8:
            raise ValueError('embedding traffic mismatch')
        if len(state['flags'])!=1 or not state['last_output']<=state['flags'][0]['start_tick']<=state['flags'][0]['done_tick']<=state['op']['tick']:
            raise ValueError('embedding completion before output ACK')
    last_output=max(e['last_output'] for e in states.values());last_done=max(e['op']['tick'] for e in states.values())
    if first_decoder is None or first_decoder<=last_output or begin[1]<=last_done:
        raise ValueError('first Decoder started before embedding global completion')
    for node in range(1,5):
        with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
            stream.seek(emb['done']);values=np.frombuffer(stream.read(32),dtype='<u8')
            if len(values)!=4 or not np.all(values==1):raise ValueError('embedding completion flag mismatch')
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    if re.findall(r'DECODER_EMBEDDING status=(\d+) tokens=(\d+) hidden=(\d+) vocab=(\d+)',stdout)!=[('0','1024',str(hidden),str(vocab))]:
        raise ValueError('embedding root completion missing/failed')
    return {'status':'PASS','tokens':1024,'hidden':hidden,'vocab':vocab,'numerics':numerical,'workers':16,
        'reset_ack_count':16,'first_token_ids_dma_tick':first_tokens,'last_embedding_output_ack_tick':last_output,
        'last_worker_completion_tick':last_done,'first_decoder_input_dma_tick':first_decoder,
        'input_token_bytes':4096,'table_read_bytes':1024*hidden*2,'output_ack_bytes':1024*hidden*2,
        'embedding_cycles':(last_output-first_tokens)/1000,'handoff_cycles':(first_decoder-last_output)/1000,
        'end_to_end_cycles':(lm_head['last_logits_ack_tick']-first_tokens)/1000,
        'contract':'first token-ID DMA send to last-token logits final output DMA ACK'}
