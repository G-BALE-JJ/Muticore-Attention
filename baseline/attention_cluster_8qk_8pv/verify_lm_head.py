"""Validate last-token head numerics, output shards, and actual DMA ACKs."""
import json
import re
import numpy as np
from run_decoder_layer import linear_reference,read_bands


def verify_lm_head(case,m,log,final,timing):
    from run_decoder_stack import fields,compare
    from analyze_resource_timeline import FIELDS,merged
    layout=m['layout'];lm=layout['lm_head'];stride=layout['node_stride'];hidden=m['hidden']
    sequence=m.get('sequence',1024);logical=m.get('logical_sequence',sequence);base_epoch=m.get('layers',2)+2
    vocab=lm['vocab'];columns=((vocab+63)//64)*64;chunks=(columns+8191)//8192
    flags={node*stride+lm['done']+lane*8 for node in range(1,5) for lane in range(4)}
    ops=[];resets=[];resident_flows=[]
    provenance=case/'decoder_stack_provenance.json'
    env=json.loads(provenance.read_text()).get('environment',{}) if provenance.exists() else {}
    resident=env.get('GOLEM_DECODER_LM_INPUT_RESIDENCY','0')=='1'
    if resident!=bool(m.get('lm_input_residency',False)):raise ValueError('LM input residency manifest/config mismatch')
    with log.open() as stream:
        for line in stream:
            if '[DECODER_OP] event=done ' in line:
                e=fields(line)
                if e['stage']==11:ops.append(e)
            if '[DECODER_SYNC] event=flag_write_ack ' in line:
                e=fields(line)
                if e['address'] in flags:resets.append(e)
            if '[LM_INPUT_RESIDENCY]' in line:resident_flows.append(fields(line))
    if len(ops)!=16*chunks:raise ValueError('LM head missing/duplicate worker chunks')
    expected_keys={(e['core'],e['epoch']) for e in ops} if resident else set()
    if len(resident_flows)!=len(expected_keys) or {(f['core'],f['epoch']) for f in resident_flows}!=expected_keys:
        raise ValueError('LM input residency lifecycle missing/duplicate/unexpected')
    resident_by_key={(f['core'],f['epoch']):f for f in resident_flows}
    for chunk in range(chunks):
        group=[e for e in ops if e['epoch']==base_epoch+chunk]
        if len(group)!=16 or {e['core'] for e in group}!=set(range(4,20)) or any(e['status'] or e['op']!=1 for e in group):
            raise ValueError('LM head chunk generation mismatch')
    start=min(e['start_tick'] for e in ops)
    if (len(resets)!=16 or {e['address'] for e in resets}!=flags or any(e['value'] or e['status'] or
        not final['last_completion_ack_tick']<=e['tick']<start for e in resets)):
        raise ValueError('LM head reset ACK/barrier mismatch')
    token=logical-1;rows=sequence//4
    input_address=(1+token//rows)*stride+layout['final_norm']['output']+(token%rows)*hidden*2
    states={(e['core'],e['epoch']):{'op':e,'writes':[],'weight_reads':[],'inputs':[],
        'flag_acks':[],'last_output':0} for e in ops}
    per_core={core:sorted((e for e in ops if e['core']==core),key=lambda e:e['epoch']) for core in range(4,20)}
    for group in per_core.values():
        for a,b in zip(group,group[1:]):
            if a['tick']>=b['start_tick']:raise ValueError('LM head reused scratch before previous chunk ACK')
    first_input=None
    with log.with_suffix('.resources.jsonl').open() as stream:
        for line in stream:
            if 'resource=dma_read_transport ' not in line and 'resource=dma_write_transport ' not in line:continue
            e={k:int(v) for k,v in FIELDS.findall(line) if k not in ('domain','resource')}
            if e['core'] not in per_core:continue
            op=next((op for op in per_core[e['core']] if op['start_tick']<=e['start_tick']<=op['tick']),None)
            if op is None:continue
            if e['ok']!=1:raise ValueError('LM head DMA failed')
            state=states[e['core'],op['epoch']];address=e['address'];stop=address+e['bytes']
            if 'resource=dma_read_transport ' in line:
                if address==input_address and e['bytes']==hidden*2:
                    if e['start_tick']<=final['last_output_ack_tick']:raise ValueError('LM input before final normalized output ACK')
                    state['inputs'].append(e)
                    first_input=e['start_tick'] if first_input is None else min(first_input,e['start_tick'])
                else:state['weight_reads'].append((address,stop))
            else:
                flag=(1+(e['core']-4)%4)*stride+lm['done']+((e['core']-4)//4)*8
                if address==flag and e['bytes']==8:state['flag_acks'].append(e)
                else:
                    state['writes'].append((address,stop));state['last_output']=max(state['last_output'],e['done_tick'])
    last_output=0;write_bytes=0;read_bytes=0;completion=[]
    for (core,epoch),state in states.items():
        chunk=epoch-base_epoch;begin=chunk*128;ntiles=min(128,columns//64-begin)
        tiles=list(range(begin+(core-4),begin+ntiles,16));node=1+(core-4)%4;op=state['op']
        wanted_writes=[(node*stride+lm['output']+tile*128,node*stride+lm['output']+(tile+1)*128) for tile in tiles]
        wanted_reads=[((1+tile%4)*stride+lm['weight']+(tile//4*(hidden//64)+k)*8192,
                      (1+tile%4)*stride+lm['weight']+(tile//4*(hidden//64)+k+1)*8192)
                      for tile in tiles for k in range(hidden//64)]
        if (merged(state['writes'])!=merged(wanted_writes) or sum(b-a for a,b in state['writes'])!=len(tiles)*128 or
            merged(state['weight_reads'])!=merged(wanted_reads) or sum(b-a for a,b in state['weight_reads'])!=len(tiles)*hidden*128):
            raise ValueError('LM head incomplete/duplicate output or weight tile coverage')
        cache_eligible=resident and chunk>0
        hit=cache_eligible and bool(tiles)
        if resident:
            f=resident_by_key[core,epoch]
            # The runtime's hit flag denotes a valid retained-input mapping.
            # An empty final vocabulary shard issues no input read or GEMM,
            # so its actual saved bytes remain zero despite that mapping.
            if f['hit']!=int(cache_eligible) or f['saved_hbm_read']!=(hidden*2 if hit else 0) or f['capacity_added'] or f['status']:
                raise ValueError('LM input residency ownership/bytes mismatch')
        if len(state['inputs'])!=int(bool(tiles) and not hit):raise ValueError('LM head last-token read/cache-hit mismatch')
        expected_read=(hidden*2 if tiles and not hit else 0)+len(tiles)*hidden*128
        expected_write=len(tiles)*128+(8 if chunk==chunks-1 else 0)
        if op['hbm_read']!=expected_read or op['hbm_write']!=expected_write:raise ValueError('LM head traffic mismatch')
        if len(state['flag_acks'])!=int(chunk==chunks-1):raise ValueError('LM head completion before last vocabulary chunk')
        for ack in state['flag_acks']:
            if not state['last_output']<=ack['start_tick']<=ack['done_tick']<=op['tick']:
                raise ValueError('LM head completion before output ACK')
            completion.append(ack['done_tick'])
        last_output=max(last_output,state['last_output']);write_bytes+=len(tiles)*128;read_bytes+=expected_read
    stdout='\n'.join(p.read_text() for p in (case/'stdout').rglob('stdout-*'))
    if re.findall(r'DECODER_LM_HEAD status=(\d+) vocab=(\d+) epoch=(\d+) token=(\d+)',stdout)!=[('0',str(vocab),str(base_epoch+chunks-1),str(logical-1))]:
        raise ValueError('LM head root completion missing/failed')
    logits=np.empty(columns,dtype='<f2')
    for node in range(1,5):
        with (case/'hbm'/f'hbm_out_node{node}.bin').open('rb') as stream:
            stream.seek(lm['output']);data=np.frombuffer(stream.read(columns*2),dtype='<f2')
            if len(data)!=columns:raise ValueError('short logits backing')
            for tile in range(node-1,columns//64,4):logits[tile*64:(tile+1)*64]=data[tile*64:(tile+1)*64]
            stream.seek(lm['done']);done=np.frombuffer(stream.read(32),dtype='<u8')
            if len(done)!=4 or not np.all(done==base_epoch+chunks-1):raise ValueError('stale LM completion epoch')
    weight=np.load(case/'lm_weight.npy',mmap_mode='r')
    actual_input=read_bands(case,layout['final_norm']['output'],hidden,rows=rows)[logical-1:logical]
    local=linear_reference(actual_input,weight).reshape(-1)
    local_result=compare(logits,local,0,'last-token LM head local')
    composed=linear_reference(np.load(case/'final_reference_last_token.npy'),weight).reshape(-1)
    composed_result=compare(logits[:vocab],composed[:vocab],6e-3,'last-token LM head composed')
    if columns>vocab and not np.all(logits[vocab:]==0):raise ValueError('nonzero padded vocabulary logits')
    np.save(case/'last_token_logits.npy',logits[:vocab])
    from select_next_token import select_next_token
    selection=select_next_token(logits,vocab)
    local_selection=select_next_token(local,vocab);composed_selection=select_next_token(composed,vocab)
    if selection['token_id']!=local_selection['token_id']:raise ValueError('host token differs from local CPU reference')
    selection.update({'status':'PASS','local_reference_token_id':local_selection['token_id'],
        'composed_reference_token_id':composed_selection['token_id'],
        'composed_reference_agrees':selection['token_id']==composed_selection['token_id'],
        'timing_scope':'host argmax only; excluded from accelerator cycles'})
    (case/'next_token.json').write_text(json.dumps(selection,indent=2)+'\n')
    precision_result={}
    if (case/'fp32_final_reference_last_token.npy').exists():
        from prefill_precision_reference import error
        fp32_logits=(np.load(case/'fp32_final_reference_last_token.npy')@weight.astype(np.float32).T).reshape(-1)[:vocab]
        precision_result=error(logits[:vocab],fp32_logits)
        precision_result['fp32_token_id']=int(np.argmax(fp32_logits))
        precision_result['greedy_agrees']=precision_result['fp32_token_id']==selection['token_id']
    return {'status':'PASS','token_index':logical-1,'M':1,'vocab':vocab,'padded_vocab':columns,
        'vocabulary_chunk_columns':8192,'chunks':chunks,'tail_chunk_columns':columns-(chunks-1)*8192,
        'worker_jobs':len(ops),'reset_ack_count':16,'last_epoch':base_epoch+chunks-1,'next_token':selection,
        'local_reference':local_result,'composed_reference':composed_result,'fp32_precision':precision_result,
        'first_input_dma_tick':first_input,'last_logits_ack_tick':last_output,'last_completion_ack_tick':max(completion),
        'input_and_weight_dma_bytes':read_bytes,'output_ack_bytes':write_bytes,
        'lm_head_cycles':(last_output-first_input)/1000,'handoff_cycles':(first_input-final['last_output_ack_tick'])/1000,
        'tail_cycles':(last_output-final['last_output_ack_tick'])/1000,
        'end_to_end_cycles':(last_output-timing['layers']['1']['first_input_dma_tick'])/1000,
        'contract':'first layer X input DMA send to last-token logits final DMA ACK'}
