import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_decoder_stack import memory_layout
from verify_embedding import verify_embedding


class EmbeddingVerifierTest(unittest.TestCase):
    def fixture(self,root,mutation=None):
        layout=memory_layout(4,2,final_norm=True,vocab=9,node_bytes=2*2**30,embedding=True)
        emb=layout['embedding'];stride=layout['node_stride'];hidden=256;events=[];lines=[]
        table=np.random.default_rng(1).normal(0,.3,(9,hidden)).astype('<f2')
        tokens=np.arange(1024,dtype='<u4')%9;tokens.tofile(root/'input_token_ids.bin')
        np.save(root/'embedding_weight.npy',table);table[tokens].tofile(root/'projection_x.bin')
        for core in range(4,20):
            node=1+(core-4)%4;lane=(core-4)//4;flag=node*stride+emb['done']+lane*8
            lines.append(f'[DECODER_SYNC] event=flag_write_ack address={flag} value=0 status=0 tick=450')
            lines.append(f'[DECODER_OP] event=done core={core} stage=12 op=6 epoch=1 status=0 start_tick=500 tick=1300 hbm_read={256+64*512} hbm_write={64*512+8}')
            def event(resource,address,size,start,done):
                events.append({'resource':resource,'core':core,'address':address,'bytes':size,'start_tick':start,'done_tick':done,'ok':1})
            event('dma_read_transport',node*stride+emb['tokens']+lane*256,256,510,520)
            for row in range(64):
                token=int(tokens[(node-1)*256+lane*64+row]);t=600+row*6
                event('dma_read_transport',(1+token%4)*stride+emb['weight']+(token//4)*512,512,t,t+2)
                event('dma_write_transport',node*stride+emb['output']+(lane*64+row)*512,512,t+3,t+4)
            event('dma_write_transport',flag,8,1000,1100)
        events.append({'resource':'dma_read_transport','core':4,'address':stride+emb['output'],'bytes':512,'start_tick':1430,'done_tick':1440,'ok':1})
        if mutation:mutation(lines,events)
        log=root/'attention-sst.log';log.write_text('\n'.join(lines)+'\n')
        log.with_suffix('.resources.jsonl').write_text('\n'.join('[RESOURCE_INTERVAL] '+' '.join(f'{k}={v}' for k,v in e.items()) for e in events)+'\n')
        (root/'stdout').mkdir();(root/'stdout'/'stdout-0').write_text('DECODER_EMBEDDING status=0 tokens=1024 hidden=256 vocab=9\n')
        (root/'hbm').mkdir()
        for node in range(1,5):
            with (root/'hbm'/f'hbm_out_node{node}.bin').open('wb') as stream:
                stream.seek(emb['output']);stream.write(table[tokens[(node-1)*256:node*256]].tobytes())
                stream.seek(emb['done']);stream.write(np.ones(4,dtype='<u8').tobytes())
        return verify_embedding(root,{'layout':layout,'hidden':hidden},log,{1:1400},{'last_logits_ack_tick':2000})

    def test_gather_verifier_accepts_exact_table_and_timed_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            r=self.fixture(Path(directory));self.assertEqual(r['numerics']['max_abs_error'],0)
            self.assertEqual(r['table_read_bytes'],524288)

    def test_rejects_wrong_stripe_duplicate_output_early_flag_and_bad_barrier(self):
        mutations=[lambda l,e:e[1].update(address=e[1]['address']+512),
            lambda l,e:e.insert(3,dict(e[2])),
            lambda l,e:e[129].update(start_tick=800),
            lambda l,e:e[-1].update(start_tick=900),
            lambda l,e:l.__setitem__(0,l[0].replace('value=0','value=1')),
            lambda l,e:e[1].update(ok=0),
            lambda l,e:l.pop()]
        for mutation in mutations:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):self.fixture(Path(directory),mutation)


if __name__=='__main__':unittest.main()
