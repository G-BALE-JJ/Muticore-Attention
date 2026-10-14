import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_decoder_stack import memory_layout,prepare,validate_inputs
from verify_lm_head import verify_lm_head


class LmHeadTest(unittest.TestCase):
    def test_full_vocab_requires_expansion_and_preserves_tensor_offsets(self):
        with self.assertRaises(subprocess.CalledProcessError):memory_layout(32,8,final_norm=True,vocab=128256)
        old=memory_layout(32,8,final_norm=True)
        new=memory_layout(32,8,final_norm=True,vocab=128256,node_bytes=2*2**30)
        for key in ('shared','activations','layers','final_norm'):self.assertEqual(old[key],new[key])
        self.assertLess(new['used_bytes'],256*2**20)
        self.assertEqual(new['node_stride'],2*2**30)
        self.assertEqual(new['gm_stride'],old['gm_stride'])
        self.assertEqual(next(r['bytes'] for r in new['regions'] if r['name']=='lm_weight'),128256*2048*2//4)
        for a,b in zip(new['regions'],new['regions'][1:]):self.assertLessEqual(a['offset']+a['bytes'],b['offset'])

    def test_partial_vocab_shards_recover_weights_and_poisoned_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=prepare(root,4,2,final_norm=True,vocab=8201);validate_inputs(root)
            weight=np.load(root/'lm_weight.npy');panels=np.empty((129,4,64,64),dtype='<f2')
            for node in range(1,5):panels[node-1::4]=np.fromfile(root/f'lm_weight_node{node}.bin',dtype='<f2').reshape(-1,4,64,64)
            np.testing.assert_array_equal(panels.transpose(0,2,1,3).reshape(8256,256),weight)
            self.assertTrue(np.all(weight[8201:]==0))
            self.assertTrue(np.all(weight[:8201,-1]!=0))
            self.assertTrue(np.isnan(np.fromfile(root/'poison_lm_output.bin',dtype='<f2')).all())
            self.assertEqual(len([e for e in m['preloads'] if e['name'].startswith('lm_')]),12)

    def fixture(self,root,mutation=None):
        layout=memory_layout(4,2,final_norm=True,vocab=9,node_bytes=2*2**30)
        lm=layout['lm_head'];stride=layout['node_stride'];hidden=256;lines=[];events=[]
        for core in range(4,20):
            node=1+(core-4)%4;lane=(core-4)//4;flag=node*stride+lm['done']+lane*8
            lines.append(f'[DECODER_SYNC] event=flag_write_ack address={flag} value=0 status=0 tick=450')
            read_bytes=33280 if core==4 else 0;write_bytes=136 if core==4 else 8
            lines.append(f'[DECODER_OP] event=done core={core} stage=11 op=1 epoch=4 status=0 start_tick=500 tick=650 hbm_read={read_bytes} hbm_write={write_bytes}')
            if core==4:
                events.append({'resource':'dma_read_transport','core':core,'address':4*stride+layout['final_norm']['output']+255*hidden*2,'bytes':512,'start_tick':510,'done_tick':520,'ok':1})
                for k in range(4):events.append({'resource':'dma_read_transport','core':core,'address':stride+lm['weight']+k*8192,'bytes':8192,'start_tick':530+k,'done_tick':540+k,'ok':1})
                events.append({'resource':'dma_write_transport','core':core,'address':stride+lm['output'],'bytes':128,'start_tick':580,'done_tick':600,'ok':1})
            events.append({'resource':'dma_write_transport','core':core,'address':flag,'bytes':8,'start_tick':610,'done_tick':620,'ok':1})
        if mutation:mutation(lines,events)
        log=root/'attention-sst.log';log.write_text('\n'.join(lines)+'\n')
        log.with_suffix('.resources.jsonl').write_text('\n'.join('[RESOURCE_INTERVAL] '+' '.join(f'{k}={v}' for k,v in e.items()) for e in events)+'\n')
        (root/'stdout').mkdir();(root/'stdout'/'stdout-0').write_text('DECODER_LM_HEAD status=0 vocab=9 epoch=4 token=1023\n')
        (root/'hbm').mkdir()
        x=np.zeros((1,hidden),dtype='<f2');x[0,0]=1
        weights=np.zeros((64,hidden),dtype='<f2');weights[:9,0]=.125
        np.save(root/'lm_weight.npy',weights);np.save(root/'final_reference_last_token.npy',x)
        for node in range(1,5):
            with (root/'hbm'/f'hbm_out_node{node}.bin').open('wb') as stream:
                stream.seek(layout['final_norm']['output']);data=np.zeros((256,hidden),dtype='<f2')
                if node==4:data[-1]=x[0]
                stream.write(data.tobytes());stream.seek(lm['output'])
                stream.write(np.array([.125]*9+[0]*55,dtype='<f2').tobytes())
                stream.seek(lm['done']);stream.write(np.full(4,4,dtype='<u8').tobytes())
        return verify_lm_head(root,{'layout':layout,'hidden':hidden},log,
            {'last_completion_ack_tick':400,'last_output_ack_tick':390},
            {'layers':{'1':{'first_input_dma_tick':100}}})

    def test_head_verifier_checks_all_workers_and_valid_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r=self.fixture(root)
            self.assertEqual(r['output_ack_bytes'],128)
            self.assertEqual(r['local_reference']['max_abs_error'],0)
            self.assertEqual(np.load(root/'last_token_logits.npy').shape,(9,))

    def test_head_verifier_rejects_wrong_token_generation_missing_tile_and_early_ack(self):
        mutations=[lambda lines,events:events[0].update(address=events[0]['address']-512),
                   lambda lines,events:lines.__setitem__(1,lines[1].replace('epoch=4','epoch=3')),
                   lambda lines,events:events[1].update(address=events[1]['address']+8192),
                   lambda lines,events:events[5].update(bytes=64),
                   lambda lines,events:events[6].update(start_tick=590),
                   lambda lines,events:lines.__setitem__(0,lines[0].replace('tick=450','tick=300'))]
        for mutation in mutations:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):self.fixture(Path(directory),mutation)


if __name__=='__main__':unittest.main()
