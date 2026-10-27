import subprocess
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_decoder_stack import memory_layout,prepare,validate_inputs,verify_final_norm,verify_epoch_resets


class DecoderStackTest(unittest.TestCase):
    def test_epoch_flags_initialize_first_generation_and_elide_only_later_resets(self):
        flags={100,200};good=[dict(address=a,value=0,status=0) for a in flags]
        for enabled in (False,True):verify_epoch_resets(good,flags,1,enabled)
        verify_epoch_resets(good,flags,2,False)
        verify_epoch_resets([],flags,2,True)
        for group,epoch,enabled in (([],1,True),([],2,False),(good,2,True),(good[:1],1,True)):
            with self.assertRaises(ValueError):verify_epoch_resets(group,flags,epoch,enabled)

    def test_two_formal_layers_fit_and_have_no_aliases(self):
        m=memory_layout(32,8)
        self.assertLessEqual(m['used_bytes'],128*2**20)
        self.assertEqual(m['layers'][0]['output'],m['layers'][1]['input'])
        self.assertEqual(m['layers'][0]['input'],m['layers'][1]['output'])
        regions=sorted(m['regions'],key=lambda r:r['offset'])
        for a,b in zip(regions,regions[1:]):self.assertLessEqual(a['offset']+a['bytes'],b['offset'])
        for name in ('qkv_weight','wo_weight','gate_weight','up_weight','down_weight','gamma1','gamma2'):
            self.assertNotEqual(m['layers'][0][name],m['layers'][1][name])

    def test_expansion_uses_same_scratch_and_rejects_insufficient_capacity(self):
        two=memory_layout(4,2,2);sixteen=memory_layout(4,2,16)
        self.assertEqual(two['shared'],sixteen['shared'])
        self.assertEqual(two['activations'],sixteen['activations'])
        for a,b in zip(sixteen['layers'],sixteen['layers'][1:]):self.assertEqual(a['output'],b['input'])
        with self.assertRaises(subprocess.CalledProcessError):memory_layout(32,8,3)

    def test_distinct_weights_and_poisoned_flags_have_verified_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=prepare(root,4,2)
            validate_inputs(root)
            for name in ('qkv','wo','gate','up','down'):
                a=np.load(root/f'layer1_{name}.npy');b=np.load(root/f'layer2_{name}.npy')
                self.assertFalse(np.array_equal(a,b))
            self.assertTrue(np.all(np.fromfile(root/'poison_flags.bin',dtype='<u8')!=0))
            path=Path(m['preloads'][0]['path']);data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'provenance'):validate_inputs(root)

    def test_final_norm_layout_preserves_decoder_addresses_and_fits(self):
        for hq,hkv in ((4,2),(32,8)):
            old=memory_layout(hq,hkv);new=memory_layout(hq,hkv,final_norm=True)
            for name in ('shared','activations','layers'):self.assertEqual(old[name],new[name])
            self.assertLessEqual(new['used_bytes'],new['node_stride'])
            for a,b in zip(new['regions'],new['regions'][1:]):self.assertLessEqual(a['offset']+a['bytes'],b['offset'])
            self.assertEqual({r['name'] for r in new['regions'][len(old['regions']):]}, {'final_gamma','final_output','final_done'})

    def test_final_gamma_and_nan_output_have_verified_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m=prepare(root,4,2,final_norm=True);validate_inputs(root)
            gamma=np.fromfile(root/'final_gamma.bin',dtype='<f2')
            for layer in (1,2):
                for name in ('gamma1','gamma2'):
                    self.assertFalse(np.array_equal(gamma,np.fromfile(root/f'layer{layer}_{name}.bin',dtype='<f2')))
            self.assertTrue(np.isnan(np.fromfile(root/'poison_final_output.bin',dtype='<f2')).all())
            self.assertEqual(len([e for e in m['preloads'] if e['name'].startswith('final_')]),12)

    def final_fixture(self,root,mutation=None):
        layout=memory_layout(4,2,final_norm=True);f=layout['final_norm'];stride=layout['node_stride']
        m={'layout':layout,'layers':2,'hidden':256};row_bytes=64*256*2
        log=root/'attention-sst.log';lines=[];resources=[]
        for core in range(4,20):
            node=1+(core-4)%4;lane=(core-4)//4;flag=node*stride+f['done']+lane*8
            lines.append(f'[DECODER_SYNC] event=flag_write_ack address={flag} value=0 status=0 tick=250')
            lines.append(f'[DECODER_OP] event=done core={core} stage=10 op=3 epoch=3 status=0 start_tick=300 tick=400 hbm_read={row_bytes+1024} hbm_write={row_bytes+8}')
            resources.append({'resource':'dma_read_transport','core':core,'address':node*stride+layout['layers'][-1]['output']+lane*row_bytes,'bytes':row_bytes,'start_tick':310,'done_tick':320,'ok':1})
            for tick in (330,340):resources.append({'resource':'dma_read_transport','core':core,'address':node*stride+f['gamma'],'bytes':512,'start_tick':tick,'done_tick':tick+5,'ok':1})
            resources.append({'resource':'dma_write_transport','core':core,'address':node*stride+f['output']+lane*row_bytes,'bytes':row_bytes,'start_tick':350,'done_tick':390,'ok':1})
            resources.append({'resource':'dma_write_transport','core':core,'address':flag,'bytes':8,'start_tick':391,'done_tick':395,'ok':1})
        if mutation:mutation(lines,resources)
        log.write_text('\n'.join(lines)+'\n')
        log.with_suffix('.resources.jsonl').write_text('\n'.join('[RESOURCE_INTERVAL] '+' '.join(f'{k}={v}' for k,v in e.items()) for e in resources)+'\n')
        (root/'stdout').mkdir();(root/'stdout'/'stdout-0').write_text('DECODER_FINAL_NORM status=0 hidden=256 epoch=3\n')
        return verify_final_norm(root,m,log,{2:200},{'layers':{'1':{'first_input_dma_tick':100},'2':{'last_y_ack_tick':190}}})

    def test_final_norm_checks_all_workers_and_output_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.final_fixture(Path(directory))
            self.assertEqual(result['output_ack_bytes'],1024*256*2)
            self.assertEqual(result['last_completion_ack_tick'],395)

    def test_reused_gamma_requires_exact_physical_load_and_complete_coverage(self):
        for invalid in (False,True):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/'decoder_stack_provenance.json').write_text(json.dumps(
                    {'environment':{'GOLEM_DECODER_GAMMA_REUSE':'1'}}))
                def mutate(lines,events):
                    lines[:]=[line.replace('hbm_read=33792','hbm_read=33280') for line in lines]
                    events[:]=[e for e in events if e['start_tick']!=340]
                    if invalid:
                        # Preserve the byte count, but read the wrong gamma half.
                        events[1]['address']+=256
                if invalid:
                    with self.assertRaises(ValueError):self.final_fixture(root,mutate)
                else:self.assertEqual(self.final_fixture(root,mutate)['status'],'PASS')

    def test_final_norm_rejects_stale_epoch_early_ack_and_missing_output(self):
        mutations=[lambda lines,events:lines.__setitem__(1,lines[1].replace('epoch=3','epoch=2')),
                   lambda lines,events:events[4].update(start_tick=380),
                   lambda lines,events:events[3].update(bytes=16384),
                   lambda lines,events:lines.__setitem__(0,lines[0].replace('tick=250','tick=2999')),
                   lambda lines,events:events[0].update(start_tick=180)]
        for mutation in mutations:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):self.final_fixture(Path(directory),mutation)

if __name__=='__main__':unittest.main()
