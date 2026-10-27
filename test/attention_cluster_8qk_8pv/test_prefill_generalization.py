import subprocess
import tempfile
from pathlib import Path
import unittest
import numpy as np
from run_decoder_layer import linear_reference, sparse_linear_reference
from run_decoder_stack import prepare, memory_layout, validate_inputs, attention_reference, compare, verify_lifecycle
from prefill_precision_reference import error as precision_error


class PrefillGeneralizationTest(unittest.TestCase):
    def test_layer_count_and_retained_output_layouts(self):
        for count in (1,3,16,32):
            ping=memory_layout(4,2,count,sequence=2048)
            keep=memory_layout(4,2,count,sequence=2048,retain_outputs=True)
            self.assertEqual(len(ping['layers']),count)
            for layout in (ping,keep):
                for a,b in zip(layout['layers'],layout['layers'][1:]):self.assertEqual(a['output'],b['input'])
                for a,b in zip(layout['regions'],layout['regions'][1:]):self.assertLessEqual(a['offset']+a['bytes'],b['offset'])
                self.assertEqual(layout['gm_stride'],2*2**20)
            self.assertEqual(len({l['output'] for l in keep['layers']}),count)
            self.assertLessEqual(len({l['output'] for l in ping['layers']}),2)
        target=memory_layout(32,8,16,final_norm=True,vocab=128256,node_bytes=2*2**30,embedding=True,retain_outputs=True)
        self.assertLessEqual(target['used_bytes'],target['node_stride'])

    def test_unsupported_physical_layout_and_capacity_fail(self):
        for sequence,count in ((0,2),(33,2),(9216,2),(1024,0),(1024,33)):
            with self.subTest(sequence=sequence,count=count),self.assertRaises(subprocess.CalledProcessError):
                memory_layout(4,2,count,sequence=sequence)
        with self.assertRaises(subprocess.CalledProcessError):memory_layout(32,8,16)

    def test_short_dense_inputs_are_padded_and_last_real_token_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            m=prepare(root,4,2,layers=3,sequence=1,weight_kind='dense',final_norm=True,vocab=65,embedding=True,retain_outputs=True)
            validate_inputs(root)
            self.assertEqual((m['logical_sequence'],m['sequence'],m['padding_rows']),(1,1024,1023))
            tokens=np.fromfile(root/'input_token_ids.bin',dtype='<u4')
            self.assertEqual(tokens[0],64);self.assertTrue(np.all(tokens[1:]==0))
            for layer in range(1,4):
                w=np.load(root/f'layer{layer}_qkv.npy')
                self.assertGreater(np.count_nonzero(w)/w.size,.99)
            self.assertFalse(np.array_equal(np.load(root/'layer1_down.npy'),np.load(root/'layer3_down.npy')))

    def test_dense_scalar_oracle_matches_numpy_rounding_across_k_tiles(self):
        rng=np.random.default_rng(123)
        for k in (64,128,256):
            x=rng.normal(0,.3,(3,k)).astype('<f2');w=rng.normal(0,.1,(7,k)).astype('<f2')
            self.assertTrue(np.array_equal(linear_reference(x,w),sparse_linear_reference(x,w)))
        with self.assertRaises(ValueError):linear_reference(x,w[:,:64])

    def test_oracle_corruption_and_invalid_logical_length_are_rejected(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=prepare(root,4,2,layers=1,sequence=33)
            file=root/'layer1_wo.npy';data=file.read_bytes();file.write_bytes(data[:-2]+b'xx')
            with self.assertRaisesRegex(ValueError,'oracle provenance'):validate_inputs(root)
            file.write_bytes(data)
            m['logical_sequence']=0
            (root/'decoder_stack_inputs.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'sequence contract'):validate_inputs(root)

    def test_nonfinite_and_shape_errors_are_not_reported_as_pass(self):
        good=np.ones((2,3),dtype='<f2')
        for bad in (np.full_like(good,np.nan),np.full_like(good,np.inf),good[:1]):
            with self.assertRaises(ValueError):compare(bad,good,1,'negative')
            with self.assertRaises(ValueError):precision_error(bad,good)

    def test_causal_prefix_ignores_future_tokens_and_uses_rope_positions(self):
        rng=np.random.default_rng(222)
        x=rng.normal(0,.2,(5,256)).astype('<f2');w=rng.normal(0,.02,(512,256)).astype('<f2')
        positions=np.arange(5)[:,None];frequency=1/(10000**(np.arange(0,64,2)/64))
        table=np.empty((5,64),dtype='<f2');table[:,0::2]=np.cos(positions*frequency);table[:,1::2]=np.sin(positions*frequency)
        all_rows=attention_reference(x,w,4,2,table)
        prefix=attention_reference(x[:3],w,4,2,table)
        np.testing.assert_array_equal(all_rows[:3],prefix)
        changed=x.copy();changed[3:]=123
        np.testing.assert_array_equal(attention_reference(changed,w,4,2,table)[:3],prefix)
        identity=table.copy();identity[:,0::2]=1;identity[:,1::2]=0
        self.assertGreater(np.max(np.abs(prefix.astype('f4')-attention_reference(x[:3],w,4,2,identity).astype('f4'))),0)

    def test_attention_partition_job_count_stays_fixed_as_rows_grow(self):
        # A QK worker owns one job per KV head; increasing S grows its row
        # blocks, not the number of worker/head partition jobs. Test the entire
        # lifecycle validator, including counts and omitted-record rejection.
        for sequence in (1024,2048):
            with self.subTest(sequence=sequence),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'logs').mkdir();(root/'stdout').mkdir()
                layout=memory_layout(4,2,1,sequence=sequence);stride=layout['node_stride'];layer=layout['layers'][0]
                m={'layers':1,'sequence':sequence,'hidden':256,'num_query_heads':4,'num_kv_heads':2,'layout':layout}
                lines=[]
                for node in range(1,5):
                    for flag in ('projection_done','decoder_done'):
                        for lane in range(4):
                            lines.append(f'[DECODER_SYNC] event=flag_write_ack address={node*stride+layout["shared"][flag]+lane*8} value=0 status=0 tick=1000')
                for core in range(4,20):
                    node=1+(core-4)%4
                    lines.append(f'[CONTROL_EVENT] stage=projection_accept core={core} epoch=1 tick=100000 input={node*stride+layer["input"]} weights={node*stride+layer["qkv_weight"]}')
                    for tick in (80000,150000):lines.append(f'[DECODER_SYNC] event=scratch_retire core={core} tick={tick}')
                    for stage in range(1,10):
                        lines.append(f'[DECODER_OP] event=done core={core} stage={stage} epoch=1 status=0 start_tick={200000+stage*100000} tick={250000+stage*100000} hbm_read=0 hbm_write=0')
                blocks=sequence//64
                tiles=sum(row+1 for row in range(blocks))*4
                for i in range(16):
                    core=4+i%8;owned=tiles//16+int(i<tiles%16)
                    lines.append(f'[ATTENTION_REUSE_WINDOW_QK] core={core} cycles=80 start=110 end=190 fusion_tiles={owned} expected_tiles={owned}')
                    lines.append(f'[ATTENTION_WORKER_CLUSTER_SFU] core={core} start=110 end=190 cycles=80')
                    lines.append(f'[ATTENTION_WORKER_CLUSTER_E2E] core={core} start=110 end=190 cycles=80')
                for i in range(4*sum((row+4)//4 for row in range(blocks))):
                    lines.append(f'[ATTENTION_WORKER_CLUSTER_PV] core={12+i%8} qk_core={4+i%8} row={i%blocks} window=0 cycles=80 start=110 end=190')
                for tag in (123,456):lines.append(f'[ATTENTION_MILESTONE] stage=root_descriptor_accept tag={tag} sst_tick=110000')
                path=root/'logs/attention-sst.log';path.write_text('\n'.join(lines)+'\n')
                (root/'stdout/stdout-0').write_text('DECODER_LAYER status=0 stage=4 hidden=256 intermediate=1024 layer=1\n'+
                    '\n'.join(f'FUSED_ATTENTION status=0 job=1 manager={i}' for i in range(4)))
                self.assertEqual(verify_lifecycle(root,m)[-1]['all_attention_jobs_checked'],16)
                path.write_text('\n'.join(line for line in lines if 'WORKER_CLUSTER_SFU' not in line)+'\n')
                with self.assertRaisesRegex(ValueError,'lifecycle incomplete'):verify_lifecycle(root,m)

    def test_replay_compatibility_rejects_changed_simulation_or_inputs(self):
        from run_prefill_validation import simulation_source_ast
        source=Path(__file__).with_name('run_decoder_stack.py').read_text()
        repaired=source.replace("expected_jobs=4*m['num_kv_heads']*2*groups", "expected_jobs=4*m['num_kv_heads']*2")
        self.assertEqual(simulation_source_ast(source),simulation_source_ast(repaired))
        for old,new in (("env.update({'GOLEM_DECODER_LAYERS'", "env.update({'WRONG_DECODER_LAYERS'"),
                        ("weight_kind='sparse'", "weight_kind='dense'")):
            self.assertIn(old,source)
            self.assertNotEqual(simulation_source_ast(source),simulation_source_ast(source.replace(old,new)))

    def test_normalization_propagates_bounded_input_error_but_rejects_bad_operator(self):
        from run_decoder_stack import compare_normalized_chain,rms
        rng=np.random.default_rng(91)
        reference=rng.normal(0,.03,(2,256)).astype('<f2')
        actual=(reference.astype('f4')+rng.uniform(-.001,.001,reference.shape)).astype('<f2')
        gamma=np.ones(256,dtype='<f2')
        got=rms(actual,gamma);want=rms(reference,gamma)
        self.assertGreater(np.max(np.abs(got.astype('f4')-want.astype('f4'))),.006)
        result=compare_normalized_chain(got,want,actual,reference,gamma)
        self.assertEqual(result['status'],'PASS')
        with self.assertRaises(ValueError):compare_normalized_chain(got+.5,want,actual,reference,gamma)
        with self.assertRaises(ValueError):compare_normalized_chain(got,want,actual+.1,reference,gamma)
        with self.assertRaises(ValueError):compare_normalized_chain(np.full_like(got,np.nan),want,actual,reference,gamma)


if __name__=='__main__':unittest.main()
