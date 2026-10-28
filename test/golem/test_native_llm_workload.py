import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.native_workload_plan import build_plan,panel_locations
from tools.native_llm_workload import MODES,encode,engine_parameters,load_config,preload_native_images,write_payloads

class NativeLLMWorkloadTests(unittest.TestCase):
    def plan(self,precision='fp16',instances=2):
        return build_plan(m_tiles=16,n_tiles=4,k_tiles=2,reuse_m=2,reuse_n=2,
            workers=16,groups=4,data_nodes=4,instances=instances,panel_mapping='reuse-striped',
            replicate_b=True,input_bits=MODES[precision][1])
    def test_exact_native_operand_encodings(self):
        x=np.array([[-2,-1,0,1,2]],dtype='<f4')
        self.assertEqual(encode(x,'int4'),bytes.fromhex('fe100200'))
        self.assertEqual(encode(x,'int8'),bytes.fromhex('feff000102000000'))
        self.assertEqual(encode(x,'fp8e4m3'),bytes.fromhex('c0b8003840000000'))
        self.assertEqual(encode(x,'fp8e5m2'),bytes.fromhex('c0bc003c40000000'))
        for precision in ('fp16','bf16','tf32','fp32'):
            self.assertEqual(len(encode(x,precision)),((5*MODES[precision][1]+31)//32)*4)
        with self.assertRaises(ValueError):encode(np.array([[0.3]]),'int8')
        varied=np.array([[-2,-1,0,1,2],[2,1,0,-1,-2]],dtype='<f4').T
        for precision in MODES:
            self.assertEqual(encode(varied,precision),encode(np.ascontiguousarray(varied),precision))
    def test_eight_precisions_partition_existing_workers_and_size_packed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            for precision,(mode,bits) in MODES.items():
                plan=self.plan(precision);path=Path(tmp)/(precision+'.json')
                path.write_text(json.dumps(dict(schema='llm.native-operator',precision=precision,plan=plan)))
                workers=[engine_parameters(str(path),core) for core in range(4,20)]
                self.assertTrue(all(w['native_precision_mode']==mode for w in workers))
                self.assertEqual({w['native_worker_count'] for w in workers},{8})
                self.assertEqual(plan['regions']['a']['stride'],64*64*bits//8)
                self.assertEqual(plan['regions']['c']['stride'],64*64*4)
                self.assertNotEqual(workers[0]['native_data_nodes'],workers[2]['native_data_nodes'])
                self.assertEqual(engine_parameters(str(path),0),{})
    def test_tampered_partition_and_capacity_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=dict(schema='llm.native-operator',precision='fp16',plan=self.plan())
            for index,mutation in enumerate(('worker','node','offset','mode')):
                c=copy.deepcopy(base)
                if mutation=='worker':c['plan']['instances'][1]['physical_workers'][0]=0
                elif mutation=='node':c['plan']['instances'][1]['data_nodes'][0]=1
                elif mutation=='offset':c['plan']['regions']['c']['offset']=0
                else:c['precision']='fp8e4m3'
                path=Path(tmp)/str(index);path.write_text(json.dumps(c))
                with self.assertRaises(ValueError):load_config(str(path))
            with self.assertRaises(ValueError):build_plan(m_tiles=16,n_tiles=4,k_tiles=2,reuse_m=2,reuse_n=2,workers=16,groups=4,data_nodes=4,input_bits=16,node_bytes=1)
    def test_distinct_payload_images_replication_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.plan()
            inputs=[(np.full((1024,128),i+1,dtype='<f4'),np.full((128,256),2-i,dtype='<f4')) for i in range(2)]
            entries=write_payloads(plan,inputs,'fp16',root/'payloads')
            path=root/'config.json';path.write_text(json.dumps(dict(schema='llm.native-operator',precision='fp16',plan=plan,payloads=entries)))
            class Memory:
                def __init__(self):self.blocks={}
                def write_block(self,offset,data):self.blocks[offset]=data
            buffers={node:Memory() for node in range(1,5)}
            preload_native_images(str(path),buffers,plan['node_bytes'])
            for i,(a,b) in enumerate(inputs):
                for kind,tile,k,node,offset in panel_locations(plan,i):
                    want=a[tile*64:(tile+1)*64,k*64:(k+1)*64] if kind=='a' else b[k*64:(k+1)*64,tile*64:(tile+1)*64].T
                    self.assertEqual(buffers[node].blocks[offset],encode(want,'fp16'))
            Path(entries[0]['path']).write_bytes(b'bad')
            with self.assertRaises(ValueError):preload_native_images(str(path),buffers,plan['node_bytes'])
if __name__=='__main__':unittest.main()
