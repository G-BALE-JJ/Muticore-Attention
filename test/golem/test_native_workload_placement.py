from pathlib import Path
import importlib.util
import subprocess
import tempfile
import unittest
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configure.ac').is_file())
spec=importlib.util.spec_from_file_location('native_workload_plan',Path(__file__).parent/'tools/native_workload_plan.py')
planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)
class NativeWorkloadPlacementTests(unittest.TestCase):
    def test_partition_mapping_replication_and_distinct_binary_inputs(self):
        for mapping in ('group-affine','reuse-striped'):
            for replicate in (False,True):
                plan=planner.build_plan(m_tiles=4,n_tiles=4,k_tiles=2,reuse_m=2,reuse_n=2,
                    workers=16,groups=4,data_nodes=4,instances=2,panel_mapping=mapping,replicate_b=replicate,
                    block_m=2,block_n=2,block_k=3,node_bytes=4096)
                self.assertEqual(sorted(n for j in plan['instances'] for n in j['data_nodes']),[1,2,3,4])
                self.assertEqual(sorted(w for j in plan['instances'] for w in j['physical_workers']),list(range(16)))
                self.assertEqual([sum(j['worker_task_counts']) for j in plan['instances']],[4,4])
                inputs=[(np.arange(48,dtype=np.float32).reshape(8,6)+100*i,
                         np.arange(48,dtype=np.float32).reshape(6,8)+200*i) for i in range(2)]
                with tempfile.TemporaryDirectory() as tmp:
                    out=Path(tmp)/'images';planner.write_images(plan,inputs,out)
                    for i,(a,b) in enumerate(inputs):
                        for kind,tile,k,node,offset in planner.panel_locations(plan,i):
                            ks=slice(k*3,(k+1)*3)
                            expected=a[tile*2:(tile+1)*2,ks] if kind=='a' else b[ks,tile*2:(tile+1)*2].T
                            with (out/('node%d.bin'%node)).open('rb') as f:
                                f.seek(offset);actual=f.read(expected.nbytes)
                            self.assertEqual(actual,expected.tobytes())
    def test_cpp_runtime_addresses_match_image_layout(self):
        plan=planner.build_plan(m_tiles=8,n_tiles=8,k_tiles=3,reuse_m=4,reuse_n=4,
            workers=16,groups=4,data_nodes=4,panel_mapping='reuse-striped',replicate_b=True)
        j=plan['instances'][0];reg=plan['regions']
        vector=lambda xs:'{'+','.join(map(str,xs))+'}'
        body='#include <sst/elements/golem/tileengine/workloadPlacement.h>\n#include <cassert>\nint main(){\nSST::Golem::WorkloadPlacement p;\n'
        for field,key in [('dataNodes','data_nodes'),('taskNodes','task_nodes'),('aNodes','a_tile_nodes'),('bNodes','b_tile_nodes')]:body+='p.%s=%s;\n'%(field,vector(j[key]))
        body+='p.replicateB=true;assert(p.valid(4,8,8,4));\n'
        for kind,tile,k,node,offset in planner.panel_locations(plan,0):
            if kind=='a':expr='p.aAddress(%d,%d,3,%d,%d,%d)'%(tile,k,plan['node_bytes'],reg['a']['offset'],reg['a']['stride'])
            else:expr='p.bAddress(%d,%d,3,%d,%d,%d,%d)'%(tile,k,node,plan['node_bytes'],reg['b']['offset'],reg['b']['stride'])
            body+='assert(%s==%dULL);\n'%(expr,node*plan['node_bytes']+offset)
        body+='p.dataNodes.push_back(1);assert(!p.valid(4,8,8,4));}\n'
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'test.cpp';src.write_text(body);binary=src.with_suffix('')
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT/'src'),str(src),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)
    def test_invalid_resource_partitions_and_capacity_rejected(self):
        base=dict(m_tiles=4,n_tiles=4,k_tiles=2,reuse_m=2,reuse_n=2,workers=16,groups=4,data_nodes=4)
        for change in [dict(instances=3),dict(node_bytes=1),dict(panel_mapping='invalid'),dict(reuse_n=1)]:
            with self.assertRaises(ValueError):planner.build_plan(**dict(base,**change))
if __name__=='__main__':unittest.main()
