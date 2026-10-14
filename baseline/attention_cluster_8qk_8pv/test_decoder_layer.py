import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_decoder_layer import validate_layout, prepare, linear_reference, storage_half, OFFSETS


class DecoderLayerTest(unittest.TestCase):
    def test_full_layer_regions_fit_without_destroying_attention_audits(self):
        for hidden,kv in ((256,2),(2048,8)):
            regions=validate_layout(hidden,kv)
            by_name={e['name']:e for e in regions}
            self.assertEqual(by_name['down_weight']['bytes'],hidden*(4*hidden)*2//4)
            self.assertLessEqual(by_name['O']['offset']+by_name['O']['bytes'],OFFSETS['wo'])
            for a,b in zip(regions,regions[1:]):self.assertLessEqual(a['offset']+a['bytes'],b['offset'])
        with self.assertRaises(ValueError):validate_layout(512,2)

    def test_weight_shards_reconstruct_actual_projection_matrices(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);prepare(root,256)
            for name,n,k in [('wo',256,256),('gate',1024,256),('up',1024,256),('down',256,1024)]:
                panels=np.empty((n//64,k//64,64,64),dtype='<f2')
                for node in range(1,5):
                    panels[node-1::4]=np.fromfile(root/f'decoder_{name}_node{node}.bin',dtype='<f2').reshape(-1,k//64,64,64)
                recovered=panels.transpose(0,2,1,3).reshape(n,k)
                np.testing.assert_array_equal(recovered,np.load(root/f'decoder_{name}_weight.npy'))
                self.assertTrue(np.all(recovered[:,-1]!=0))

    def test_reference_handles_down_last_tile_and_fp16_rounding(self):
        x=np.zeros((3,8192),dtype='<f2');x[:,0]=[.125,.5,-.5];x[:,-1]=[1,2,3]
        w=np.zeros((64,8192),dtype='<f2');w[:,0]=.25;w[:,-1]=-.125
        expected=storage_half(storage_half(x[:,0]*np.float32(.25)).astype('f4')+
                              storage_half(x[:,-1]*np.float32(-.125)).astype('f4'))
        np.testing.assert_array_equal(linear_reference(x,w),np.repeat(expected[:,None],64,axis=1))

    def test_tiny_values_follow_existing_converter(self):
        values=np.array([2**-25,3*2**-26,2**-24,-2**-25],dtype='f4')
        np.testing.assert_array_equal(storage_half(values),np.array([0,0,2**-24,0],dtype='<f2'))
        self.assertEqual(int(storage_half(values).view('<u2')[-1]),0x8000)

if __name__=='__main__':unittest.main()
