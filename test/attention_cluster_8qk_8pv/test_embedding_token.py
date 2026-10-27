import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from run_decoder_stack import ROOT,memory_layout,prepare,validate_inputs
from select_next_token import select_next_token

sys.path.insert(0,str(ROOT/'test/golem/architecture'))
from partition_planner import attention_query_block_placements


class EmbeddingTokenTest(unittest.TestCase):
    def test_greedy_valid_vocab_padding_ties_and_invalid_logits(self):
        self.assertEqual(select_next_token([1,3,3,100,float('nan')],3)['token_id'],1)
        self.assertEqual(select_next_token([-9,-4,-8],3)['token_id'],1)
        for values,vocab in [([float('nan')],1),([float('inf')],1),([],0),([[1]],1),([1],2)]:
            with self.subTest(values=values,vocab=vocab),self.assertRaises(ValueError):select_next_token(values,vocab)

    def test_embedding_table_stripes_input_oracle_and_no_host_activation_preload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            m=prepare(root,4,2,final_norm=True,vocab=9,node_bytes=2*2**30,embedding=True)
            validate_inputs(root)
            self.assertFalse(any(e['name']=='X0' for e in m['preloads']))
            table=np.load(root/'embedding_weight.npy');recovered=np.empty_like(table)
            for node in range(1,5):recovered[node-1::4]=np.fromfile(root/f'embedding_weight_node{node}.bin',dtype='<f2').reshape(-1,256)
            np.testing.assert_array_equal(recovered,table)
            tokens=np.fromfile(root/'input_token_ids.bin',dtype='<u4')
            self.assertEqual(tokens[-1],8)
            np.testing.assert_array_equal(tokens[:4],tokens[4:8])
            np.testing.assert_array_equal(np.fromfile(root/'projection_x.bin',dtype='<f2').reshape(1024,256),table[tokens])
            layout=m['layout']
            self.assertEqual(layout['layers'][0]['input'],layout['embedding']['output'])
            self.assertNotIn(layout['embedding']['output'],layout['activations'])

    def test_formal_embedding_capacity_and_region_disjointness(self):
        layout=memory_layout(32,8,final_norm=True,vocab=128256,node_bytes=2*2**30,embedding=True)
        self.assertLess(layout['used_bytes'],2*2**30)
        for a,b in zip(layout['regions'],layout['regions'][1:]):self.assertLessEqual(a['offset']+a['bytes'],b['offset'])
        self.assertEqual(next(r['bytes'] for r in layout['regions'] if r['name']=='embedding_weight'),128256*2048*2//4)

    def test_query_block_eight_ranks_balance_workers_and_keep_memory_qk_local(self):
        for ranks in (2,4):
            self.assertEqual(attention_query_block_placements(25,20,ranks),[(r%ranks if r<24 else 0,0) for r in range(25)])
        p=attention_query_block_placements(25,20,8)
        self.assertEqual({rank for rank,thread in p},set(range(8)))
        for column in range(4):
            self.assertEqual([p[r][0] for r in (column,4+column,8+column,12+column)],[2*column]*4)
            self.assertEqual([p[r][0] for r in (16+column,20+column)],[2*column+1]*2)
        for args in ((25,19,8),(25,20,3),(23,20,8)):
            with self.assertRaises(ValueError):attention_query_block_placements(*args)


if __name__=='__main__':unittest.main()
