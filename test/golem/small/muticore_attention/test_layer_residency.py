"""Retained FFN output, timed final Norm consumer and metadata validation."""
import os
from unittest.mock import patch
import unittest
import test_decoder_stage_chain


class LayerResidencyTest(unittest.TestCase):
    def test_final_norm_reads_retained_y_with_delayed_acks(self):
        with patch.dict(os.environ, GOLEM_DECODER_LAYER_RESIDENCY='1', GOLEM_TEST_LAYER_CHECK='1'):
            test_decoder_stage_chain.DecoderStageChainTest().test_bits_traffic_lifetimes_and_shape_fallback()


if __name__ == '__main__': unittest.main()
