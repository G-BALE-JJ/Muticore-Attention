"""A shard's own Norm completion must not authorize reading another shard's token."""
import json
from pathlib import Path
import tempfile
import unittest

from verify_lm_head import verify_lm_head
from verification_retained_input import _inputs


class ProducerDependencyTest(unittest.TestCase):
    def test_reference_accepts_only_equivalent_tail_schedule(self):
        reference = dict(sequence=1024, hidden=2048, oracle_sha256='input-hash')
        candidate = dict(reference, final_norm_lm_stream=True)
        self.assertEqual(_inputs(reference), _inputs(candidate))
        for key, value in (('oracle_sha256', 'changed'), ('hidden', 4096)):
            changed = dict(candidate, **{key: value})
            self.assertNotEqual(_inputs(reference), _inputs(changed))

    def verify_trace(self, read_tick, notify=False, worker_poll=False, priority=False):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            stride = 1 << 30
            hidden = 64
            lm = dict(vocab=1024, done=4096, output=8192, weight=16384)
            manifest = dict(sequence=1024, hidden=hidden, layers=2,
                            final_norm_lm_stream=True,
                            layout=dict(lm_head=lm, node_stride=stride,
                                        final_norm=dict(output=32768,done=2048)))
            if notify:
                manifest.update(lm_handoff_protocol='producer_remote_store_v1',
                                final_norm_producer_first=priority)
            final = dict(last_completion_ack_tick=310000, last_output_ack_tick=300000,
                         worker_output_ack_ticks={str(c): (200000 if c == 19 else 100000)
                                                  for c in range(4, 20)})
            (case/'decoder_stack_provenance.json').write_text(json.dumps(dict(
                environment={'GOLEM_DECODER_FINAL_NORM_LM_STREAM': '1',
                             'GOLEM_DECODER_FINAL_NORM_PRODUCER_FIRST':str(int(priority))})))
            lines = ['[DECODER_OP] event=done stage=1 epoch=2 start_tick=50000']
            for node in range(1, 5):
                for lane in range(4):
                    lines.append('[DECODER_SYNC] event=flag_write_ack '
                                 f'address={node*stride+lm["done"]+lane*8} value=0 status=0 tick=1000')
            for core in range(4, 20):
                if notify:
                    lines.append(f'[DECODER_OP] event=done core={core} stage=10 epoch=3 '
                                 f'start_tick=80000 tick={200000 if core==19 else 100000}')
                lines.append(f'[DECODER_OP] event=done core={core} stage=11 epoch=4 '
                             f'op=1 status=0 start_tick={210000 if notify else 120000} tick=400000')
            log = case/'attention-sst.log'
            log.write_text('\n'.join(lines)+'\n')
            address = 4*stride+32768+255*hidden*2
            log.with_suffix('.resources.jsonl').write_text(
                (f'domain=decoder resource=dma_read_transport core=4 address={4*stride+2048+24} '
                 'bytes=8 start_tick=150000 done_tick=151000 ok=1\n' if worker_poll else '')+
                f'domain=decoder resource=dma_read_transport core=4 address={address} '
                f'bytes={hidden*2} start_tick={read_tick} done_tick={read_tick+1000} ok=1\n')
            # Deliberately omit the weight/output traffic. A valid dependency
            # must proceed to the independent coverage check and fail there.
            verify_lm_head(case, manifest, log, final, {})

    def test_own_norm_ready_but_common_token_not_ready_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'before final normalized output ACK'):
            self.verify_trace(150000)

    def test_equal_to_producer_ack_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'before final normalized output ACK'):
            self.verify_trace(200000)

    def test_actual_token_ready_can_precede_unrelated_global_norm_ack(self):
        with self.assertRaisesRegex(ValueError, 'incomplete/duplicate output or weight tile coverage'):
            self.verify_trace(250000)

    def test_notification_rejects_any_worker_hbm_flag_poll(self):
        with self.assertRaisesRegex(ValueError, 'worker HBM polling forbidden'):
            self.verify_trace(250000,notify=True,worker_poll=True)

    def test_notification_still_checks_tensor_coverage(self):
        with self.assertRaisesRegex(ValueError, 'incomplete/duplicate output or weight tile coverage'):
            self.verify_trace(250000,notify=True)

    def test_producer_priority_is_not_just_a_manifest_label(self):
        with self.assertRaisesRegex(ValueError, 'producer-first ordering violated'):
            self.verify_trace(250000,notify=True,priority=True)


if __name__ == '__main__':
    unittest.main()
