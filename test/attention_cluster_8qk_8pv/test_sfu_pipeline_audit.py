import json
from pathlib import Path
import tempfile
import unittest
from summarize_sfu_vector_pipeline import audit


class SFUPipelineAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.case, self.baseline = root/'new', root/'old'
        result = {'status': 'PASS', 'shape': {'hidden': 256},
                  'dma_timing': {'end_to_end_cycles': 1000000},
                  'lifecycle': {'stage_spans': {str(e): {'6': {'span_cycles': 1000}} for e in (1, 2)}}}
        manifest = {'layout': {'node_stride': 134217728},
                    'preloads': [{'name': 'weight', 'node': 1, 'offset': 0, 'bytes': 32, 'sha256': 'a'}]}
        for case in (self.case, self.baseline):
            (case/'logs').mkdir(parents=True)
            (case/'decoder_stack_result.json').write_text(json.dumps(result))
            (case/'decoder_stack_inputs.json').write_text(json.dumps(manifest))
        logs = []
        for core in range(4, 20):
            job, tick = 0, 0
            for epoch in (1, 2):
                for stage, opcode, chunks in ((2, 3, 2), (6, 5, 8), (7, 4, 8), (9, 3, 2)):
                    start = tick
                    for _ in range(chunks):
                        job += 1
                        cycles = 8217 if opcode == 5 else 515
                        logs.append(f'[SFU_VECTOR_OP] core={core} op={opcode} job={job} elements=8192 '
                                    f'cycles={cycles} submit_tick={tick+1} done_tick={tick+cycles*1000+1}\n')
                        tick += (cycles+1)*1000
                    logs.append(f'[DECODER_OP] event=done core={core} stage={stage} epoch={epoch} '
                                f'start_tick={start} tick={tick}\n')
            logs.append(f'[SFU_VECTOR_PIPELINE] core={core} fifo_elements=64 fifo_high_water=64 '
                        'request_high_water=1 fifo_backpressure_events=20 resource_backpressure_events=0\n')
        old = []
        for core in range(4, 20):
            for unit, lanes, latency, tokens in (('vector_op', 16, 3, 36864),
                                                 ('exp', 16, 8, 8192),
                                                 ('reciprocal', 1, 8, 131072)):
                line = (f'GOLEM_SFU_HW_PIPELINE core={core} unit={unit} lanes={lanes} latency={latency} '
                        f'ii=1 depth=16 accepted_tokens={tokens} max_occupancy=8\n')
                logs.append(line)
                old.append(line.replace(f'accepted_tokens={tokens}', 'accepted_tokens=0'))
        self.log = self.case/'logs/attention-sst.log'
        self.log.write_text(''.join(logs))
        (self.baseline/'logs/attention-sst.log').write_text(''.join(old))

    def tearDown(self):
        self.temp.cleanup()

    def test_balanced_work_and_shared_reservations(self):
        r = audit(self.case, self.baseline)
        self.assertEqual(r['vector_completions'], 640)
        self.assertEqual(r['old_silu_formula_cycles_8192'], 10262)

    def test_rejects_formula_only_exp_without_reservations(self):
        self.log.write_text(self.log.read_text().replace('accepted_tokens=8192', 'accepted_tokens=0'))
        with self.assertRaisesRegex(ValueError, 'actual shared hardware'):
            audit(self.case, self.baseline)

    def test_rejects_missing_completions(self):
        self.log.write_text('\n'.join(line for line in self.log.read_text().splitlines()
                                     if not ('[SFU_VECTOR_OP]' in line and 'core=4 ' in line and 'job=1 ' in line)))
        with self.assertRaisesRegex(ValueError, 'operation completions'):
            audit(self.case, self.baseline)

    def test_rejects_fifo_overflow(self):
        self.log.write_text(self.log.read_text().replace('fifo_high_water=64', 'fifo_high_water=65'))
        with self.assertRaisesRegex(ValueError, 'FIFO capacity'):
            audit(self.case, self.baseline)

    def test_rejects_completion_before_scalar_reciprocal_can_finish(self):
        self.log.write_text(self.log.read_text().replace('cycles=8217', 'cycles=515'))
        with self.assertRaisesRegex(ValueError, 'physical pipeline'):
            audit(self.case, self.baseline)

    def test_rejects_comparison_with_different_weights(self):
        path = self.case/'decoder_stack_inputs.json'
        manifest = json.loads(path.read_text())
        manifest['preloads'][0]['sha256'] = 'b'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'inputs or memory layout differ'):
            audit(self.case, self.baseline)


if __name__ == '__main__':
    unittest.main()
