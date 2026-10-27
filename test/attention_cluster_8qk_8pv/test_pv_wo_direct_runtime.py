"""Keep a direct Wo landing leased until explicit consumer release."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_pv_wo_runtime import ROOT, STUB, SOURCE


class DirectLeaseTest(unittest.TestCase):
    def test_two_generations_cannot_reuse_the_landing_while_leased(self):
        source=SOURCE.replace(
            'auto& s=service[4+manager+lane*4];assert(!s->consume(epoch+1,manager,lane));\n'
            '   assert(s->consume(epoch,manager,lane));assert(!s->consume(epoch,manager,lane));',r'''
   const uint32_t target=4+manager+lane*4;
   auto& s=service[target];assert(!s->acquire(epoch+1,manager,lane));
   assert(s->acquire(epoch,manager,lane));
   assert(!s->consume(epoch,manager,lane));assert(!s->acquire(epoch,manager,lane));
   ControlTransportMessage next{};next.kind=ControlTransportMessageKind::AttentionPvWoData;
   next.workerCore=target;next.ownerCore=12;next.workerSlot=0;next.outputAddr=manager;
   next.inputAddr=epoch+1;next.queryLength=256;next.headDim=64;next.tensorPayload.resize(8192);
   bool rejected=false;try{s->receive(next);}catch(const std::logic_error&){rejected=true;}assert(rejected);
   assert(!s->release(epoch+1,manager,lane));
   assert(s->release(epoch,manager,lane));assert(!s->release(epoch,manager,lane));
   assert(!s->consume(epoch,manager,lane));''')
        self.assertIn('next.inputAddr=epoch+1',source)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);stub=path/'sst/elements/golem/globalmemory/globalmemory.h'
            stub.parent.mkdir(parents=True);stub.write_text(STUB)
            (path/'test.cc').write_text(source)
            includes=['-I',str(path)]
            if os.environ.get('GOLEM_TEST_SOURCE_OVERLAY'):
                includes+=['-I',os.environ['GOLEM_TEST_SOURCE_OVERLAY']]
            subprocess.run(['g++','-std=c++17','-O2',*includes,'-I',str(ROOT/'src'),str(path/'test.cc'),'-o',str(path/'test')],check=True)
            result=subprocess.run([str(path/'test')],capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            self.assertEqual(result.stdout.count(b'[PV_WO_HANDOFF]'),32)
            self.assertIn(b'relocation_read_bytes=0 relocation_write_bytes=0',result.stdout)
