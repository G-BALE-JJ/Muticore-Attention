"""Wire ABI, packing identity and rejection checks independent of SST."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from tools.native_descriptor import encode_command,checksum,COMMAND,write_bundle
from tools.native_workload_plan import build_plan
ROOT=Path(__file__).resolve().parents[2]

class NativeDescriptorTests(unittest.TestCase):
    def config(self):
        return dict(schema='llm.native-operator',precision='int8',plan=build_plan(
            m_tiles=16,n_tiles=4,k_tiles=2,reuse_m=2,reuse_n=2,workers=16,groups=4,
            data_nodes=4,instances=2,panel_mapping='reuse-striped',replicate_b=True,
            node_bytes=128*1024**2,elem_bytes=4,input_bits=8))

    def test_host_and_cpp_validation_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'check.cc'
            source.write_text('''#include <sst/elements/golem/isa/nativeDescriptor.h>
#include <iostream>
#include <vector>
int main() {
 std::vector<char> b((std::istreambuf_iterator<char>(std::cin)),{});
 if(b.size()<sizeof(GolemNative::Command)) return 20;
 GolemNative::Command c;std::memcpy(&c,b.data(),sizeof(c));
 auto s=GolemNative::validate(c,b.size());
 if(s!=GolemNative::Status::Success) return int(s);
 return GolemNative::checksum(b.data(),b.size())==c.checksum?0:21;
}''')
            exe=root/'check'
            subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'src'),str(source),'-o',str(exe)],check=True)
            def code(data):return subprocess.run([str(exe)],input=data).returncode
            config=self.config()
            for worker in range(16):
                data=encode_command(config,worker)
                self.assertEqual(code(data),0)
                fields=COMMAND.unpack_from(data)
                self.assertEqual(fields[6],worker)
                self.assertEqual(fields[7],(worker//4)*2+worker%2)
                self.assertEqual(fields[18],0)
            data=encode_command(config,0)
            for offset,value,status in ((0,0,1),(4,99,2),(8,99,2),(20,2,1),(24,16,1),
                                        (32,0,1),(36,8,2),(40,65,1),(52,3,1),(96,0,1),
                                        (108,0xffffffff,1)):
                broken=bytearray(data);struct.pack_into('<I',broken,offset,value)
                struct.pack_into('<I',broken,16,checksum(broken))
                self.assertEqual(code(broken),status,(offset,value))
            self.assertEqual(code(data[:-8]),1)
            broken=bytearray(data);broken[-1]^=1;self.assertEqual(code(broken),21)

    def test_node0_reserved_pages_are_not_allocated(self):
        with tempfile.TemporaryDirectory() as directory:
            exe=Path(directory)/'pages'
            subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'src'),
                '-I'+str(ROOT/'src/sst/elements/vanadis'),'-I/local/sstcore/include',
                '-I/local/sstcore/include/sst/core',str(ROOT/'test/golem/unit/native_pages_test.cpp'),
                '-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True)

    def test_bundle_preserves_each_worker_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=self.config();p=root/'config.json'
            p.write_text(json.dumps(config));manifest=write_bundle(p,root)
            data=Path(manifest['bundle']).read_bytes()
            header=struct.unpack_from('<38I',data)
            self.assertEqual(header[4],checksum(data[:152]))
            for worker in range(16):
                offset,size=header[6+worker],header[22+worker]
                self.assertEqual(data[offset:offset+size],encode_command(config,worker))
            with Path(manifest['node0_image']).open('rb') as stream:
                stream.seek(manifest['physical_address']);self.assertEqual(stream.read(len(data)),data)
            self.assertEqual(manifest['command_count'],16)

if __name__=='__main__':unittest.main()
