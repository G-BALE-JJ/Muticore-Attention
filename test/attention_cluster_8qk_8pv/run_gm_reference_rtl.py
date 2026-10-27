#!/usr/bin/env python3
"""Compile the unmodified reference bank/SRAM RTL and compare it to the C++ scheduler."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,default=Path('/data/jjgong/RISC-V-CIM-Manycore-SST'))
    p.add_argument('--verilator',default='verilator')
    p.add_argument('--artifact-root',type=Path,required=True)
    a=p.parse_args();out=a.artifact_root.resolve();out.mkdir(parents=True,exist_ok=True)
    rtl=a.reference_root/'test/golem/architecture/HDL/rtl'
    sources=[rtl/(s+'.sv') for s in ('golem_node_params_pkg','golem_sram_macros','golem_gm_bank_cluster')]
    tests=ROOT/'src/sst/elements/golem/globalmemory'
    command=[a.verilator,'--cc','--exe','--unroll-count','1024','-Wno-fatal','-DGOLEM_SRAM_SIM',
             '--top-module','golem_gm_bank_cluster','-GCLIENTS=2','-GGM_SIZE_BYTES=2097152',
             '--Mdir',str(out/'obj'),'-CFLAGS','-I'+str(tests),*[str(s) for s in sources],str(tests/'test_localGmReferenceRtl.cpp')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['make','-C',str(out/'obj'),'-f','Vgolem_gm_bank_cluster.mk','-j4'],check=True,stdout=log,stderr=subprocess.STDOUT)
    result=subprocess.check_output([str(out/'obj/Vgolem_gm_bank_cluster')],text=True)
    counts={k:int(v) for k,v in re.findall(r'(cycles|grants|reads|conflicts)=(\d+)',result)}
    if not result.startswith('PASS') or counts.get('cycles')!=9999:raise ValueError(result)
    report={'status':'PASS',**counts,'stdout':result,'command':command,'tool':subprocess.check_output([a.verilator,'--version'],text=True).strip(),
        'reference_head':subprocess.check_output(['git','-C',str(a.reference_root),'rev-parse','HEAD'],text=True).strip(),
        'reference_sha256':{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        'test_sha256':hashlib.sha256((tests/'test_localGmReferenceRtl.cpp').read_bytes()).hexdigest(),
        'scheduler_sha256':hashlib.sha256((tests/'localGmBankScheduler.h').read_bytes()).hexdigest(),
        'scope':'unaltered reference bank/SRAM RTL; no whole-reference DMA/NoC/HBM timing equivalence claim'}
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(result,end='')
if __name__=='__main__':main()
