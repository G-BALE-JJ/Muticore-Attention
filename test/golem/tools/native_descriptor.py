"""Versioned fixed-launcher commands; no compiler/TileBin dependency."""
import json
from pathlib import Path
import struct
from tools.native_llm_workload import load_config, MODES, digest
MAGIC=0x4e444d47
BUNDLE_MAGIC=0x4e424d47
VERSION=1
REGION_BYTES=1024*1024
COMMAND=struct.Struct('<14I5Q8I')
BUNDLE=struct.Struct('<38I')
assert COMMAND.size==128 and BUNDLE.size==152


def checksum(data):
    value=2166136261
    for index,byte in enumerate(data):
        value=((value^(0 if 16<=index<20 else byte))*16777619)&0xffffffff
    return value


def encode_command(config,worker):
    p=config['plan'];g=p['geometry'];r=p['regions']
    job=next(j for j in p['instances'] if worker in j['physical_workers'])
    maps=[job['data_nodes'],job['task_nodes'],job['a_tile_nodes'],job['b_tile_nodes']]
    offsets=[];payload=bytearray()
    for mapping in maps:
        offsets.extend((COMMAND.size+len(payload),len(mapping)))
        payload+=struct.pack('<'+'I'*len(mapping),*mapping)
    payload+=bytes((-len(payload))%8)
    size=COMMAND.size+len(payload)
    if size>32768:raise ValueError('native command exceeds reserved 32 KiB command capacity')
    fields=[MAGIC,VERSION,1,size,0,int(p['replicate_b']),worker,
            job['physical_workers'].index(worker),len(job['physical_workers']),MODES[config['precision']][0],
            g['m_tiles']*64,g['n_tiles']*64,g['k_tiles']*64,g['reuse_m'],
            p['node_bytes'],r['a']['offset'],r['b']['offset'],r['c']['offset'],0,*offsets]
    data=bytearray(COMMAND.pack(*fields)+payload)
    struct.pack_into('<I',data,16,checksum(data))
    return bytes(data)


def write_bundle(config_path,root):
    root=Path(root);config=load_config(str(Path(config_path).resolve()))
    commands=[encode_command(config,i) for i in range(16)]
    offsets=[];offset=BUNDLE.size
    for command in commands:offsets.append(offset);offset+=len(command)
    if offset>REGION_BYTES:raise ValueError('native bundle exceeds reserved Node0 mapping')
    header=BUNDLE.pack(BUNDLE_MAGIC,VERSION,offset,16,0,0,*offsets,*map(len,commands))
    data=bytearray(header+b''.join(commands));struct.pack_into('<I',data,16,checksum(data[:BUNDLE.size]))
    bundle=root/'native_commands.bin';bundle.write_bytes(data)
    image=root/'native_node0_init.bin'
    with image.open('wb') as stream:
        stream.truncate(config['plan']['node_bytes'])
        stream.seek(config['plan']['node_bytes']-REGION_BYTES);stream.write(data)
    manifest=dict(schema='llm.native-bundle-v1',bundle=str(bundle),bundle_sha256=digest(bundle),
                  node0_image=str(image),node0_image_sha256=digest(image),command_count=16,
                  command_bytes=list(map(len,commands)),region_bytes=REGION_BYTES,
                  physical_address=config['plan']['node_bytes']-REGION_BYTES,virtual_address=0x50000000)
    (root/'native_bundle.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
