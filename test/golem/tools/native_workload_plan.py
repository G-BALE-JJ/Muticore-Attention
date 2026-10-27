"""Native independent workload/HBM placement; no compiler or TileBin dependency.

Node 0 remains OS-only. Plan node/worker sets partition the given hardware;
all tensor offsets are local to a node. The runtime accepts each instance's
maps through WorkloadPlacement and startPlacedWindow, with physical scheduler
worker slot kept separate from logical per-instance worker slot.
"""
import argparse
import json
from pathlib import Path


def build_plan(*, m_tiles, n_tiles, k_tiles, reuse_m, reuse_n, workers,
               groups, data_nodes, instances=1, panel_mapping='group-affine',
               replicate_b=False, block_m=64, block_n=64, block_k=64,
               elem_bytes=4, node_bytes=64*1024**2):
    counts=(m_tiles,n_tiles,k_tiles,reuse_m,reuse_n,workers,groups,data_nodes,
            instances,block_m,block_n,block_k,elem_bytes,node_bytes)
    if any(not isinstance(x,int) or x<=0 for x in counts):
        raise ValueError('all geometry and resource counts must be positive integers')
    if workers%groups or groups%instances or data_nodes%instances:
        raise ValueError('instances must partition whole groups, workers and data nodes')
    if panel_mapping not in ('group-affine','reuse-striped'):
        raise ValueError('unsupported panel mapping')
    if (reuse_m>1)!=(reuse_n>1):
        raise ValueError('native placement requires no reuse or two-dimensional reuse')
    mg=(m_tiles+reuse_m-1)//reuse_m;ng=(n_tiles+reuse_n-1)//reuse_n
    wp=workers//instances;gp=groups//instances;dp=data_nodes//instances
    mat_stride=block_m*block_k*elem_bytes
    vec_stride=block_n*block_k*elem_bytes
    c_stride=((block_m*block_n*elem_bytes+255)//256)*256
    jobs=[]
    for instance in range(instances):
        nodes=list(range(1+instance*dp,1+(instance+1)*dp))
        physical_workers=[s for s in range(workers) if (s%groups)//gp==instance]
        task_nodes=[nodes[(task%wp)%gp%dp] for task in range(mg*ng)]
        def panels(count,reuse,gcount,is_b):
            if panel_mapping=='group-affine':return [task_nodes[t//reuse] for t in range(count)]
            stripe=min(reuse,max(1,(dp+gcount-1)//gcount))
            phase=dp//2 if is_b and stripe>1 else 0
            return [nodes[((t//reuse)*stripe+(t%reuse)%stripe+phase)%dp] for t in range(count)]
        a=panels(m_tiles,reuse_m,mg,False);b=panels(n_tiles,reuse_n,ng,True)
        jobs.append(dict(instance=instance,data_nodes=nodes,physical_workers=physical_workers,
                         task_nodes=task_nodes,a_tile_nodes=a,b_tile_nodes=b,
                         worker_task_counts=[len(range(s,mg*ng,wp)) for s in range(wp)]))
    max_a=max(max(job['a_tile_nodes'].count(n) for n in job['data_nodes']) for job in jobs)
    max_b=n_tiles if replicate_b else max(max(job['b_tile_nodes'].count(n) for n in job['data_nodes']) for job in jobs)
    max_c=max(max(job['task_nodes'].count(n) for n in job['data_nodes']) for job in jobs)
    a_offset=0;b_offset=max_a*k_tiles*mat_stride
    c_offset=b_offset+max_b*k_tiles*vec_stride
    end=c_offset+max_c*reuse_m*reuse_n*c_stride
    if end>node_bytes:raise ValueError('HBM images exceed physical node capacity')
    plan=dict(schema='llm.native-workload-placement',version=1,instances=jobs,
              panel_mapping=panel_mapping,replicate_b=bool(replicate_b),node_bytes=node_bytes,
              geometry=dict(m_tiles=m_tiles,n_tiles=n_tiles,k_tiles=k_tiles,reuse_m=reuse_m,reuse_n=reuse_n,
                            block_m=block_m,block_n=block_n,block_k=block_k,elem_bytes=elem_bytes),
              regions=dict(a=dict(offset=a_offset,stride=mat_stride),b=dict(offset=b_offset,stride=vec_stride),
                           c=dict(offset=c_offset,stride=c_stride),end=end))
    return plan


def panel_locations(plan,instance):
    job=plan['instances'][instance];g=plan['geometry'];reg=plan['regions']
    for kind,mapping in [('a',job['a_tile_nodes']),('b',job['b_tile_nodes'])]:
        for tile,node in enumerate(mapping):
            nodes=job['data_nodes'] if kind=='b' and plan['replicate_b'] else [node]
            slot=tile if kind=='b' and plan['replicate_b'] else mapping[:tile].count(node)
            for k in range(g['k_tiles']):
                offset=reg[kind]['offset']+(slot*g['k_tiles']+k)*reg[kind]['stride']
                for target in nodes:yield kind,tile,k,target,offset


def write_images(plan,inputs,output_dir,dtype='float32'):
    """inputs contains one distinct (A[M,K], B[K,N]) pair per instance."""
    import numpy as np
    if len(inputs)!=len(plan['instances']):raise ValueError('one input pair per instance is required')
    dt=np.dtype(dtype).newbyteorder('<');g=plan['geometry']
    if dt.itemsize!=g['elem_bytes']:raise ValueError('input dtype does not match planned element bytes')
    target=Path(output_dir);target.mkdir(parents=True,exist_ok=False)
    shape_a=(g['m_tiles']*g['block_m'],g['k_tiles']*g['block_k'])
    shape_b=(shape_a[1],g['n_tiles']*g['block_n'])
    # Validate every instance before creating any partial images.
    arrays=[]
    for a,b in inputs:
        a=np.asarray(a,dtype=dt);b=np.asarray(b,dtype=dt)
        if a.shape!=shape_a or b.shape!=shape_b:raise ValueError('input tensor shape mismatch')
        arrays.append((a,b))
    handles={}
    try:
        for job in plan['instances']:
            for node in job['data_nodes']:
                f=(target/('node%d.bin'%node)).open('xb');f.truncate(plan['node_bytes']);handles[node]=f
        for instance,(a,b) in enumerate(arrays):
            for kind,tile,k,node,offset in panel_locations(plan,instance):
                ks=slice(k*g['block_k'],(k+1)*g['block_k'])
                if kind=='a':values=a[tile*g['block_m']:(tile+1)*g['block_m'],ks]
                else:values=b[ks,tile*g['block_n']:(tile+1)*g['block_n']].T
                handles[node].seek(offset);handles[node].write(values.tobytes(order='C'))
    finally:
        for f in handles.values():f.close()
    (target/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    for key,default in [('m-tiles',4),('n-tiles',4),('k-tiles',2),('reuse-m',2),('reuse-n',2),
                        ('workers',16),('groups',4),('data-nodes',4),('instances',1),
                        ('block-m',64),('block-n',64),('block-k',64),('elem-bytes',4),('node-bytes',64*1024**2)]:
        p.add_argument('--'+key,type=int,default=default)
    p.add_argument('--panel-mapping',choices=('group-affine','reuse-striped'),default='group-affine')
    p.add_argument('--replicate-b',action='store_true')
    args=vars(p.parse_args());output=args.pop('output');plan=build_plan(**args)
    output.write_text(json.dumps(plan,indent=2)+'\n')
if __name__=='__main__':main()
