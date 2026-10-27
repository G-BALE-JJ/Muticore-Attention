#!/usr/bin/env python3
"""Plot actual physical Projection service and operand readiness for selected workers."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(case, output):
    case=Path(case)
    selected={12:[],7:[]}
    with (case/'projection_tile_timeline.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['core'] in selected and row['block']==0 and row['kind']==0 and row['head']==0 and row['tile']<16:
                selected[row['core']].append(row)
    fig,axs=plt.subplots(2,1,figsize=(14,6),layout='constrained')
    for ax,(core,rows) in zip(axs,selected.items()):
        if len(rows)!=16:raise ValueError('sample tiles incomplete')
        rows.sort(key=lambda r:r['tile'])
        origin=rows[0]['matrix_submit_cycle']
        colors={'Matrix':'#cd8a27','Scatter':'#568cb3','MAC':'#238c6b'}
        for row in rows:
            for label,lo,hi,y in (
                ('Matrix',row['matrix_submit_cycle']+row['matrix_command_before_resource_cycles'],row['matrix_done_cycle'],2),
                ('Scatter',row['scatter_submit_cycle']+row['scatter_command_before_resource_cycles'],row['scatter_done_cycle'],1),
                ('MAC',row['physical_mac_start_cycle'],row['physical_mac_done_cycle'],0)):
                ax.broken_barh([(lo-origin,hi-lo)],(y-.24,.48),facecolors=colors[label])
            if 'weight_read_done_cycle' in row:
                ax.plot(row['weight_read_done_cycle']-origin,3,'o',color='#942f4a',markersize=4)
                ax.annotate(str(row['tile']), (row['weight_read_done_cycle']-origin,3),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
        ax.set_yticks([0,1,2,3],['MAC','Input scatter','Matrix program','Weight read ready'])
        ax.set_title(f'Worker {core}, first head, tiles 0–15 (actual service intervals)',loc='left',fontsize=11)
        ax.grid(axis='x',alpha=.2)
        ax.set_xlabel(f'Cycles relative to worker {core} first matrix submit (cycle {int(origin)})')
        ax.set_ylim(-.5,3.7)
    fig.suptitle('Projection: operand supply and serial array preparation',fontsize=14)
    fig.savefig(output)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.case,args.output)
