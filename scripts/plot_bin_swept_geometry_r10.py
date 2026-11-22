"""Static scientific view of immutable TRAIN CAD sweep diagnostic results."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(report_path,output):
    report_path=Path(report_path).resolve();output=Path(output).resolve();report=json.loads(report_path.read_text())
    if (report.get('complete') is not True or report.get('TRAIN_only') is not True or output.exists()
            or not output.is_relative_to(ROOT/'artifacts/framework/multitask_r10')):
        raise ValueError('complete fixed TRAIN evidence and new project plot only')
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True,layout='constrained')
    for row,(task,record) in enumerate(report['tasks'].items()):
        asset_path=ROOT/record['source_asset'];slab_path=ROOT/record['slabs']
        if sha(asset_path)!=record['source_asset_sha256'] or sha(slab_path)!=record['slabs_sha256']:
            raise ValueError('immutable geometry/slab evidence differs')
        asset=json.loads(asset_path.read_text());slabs=json.loads(slab_path.read_text())
        vertices=np.asarray(asset['bodies']['bin']['collision_vertices_body']);radius=float(np.linalg.norm(vertices,axis=1).max())
        for axis in axes[row]:
            axis.axhline(0,color='#8a8a8a',lw=.6)
            axis.set_aspect('equal',adjustable='box');axis.grid(alpha=.16)
            axis.set_xlabel('Body-local X (mm); mouth to the right')
        axes[row,0].fill_between([-radius*1000,radius*1000],-radius*1000,radius*1000,
            color='#d6d6d6',label='Full-rotation uncertainty cube')
        for slab in slabs:
            xs=np.asarray(slab['stored_axial_interval'])*1000
            high=slab['outer_vertex_radius']*1000;low=slab['inner_vertex_radius']*1000
            axes[row,1].fill_between(xs,low,high,color='#487fba')
            axes[row,1].fill_between(xs,-high,-low,color='#487fba')
        axes[row,0].set_ylabel(('Ordinary' if row==0 else 'High speed')+'\nSigned radial section (mm)')
        axes[row,1].annotate('Closed bottom',xy=(-85,0),xytext=(-150,120),
            arrowprops={'arrowstyle':'->','color':'#333'},fontsize=9)
        axes[row,1].annotate('Retained mouth\n169.1 mm clear diameter',xy=(90,0),xytext=(-25,-150),
            arrowprops={'arrowstyle':'->','color':'#333'},fontsize=9)
    axes[0,0].set_title('Current default: cavity filled')
    axes[0,1].set_title('Uninstalled axis-yaw swept prototype')
    axes[0,0].set_xlim(-180,180);axes[0,0].set_ylim(-180,180)
    fig.suptitle('TRAIN CAD only | 32 slabs, 64 angular facets | directed local-X axis required',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)
    receipt=dict(report=str(report_path.relative_to(ROOT)),report_sha256=sha(report_path),
        plot=str(output.relative_to(ROOT)),plot_sha256=sha(output),source=str(Path(__file__).relative_to(ROOT)),
        source_sha256=sha(__file__),runtime_installed=False,GT_pose_used=False,
        note='Signed radial schematic of conservative prism sections, not a native render or exact CAD.')
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();print(json.dumps(plot(args.report,args.out)))
