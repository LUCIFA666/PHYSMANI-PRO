"""Score a frozen real-clock motion checkpoint on a compiled selection/heldout role."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pointer(p):return dict(path=str(Path(p).resolve(strict=True)),sha256=sha(p))


def evaluate_selected(dataset,selected,digest,status,status_sha,output):
    if sha(status)!=status_sha or sha(selected)!=digest:
        raise ValueError('External frozen selected weight and completed-fit SHA required')
    completed=read(status)
    if (completed.get('training_complete')is not True or completed.get('source_unchanged')is not True
            or completed.get('heldout_payloads_read')is not False
            or completed.get('selected_checkpoint')!=pointer(selected)):
        raise ValueError('Actual completed selected fit, without heldout access, required')
    from train_cube_motion_residual_20261104_v1 import load_role,to_device,evaluate,write
    from our_mani.cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
    import torch
    metadata=read(Path(dataset)/'report.json')
    if metadata.get('role')not in('selection','heldout'):
        raise ValueError('Explicit development selection or heldout role required')
    payload=torch.load(selected,map_location='cpu',weights_only=False)
    role_key='selection_parents'if metadata['role']=='selection'else'heldout_parents'
    if metadata.get('parents')!=payload[role_key]:
        raise ValueError('Exact whole-parent role required before dataset payload access')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    request=dict(dataset=pointer(Path(dataset)/'dataset.npz'),dataset_report=pointer(Path(dataset)/'report.json'),
        selected=pointer(selected),completed_fit=pointer(status),role=metadata['role'],script=pointer(__file__),
        Native_started=False,GPU_used=False,new_training=False)
    write(output/'request.json',request)
    started=time.perf_counter()
    torch.set_num_threads(2)
    runtime=CubeResidualMotionDynamics(selected,expected_sha256=digest)
    arrays,report=load_role(dataset,expected_role=metadata['role'],expected_parents=payload[role_key])
    runtime._assert_live()
    result=evaluate(runtime.model,to_device(arrays,'cpu'),report['parents'])
    runtime._assert_live()
    write(output/'report.json',dict(complete=True,role=report['role'],metrics=result,
        selected=request['selected'],dataset=request['dataset'],wall_seconds=time.perf_counter()-started,
        source_unchanged=True,Native_started=False,GPU_used=False,training_started=False,
        formal_TEST_package_read=False,heldout_scored=report['role']=='heldout'))
    write(output/'completion.json',dict(complete=True,role=report['role'],report=pointer(output/'report.json'),
        model_forward_batches=result['NN_forward_calls'],source_unchanged=True,Native_started=False,GPU_used=False))
    print(json.dumps(dict(report=pointer(output/'report.json'),horizons=result['horizons_seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--selected-sha256',required=True);p.add_argument('--fit-status',type=Path,required=True)
    p.add_argument('--fit-status-sha256',required=True);p.add_argument('--output',type=Path)
    p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.evaluate:
        if a.output is None:p.error('Fresh output required')
        evaluate_selected(a.dataset,a.selected,a.selected_sha256,a.fit_status,a.fit_status_sha256,a.output)
    else:
        print(json.dumps(dict(preparation_only=True,role=read(a.dataset/'report.json')['role'],
            model_constructed=False,Native_started=False,GPU_used=False)))


if __name__=='__main__':main()
