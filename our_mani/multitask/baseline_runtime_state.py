"""One explicitly reviewed CoppeliaSim crash-bookkeeping state exception.

Every other runtime file remains immutable.  This observer never writes the
runtime, changes a simulator setting, steps physics, or reads task ground truth.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys

from .evaluation import file_sha

SCHEMA = 'r10_exact_Coppelia_crash_marker_exclusion_v1'
WITNESS = 'r10_exact_Coppelia_crash_marker_case_witness_v1'
MARKER = b'SIMSETTINGS_SIM_CRASHED'
STATE_FILE = Path('/home/ubuntu/projects/physmani_reproduction/runtime/'
    'CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/system/settings.dat').resolve()
LIMITATIONS = [
    'Concurrent workers share this bookkeeping file; no writer PID is inferred.',
    'Before/after observations do not prove that no intermediate state occurred.',
    'Only the declared single crash marker is mutable; all other runtime files remain immutable.']
BASE_CONDA=Path('/home/ubuntu/physmani-conda')
SHELL_PROGRAMS=('bash','python','Xvfb','xvfb-run','xauth','awk','stty','fmt','cat',
                'getopt','mktemp','touch','mcookie','rm','dirname','mkdir')
LAUNCH_ENV_KEYS=('PATH','LD_LIBRARY_PATH','PYTHONPATH','COPPELIASIM_ROOT','CONDA_PREFIX',
    'CUDA_HOME','CC','CXX','QT_QPA_PLATFORM_PLUGIN_PATH','PHYSMANI_PROJECT_ROOT','PHYS_MANI_REPO_ROOT',
    'TORCH_CUDA_ARCH_LIST','MAX_JOBS','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTHONUTF8','XDG_CACHE_HOME','TORCH_HOME','TMPDIR')
JIT_ENV_KEYS=('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')


def require(value, message):
    if not value:raise ValueError('baseline exact runtime state: '+message)


def parse_crash_marker(payload):
    require(isinstance(payload,bytes) and len(payload) in (39,40),'unexpected payload size/type')
    require(struct.unpack_from('<i',payload,0)[0]==1 and payload[4]==1,'exact single entry/tag required')
    require(struct.unpack_from('<i',payload,5)[0]==len(MARKER)
        and payload[9:9+len(MARKER)]==MARKER,'unknown persistent-data key')
    offset=9+len(MARKER);length=struct.unpack_from('<i',payload,offset)[0];offset+=4
    value=payload[offset:offset+length]
    require(length in (2,3) and value in (b'No',b'Yes') and len(value)==length
        and payload[offset+length:]==b'\x00','unknown value, extra setting, or malformed terminator')
    return dict(entry_count=1,key=MARKER.decode(),value=value.decode(),size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),hex=payload.hex())


def snapshot(path):
    path=Path(path).resolve();require(path==STATE_FILE,'undeclared mutable file path')
    stat=lambda s:dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
    before=path.stat();payload=path.read_bytes();after=path.stat()
    require(stat(before)==stat(after),'concurrent unstable bookkeeping read')
    return dict(resolved_path=str(path),before_stat=stat(before),after_stat=stat(after),
                parsed=parse_crash_marker(payload),read_only=True)


def diagnostic_snapshot():
    """Retain failed reads honestly; this result cannot pass validate_snapshot."""
    value=dict(resolved_path=str(STATE_FILE),complete=False,read_only=True)
    try:
        before=STATE_FILE.stat();payload=STATE_FILE.read_bytes();after=STATE_FILE.stat()
        stat=lambda s:dict(size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)
        value.update(before_stat=stat(before),after_stat=stat(after),hex=payload.hex(),
            sha256=hashlib.sha256(payload).hexdigest(),stable_read=stat(before)==stat(after))
        try:value['parsed']=parse_crash_marker(payload)
        except Exception as error:value['parse_error']=type(error).__name__+': '+str(error)
    except Exception as error:value['read_error']=type(error).__name__+': '+str(error)
    return value


def source_bindings(root, config):
    repo=Path(config['baseline_repo']).resolve()
    paths=[Path(__file__).resolve(),Path(root)/'our_mani/multitask/simulation.py',
        repo/'third_party/PyRep/pyrep/pyrep.py',repo/'third_party/PyRep/pyrep/backend/sim.py',
        repo/'third_party/PyRep/pyrep/backend/simConst.py',STATE_FILE.parent.parent/'libcoppeliaSim.so']
    return {str(path.resolve()):file_sha(path) for path in paths}


def draft_permission(root, config, evidence_report):
    snapshot(STATE_FILE)
    return dict(schema=SCHEMA,reviewed_before_execution=False,resolved_path=str(STATE_FILE),
        allowed_values=['No','Yes'],native_headless_required=True,
        directory_integrity_for_every_other_file=True,source_bindings_sha256=source_bindings(root,config),
        evidence_report=deepcopy(evidence_report),limitations=deepcopy(LIMITATIONS))


def load_permission(root, pointer, config, *, require_reviewed=True):
    root=Path(root).resolve()
    require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},'exact permission pointer required')
    path=(root/pointer['path']).resolve()
    require(path.is_relative_to(root) and file_sha(path)==pointer['sha256'],'permission bytes/path changed')
    value=json.loads(path.read_text())
    require(set(value)=={'schema','reviewed_before_execution','resolved_path','allowed_values',
        'native_headless_required','directory_integrity_for_every_other_file','source_bindings_sha256',
        'evidence_report','limitations'},'unknown/missing permission field')
    require(value['schema']==SCHEMA and value['reviewed_before_execution'] is require_reviewed
        and value['resolved_path']==str(STATE_FILE) and value['allowed_values']==['No','Yes']
        and value['native_headless_required'] is True
        and value['directory_integrity_for_every_other_file'] is True
        and value['limitations']==LIMITATIONS,'unreviewed or widened bookkeeping exception')
    require(value['source_bindings_sha256']==source_bindings(root,config),'observer/native API/source bytes changed')
    evidence=value['evidence_report']
    require(isinstance(evidence,dict) and set(evidence)=={'path','sha256'},'exact evidence pointer required')
    evidence_path=(root/evidence['path']).resolve()
    require(evidence_path.is_relative_to(root) and file_sha(evidence_path)==evidence['sha256'],
            'bookkeeping source/binary review evidence changed')
    snapshot(STATE_FILE)
    return value


def observe_headless(*, sim=None):
    native_root=os.environ.get('COPPELIASIM_ROOT')
    require(isinstance(native_root,str) and Path(native_root).resolve()==STATE_FILE.parent.parent,
            'actual native runtime root differs from the exact declared bookkeeping file')
    if sim is None:
        from pyrep.backend import sim
    require(sim.sim_boolparam_headless==34 and sim.sim_gui_headless==65536,'native headless API constants differ')
    before=sim.simGetSimulationTime()
    result=sim.simGetBoolParameter(sim.sim_boolparam_headless)
    after=sim.simGetSimulationTime()
    # Installed PyRep returns the C API integer directly (not Python bool).
    require(type(result) in (int,bool) and result==1,'actual native simulator is not headless')
    require(type(before) in (int,float) and math.isfinite(before) and before>=0
        and type(after) in (int,float) and math.isfinite(after) and before==after,
        'headless read unexpectedly advanced or invalidated physical time')
    return dict(public_API='simGetBoolParameter',parameter_name='sim_boolparam_headless',
        parameter_id=34,value=True,raw_value=result,raw_type=type(result).__name__,launch_headless_option=65536,
        native_runtime_root=str(Path(native_root).resolve()),
        physical_time_before=before,physical_time_after=after,physical_time_unchanged=True,
        no_physics_step_or_setting_mutation=True,task_ground_truth_read=False)


def validate_snapshot(value, permission):
    require(isinstance(value,dict) and set(value)=={'resolved_path','before_stat','after_stat','parsed','read_only'}
        and value['resolved_path']==permission['resolved_path'] and value['read_only'] is True,
        'raw bookkeeping observation schema/path differs')
    fields={'size','mtime_ns','inode','device'}
    require(set(value['before_stat'])==fields and value['before_stat']==value['after_stat']
        and all(type(item) is int and item>=0 for item in value['before_stat'].values()),
        'raw bookkeeping read was unstable or stat is malformed')
    parsed=value['parsed']
    try:payload=bytes.fromhex(parsed['hex'])
    except (KeyError,ValueError,TypeError):raise ValueError('baseline exact runtime state: invalid raw bytes')
    require(parsed==parse_crash_marker(payload) and value['before_stat']['size']==len(payload),
            'raw bookkeeping bytes/parse/SHA/stat differ')


def validate_case_witness(value, permission, pointer, initial):
    require(isinstance(value,dict) and set(value)=={'schema','permission','before','after','headless','limitations'}
        and value['schema']==WITNESS and value['permission']==pointer and value['limitations']==LIMITATIONS,
        'exact bookkeeping case witness missing or widened')
    require(initial=={key:deepcopy(value[key]) for key in ('permission','before','headless')},
        'case before/headless witness differs from original native reset evidence')
    validate_snapshot(value['before'],permission);validate_snapshot(value['after'],permission)
    headless=value['headless']
    require(set(headless)=={'public_API','parameter_name','parameter_id','value','raw_value','raw_type','launch_headless_option','native_runtime_root',
        'physical_time_before','physical_time_after','physical_time_unchanged',
        'no_physics_step_or_setting_mutation','task_ground_truth_read'}
        and headless['public_API']=='simGetBoolParameter' and headless['parameter_name']=='sim_boolparam_headless'
        and headless['parameter_id']==34 and headless['launch_headless_option']==65536
        and headless['native_runtime_root']==str(Path(permission['resolved_path']).parent.parent)
        and type(headless['raw_value']) in (int,bool) and headless['raw_value']==1
        and headless['raw_type']==type(headless['raw_value']).__name__
        and headless['value'] is True and headless['physical_time_unchanged'] is True
        and type(headless['physical_time_before']) in (int,float) and math.isfinite(headless['physical_time_before'])
        and headless['physical_time_before']>=0 and type(headless['physical_time_after']) in (int,float)
        and headless['physical_time_before']==headless['physical_time_after']
        and headless['no_physics_step_or_setting_mutation'] is True and headless['task_ground_truth_read'] is False,
        'actual public native headless observation is absent, false, or changed physics')


def validate_launcher_environment(value, dependencies, root, config, interpreter, *, jit_profile=None):
    root=Path(root).resolve();prefix=Path(interpreter).resolve().parent.parent
    require(isinstance(value,dict) and set(value)=={'schema','source','environment','programs'}
        and value['schema']=='r10_actual_PhysMani_child_launcher_environment_v1'
        and value['source']=='actual_python_child_after_runtime_script','actual child launcher environment is missing')
    environment=value['environment'];programs=value['programs'];repo=Path(config['baseline_repo']).resolve()
    keys=LAUNCH_ENV_KEYS+(JIT_ENV_KEYS if jit_profile is not None else ())
    require(set(environment)==set(keys) and all(isinstance(v,str) for v in environment.values()),
            'launcher environment has missing or unknown fields')
    expected={'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
        'TORCH_CUDA_ARCH_LIST':'12.0','MAX_JOBS':'6','PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1'}
    require(all(environment[key]==item for key,item in expected.items()),'actual frozen hardware/runtime export differs')
    path_fields={'COPPELIASIM_ROOT':STATE_FILE.parent.parent,'CONDA_PREFIX':prefix,'CUDA_HOME':prefix,
        'QT_QPA_PLATFORM_PLUGIN_PATH':STATE_FILE.parent.parent,'PHYSMANI_PROJECT_ROOT':repo,'PHYS_MANI_REPO_ROOT':repo,
        'CC':prefix/'bin/x86_64-conda-linux-gnu-cc','CXX':prefix/'bin/x86_64-conda-linux-gnu-c++',
        'XDG_CACHE_HOME':root/'runtime/framework/cache','TORCH_HOME':root/'runtime/framework/torch','TMPDIR':root/'runtime/framework/tmp'}
    if jit_profile is not None:
        from .baseline_jit_artifacts import cache_paths
        expected_jit=cache_paths(root)
        require(jit_profile.get('cache_paths')==expected_jit,'launcher JIT profile widened ROOT paths')
        path_fields.update(TORCHINDUCTOR_CACHE_DIR=Path(expected_jit['inductor']),TRITON_CACHE_DIR=Path(expected_jit['triton']))
    require(all(Path(environment[key]).resolve()==item.resolve() for key,item in path_fields.items()),
            'actual worker runtime/project/compiler/cache path differs')
    split=lambda key:[Path(item).resolve() for item in environment[key].split(':') if item]
    resolved=lambda paths:[path.resolve() for path in paths]
    require(split('PYTHONPATH')==resolved([root,repo,repo/'third_party/RLBench',repo/'third_party/PyRep'])
        and split('PATH')[:2]==resolved([STATE_FILE.parent.parent.parent/'usr/bin',prefix/'bin'])
        and split('LD_LIBRARY_PATH')[:4]==resolved([STATE_FILE.parent.parent,
            STATE_FILE.parent.parent.parent/'usr/lib/x86_64-linux-gnu',prefix/'lib',prefix/'targets/x86_64-linux/lib']),
        'actual launcher PATH/PYTHONPATH/LD_LIBRARY_PATH differs from reviewed source chain')
    expected_programs=set(SHELL_PROGRAMS)|{'base_conda','base_python','xvfb_shell_interpreter'}
    require(set(programs)==expected_programs,'launcher external program coverage differs')
    for item in programs.values():
        require(isinstance(item,dict) and set(item)=={'path','sha256'} and Path(item['path']).is_absolute()
            and dependencies.get(str(Path(item['path']).resolve()))==item['sha256'],
            'actual launcher external program is outside the immutable runtime closure')
    require(programs['python']['path']==str(Path(interpreter).resolve())
        and programs['base_conda']['path']==str((BASE_CONDA/'bin/conda').resolve())
        and programs['base_python']['path']==str((BASE_CONDA/'bin/python').resolve())
        and programs['xvfb_shell_interpreter']['path']==str(Path('/bin/sh').resolve()),
        'actual launcher interpreter/base-conda identity differs')


def observe_launcher_environment(dependencies, root, config, interpreter, *, jit_profile=None):
    programs={}
    for name in SHELL_PROGRAMS:
        path=shutil.which(name);require(path is not None,'launcher external program is missing: '+name)
        path=Path(path).resolve();programs[name]=dict(path=str(path),sha256=file_sha(path))
    for name,path in {'base_conda':BASE_CONDA/'bin/conda','base_python':BASE_CONDA/'bin/python',
                      'xvfb_shell_interpreter':Path('/bin/sh')}.items():
        path=path.resolve();programs[name]=dict(path=str(path),sha256=file_sha(path))
    require(Path(sys.executable).resolve()==Path(interpreter).resolve(),'actual child Python differs from frozen native interpreter')
    value=dict(schema='r10_actual_PhysMani_child_launcher_environment_v1',source='actual_python_child_after_runtime_script',
        environment={key:os.environ.get(key) for key in LAUNCH_ENV_KEYS+(JIT_ENV_KEYS if jit_profile is not None else ())},programs=programs)
    validate_launcher_environment(value,dependencies,root,config,interpreter,jit_profile=jit_profile)
    return value
