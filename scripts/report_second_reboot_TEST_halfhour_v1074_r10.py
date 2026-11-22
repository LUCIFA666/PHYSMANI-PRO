"""Inactive metadata-only seven-tier TEST observer after the recorded host reboot.

An explicit frozen small source lock activates observation only. No case, raw,
model, source manifest or scientific graph is opened. Prepared chat is never sent
or acknowledged. Every update retains the original :08/:38 UTC phase.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import types

ROOT=Path(__file__).resolve().parents[1]
SOURCE='scripts/report_second_reboot_TEST_halfhour_v1074_r10.py'
CORE='our_mani/multitask/composite_TEST_reboot_resume_v1070.py'
OWNER_SOURCE='scripts/run_second_reboot_TEST_owner_v1071_r10.py'
AUDITOR_SOURCE='scripts/audit_second_reboot_TEST_results_v1072_r10.py'
CONTROLLER_SOURCE='scripts/continue_second_reboot_TEST_v1073_r10.py'
HELPER_SOURCE='scripts/report_composite_TEST_halfhour_v1044_r10.py'
LEGACY_IDENTITY_SOURCE='scripts/report_reboot_TEST_halfhour_v1060_r10.py'
AUDIT_SOURCE='scripts/audit_second_reboot_TEST_evidence_v1067_r10.py'
CHECK_SOURCE='scripts/check_frozen_TEST_environment_v954_r10.py'
SOURCES=(SOURCE,CORE,OWNER_SOURCE,AUDITOR_SOURCE,CONTROLLER_SOURCE,HELPER_SOURCE,LEGACY_IDENTITY_SOURCE,AUDIT_SOURCE,CHECK_SOURCE)
STATE='runtime/framework/second_reboot_TEST_report_v1074'
OUT='TEST_结果更新_第二次重启续跑.txt'
RUNTIME='runtime/framework/formal_TEST_reboot_resume_v1071'
CONTROLLER='runtime/framework/ordered_second_reboot_TEST_v1073'
OLD='runtime/framework/formal_TEST_reboot_resume_v1056'
FIFTH='runtime/framework/formal_TEST_composite_resume_v1041'
INTERRUPTED=(FIFTH,OLD)
LEGACY='runtime/framework/reboot_TEST_report_v1060'
SETUP='artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071_setup'
CONTROL=SETUP+'/resume_control.json';CONTROL_LOCK=SETUP+'/resume_inputs.json'
RESULTS='artifacts/framework/multitask_r10/formal_TEST_reboot_resume_v1071/TEST_results.json'
REBOOT='runtime/framework/second_reboot_TEST_evidence_v1067.json'
BOOT='runtime/framework/reboot_hardware_boot_observation_v1068.json'
RECHECK='runtime/framework/frozen_TEST_environment_recheck_after_second_reboot_v1070.json'
SCHEMA='r10_readonly_second_reboot_TEST_report_source_v1074'
LOCK_SCHEMA='r10_readonly_second_reboot_TEST_report_inputs_v1074'
RESULT_SCHEMA='r10_seven_tier_second_reboot_formal_TEST_results_v1072'
METHODS=('cv_geometry','context_geometry','physmani')
BINDINGS=dict(resume_control=CONTROL,resume_control_lock=CONTROL_LOCK,reboot_audit=REBOOT,
    host_reboot_observation=BOOT,environment_recheck=RECHECK,previous_plan=OLD+'/plan.json',
    previous_status=OLD+'/status.json',legacy_reporter_launch=LEGACY+'/launch.json',
    owner_launch=RUNTIME+'/launch.json',
    owner_plan=RUNTIME+'/plan.json',controller_launch=CONTROLLER+'/launch.json')
PHASE_US=501384754;PERIOD_US=1800000000
STOP_PATHS=('runtime/framework/multitask_r10_STOP',STATE+'/STOP')
TIERS=('runtime/framework/formal_TEST_after_environment_resume_v753',
    'runtime/framework/formal_TEST_concurrent_resume_v902','runtime/framework/formal_TEST_composite_resume_v956',
    'runtime/framework/formal_TEST_composite_resume_v1011',FIFTH,OLD,RUNTIME)


def require(value,message):
    if not value:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
def valid_sha(value):return isinstance(value,str) and len(value)==64 and not set(value)-set('0123456789abcdef')
def valid_pointer(value,name=None):
    return (isinstance(value,dict) and set(value)=={'path','sha256'} and isinstance(value['path'],str)
        and bool(value['path']) and not Path(value['path']).is_absolute() and '..' not in Path(value['path']).parts
        and str(Path(value['path']))==value['path'] and valid_sha(value['sha256']) and (name is None or value['path']==name))


def phase_deadline(now):
    current=int(round(now*1000000))
    return (PHASE_US+((current-PHASE_US)//PERIOD_US+1)*PERIOD_US)/1000000


def is_phase(value):return (int(round(value*1000000))-PHASE_US)%PERIOD_US==0


def allowed_names():
    names={*SOURCES,*BINDINGS.values(),STATE+'/source.json',STATE+'/source_inputs.json',RESULTS}
    for runtime in (RUNTIME,CONTROLLER):names.update(runtime+'/'+leaf for leaf in ('launch.json','plan.json','status.json','terminal.json','failure.json'))
    for stage in ('formal_TEST_1920','independent_TEST_results'):
        names.update(CONTROLLER+'/stages/'+stage+'/'+leaf for leaf in ('launch.json','terminal.json','stdout.log'))
    return names


class Metadata:
    def __init__(self,*,root=ROOT,read_bytes=None,stop=lambda:False):
        self.root=Path(root);self.callback=read_bytes;self.stop=stop;self.raw={};self.checksums={}
    def bytes(self,name,expected=None,*,missing=False):
        if self.stop():raise InterruptedError('observer STOP')
        require(isinstance(name,str) and name in allowed_names(),'metadata outside fixed allowlist: '+str(name))
        if name not in self.raw:
            try:
                path=self.root/name
                if self.callback is None:
                    require(path.resolve(strict=True)==path.absolute(),'metadata alias refused');raw=path.read_bytes()
                else:raw=self.callback(name)
            except FileNotFoundError:
                if missing:return None
                raise
            require(type(raw) is bytes,'exact metadata bytes required');self.raw[name]=raw;self.checksums[name]=sha(raw)
        require(expected is None or self.checksums[name]==expected,'frozen metadata changed: '+name)
        return self.raw[name]
    def json(self,name,expected=None,*,missing=False):
        raw=self.bytes(name,expected,missing=missing)
        if raw is None:return None
        value=json.loads(raw);require(isinstance(value,dict),'metadata JSON object required');return value
    def pointer(self,p,name=None):
        require(valid_pointer(p,name),'exact allowed metadata pointer required');return self.json(p['path'],p['sha256'])
    def point(self,name):
        require(name in self.checksums,'pointer needs observed bytes');return dict(path=name,sha256=self.checksums[name])


def source_bundle(config,lock,reader,*,config_sha,lock_sha):
    require(config.get('schema')==SCHEMA and config.get('runtime')==RUNTIME and config.get('controller')==CONTROLLER
        and config.get('test_results')==RESULTS,'fixed reboot reporter scope required')
    require(lock.get('schema')==LOCK_SCHEMA and lock.get('approved_for_observation') is True,'small observation lock required')
    require('legacy_reporter_checkpoint' not in config, '1060 identity must come directly from its own launch, never the1044 checkpoint')
    sources=config.get('source_bindings');require(isinstance(sources,dict) and set(sources)==set(SOURCES),'exact observer sources required')
    pointers=[dict(path=STATE+'/source.json',sha256=config_sha)]
    for key,name in BINDINGS.items():
        require(valid_pointer(config.get(key),name),'fixed pointer required: '+key);pointers.append(config[key])
    for name,p in sources.items():require(valid_pointer(p,name),'exact source pointer required');pointers.append(p)
    expected={p['path']:p['sha256'] for p in pointers}
    require(lock.get('files')==expected,'observer lock must contain only the exact small metadata map')
    for p in pointers:reader.bytes(p['path'],p['sha256'])
    return dict(config=config,config_sha256=config_sha,lock_sha256=lock_sha,frozen=expected)


def load_source(expected,*,root=ROOT,read_bytes=None,stop=lambda:False):
    require(valid_sha(expected),'external small source-lock SHA required')
    r=Metadata(root=root,read_bytes=read_bytes,stop=stop)
    raw=r.bytes(STATE+'/source_inputs.json',expected);cfg=r.bytes(STATE+'/source.json')
    return source_bundle(json.loads(cfg),json.loads(raw),r,config_sha=sha(cfg),lock_sha=sha(raw))


def load_module(reader,name):
    require(name in (CORE,HELPER_SOURCE),'only the verified pure observer/core helpers may be loaded')
    value=types.ModuleType('verified_report1074_'+Path(name).stem);value.__file__=str(reader.root/name)
    exec(compile(reader.bytes(name),value.__file__,'exec'),value.__dict__);return value


def process_alive(identity):
    pid,start=identity['pid'],identity['start_ticks']
    require(type(pid) is int and pid>0 and type(start) is int and start>0,'real historical identity required')
    try:raw=Path('/proc',str(pid),'stat').read_text()
    except FileNotFoundError:return False
    return int(raw.rsplit(')',1)[1].split()[19])==start


def legacy_reboot(config,reader,audit,boot,*,alive=process_alive):
    old=reader.pointer(config['legacy_reporter_launch'],LEGACY+'/launch.json')
    identity=dict(pid=old.get('pid'),start_ticks=old.get('start_ticks'))
    require(type(identity['pid']) is int and identity['pid']>0
        and type(identity['start_ticks']) is int and identity['start_ticks']>0
        and old.get('source_sha256')==config['source_bindings'][LEGACY_IDENTITY_SOURCE]['sha256']
        and old.get('observer_only') is True and old.get('Native_started') is False,
        'genuine interrupted1060 reporter PID/start from its actual launch required')
    require(boot.get('schema')=='reboot_hardware_boot_point_observation_v1068'
        and boot.get('boot_id')==audit.get('current_boot_id')
        and boot.get('old_owner_created_before_current_boot') is True
        and boot.get('old_start_ticks_impossible_in_current_boot') is True
        and datetime.fromisoformat(old['created_utc']).timestamp()<boot['btime_unix']
        and type(identity['start_ticks']) is int and identity['start_ticks']>boot['uptime_seconds']*boot['clock_ticks_per_second']
        and alive(identity) is False,'old1060 must be genuinely lost across this boot, without an invented exit receipt')
    return dict(runtime=LEGACY,identity=identity,interrupted_by_host_reboot=True,
        private_STOP_invented=False,successful_exit_receipt_invented=False)


def common(value,control,config):
    require(value.get('split')=='test' and value.get('policy_sha256')==control['policy_sha256']
        and value.get('author_outcome_protocol')==control['author_outcome_protocol'] and value.get('automatic_retry') is False
        and value.get('resume_control')==config['resume_control'] and value.get('resume_control_lock')==config['resume_control_lock'],
        'real owner metadata must retain frozen control/policy/protocol')


def collect(bundle,*,root=ROOT,read_bytes=None,stop=lambda:False,alive=process_alive,boot_id=None):
    r=Metadata(root=root,read_bytes=read_bytes,stop=stop);cfg=bundle['config']
    r.bytes(STATE+'/source_inputs.json',bundle['lock_sha256'])
    for name,h in bundle['frozen'].items():r.bytes(name,h)
    core,old=load_module(r,CORE),load_module(r,HELPER_SOURCE)
    control=r.pointer(cfg['resume_control'],CONTROL)
    # Only the complete lock byte SHA is observed. Its transitive science map is never traversed.
    r.bytes(CONTROL_LOCK,cfg['resume_control_lock']['sha256'])
    require(control.get('schema')==core.CONTROL_SCHEMA and control.get('resume_owner')==cfg['source_bindings'][OWNER_SOURCE]
        and control.get('resume_auditor')==control.get('result_auditor')==cfg['source_bindings'][AUDITOR_SOURCE]
        and control.get('resume_controller')==cfg['source_bindings'][CONTROLLER_SOURCE],'frozen1070 selected source binding differs')
    for key in ('reboot_audit','host_reboot_observation','environment_recheck','previous_plan','previous_status'):
        require(control.get(key)==cfg[key],'actual recovery control selection differs: '+key)
    audit=r.pointer(cfg['reboot_audit'],REBOOT);boot=r.pointer(cfg['host_reboot_observation'],BOOT)
    require(audit.get('source')==cfg['source_bindings'][AUDIT_SOURCE],'actual1067 evidence source differs')
    actual_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip() if boot_id is None else boot_id
    require(actual_boot==audit.get('current_boot_id')==boot.get('boot_id'),'reporter observation belongs to another boot')
    legacy=legacy_reboot(cfg,r,audit,boot,alive=alive)
    observation=r.pointer(cfg['environment_recheck'],RECHECK);core.require_environment_recheck(observation,control)
    require(observation.get('source')==cfg['source_bindings'][CHECK_SOURCE],'recorded fresh954 source differs')
    parent_plan=r.pointer(cfg['previous_plan'],OLD+'/plan.json');parent_status=r.pointer(cfg['previous_status'],OLD+'/status.json')
    plan=r.pointer(cfg['owner_plan'],RUNTIME+'/plan.json');core.require_resume_plan(control,parent_plan,parent_status,audit,plan)
    jobs=old.grid(plan);owner=r.pointer(cfg['owner_launch'],RUNTIME+'/launch.json')
    ctl=r.pointer(cfg['controller_launch'],CONTROLLER+'/launch.json');status=r.json(RUNTIME+'/status.json');ctl_status=r.json(CONTROLLER+'/status.json')
    profile=old.recorded_profiles(control)
    for value in (owner,plan,status):
        common(value,control,cfg)
        require(value.get('resource_profile')==profile and value.get('reboot_audit')==cfg['reboot_audit']
            and value.get('boot_id')==actual_boot and value.get('infrastructure_recovery') is True,'same real seven-tier owner scope required')
    require(owner.get('source_sha256')==cfg['source_bindings'][OWNER_SOURCE]['sha256']
        and ctl.get('source_sha256')==cfg['source_bindings'][CONTROLLER_SOURCE]['sha256']
        and ctl.get('resume_control')==cfg['resume_control'] and ctl.get('resume_control_lock')==cfg['resume_control_lock']
        and ctl.get('automatic_retry') is False,'actual1071/1073 launches required')
    old.recorded_environment(ctl.get('frozen_environment'),control,pid=ctl.get('pid'),start=ctl.get('start_ticks'))
    old.recorded_environment(owner.get('frozen_environment'),control,pid=owner.get('pid'),start=owner.get('start_ticks'),expected=ctl['frozen_environment'])
    stage=r.json(CONTROLLER+'/stages/formal_TEST_1920/launch.json')
    require(stage.get('pid')==owner['pid'] and stage.get('start_ticks')==owner['start_ticks']
        and stage.get('source_sha256')==cfg['source_bindings'][OWNER_SOURCE]['sha256']
        and old.stage_inner(stage,OWNER_SOURCE,r.root)==owner.get('argv'),'real controller-owned1071 stage required')
    imported=set(control['completed_job_ids']);adopted=dict(completed={k:parent_status['completed'][k] for k in imported})
    data=old.count_status(status,jobs,adopted,imported,profile)
    data.update(admitted=True,release_complete=False,errors=[],reader=r,config=cfg,control=control,profile=profile,
        controller=ctl_status,legacy_interruption=legacy,jobs=jobs,owner=owner,controller_launch=ctl,helpers=old,
        resource_capacity_envelope=control['resource_capacity_envelope'],resource_capacity_source=control['resource_capacity_source'])
    if data['failed']:
        data['errors'].append('本轮基础设施失败 '+str(data['failed'])+' 条，未计作NativeFalse')
    if status.get('owner_error') is not None:
        data['errors'].append('owner_error: '+str(status['owner_error']))
    for flag in ('drain_requested','stop_requested'):
        if status.get(flag) is True:data['errors'].append('owner '+flag+'=True')
    stage_name=ctl_status.get('stage')
    if stage_name in ('requires_concrete_repair','user_stopped') or (isinstance(stage_name,str) and stage_name.startswith('draining')):
        data['errors'].append('controller stage: '+stage_name)
    if ctl_status.get('error') is not None:
        data['errors'].append('controller error: '+str(ctl_status['error']))
    data['release_complete']=final_gate(data,old,r)
    return data


def final_gate(data,old,r):
    cfg,control,status,jobs=(data[k] for k in ('config','control','status','jobs'))
    if data['controller'].get('stage')!='complete':return False
    require(len(status['completed'])==1920 and not data['failed'] and not sum(data['unknown_native'].values())
        and status.get('active')==status.get('checking')==status.get('ready_jobs')==[]
        and status.get('owner_error') is None,'a partial or unknown status is never a final result')
    terminal=r.json(RUNTIME+'/terminal.json');common(terminal,control,cfg)
    require(terminal.get('complete') is True and terminal.get('expected')==terminal.get('completed')==1920
        and terminal.get('failed')==0 and terminal.get('owner_error') is None
        and all(terminal.get(k) is False for k in ('stop_requested','drain_requested','drained','automatic_retry'))
        and terminal.get('CPU_not_launched')==0 and terminal.get('CPU_ready_unexecuted')==[]
        and terminal.get('imported_completed_cases')==len(control['completed_job_ids'])
        and terminal.get('resumed_new_cases')==1920-len(control['completed_job_ids']),'clean full1071 terminal required')
    end=r.json(CONTROLLER+'/terminal.json')
    require(end.get('complete') is True and end.get('automatic_retry') is False and valid_pointer(end.get('TEST_results'),RESULTS),
        'actual1073 final independent-result pointer required')
    stages={}
    for stage,source in (('formal_TEST_1920',OWNER_SOURCE),('independent_TEST_results',AUDITOR_SOURCE)):
        leaf=CONTROLLER+'/stages/'+stage;launch=r.json(leaf+'/launch.json');ended=r.json(leaf+'/terminal.json')
        require(launch.get('source_sha256')==cfg['source_bindings'][source]['sha256'] and launch.get('automatic_retry') is False
            and ended.get('complete') is True and type(ended.get('returncode')) is int and ended['returncode']==0
            and ended.get('automatic_retry') is False and launch.get('inherited_frozen_environment')==data['controller_launch']['frozen_environment'],
            'actual clean1073 owner/auditor stages required')
        old.stage_inner(launch,source,r.root);stages[stage]=launch
    stdout=r.json(CONTROLLER+'/stages/independent_TEST_results/stdout.log')
    require(stdout.get('complete') is True and stdout.get('schema')==RESULT_SCHEMA and stdout.get('cases')==1920
        and stdout.get('split')=='test' and stdout.get('out')==RESULTS,'actual full1072 stdout required')
    result=r.pointer(end['TEST_results'],RESULTS)
    require(result.get('schema')==RESULT_SCHEMA and result.get('complete') is True and result.get('split')=='test'
        and result.get('source_cases')==640 and result.get('method_cases')==result.get('completed_method_cases')==1920
        and result.get('methods')==list(METHODS) and set(result.get('tasks',[]))==set(old.TASK_LABELS)
        and result.get('source_sha256')==cfg['source_bindings'][AUDITOR_SOURCE]['sha256']
        and result.get('resume_control')==cfg['resume_control'] and result.get('resume_control_lock')==cfg['resume_control_lock']
        and result.get('policy_sha256')==control['policy_sha256'] and result.get('author_outcome_protocol')==control['author_outcome_protocol']
        and result.get('source_input_lock')==control['source_inputs'] and result.get('validation_release')==control['validation_release']
        and result.get('reboot_audit')==cfg['reboot_audit'] and result.get('host_reboot_observation')==cfg['host_reboot_observation']
        and all(result.get(k) is True for k in ('independent_composite_audit','used_evidence_rechecked_at_close','native_failures_retained',
            'motion_unknown_retained','zero_tick_cases_retained','motion_is_diagnostic_only','adopted_completed_results_not_rerun'))
        and all(result.get(k) is False for k in ('automatic_retry','source_outcome_filtering','Value_NGFF_FASTER_used')),
        'only the actual full1920 independent1072 result can release TEST')
    invocation=result.get('auditor_invocation',{});launched=stages['independent_TEST_results']
    require(invocation.get('pid')==launched.get('pid') and invocation.get('start_ticks')==launched.get('start_ticks')
        and invocation.get('source')==cfg['source_bindings'][AUDITOR_SOURCE]
        and invocation.get('argv')==old.stage_inner(launched,AUDITOR_SOURCE,r.root),'actual independent auditor invocation differs')
    totals={m:sum(data['success'][t,m] for t in old.TASK_LABELS) for m in METHODS}
    require(result.get('native_successes_by_method')==stdout.get('native_successes_by_method')==totals,'actual result totals differ')
    pointers,drivers,tiers=(result.get(k) for k in ('cases','admission_driver_receipts','evidence_tiers'))
    require(all(isinstance(v,list) and len(v)==1920 for v in (pointers,drivers,tiers)),'full1920 pointer grid required')
    for index,row in enumerate(jobs.values()):
        evidence=tiers[index];record=status['completed'][row['job_id']]
        require(evidence.get('job_id')==row['job_id'] and evidence.get('evidence_runtime')==row['evidence_runtime']
            and all(evidence.get(k)==row.get(k) for k in ('resume_lineage','resume_lineage_v1010','resume_lineage_v1040','resume_lineage_v1055','resume_lineage_v1070'))
            and evidence.get('case')==pointers[index]==record.get('case')
            and evidence.get('admission_driver_receipt')==drivers[index]==record.get('admission_driver_receipt'),
            'independent result must match the complete same status/plan pointer grid')
    owners=result.get('actual_stage_owners');counts=Counter(row['evidence_runtime'] for row in jobs.values())
    require(isinstance(owners,list) and len(owners)==7 and {x.get('runtime') for x in owners}==set(TIERS)
        and all(x.get('completed')==counts[x['runtime']] for x in owners)
        and all(x.get('terminal') is None for x in owners if x['runtime'] in INTERRUPTED), 'seven real physical tiers and both interrupted owners remain explicit')
    for value in owners:
        runtime=value['runtime']
        require(valid_pointer(value.get('launch'),runtime+'/launch.json')
            and type(value.get('pid')) is int and value['pid']>0
            and type(value.get('start_ticks')) is int and value['start_ticks']>0
            and (value.get('terminal') is None if runtime in INTERRUPTED else valid_pointer(value.get('terminal'),runtime+'/terminal.json')),
            'actual owner identity and genuine terminal pointer required')
        if runtime==RUNTIME:
            require(value['launch']==r.point(RUNTIME+'/launch.json') and value['terminal']==r.point(RUNTIME+'/terminal.json')
                and value['pid']==data['owner']['pid'] and value['start_ticks']==data['owner']['start_ticks'],
                'final current owner identity differs from actual1071 metadata')
    for key in ('resource_profile','resource_qualification','resource_qualifier','resource_capacity_envelope',
            'resource_profile_within_capacity','resource_capacity_source','resource_profile_changed','previous_resource_profile',
            'previous_resource_qualification','execution_implementation','environment_recheck'):
        require(result.get(key)==control.get(key),'final independent resource/control binding differs: '+key)
    return True


def empty_data(error):
    return dict(admitted=False,release_complete=False,errors=[str(error)],completed=None,success=None,
        unknown_native=None,unknown_motion=None,motion_false=None,failed=None,imported_completed=None,new_completed=None,
        status={},controller={},legacy_interruption=None)


def atomic(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    # Make both file publication and any newly-created ancestor entries durable.
    current=path.parent
    while True:
        descriptor=os.open(current,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if current==current.parent:break
        current=current.parent


def save_report(data,bundle,history,*,now,next_due,reason,root=ROOT):
    require(is_phase(next_due) and next_due>now,'original strictly-future half-hour phase required')
    old=data.get('helpers');methods={};total=None
    if data['admitted']:
        methods={m:{field:sum(data[key][t,m] for t in old.TASK_LABELS) for field,key in
            (('completed','completed'),('successes','success'),('unknown_native','unknown_native'),('unknown_motion','unknown_motion'),('motion_false','motion_false'))} for m in METHODS}
        total=sum(v['completed'] for v in methods.values())
    else:methods={m:dict(completed=None,successes=None,unknown_native=None,unknown_motion=None,motion_false=None) for m in METHODS}
    item=dict(schema='r10_second_reboot_TEST_halfhour_snapshot_v1074',created_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),
        next_update_utc=datetime.fromtimestamp(next_due,timezone.utc).isoformat(),admitted=data['admitted'],release_complete=data['release_complete'],
        total_completed=total,expected=1920,methods=methods,errors=data['errors'],reason=reason,
        source_config_sha256=bundle['config_sha256'],source_lock_sha256=bundle['lock_sha256'],
        imported_completed=data.get('imported_completed'),new_completed=data.get('new_completed'),
        current1071_infrastructure_failed=data.get('failed'),resource_profile=data.get('profile'),
        resource_capacity_envelope=data.get('resource_capacity_envelope'),resource_capacity_source=data.get('resource_capacity_source'),
        controller_stage=data.get('controller',{}).get('stage'),owner_error=data['status'].get('owner_error'),
        owner_drain_requested=data['status'].get('drain_requested'),owner_stop_requested=data['status'].get('stop_requested'),
        evidence_tiers=dict(data['evidence_tiers']) if data.get('evidence_tiers') is not None else None,
        legacy_reporter_interruption=data.get('legacy_interruption'),Native_started=False,GPU_started=False,
        chat_sent=False,delivery_acknowledged=False,owner_updated_utc=data['status'].get('updated_utc'))
    number=lambda value:'未知' if value is None else str(value)
    phase='完整1920已由1072独立审计及1073终态确认' if item['release_complete'] else '运行状态统计，尚未通过完整独立审计'
    summary=item['created_utc']+'｜TEST '+number(total)+'/1920｜'+'；'.join(m+' '+number(v['successes'])+'/'+number(v['completed'])+'（未知'+number(v['unknown_native'])+'）' for m,v in methods.items())
    lines=['TEST结果更新（第二次重启续跑）',phase,summary,
        '控制器阶段：'+number(item['controller_stage'])+'；本轮infra：'+number(item['current1071_infrastructure_failed'])+
        '；owner_error：'+('未知／未观测' if not data['admitted'] else str(item['owner_error']))+
        '；drain：'+number(item['owner_drain_requested'])+'；stop：'+number(item['owner_stop_requested']),
        '下次固定更新：'+item['next_update_utc'],
        '1060因第二次真实主机重启中断；未伪造STOP或退出回执，旧文件全部保留。',
        '仅观察计划与状态元数据；NativeFalse保留，未知为None；未读取case/raw/model或遍历科学图。',
        '历史1031容量资格沿用，未在本次开机重跑；GPU独占及机器极限均未声明。',
        '对话文本仅准备；root真实发送后另行ack，报告器不会发送或自行ack。']
    if data['errors']:lines+=['观察异常：'+'；'.join(data['errors'])]
    if data['admitted']:
        lines+=['','任务｜CV+几何 成功/完成/未知｜Context+几何 成功/完成/未知｜PhysMani 成功/完成/未知']
        for task,label in old.TASK_LABELS.items():
            lines.append(label+'｜'+'｜'.join(f"{data['success'][task,m]}/{data['completed'][task,m]}/{data['unknown_native'][task,m]}" for m in METHODS))
    history=[*history,item];r=data.get('reader')
    if r:
        item['observed_metadata_sha256']=dict(r.checksums)
        stamp=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');item['input_snapshots']={}
        for original,leaf in ((RUNTIME+'/status.json','owner_status.json'),(CONTROLLER+'/status.json','controller_status.json')):
            if original in r.raw:
                name=STATE+'/snapshots/'+stamp+'/'+leaf;require(not (Path(root)/name).exists(),'preserve earlier metadata snapshot')
                atomic(Path(root)/name,r.raw[original]);item['input_snapshots'][original]=dict(path=name,sha256=r.checksums[original])
    for name,payload in ((OUT,('\n'.join(lines)+'\n').encode()),(STATE+'/status.json',encoded(item)),
            (STATE+'/history.jsonl',b''.join(encoded(x) for x in history)),(STATE+'/chat_ready.json',encoded(dict(
                schema='r10_chat_ready_only_v1074',snapshot_id=item['created_utc'],created_utc=item['created_utc'],
                next_update_utc=item['next_update_utc'],text=summary+'\n'+phase,chat_sent=False,delivery_acknowledged=False)))):
        atomic(Path(root)/name,payload)
    return history


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--source-lock-sha256')
    args=parser.parse_args(argv)
    if not args.execute:print(json.dumps(dict(active=False,Native_started=False,GPU_started=False)));return 0
    require(valid_sha(args.source_lock_sha256),'external small observation lock SHA required')
    event=threading.Event()
    for number in (signal.SIGINT,signal.SIGTERM):signal.signal(number,lambda *_:event.set())
    stop=lambda:event.is_set() or any((ROOT/name).exists() for name in STOP_PATHS)
    if stop():return 0
    bundle=load_source(args.source_lock_sha256,stop=stop)
    import fcntl
    state=ROOT/STATE;state.mkdir(parents=True,exist_ok=True)
    with (state/'observer.lock').open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any((state/name).exists() for name in ('launch.json','status.json','history.jsonl','terminal.json','chat_ready.json','snapshots'))
            and not (ROOT/OUT).exists(),'preserve an earlier1074 observer attempt')
        atomic(state/'launch.json',encoded(dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
            created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=bundle['config']['source_bindings'][SOURCE]['sha256'],
            source_config_sha256=bundle['config_sha256'],source_lock_sha256=bundle['lock_sha256'],observer_only=True,Native_started=False,GPU_started=False)))
        history=[];due=phase_deadline(time.time());first=True;signature=None;data=empty_data('not observed yet')
        while not stop():
            try:data=collect(bundle,stop=stop)
            except InterruptedError:break
            except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError) as error:data=empty_data(error)
            marker=encoded(dict(admitted=data['admitted'],errors=data['errors'],release=data['release_complete']))
            now=time.time()
            if first or marker!=signature or now>=due or data['release_complete']:
                due=phase_deadline(now);history=save_report(data,bundle,history,now=now,next_due=due,
                    reason='完整独立终态' if data['release_complete'] else '启动快照' if first else '固定30分钟或观察变化')
                first=False
            signature=marker
            if data['release_complete']:break
            event.wait(min(10,max(.1,due-time.time())))
        atomic(state/'terminal.json',encoded(dict(complete=True,observer_only=True,final_TEST_independently_released=data['release_complete'],
            user_or_signal_stop=stop(),Native_started=False,GPU_started=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
