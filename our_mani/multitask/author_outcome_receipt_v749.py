"""Read-only validation of explicit author-action outcome receipts.

Task failure remains a valid measured outcome. Infrastructure, an aborted
action or uncovered physical ticks cannot impersonate a completed native run.
Internally handled planner/public-geometry holds retain their existing meaning;
this verifier never converts those diagnostics into native task failures.
"""
import json
from .joint16_tracks_v31 import BLOCKS

VERSION='r10_full16_explicit_author_action_outcome_v744'
CORE_SCHEMA='r10_explicit_author_action_outcome_sampling_v741'
BLOCK_SCHEMA='r10_original_no_failure_block_action_sampling_v744'


def _same(actual,expected):
    # JSON equality is type-sensitive for booleans versus integer flags.
    try:return json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError):return False


def _require(condition,message):
    if not condition:raise ValueError('author outcome receipt: '+message)


def _integer(value,*,minimum=0,maximum=None):
    return type(value) is int and value>=minimum and (maximum is None or value<=maximum)


def _outcome(value):
    _require(isinstance(value,dict) and set(value)=={'success','terminate','terminal_failure'},'explicit native result fields required')
    _require(all(type(v) is bool for v in value.values()),'native result fields must be booleans')
    triple=(value['success'],value['terminate'],value['terminal_failure'])
    _require(triple in ((False,False,False),(False,True,True),(True,True,False)),'inconsistent native result')
    return value


def require_sampling_case(case,expected_protocol,*,block):
    """Return the validated receipt; never read files or require task success."""
    _require(type(block) is bool,'explicit core/block route required')
    _require(isinstance(case,dict) and case.get('version')==VERSION,'actual versioned744 native case required')
    _require(case.get('complete') is True and case.get('cleanup_completed') is True
        and case.get('infrastructure_error') is None,'clean completed original native case required')
    _require((case.get('task') in BLOCKS)==block,'actual registered block/core route differs')
    _require(isinstance(expected_protocol,dict) and expected_protocol.get('schema')=='r10_author_action_outcome_protocol_v741'
        and _same(case.get('author_outcome_protocol'),expected_protocol),'exact frozen protocol binding required')
    method=case.get('method')
    _require(method in ('cv_geometry','context_geometry','physmani'),'exactly the three authorized methods')
    success,fatal=case.get('official_task_success'),case.get('native_terminal_failure')
    _require(type(success) is bool and type(fatal) is bool and not(success and fatal),'explicit consistent original native outcomes required')
    terminal=success or fatal;clock=case.get('counted_clock',{});ticks=clock.get('ticks')
    _require(_integer(ticks,maximum=200),'actual counted physical tick total required')
    receipt=case.get('native_outcome_sampling')
    _require(isinstance(receipt,dict) and receipt.get('schema')==(BLOCK_SCHEMA if block else CORE_SCHEMA),'actual core/block sampler schema required')
    _require('active_action' in receipt and receipt['active_action'] is None and receipt.get('aborted') is False,'unfinished or aborted actions are not native outcomes')
    _require(receipt.get('aborted_actions',[] if block else None)==[],'aborted action evidence cannot release a clean native run')
    _require(type(receipt.get('last_tick')) is int and receipt['last_tick']==ticks,'sampler and physical clock differ')
    _require(receipt.get('global_tick_modulus_used') is False,'explicit action boundaries required instead of a global tick modulus')
    alignment=case.get('alignment_receipt',{})
    _require(_same(alignment.get('native_outcome_sampling'),receipt),'alignment and top-level action sampling receipts differ')
    _require(alignment.get('native_success_latched') is success and alignment.get('native_terminal_failure_latched') is fatal,
        'original alignment success/failure differs from the case')
    expected_outcome=dict(success=success,terminate=terminal,terminal_failure=fatal)
    if block:
        _require(not fatal,'original blocks have no native fatal conditions')
        _require(receipt.get('original_native_success_latch_preserved') is True,'original block success latch must be preserved')
        for key in ('native_failure_conditions','extra_task_success_calls'):
            _require(type(receipt.get(key)) is int and receipt[key]==0,'blocks must retain zero native fail conditions and extra success calls')
    else:
        for key in ('success_sampled_each_tick','failure_sampled_at_explicit_action_end',
                    'reset_failure_is_diagnostic_only','reset_success_semantics_preserved'):
            _require(receipt.get(key) is True,'required core native sampling rule missing: '+key)
        _require(_same(_outcome(receipt.get('sampled_outcome')),expected_outcome),'sampled terminal outcome differs from the original case')
        _require(alignment.get('native_terminal_latched') is terminal,'native terminal flag differs from the sampled result')
        _require(_same(alignment.get('native_outcome'),expected_outcome),'original alignment native result differs from the sampler')
        _require(_integer(receipt.get('intermediate_failure_samples')),'intermediate failure count must remain diagnostic')
        initial=_outcome(receipt.get('initial_native_result'))
    actions=receipt.get('completed_actions')
    _require(isinstance(actions,list),'complete explicit action list required')
    cursor=0;previous_terminal=False;last=None
    allowed={'baseline_execute'} if method=='physmani' else {'geometry_execute','public_hold'}
    for index,row in enumerate(actions):
        _require(isinstance(row,dict) and row.get('complete') is True,'incomplete action is not a native outcome')
        _require(not previous_terminal,'no action may follow a sampled native terminal')
        label=row.get('label');maximum=row.get('max_ticks');start=row.get('start_tick');end=row.get('end_tick');count=row.get('ticks')
        _require(label in allowed,'action label differs from the authorized method')
        _require(_integer(maximum,minimum=1,maximum=10) and (label!='public_hold' or maximum==1),'bounded pose/hold action allowance required')
        _require(type(start) is int and start==cursor and _integer(end,minimum=start+1,maximum=ticks),'action boundaries must be continuous from physical tick zero')
        _require(_integer(count,minimum=1,maximum=maximum) and count==end-start,'action tick count differs from its physical interval')
        sampled=_outcome(row.get('sampled_outcome'))
        if block:
            _require(type(row.get('extra_task_success_calls')) is int and row['extra_task_success_calls']==0
                and sampled['terminal_failure'] is False,'block action must preserve original success-only evaluation')
        else:
            _require(type(row.get('action_index')) is int and row['action_index']==index,'core action indices must remain sequential')
            current=_outcome(row.get('current_native_result'))
            if not sampled['success']:
                _require(_same(sampled,current),'failure must consume the current action-end result, never OR past failures')
        cursor=end;last=sampled;previous_terminal=sampled['terminate']
    _require(cursor==ticks,'completed actions must cover every counted physical tick exactly once')
    if actions:
        _require(_same(last,expected_outcome),'last completed action and original terminal outcome differ')
        if not block:_require(not initial['success'],'an initially successful episode cannot execute another action')
    else:
        _require(ticks==0 and not fatal,'zero-action cases cannot claim a native action-end failure')
        if not block:_require(initial['success'] is success,'zero-action success must equal the preserved reset success')
    if not block:
        terminal_tick=alignment.get('native_terminal_tick')
        _require((type(terminal_tick) is int and terminal_tick==ticks) if terminal else terminal_tick is None,
            'terminal tick must be its actual success/action-end tick')
    return receipt
