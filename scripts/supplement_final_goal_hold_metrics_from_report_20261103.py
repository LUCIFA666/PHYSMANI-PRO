"""Derive explicit final-goal AND current-hold metrics from a sealed report only.

Never opens referenced case/GT/source files; never changes a success judge.
"""
import argparse
import hashlib
import json
from pathlib import Path

SCHEMA='final_goal_and_current_hold_report_only_supplement_v1'
FIELDS=('Native_goal_any','Native_goal_final','actual_current_lifted_hold_passed',
    'everGoalAndCurrentHold','finalNativeGoalAndCurrentHold',
    'public_everhold','public_finalhold','public_finalgoalAndhold')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(value):
    if type(value)is not bool:raise ValueError('Explicit original boolean metric required')
    return value


def supplement(report_path,output,expected_sha256):
    path=Path(report_path).resolve(strict=True);digest=sha256(path)
    if digest!=expected_sha256:raise ValueError('Sealed source report SHA differs')
    source=json.loads(path.read_text())
    if (source.get('schema')!='closed_public_position_development_cohort_evaluation_v1'
        or source.get('source_audit',{}).get('complete')is not True
        or source['source_audit'].get('all_actual_exits0_and_nativeclosed')is not True):
        raise ValueError('Previously closed/source-audited development report required')
    positive=[];negative=[]
    for case in source['cases']:
        n=case['original_native_task_and_hold'];public=case['public']
        goal_any=boolean(n['native_success_any']);goal_final=boolean(n['native_success_final'])
        current_hold=boolean(n['final']['current_lifted_hold_passed'])
        old=boolean(case['actual_goalAny_and_strict_final_hold'])
        if old!=(goal_any and current_hold):raise ValueError('Original goalAny/currenthold metric differs')
        row=dict(id=case['id'],Native_goal_any=goal_any,Native_goal_final=goal_final,
            actual_current_lifted_hold_passed=current_hold,everGoalAndCurrentHold=old,
            finalNativeGoalAndCurrentHold=goal_final and current_hold,
            public_everhold=boolean(public['public_lift_hold_ever']),
            public_finalhold=boolean(public['public_lift_hold_final']),
            public_finalgoalAndhold=boolean(public['public_goal_complete_final'])and boolean(public['public_lift_hold_final']),
            public_finalgoal=boolean(public['public_goal_complete_final']))
        if boolean(case['expected_negative_no_proxy']):
            row.update(expected_negative_no_proxy=True,excluded_from_positive_denominator=True,
                ever_effective_public_false_confirmation=boolean(public['ever_effective_public_attachment_confirmed']))
            negative.append(row)
        else:positive.append(row)
    if len({r['id']for r in positive+negative})!=len(positive)+len(negative):raise ValueError('Duplicate case ID')
    aggregate=dict(positive_denominator=len(positive),negative_denominator=len(negative),
        positive_counts={key:sum(r[key]for r in positive)for key in FIELDS})
    result=dict(schema=SCHEMA,source_report=dict(path=str(path),sha256=digest),
        definitions=dict(everGoalAndCurrentHold='Native_goal_any AND actual_current_lifted_hold_passed at final sample; ever applies only to the Native goal flag, not a final-goal claim',
            finalNativeGoalAndCurrentHold='Native_goal_final AND actual_current_lifted_hold_passed at final sample',
            public_finalgoalAndhold='public_goal_complete_final AND public_lift_hold_final',
            current_hold='Original unchanged Native current lifted hold: .5s duration, .04m lift, .02m relative drift',
            reporting='Report final Native goal+current hold beside public final goal+hold; do not call goalAny/currenthold full task success'),
        positives=positive,negative_no_proxy=negative,aggregate=aggregate,
        sealed_source_audit=source['source_audit'],original_judge=source['original_judge'],
        original_report_or_success_criteria_changed=False,source_case_or_raw_GT_opened=False,
        NN_forward_calls=0,Native_started=False,fit_started=False,TEST_read=False,formal_qualification=False)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    report=out/'report.json';report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if sha256(path)!=digest:raise ValueError('Original report changed while deriving supplement')
    receipt=dict(schema='final_goal_hold_report_only_supplement_receipt_v1',complete=True,
        source_report=result['source_report'],report=dict(path=str(report),sha256=sha256(report)),
        script=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)),
        positive_cases=len(positive),negative_cases=len(negative),
        reads_only_sealed_report=True,raw_GT_or_case_reads=0,success_judge_changed=False,
        source_report_after_sha256=sha256(path),original_report_unchanged=True,
        Native_started=False,fit_started=False,NN_forward_calls=0,TEST_read=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(aggregate=aggregate,positives=positive,negative_no_proxy=negative,report_sha256=sha256(report),receipt_sha256=sha256(out/'receipt.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(supplement(args.report,args.output,args.expected_sha256),indent=2))

if __name__=='__main__':main()
