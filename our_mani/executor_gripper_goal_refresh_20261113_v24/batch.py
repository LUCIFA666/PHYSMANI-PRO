"""Paired TRAIN executor regression, after the preceding Native owner exits."""
import json,time
from pathlib import Path
from .common import OUT,read,write,append,pointer,sources,sha,verify,stopped
from .owner import run_case


def main():
    stopped();plan_path=OUT/'predeclared_native_validation.json';plan=read(plan_path)
    frozen=sources();frozen[str(plan_path)]=sha(plan_path)
    frozen.update({name:sha(name)for name in plan['cases']})
    write(OUT/'native_validation/request.json',dict(plan=pointer(plan_path),source_sha256=frozen))
    dependency=Path(plan['Native_slot_dependency'])
    while not dependency.exists():
        stopped();time.sleep(.25)
    previous=read(dependency)
    if not previous['complete']or not all(row['process_lifecycle']['complete']
        and row['process_lifecycle']['actual_wait_completed']
        and not row['process_lifecycle']['unreaped_descendants']for row in previous['rows']):
        raise RuntimeError('Previous Native owner did not release its slot')
    write(OUT/'native_validation/Native_slot_dependency_receipt.json',dict(dependency=pointer(dependency)))
    rows=[]
    for name in plan['cases']:
        stopped();verify(frozen);case=Path(name);print('START',case.stem,flush=True)
        result,life=run_case(case,OUT/'native_validation'/case.stem,mode='online_teacher')
        row=dict(case=name,result=result,process_lifecycle=life);rows.append(row)
        append(OUT/'native_validation/batch_progress.jsonl',row)
        print('DONE',case.stem,None if result is None else
            {k:result[k]for k in('original_native_success','actual_seconds','stop_reason')},
            'owner_complete',life['complete'],flush=True)
        if not life['complete']:raise RuntimeError('Incomplete Native lifecycle')
    verify(frozen)
    write(OUT/'native_validation/batch_receipt.json',dict(complete=True,rows=rows))


if __name__=='__main__':main()
