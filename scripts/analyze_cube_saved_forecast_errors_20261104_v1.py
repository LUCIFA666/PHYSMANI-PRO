"""Read-only same-time forecast decomposition on the eight closed Cube cases.

No model, simulator or new predictions. Existing causal public predictions are
compared with private labels only after the completed paired-run audit.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/local_mpc_20261102'


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, float)
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.median(a)),
        p95=float(np.percentile(a, 95)), maximum=float(a.max())) if len(a) else dict(count=0)


def main():
    owner = BASE / 'cube_v72_paired_development_run_v1'
    done = read(owner / 'completion.json'); root_wait = read(owner / 'Root_actual_exit.json')
    if not (done['complete'] and done['cases_completed'] == 8 and done['all_actual_waits_completed']
            and root_wait['actual_process_returncode'] == 0 and root_wait['actual_wait_completed']):
        raise ValueError('Actual completed eight-case run required')
    report_path = BASE / 'cube_candidate_paired_closed8_posthoc_v1/report.json'
    audited = read(report_path)
    if not audited['complete']: raise ValueError('Completed source-bound posthoc audit required')
    bound = {str(report_path): sha(report_path)}; cases = []
    for case in audited['cases']:
        path = ROOT / 'runtime/framework/cube_v72_live_20261104_v1' / case['profile'] / case['case_id']
        files = [path / n for n in ('initial_snapshot.json','native_ticks.jsonl',
            'public_candidate_position_v72_decisions.jsonl','public_shadow/public_outputs.jsonl','worker_exit.json')]
        bound.update({str(p): sha(p) for p in files})
        worker = read(path / 'worker_exit.json')
        if worker['exit_code'] != 0 or not worker['native_closed']:
            raise ValueError('Closed Native worker required')
        truth = {0: read(path / 'initial_snapshot.json')['state']}
        for tick in lines(path / 'native_ticks.jsonl'):
            state = tick.get('state', tick)
            truth[int(round(float(state['time']) / .05))] = state
        commands = lines(path / 'public_candidate_position_v72_decisions.jsonl')
        first_close = min((float(d['time']) for d in commands
            if float(d['executed_action']['opening']) < .5), default=float('inf'))
        collected = {h: dict(current=[], future=[], displacement=[], samples=[]) for h in (2,8,16)}
        for row in lines(path / 'public_shadow/public_outputs.jsonl'):
            r = row['result']; forecast = r.get('forecast')
            if r.get('mode') != 'FREE' or not r.get('free_forecast_available') or not forecast:
                continue
            now = float(r['timestamp']); k = int(round(now / .05))
            if abs(k*.05 - now) > 1e-7 or k not in truth:
                continue
            prediction = np.asarray(forecast['positions'], float)
            if prediction.shape != (17,1,3): raise ValueError('Original H17 target-only prediction required')
            ft = np.asarray(forecast['times'], float)
            if not np.allclose(ft, now + np.arange(17)*.05, atol=1e-7, rtol=0):
                raise ValueError('Same physical timestamp forecast required')
            for h, values in collected.items():
                if now+h*.05 >= first_close-1e-8 or any(j not in truth for j in range(k,k+h+1)):
                    continue
                if any(truth[j]['held'] for j in range(k,k+h+1)):
                    continue
                p0 = np.asarray(truth[k]['object_position'], float)
                p1 = np.asarray(truth[k+h]['object_position'], float)
                current = np.linalg.norm(prediction[0,0]-p0)
                future = np.linalg.norm(prediction[h,0]-p1)
                displacement = np.linalg.norm((prediction[h,0]-prediction[0,0])-(p1-p0))
                values['current'].append(float(current));values['future'].append(float(future))
                values['displacement'].append(float(displacement))
                values['samples'].append(dict(t=now,current_error_m=float(current),
                    same_time_future_error_m=float(future),displacement_error_m=float(displacement)))
        cases.append(dict(profile=case['profile'], case_id=case['case_id'],
            first_close_time=None if not np.isfinite(first_close) else first_close,
            horizons={str(h*.05):dict(current_position_error_m=stats(v['current']),
                same_time_future_position_error_m=stats(v['future']),
                motion_displacement_error_m=stats(v['displacement']),samples=v['samples']) for h,v in collected.items()}))
    for p,h in bound.items():
        if sha(p)!=h: raise ValueError('Input changed during read-only analysis')
    out = BASE / 'cube_failure_mechanism_analysis_20261104_v1'
    out.mkdir(parents=True,exist_ok=True)
    report = dict(complete=True,scope='pre-first-close unattached intervals only; retained public forecast vs private same-time labels',
        definitions=dict(current='perception position error at anchor',future='total position error at future timestamp',
            displacement='forecast motion minus actual motion; subtracts each anchor position bias'),
        scope_limits=['Different policy trajectories create different visibility and available anchors.',
            'No neural forwards or simulator execution; historical measurements do not diagnose unseen images.'],
        cases=cases, input_sha256=bound, script_sha256=sha(__file__),NN_forward_calls=0,Native_started=False,GPU=False)
    with (out/'saved_forecast_error_decomposition.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(output=str(out/'saved_forecast_error_decomposition.json'),cases=len(cases))))


if __name__ == '__main__': main()
