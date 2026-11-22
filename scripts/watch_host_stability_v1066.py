"""Read-only, durable host telemetry; never launches evaluation or changes hardware."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,signal,subprocess,threading

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'runtime/framework/host_stability_watch_v1066'
EVENT=threading.Event()

def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=4,check=False)
        return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip()[:500])
    except (OSError,subprocess.TimeoutExpired) as e:return dict(error=repr(e))

def sample():
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    memory={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')}
    temps={}
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:temps[str(p.parent.name)]=dict(kind=(p.parent/'type').read_text().strip(),millidegrees_C=int(p.read_text()))
        except OSError:pass
    for p in Path('/sys/class/hwmon').glob('hwmon*/temp*_input'):
        try:
            label=p.with_name(p.name.replace('_input','_label'))
            temps[str(p.parent.name)+'/'+p.name]=dict(kind=(p.parent/'name').read_text().strip(),
                label=label.read_text().strip() if label.exists() else p.stem,millidegrees_C=int(p.read_text()))
        except (OSError,ValueError):pass
    psi={}
    for kind in ('cpu','memory','io'):
        p=Path('/proc/pressure')/kind
        if p.exists():psi[kind]=p.read_text().strip()
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),boot_id=boot,
        uptime_seconds=float(Path('/proc/uptime').read_text().split()[0]),memory_kib=memory,temperatures=temps,
        pressure=psi,loadavg=Path('/proc/loadavg').read_text().strip(),
        GPU=command(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu',
            '--format=csv,noheader,nounits']),
        GPU_compute=command(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits']),
        task_or_hardware_configuration_changed=False,Native_started=False)

def save(value):
    STATE.mkdir(parents=True,exist_ok=True)
    p=STATE/(value['boot_id']+'.jsonl')
    with p.open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    d=STATE
    while d.is_relative_to(ROOT):
        fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if d==ROOT:break
        d=d.parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=10)
    a=p.parse_args();assert 5<=a.interval<=60
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:EVENT.set())
    while not EVENT.is_set() and not (STATE/'STOP').exists() and not (ROOT/'runtime/framework/multitask_r10_STOP').exists():
        value=sample();save(value)
        if a.once:print(json.dumps(value,ensure_ascii=False));return
        EVENT.wait(a.interval)

if __name__=='__main__':main()
