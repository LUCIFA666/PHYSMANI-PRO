"""Host-side actual exit receipt for a new serial local MPC diagnostic."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    argv = sys.argv[1:]
    index = argv.index("--output")
    output = Path(argv[index + 1]).resolve()
    root = Path(__file__).resolve().parents[1]
    family = root / "runtime/framework/local_mpc_20261102"
    if family not in output.parents:
        raise ValueError("new local MPC runtime output required")
    family.mkdir(parents=True, exist_ok=True)
    prefix = output.parent / (output.name + ".owner")
    with Path(str(prefix) + ".log").open("x") as log:
        start = datetime.now(timezone.utc).isoformat()
        process = subprocess.Popen(["bash", str(root / "scripts/run_local_mpc_20261102.sh"), *argv],
                                   cwd=root, stdout=log, stderr=subprocess.STDOUT)
        with Path(str(prefix) + ".json").open("x") as stream:
            json.dump(dict(started_utc=start, child_pid=process.pid, argv=argv), stream, indent=2)
        print(json.dumps(dict(child_pid=process.pid, log=str(prefix) + ".log")), flush=True)
        code = process.wait()
    with Path(str(prefix) + ".actual_exit.json").open("x") as stream:
        json.dump(dict(started_utc=start, finished_utc=datetime.now(timezone.utc).isoformat(),
                       actual_process_returncode=code, output=str(output)), stream, indent=2)
    print(json.dumps(dict(actual_process_returncode=code, output=str(output))), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
