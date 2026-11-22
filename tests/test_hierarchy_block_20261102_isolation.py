"""CPU-only real flock/socket/namespace tests; never import a simulator/model."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.hierarchy_block_20261102 import isolation as i
from scripts import run_hierarchy_frozen_environment_block_20261102 as wrapper


PROJECT = Path(__file__).resolve().parents[1]
SHELL = "scripts/run_hierarchy_isolated_block_20261102.sh"


class TestIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def scope(self, slot=0, *, parallel=True, label=None):
        label = label or "c"+str(slot)
        runtime = self.root / i.RUNTIME / "evaluation_v1/train/cases" / label
        output = self.root / i.ARTIFACTS / "evaluation_v1/train/cases" / label
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "shell_environment.json").write_text("{}")
        paths = {key: str(runtime / name) for key, name in i.PRIVATE_NAMES.items()}
        for path in paths.values():
            Path(path).mkdir(exist_ok=True)
        shell = dict(assigned_cpus=sorted(os.sched_getaffinity(0)), private_paths=paths,
                     outer_process=i.process_identity(), host_network_namespace="synthetic-only-host")
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, {**paths, "DISPLAY": ":"+str(900+slot)}))
        stack.enter_context(patch.object(i, "_validate_shell", return_value=(shell, i._sha(runtime / "shell_environment.json"))))
        stack.enter_context(patch.object(i, "_display_evidence", return_value={"synthetic_CPU_only": True}))
        stack.enter_context(patch.object(i, "_baseline_snapshot", return_value={"synthetic_CPU_only": "unchanged"}))
        self.addCleanup(stack.close)
        try:
            return stack.enter_context(i.resource_scope(self.root, runtime, slot=slot, parallel=parallel, case_output=output)), stack
        except BaseException:
            stack.close()
            raise

    def test_dynamic_four_way_cpu_partition_and_ceiling(self):
        masks = [i.partition_cpus(range(32), slot) for slot in range(4)]
        self.assertEqual([len(mask) for mask in masks], [8]*4)
        self.assertEqual(set().union(*map(set, masks)), set(range(32)))
        self.assertTrue(all(set(masks[a]).isdisjoint(masks[b]) for a in range(4) for b in range(a)))
        for slot in (True, -1, 4):
            with self.assertRaises(i.IsolationError):
                i.partition_cpus(range(32), slot)
        with self.assertRaises(i.IsolationError):
            i.partition_cpus(range(16), 0)

    def test_parallel_shared_guards_reject_old_ex_and_exclude_other_ex(self):
        for name in i.SHARED_LOCKS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"old bytes remain unchanged")
        old = (self.root / i.SHARED_LOCKS[0]).open("r+b")
        self.addCleanup(old.close)
        fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(BlockingIOError):
            self.scope()
        fcntl.flock(old, fcntl.LOCK_UN)
        receipt, stack = self.scope(label="second")
        self.assertEqual(receipt["lock_proofs"][0]["mode"], "READ")
        with self.assertRaises(BlockingIOError):
            fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stack.close()
        self.assertTrue(receipt["complete"])
        for name in i.SHARED_LOCKS:
            self.assertEqual((self.root / name).read_bytes(), b"old bytes remain unchanged")

    def test_slot_and_output_leases_are_exclusive(self):
        receipt, stack = self.scope()
        with self.assertRaises(BlockingIOError):
            self.scope(slot=0, label="different_case_same_slot")
        self.assertTrue(i.assert_live(receipt))
        self.assertFalse(Path(receipt["case_output"]).exists())
        stack.close()
        self.assertTrue(receipt["restored"])

    def test_listener_audit_proves_self_socket_inode_and_unknown_port_not_invented(self):
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0)); listener.listen(1)
        snapshot = i.listener_snapshot()
        port = listener.getsockname()[1]
        rows = [row for row in snapshot["listeners"] if row["local_port"] == port]
        self.assertEqual(len(rows), 1)
        self.assertIn(listener.fileno(), rows[0]["fds"])
        self.assertEqual(os.readlink(f"/proc/self/fd/{listener.fileno()}"), "socket:["+str(rows[0]["inode"])+"]")
        self.assertEqual(snapshot["network_namespace"], os.readlink("/proc/self/ns/net"))

    def test_interval_and_final_audit_failure_raise(self):
        receipt, stack = self.scope()
        i.record_native_interval(receipt, "start")
        with self.assertRaises(i.IsolationError):
            i.record_native_interval(receipt, "start")
        i.record_native_interval(receipt, "end")
        stack.close()
        i.validate_receipt(receipt)
        self.assertGreaterEqual(receipt["native_interval"]["end_monotonic"], receipt["native_interval"]["start_monotonic"])
        receipt, stack = self.scope(slot=1, label="failed_final_audit")
        with patch.object(i, "listener_snapshot", side_effect=OSError("synthetic proc failure")):
            with self.assertRaises(i.IsolationError):
                stack.close()
        self.assertFalse(receipt["complete"])
        self.assertTrue(receipt["errors"])

    def test_namespace_alias_path_refused(self):
        bad = self.root / "old_namespace/cases/case1"
        with self.assertRaises(i.IsolationError):
            with i.resource_scope(self.root, bad, slot=0, parallel=True,
                                  case_output=self.root / i.ARTIFACTS / "cases/case1"):
                pass

    def test_new_wrapper_marker_is_explicit_and_never_an_old_marker_alias(self):
        with patch.dict(os.environ, {wrapper.MARKER: "{}"}):
            with self.assertRaisesRegex(ValueError, "attestation differs"):
                wrapper.namespace_context("a"*64)

    def test_new_wrapper_uses_one_bwrap_and_original_mount_arguments(self):
        captured = {}
        def run(command, **kwargs):
            captured.update(command=command, env=kwargs["env"])
            fd = kwargs["pass_fds"][0]
            captured["options"] = os.read(fd, 100000).decode().split("\0")[:-1]
            return SimpleNamespace(returncode=0)
        original_bytes = wrapper.original.sha(wrapper.original.__file__)
        with patch.object(wrapper.subprocess, "run", side_effect=run):
            self.assertEqual(wrapper.run_namespace({"files": []}, ["python", "CPU_only.py"]), 0)
        self.assertEqual(captured["options"], ["--unshare-net", "--bind", "/", "/", "--dev-bind", "/dev", "/dev"])
        self.assertEqual(captured["command"][-4:], ["xvfb-run", "-a", "python", "CPU_only.py"])
        marker = json.loads(captured["env"][wrapper.MARKER])
        self.assertFalse(marker["host_network_preserved"])
        self.assertEqual(marker["schema"], wrapper.SCHEMA)
        self.assertNotIn(wrapper.original.MARKER, captured["env"])
        self.assertEqual(wrapper.original.sha(wrapper.original.__file__), original_bytes)

    def test_actual_shell_single_cpu_net_wrapper_isolates_port_display_tmp_and_affinity(self):
        xvfb = shutil.which("xvfb-run") or Path("/home/ubuntu/projects/physmani_reproduction/runtime/usr/bin/xvfb-run").is_file()
        if not all(shutil.which(name) for name in ("bwrap", "taskset")) or not xvfb:
            self.skipTest("CPU namespace/Xvfb tools not available")
        for relative in (SHELL, "our_mani/hierarchy_block_20261102/isolation.py"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / relative, target)
        # Fast real-kernel resource test. The separately recorded full-wrapper
        # CPU integration probe verifies the actual v820 frozen OS materials.
        # This temporary CPU-only wrapper never loads a simulator/model.
        cpu_wrapper = self.root / "scripts/CPU_net_only.py"
        cpu_wrapper.write_text("import subprocess,sys\nraise SystemExit(subprocess.run(['/usr/bin/bwrap','--unshare-net','--bind','/','/','--dev-bind','/dev','/dev','--','xvfb-run','-a',sys.executable,*sys.argv[1:]]).returncode)\n")
        code = '''import importlib.util,json,os,socket,sys,time
from pathlib import Path
root,runtime,output=map(Path,sys.argv[1:4]); slot=int(sys.argv[-1])
spec=importlib.util.spec_from_file_location("private_isolation",root/"our_mani/hierarchy_block_20261102/isolation.py")
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=socket.socket(); s.bind(("0.0.0.0",19997)); s.listen(1)
with m.resource_scope(root,runtime,slot=slot,parallel=True,case_output=output) as r:
    r["CPU_test_only"]=True; r["Native_started"]=False
    m.record_native_interval(r,"start")
    (runtime/"ready").write_text("CPU only")
    deadline=time.monotonic()+15
    while not all((runtime.parent/("c"+str(n))/"ready").exists() for n in (0,1)):
        if time.monotonic()>deadline: raise RuntimeError("CPU rendezvous timeout")
        time.sleep(.01)
    time.sleep(.1); s.close(); m.record_native_interval(r,"end")
print(json.dumps(r))
'''
        processes = []
        for slot in (0, 1):
            runtime = self.root / i.RUNTIME / "CPU_probes/cases" / ("c"+str(slot))
            runtime.mkdir(parents=True)
            output = self.root / i.ARTIFACTS / "CPU_probes/cases" / ("c"+str(slot))
            command = ["bash", str(self.root / SHELL), str(runtime), "--", str(cpu_wrapper), "-c", code,
                       str(self.root), str(runtime), str(output), "--resource-slot", str(slot)]
            processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True))
        receipts = []
        results = []
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                results.append((process.returncode, stdout, stderr))
        finally:
            for process in processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
        for returncode, stdout, stderr in results:
            self.assertEqual(returncode, 0, stderr+stdout)
            receipts.append(json.loads(stdout.splitlines()[-1]))
        a, b = receipts
        for receipt in receipts:
            i.validate_receipt(receipt)
            self.assertFalse(receipt["Native_started"])
            self.assertEqual(len(receipt["assigned_cpus"]), 8)
            self.assertTrue(any(row["local_port"] == 19997 for row in receipt["listeners_active"]))
            self.assertTrue(receipt["display_evidence"]["server"]["pid"] > 0)
            self.assertEqual(receipt["baseline_sources_before"], receipt["baseline_sources_after"])
        self.assertNotEqual(a["network_namespace"], b["network_namespace"])
        self.assertNotEqual(a["display"], b["display"])
        self.assertNotEqual(a["private_paths"]["TMPDIR"], b["private_paths"]["TMPDIR"])
        self.assertNotEqual(a["process"], b["process"])
        self.assertTrue(set(a["assigned_cpus"]).isdisjoint(b["assigned_cpus"]))
        self.assertLess(max(r["native_interval"]["start_monotonic"] for r in receipts), min(r["native_interval"]["end_monotonic"] for r in receipts))


if __name__ == "__main__":
    unittest.main()
