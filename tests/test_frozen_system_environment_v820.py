"""CPU-only failure-boundary tests; never invoke bwrap, simulation or a model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1]/'scripts/run_frozen_system_environment_v820_r10.py'
spec = importlib.util.spec_from_file_location('frozen_os_v820_test_subject', MODULE)
v820 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v820)


class FrozenSystemEnvironmentTests(unittest.TestCase):
    def test_non_OS_and_traversal_targets_rejected(self):
        for target in ('/home/ubuntu/project/a.py', '/proc/1/mem', '/dev/null',
                       '/usr/../home/ubuntu/a.py', '/usr//bin/git', 'usr/bin/git'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                v820.os_target(target)

    def test_actual_existing_OS_target_is_canonical(self):
        self.assertEqual(v820.os_target('/usr/bin/git'), Path('/usr/bin/git'))

    def test_arguments_preserve_literal_argv_and_host_surfaces(self):
        command = ['/bin/echo', '$(false); literal with spaces']
        args = v820.bwrap_arguments(dict(files=[dict(snapshot_file='files/usr/bin/git', target='/usr/bin/git')]), command)
        self.assertEqual(args[:3], ['--bind', '/', '/'])
        self.assertEqual(args[3:6], ['--dev-bind', '/dev', '/dev'])
        self.assertEqual(args[-3:], ['--', *command])
        for forbidden in ('--unshare-pid', '--unshare-net', '--unshare-user', '--proc', '--dev', '--uid', '--gid'):
            self.assertNotIn(forbidden, args)

    def test_empty_or_NUL_command_rejected(self):
        for command in ([], ['/bin/echo', 'bad\0arg']):
            with self.assertRaises(ValueError):
                v820.bwrap_arguments(dict(files=[]), command)

    def fixture_manifest(self, temporary):
        snapshot = Path(temporary)/'snapshot'
        data = snapshot/'files/usr/bin/git'
        data.parent.mkdir(parents=True)
        data.write_bytes(b'exact frozen CPU fixture bytes\n')
        digest = v820.sha(data)
        bwrap = Path(temporary)/'bwrap'
        bwrap.write_bytes(b'CPU fixture bwrap; never executed\n')
        bwrap_sha = v820.sha(bwrap)
        manifest = snapshot/'manifest.json'
        sources = [dict(path='/fixture/lock', sha256='a'*64, OS_file_count=1)]
        value = dict(schema=v820.SCHEMA, complete=True, source_locks=sources,
            frozen_OS_file_count=1, bwrap=dict(path=str(bwrap), sha256=bwrap_sha),
            files=[dict(target='/usr/bin/git', snapshot_file='files/usr/bin/git', sha256=digest,
                        bytes=data.stat().st_size)])
        manifest.write_text(json.dumps(value))
        return snapshot, data, bwrap, bwrap_sha, manifest, sources, value

    def test_complete_manifest_uses_expected_frozen_set(self):
        with tempfile.TemporaryDirectory() as t:
            snapshot, data, bwrap, bwrap_sha, manifest, sources, value = self.fixture_manifest(t)
            with patch.multiple(v820, SNAPSHOT=snapshot, MANIFEST=manifest, BWRAP=bwrap, BWRAP_SHA=bwrap_sha), \
                    patch.object(v820, 'frozen_os_files', return_value=({'/usr/bin/git': v820.sha(data)}, sources)), \
                    patch.object(v820, 'validate_library_binding'):
                self.assertEqual(v820.validate_manifest(manifest, v820.sha(manifest)), value)

    def test_snapshot_byte_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            snapshot, data, bwrap, bwrap_sha, manifest, sources, value = self.fixture_manifest(t)
            expected = v820.sha(data)
            data.write_bytes(b'changed bytes\n')
            with patch.multiple(v820, SNAPSHOT=snapshot, MANIFEST=manifest, BWRAP=bwrap, BWRAP_SHA=bwrap_sha), \
                    patch.object(v820, 'frozen_os_files', return_value=({'/usr/bin/git': expected}, sources)), \
                    patch.object(v820, 'validate_library_binding'):
                with self.assertRaisesRegex(ValueError, 'snapshot bytes/identity changed'):
                    v820.validate_manifest(manifest, v820.sha(manifest))

    def test_manifest_SHA_change_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as t:
            manifest = Path(t)/'manifest.json'
            manifest.write_text('{}')
            with patch.object(v820, 'MANIFEST', manifest):
                with self.assertRaisesRegex(ValueError, 'manifest SHA mismatch'):
                    v820.validate_manifest(manifest, '0'*64)

    def test_missing_namespace_marker_is_not_environment_proof(self):
        with patch.object(v820, 'validate_manifest', return_value={}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'namespace marker required'):
                v820.require_environment('/fixture/manifest', '0'*64)

    def test_cross_user_namespace_link_denial_is_explicit_unknown(self):
        marker = dict(schema='r10_frozen_OS_outer_marker_v820',
            manifest=v820.project_pointer(v820.MANIFEST, 'a'*64),
            wrapper=v820.project_pointer(MODULE),
            outer_wrapper_pid=12345, outer_wrapper_start_ticks=99,
            reference_namespaces=dict(mnt='mnt:[1]', pid='pid:[3]', net='net:[4]'),
            uid=os.getuid(), gid=os.getgid())
        def namespace(kind, pid='self'):
            if pid != 'self':
                raise PermissionError('fixture EACCES')
            return dict(mnt='mnt:[2]', pid='pid:[3]', net='net:[4]')[kind]
        with patch.dict(os.environ, {v820.MARKER: json.dumps(marker)}), \
                patch.object(v820, 'namespace_id', side_effect=namespace), \
                patch.object(v820, 'process_start_ticks', return_value=99):
            result = v820.namespace_context('a'*64)
        self.assertFalse(result[-1]['available'])
        self.assertIn('EACCES', result[-1]['reason'])

    def test_marker_cannot_substitute_for_actual_OS_bytes(self):
        with patch.object(v820, 'validate_manifest', return_value=dict(files=[dict(target='/usr/bin/git', sha256='0'*64)])), \
                patch.object(v820, 'namespace_context', return_value=({}, {}, {}, {}, {})):
            with self.assertRaisesRegex(ValueError, 'actual namespace frozen OS target SHA changed'):
                v820.require_environment('/fixture/manifest', '0'*64)

    def test_large_argument_list_uses_memfd_no_shell(self):
        rows = [dict(snapshot_file='files/usr/lib/x86_64-linux-gnu/fixture'+str(n), target='/usr/lib/x86_64-linux-gnu/fixture'+str(n)) for n in range(4828)]
        rows += [dict(snapshot_file='files/usr/bin/fixture'+str(n), target='/usr/bin/fixture'+str(n)) for n in range(1590)]
        captured = {}
        def fake_run(argv, **kwargs):
            captured.update(argv=argv, kwargs=kwargs)
            fd = kwargs['pass_fds'][0]
            captured['payload'] = os.read(fd, 4*1024*1024)
            return type('Result', (), {'returncode': 0})()
        with patch.object(v820.subprocess, 'run', side_effect=fake_run), patch.object(v820, 'sha', return_value='a'*64):
            self.assertEqual(v820.run_namespace(dict(files=rows,library_binding=True), ['/bin/true']), 0)
        self.assertEqual(captured['argv'][1], '--args')
        self.assertFalse(captured['kwargs']['shell'])
        self.assertGreater(len(captured['payload']), 100000)
        self.assertTrue(captured['payload'].endswith(b'/usr/bin/fixture1589\0'))
        self.assertEqual(captured['payload'].count(b'--ro-bind\0'), 1591)
        self.assertNotIn('--mount-stage', captured['argv'])
        self.assertEqual(captured['argv'][-2:], ['--', '/bin/true'])


if __name__ == '__main__':
    unittest.main()
