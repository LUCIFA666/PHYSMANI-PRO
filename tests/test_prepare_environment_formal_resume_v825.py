"""No real outputs: exact591/354 environment-failure partitions and tier PID checks."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json,tempfile,unittest
from scripts import prepare_environment_formal_resume_v825_r10 as prep


class PreparationTests(unittest.TestCase):
    def fixture(self):
        names=['fake'+str(i) for i in range(942)]+sorted(prep.FAILED)
        rows=[dict(job_id=name,evidence_runtime='runtime/framework/formal_VAL_v753' if i<95 else prep.PREVIOUS,
                   imported_completed=i<95) for i,name in enumerate(names)]
        done={name:dict(complete=True,native_success=False) for name in names[:591]}
        failed={name:dict(stage='native',complete=False,returncode=1,interrupted=False,timed_out=False,automatic_retry=False)
                for name in prep.FAILED}
        plan=dict(split='validation',expected_cases=945,jobs=rows,source_outcome_filtering=False,automatic_retry=False,policy_sha256='p'*64)
        status=dict(split='validation',expected=945,completed=done,failed=failed,active=[],checking=[],native_slots=0,GPU_slots=0,
                    drain_requested=True,stop_requested=False)
        terminal=dict(split='validation',expected=945,complete=False,completed=591,failed=3,drained=True,drain_requested=True,
                      stop_requested=False,policy_sha256='p'*64)
        return plan,status,terminal

    def test591_includes_every_nativeFalse_and354_contains_exact3failed(self):
        plan,status,terminal=self.fixture();jobs,done,failed,remaining,policy=prep.partition(plan,status,terminal)
        self.assertEqual(len(done),591);self.assertEqual(set(done),set(status['completed']))
        self.assertEqual(set(failed),prep.FAILED);self.assertEqual(len(remaining),354)
        self.assertEqual(len(set(remaining)-set(failed)),351);self.assertEqual(set(done)|set(remaining),set(jobs))

    def test_user_STOP_cannot_be_relabelled_as_environment_DRAIN(self):
        for where in ('status','terminal'):
            plan,status,terminal=self.fixture();(status if where=='status' else terminal)['stop_requested']=True
            with self.subTest(where=where),self.assertRaisesRegex(ValueError,'DRAIN'):prep.partition(plan,status,terminal)

    def test_live_work_or_changed_failure_reason_cannot_be_adopted(self):
        for change in ('active','interrupted','missing_failed'):
            plan,status,terminal=self.fixture()
            if change=='active':status['active']=['still-live']
            elif change=='interrupted':status['failed'][next(iter(prep.FAILED))]['interrupted']=True
            else:status['failed'].pop(next(iter(prep.FAILED)))
            with self.subTest(change=change),self.assertRaises(ValueError):prep.partition(plan,status,terminal)

    def test_flattening_all591_to772_loses_the95_real_tier_and_is_rejected(self):
        plan,status,terminal=self.fixture()
        for row in plan['jobs']:row.update(evidence_runtime=prep.PREVIOUS,imported_completed=False)
        with self.assertRaisesRegex(ValueError,'95'):prep.partition(plan,status,terminal)

    def test_wrong_completed_count_cannot_select_a_convenient_subset(self):
        plan,status,terminal=self.fixture();status['completed'].pop(next(iter(status['completed'])))
        with self.assertRaisesRegex(ValueError,'591'):prep.partition(plan,status,terminal)

    def test_STOP_prevents_any_environment_capture_or_setup(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(prep,'stopped',return_value=True):
            with patch.object(prep.owner,'require_frozen_environment') as env:
                with self.assertRaisesRegex(ValueError,'STOP'):prep.prepare(Path(tmp))
                env.assert_not_called();self.assertFalse((Path(tmp)/prep.SETUP).exists())

    def test_tier_child_ownerPID_is_not_rewritten_to_current_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tier=root/'runtime/framework/formal_VAL_v753';source=root/'scripts/run_formal_suite16_owner_v753_r10.py'
            source.parent.mkdir(parents=True);source.write_text('fake-source')
            tier.mkdir(parents=True);(tier/'launch.json').write_text(json.dumps(dict(source_sha256=prep.file_sha(source),pid=123)))
            for stage in ('CPU','native'):
                leaf=tier/stage/'job';leaf.mkdir(parents=True);(leaf/'launch.json').write_text(json.dumps(dict(owner_pid=123)))
            row=dict(job_id='job',evidence_runtime='runtime/framework/formal_VAL_v753')
            self.assertEqual(prep.freeze_tier_owner(root,{},row),tier)
            (tier/'native/job/launch.json').write_text(json.dumps(dict(owner_pid=999)))
            with self.assertRaisesRegex(ValueError,'ownerPID'):prep.freeze_tier_owner(root,{},row)

    def test_only_exact_frozen_OS_error_can_be_replaced_not_an_arbitrary_native_failure(self):
        from scripts.run_frozen_system_environment_v820_r10 import RESTORED_TARGETS
        target=next(iter(RESTORED_TARGETS))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime=root/'old';leaf=runtime/'native/job';leaf.mkdir(parents=True)
            row=dict(job_id='job')
            prefix='our_mani.multitask.protocol.MultitaskProtocolError: frozen input SHA changed: '
            with patch.object(prep.base,'freeze_interrupted'):
                (leaf/'stderr.log').write_text('Traceback\n'+prefix+target+'\n')
                prep.freeze_environment_failure(root,{},runtime,row,{},{} )
                for final in ('task failed',prefix+'/not/an/approved/OS/file'):
                    (leaf/'stderr.log').write_text(final+'\n')
                    with self.subTest(final=final),self.assertRaisesRegex(ValueError,'actual frozen OS'):
                        prep.freeze_environment_failure(root,{},runtime,row,{},{} )

    def test_complete_snapshot_bytes_and_library_topology_inventory_are_all_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest_pointer=dict(path='env/hybrid.json',sha256='a'*64)
            original_pointer=dict(path='env/original/manifest.json',sha256='b'*64)
            inventory_pointer=dict(path='env/library_inventory.json',sha256='c'*64)
            manifest=dict(schema='r10_exact_frozen_OS_hybrid_snapshot_v820',complete=True,
                base_individual_file_manifest=original_pointer,
                library_binding=dict(inventory=inventory_pointer,snapshot_directory=str(root/'env/library'),locked_regular_files=4828),
                restore_materials=dict(path='env/materials.json',sha256='d'*64))
            original=dict(files=[dict(snapshot_file='files/'+str(i),sha256='e'*64) for i in range(6418)])
            inventory=dict(complete=True,snapshot_directory=str(root/'env/library'),
                snapshot=dict(entries={**{str(i):dict(kind='file',sha256='f'*64) for i in range(4828)},
                                       'link':dict(kind='symlink',target='0'),'dir':dict(kind='directory')}))
            documents={str(root/'env/hybrid.json'):manifest,str(root/'env/original/manifest.json'):original,
                       str(root/'env/library_inventory.json'):inventory}
            with patch.object(prep,'read_evidence',side_effect=lambda r,f,p:(documents[str(p)],{})),patch.object(prep,'freeze_file') as freeze:
                result=prep.freeze_environment_materials(root,{},manifest_pointer)
            self.assertEqual(result['individual_snapshot_files'],6418)
            self.assertEqual(result['complete_library_regular_files'],4828)
            paths=[str(call.args[2]) for call in freeze.call_args_list]
            self.assertIn(str(root/'env/original/files/6417'),paths);self.assertIn(str(root/'env/library/4827'),paths)
            self.assertIn(str(root/'env/library_inventory.json'),paths)
            self.assertNotIn(str(root/'env/library/link'),paths)


if __name__=='__main__':unittest.main()
