"""Official fixed-case aggregation boundaries using CPU-only temporary proofs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import test_report as report
from our_mani.multitask import validation_release
from our_mani.multitask.evaluation import METHODS,digest,file_sha,native_completion_record,comparison_protocol_sha256
from our_mani.multitask.specs import TASK_SPECS
from scripts.run_multitask_fresh_queue_r10 import build_jobs


class OfficialReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=self.root/'artifacts/framework/multitask_r10';self.base.mkdir(parents=True)
    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return path
    def pointer(self,path):return dict(path=str(path.relative_to(self.root)),sha256=file_sha(path))
    def fixture(self):
        artifact=self.write('artifact.json',dict(CPU_only=True))
        profile=self.write('profile.json',dict(TRAIN_evidence=self.pointer(artifact)))
        child=self.write('child.py',{});parent=self.write('parent.py',{})
        config=self.write('config.json',dict(context_checkpoint=self.pointer(artifact),baseline_checkpoint='artifact.json',
            baseline_executor='CPU_V7_fixture',initial_condition_tolerance_profile=self.pointer(profile),
            r10_validation_release=self.pointer(artifact),formal_comparison_ready=True,
            continuation_identities={task:{p:'a'*64 for p in ('cv','context')} for task in TASK_SPECS}))
        rows=[]
        for task,spec in TASK_SPECS.items():
            for n in range(40):
                # Raw IDs deliberately do not sort like official alias indices.
                rows.append(dict(task=task,raw_episode_id='episode'+str(37+n),path='/CPU/never_read/data.h5',
                    variation=spec.supported_variations[n%len(spec.supported_variations)],
                    official_episode_index=n//len(spec.supported_variations),mapping_status='verified',
                    source_required_files={'target_state.pkl':dict(path='artifact.json',sha256=file_sha(artifact))},
                    target_state_sha256=file_sha(artifact),task_source_path='child.py',task_source_sha256=file_sha(child),
                    task_scene_path='artifact.json',task_scene_sha256=file_sha(artifact)))
        manifest=self.write('official_manifest.json',dict(split='test',mapping_status='verified',episode_count=560,episodes=rows))
        lock=self.write('lock.json',dict(files={str(p.relative_to(self.root)):file_sha(p) for p in
            (config,artifact,profile,child,parent,manifest)}))
        jobs=build_jobs(rows,'test',METHODS,file_sha(config),file_sha(lock),record_demos=True)
        plan=dict(version='r10_fresh_policy_queue_v1',split='test',methods=list(METHODS),replicas=1,
            source_episodes=560,expected_cases=2800,official_TEST_used_for_fitting=False,
            case_isolation='fresh_simulator_process_per_episode',config=self.pointer(config),manifest=self.pointer(manifest),
            frozen_inputs=self.pointer(lock),parent_source=self.pointer(parent),child_source=self.pointer(child),
            output=str((self.base/'TEST').relative_to(self.root)),jobs=jobs)
        plan['sha256']=digest(plan);path=self.write('plan.json',plan)
        self.write(plan['output']+'/status.json',dict(stage='incomplete',complete=False,completed=2799,expected=2800,
            plan_sha256=plan['sha256'],errors=[],stop_requested=False))
        return plan,path,manifest,config

    def test_complete_official_plan_keeps_all560_and_exact_first70(self):
        plan,path,manifest,config=self.fixture();proof=report.Proof(self.root)
        canonical,first=report.official_inventory(proof,manifest)
        self.assertEqual(len(report.official_plan(proof,path,canonical,first)['jobs']),2800)
        self.assertEqual(sum(j['record_demo'] for j in plan['jobs']),70)
        self.assertEqual(set(first.values()),{'episode37'})
        plan['jobs'].pop();plan['sha256']=digest({k:v for k,v in plan.items() if k!='sha256'});path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'complete 2800'):
            report.official_plan(report.Proof(self.root),path,canonical,first)

    def test_outcome_selected_or_extra_demo_is_rejected_before_reading_cases(self):
        plan,path,manifest,config=self.fixture();proof=report.Proof(self.root)
        canonical,first=report.official_inventory(proof,manifest)
        next(j for j in plan['jobs'] if j['source_raw_episode_id']=='episode38')['record_demo']=True
        plan['sha256']=digest({k:v for k,v in plan.items() if k!='sha256'});path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'fixed first-demo'):
            report.official_plan(report.Proof(self.root),path,canonical,first)

    def test_partial_official_queue_writes_pending_never_formal_report(self):
        plan,path,manifest,config=self.fixture();out=self.base/'official_report.json'
        with patch.object(report,'_VAL_release',return_value=dict(model_coverage_certificate={})):
            result=report.finalize_official_test(self.root,path,out,official_manifest=manifest)
        self.assertFalse(result['complete']);self.assertEqual(result['stage'],'complete_official_cases')
        self.assertFalse(out.exists());self.assertTrue((self.root/result['pending_report']['path']).is_file())
        self.assertTrue(result['no_infrastructure_failure_scored_as_policy_negative'])

    def groups(self,manifest):
        rows=json.loads(manifest.read_text())['episodes'];groups={};video=self.write('video.mp4',dict(CPU_dummy_not_decoded=True))
        decoder=self.write('decoder',dict(CPU_dummy_not_run=True))
        for row in rows:
            reset=dict(task=row['task'],raw_episode_id=row['raw_episode_id'],variation=row['variation'],
                official_episode_index=row['official_episode_index'],reset_kind='released_official_TEST_family_dispatch',
                original_source_replayed=True,expert_actions_used=False,future_observations_used=False,
                target_state_sha256=row['target_state_sha256'],verified_source_inputs=dict(
                    source_required_sha256={n:x['sha256'] for n,x in row['source_required_files'].items()},
                    task_source_sha256=row['task_source_sha256'],task_scene_sha256=row['task_scene_sha256']))
            groups[(row['task'],row['raw_episode_id'])]={method:dict(
                official_task_success=method!='context_value',native_terminal_failure=False,
                recorded_initial_conditions=dict(reset_receipt=deepcopy(reset)),counted_clock=dict(ticks=3),
                identity=dict(sha256='a'*64),video=dict(**self.pointer(video),decoded_frame_count=4),
                demo_selection='fixed first official manifest source, independent of outcome') for method in METHODS}
        return groups,dict(passed=True,decoded_frame_count=4,decoder='decoder',decoder_sha256=file_sha(decoder))

    def test_fixed_demos_keep_a_first_failure_even_when_later_case_succeeds(self):
        plan,path,manifest,config=self.fixture();groups,decoded=self.groups(manifest)
        proof=report.Proof(self.root);canonical,first=report.official_inventory(proof,manifest)
        task=next(iter(TASK_SPECS));groups[(task,'episode37')]['cv_geometry']['official_task_success']=False
        with patch.object(report,'validate_video',return_value=decoded):
            demos=report._demos(proof,groups,canonical,first)
        self.assertEqual(demos['completed'],70)
        entry=next(e for e in demos['entries'] if e['task']==task and e['method']=='cv_geometry')
        self.assertEqual(entry['raw_episode_id'],'episode37');self.assertFalse(entry['formal_result']['success'])
        self.assertFalse(demos['demo_selection_uses_outcome'])

    def test_reset_receipt_must_prove_target_schedule_source_not_just_camera(self):
        plan,path,manifest,config=self.fixture();groups,decoded=self.groups(manifest);row=json.loads(manifest.read_text())['episodes'][0]
        records=groups[(row['task'],row['raw_episode_id'])]
        report._official_source_evidence(report.Proof(self.root),row,records)
        records['context_value']['recorded_initial_conditions']['reset_receipt']['target_state_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'canonical alias/initial schedule'):
            report._official_source_evidence(report.Proof(self.root),row,records)

    def test_complete_composition_requires_parity_trace_and_all70_demos(self):
        plan,path,manifest,config=self.fixture();groups,decoded=self.groups(manifest)
        for name in ('test_report.py','trace_evidence.py','validation_release.py','video_validation.py'):
            self.write('our_mani/multitask/'+name,{})
        self.write('scripts/finalize_multitask_test_r10.py',{})
        out=self.base/'official_report.json';comparison=dict(complete=True,accepted=True,checks=[{}]*10)
        with patch.object(report,'_VAL_release',return_value=dict(model_coverage_certificate={},development_performance_screen_passed=None)),\
             patch.object(report,'_cases',return_value=(groups,[dict(CPU_mock_receipt=True)]*2800,dict(complete=True))),\
             patch.object(report,'profile_problems',return_value=[]),\
             patch.object(report,'compare_five_way',return_value=comparison),\
             patch.object(report,'audit_selection_usage',return_value=dict(complete=True,diagnostics_valid=True)),\
             patch.object(report,'validate_video',return_value=decoded),\
             patch.object(report,'__file__',str(self.root/'our_mani/multitask/test_report.py')):
            result=report.finalize_official_test(self.root,path,out,official_manifest=manifest)
        self.assertTrue(result['complete'],result.get('errors'));self.assertEqual(result['completed_method_cases'],2800)
        self.assertEqual(result['recorded_initial_conditions']['sha256'],file_sha(self.root/result['recorded_initial_conditions']['path']))
        parity=json.loads((self.root/result['recorded_initial_conditions']['path']).read_text())
        self.assertEqual(parity['pairs'],5600);self.assertEqual(result['by_speed']['ordinary']['cv_geometry']['episodes'],280)
        self.assertEqual(result['by_speed']['high_speed']['physmani']['episodes'],280)
        self.assertEqual(len(result['TEST_negative_value_geometry_differences']),14)
        self.assertFalse(result['success_module_benefit_assumed']);self.assertIsNone(result['development_performance_screen_passed'])
        self.assertNotIn(str(out.relative_to(self.root)),result['input_files_sha256'])

    def test_full_case_count_cannot_override_bad_selection_trace(self):
        plan,path,manifest,config=self.fixture();groups,decoded=self.groups(manifest);out=self.base/'official_report.json'
        with patch.object(report,'_VAL_release',return_value={}),\
             patch.object(report,'_cases',return_value=(groups,[],dict(complete=True))),\
             patch.object(report,'profile_problems',return_value=[]),\
             patch.object(report,'compare_five_way',return_value=dict(complete=True,accepted=True,checks=[{}]*10)),\
             patch.object(report,'audit_selection_usage',side_effect=ValueError('missing actual override trace')):
            result=report.finalize_official_test(self.root,path,out,official_manifest=manifest)
        self.assertFalse(result['complete']);self.assertEqual(result['stage'],'selection_trace');self.assertFalse(out.exists())

    def test_TEST_lock_preserves_VAL_model_semantics_and_original_proof_graph(self):
        artifact=self.write('artifact',{});oldconfig=self.write('val.json',dict(policy='frozen',formal_comparison_ready=False))
        oldlock=self.write('val_lock.json',dict(files={'val.json':file_sha(oldconfig),'artifact':file_sha(artifact)}))
        coverage=self.write('coverage.json',{})
        release=self.write('release.json',dict(complete=True,protocol_acceptance_passed=True,formal_test_permitted=True,
            split='validation',TEST_used=False,expected_source_cases=275,expected_method_cases=1375,completed_method_cases=1375,
            validated_input_lock=self.pointer(oldlock),validated_config=self.pointer(oldconfig),config_sha256=file_sha(oldconfig),
            comparison_protocol_sha256=comparison_protocol_sha256(json.loads(oldconfig.read_text())),
            input_files_sha256={'artifact':file_sha(artifact)},model_coverage_certificate=self.pointer(coverage)))
        testconfig=dict(policy='frozen',formal_comparison_ready=True,r10_validation_release=self.pointer(release))
        files={str(p.relative_to(self.root)):file_sha(p) for p in (artifact,oldconfig,oldlock,coverage,release)}
        with patch.object(report,'require_frozen_coverage_certificate',return_value={}):
            report._VAL_release(report.Proof(self.root),testconfig,files)
            changed=dict(testconfig,policy='changed')
            with self.assertRaisesRegex(ValueError,'changed a policy/model'):
                report._VAL_release(report.Proof(self.root),changed,files)
            omitted=dict(files);omitted.pop('artifact')
            with self.assertRaisesRegex(ValueError,'omitted or changed an original VAL input'):
                report._VAL_release(report.Proof(self.root),testconfig,omitted)

    def test_raw_TEST_receipt_preserves_formal_recording_and_exact_launcher(self):
        plan,path,manifest,config=self.fixture();job=plan['jobs'][0]
        plan['jobs']=[job];plan['_config']=json.loads(config.read_text());identity=job['identity'];ticks=3
        alignment=dict(clock=dict(ticks=ticks,dt=.05,budget_seconds=10.,simulation_seconds=ticks*.05),
            actual_physics_seconds=ticks*.05,actual_task_seconds=ticks*.05,native_success_latched=False,
            native_terminal_failure_latched=False)
        record=native_completion_record(identity,process_id=42,success=False,fatal=False,ticks=ticks,
            initial_conditions=dict(metric_only=True,policy_use_allowed=False),alignment_receipt=alignment)
        record.update(cleanup_completed=True,evaluation_process_pid=42,entrypoint_sha256=plan['child_source']['sha256'],
            config_sha256=plan['config']['sha256'],manifest_sha256=plan['manifest']['sha256'],
            source_row_sha256=identity['source_row_sha256'],frozen_input_lock=plan['frozen_inputs'],
            source_motion_replayed=True,recorded_video_in_formal_metrics=True,video=dict(CPU_dummy_not_decoded=True),
            compute_seconds=.01,wall_seconds=.02,simulation_seconds=ticks*.05,predictor='cv',
            continuation_sha256='a'*64,continuation_identity=dict(sha256='a'*64),
            unexecuted_candidates_labelled=False,first_selection_policy='geometry',downstream_selection_policy='geometry')
        output=self.root/plan['output'];child=output/'cases'/job['job_id']/'attempt0000/case.json';self.write(child,record)
        log=child.parent.parent/'attempt0000.log';log.write_text('CPU fixture never launched')
        command=['bash','scripts/run_local_sim.sh',plan['child_source']['path'],'--config',plan['config']['path'],
            '--manifest',plan['manifest']['path'],'--frozen-inputs',plan['frozen_inputs']['path'],
            '--split','test','--task',identity['task'],'--raw-episode-id',job['source_raw_episode_id'],
            '--method',identity['method'],'--replica','0','--out',str(child.parent.relative_to(self.root)),'--record-demo']
        record['parent_case_process_receipt']=dict(child_path=str(child.relative_to(self.root)),child_sha256=file_sha(child),
            child_process_id=42,launcher_process_id=41,command=command,log_path=str(log.relative_to(self.root)),
            log_sha256=file_sha(log),plan_sha256=plan['sha256'],parent_source_sha256=plan['parent_source']['sha256'])
        canonical=output/'results'/(job['job_id']+'.json');self.write(canonical,record)
        self.write(output/'status.json',dict(stage='complete',complete=True,completed=2800,expected=2800,
            plan_sha256=plan['sha256'],errors=[],stop_requested=False))
        groups,receipts,status=validation_release._cases(report.Proof(self.root),plan,split='test',expected_cases=2800)
        self.assertEqual(len(groups),1);self.assertIsNone(receipts[0]['parent_raw_episode_id'])
        record['parent_case_process_receipt']['command'].pop();canonical.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'launcher command differs'):
            validation_release._cases(report.Proof(self.root),plan,split='test',expected_cases=2800)

    def test_publication_retry_preserves_unfinished_proof_directory(self):
        out=self.base/'report.json';old=self.base/'report_proofs';old.mkdir();(old/'partial').write_text('keep')
        new=validation_release.proof_directory(out)
        self.assertNotEqual(new,old);self.assertEqual((old/'partial').read_text(),'keep')

    def published_blocks(self):
        video=self.write('video.mp4',dict(CPU_dummy_not_decoded=True));artifact=self.write('artifact',{})
        decoder=self.write('decoder',{})
        for name in ('test_report.py','video_validation.py'):self.write('our_mani/multitask/'+name,{})
        self.write('scripts/finalize_multitask_test_r10.py',{})
        new_entries=[dict(task=task,method=method,raw_episode_id='fixed',formal_metrics_included=True,
            demo_only_replay=False,replay_result_in_formal_statistics=False,video=dict(**self.pointer(video),decoded_frame_count=4))
            for task in TASK_SPECS for method in METHODS]
        demos=self.write('new_demo_index.json',dict(complete=True,expected=70,completed=70,entries=new_entries))
        pointers={}
        for field in ('canonical_cases','recorded_initial_conditions','selection_usage'):
            path=self.write(field+'.json',dict(complete=True,diagnostics_valid=True));pointers[field]=self.pointer(path)
        new=self.write('r10_report.json',dict(schema=report.SCHEMA,complete=True,protocol_acceptance_passed=True,
            expected_source_cases=560,source_cases=560,expected_method_cases=2800,completed_method_cases=2800,
            baseline_executor='V7',demo_index=self.pointer(demos),input_files_sha256={'artifact':file_sha(artifact)},**pointers))
        lock=self.write('oldlock.json',dict(files={'artifact':file_sha(artifact)}))
        old=self.write('r09_report.json',dict(complete=True,protocol_acceptance_passed=True,total_test_runs=400,
            expected_test_runs=400,acceptance_errors=[],baseline_executor='V6',
            parallel_evaluation_inputs='oldlock.json',parallel_evaluation_inputs_sha256=file_sha(lock)))
        entries=[]
        for task in ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed'):
            for method in METHODS:
                name=task+'_'+method;formal=self.write(name+'_formal.json',dict(official_task_success=False))
                replay=None if method=='context_value' else self.write(name+'_replay.json',dict(official_task_success=True))
                receipt=self.write(name+'_receipt.json',{}) if replay else None
                entries.append(dict(task=task,method=method,raw_episode_id='fixed',
                    source='independent_demo_only_replay' if replay else 'formal_rollout_video',
                    formal_metrics_included=replay is None,replay_included_in_formal_statistics=False,
                    formal_record=str(formal.relative_to(self.root)),formal_record_sha256=file_sha(formal),
                    original_formal_child_record=str(formal.relative_to(self.root)),original_formal_child_record_sha256=file_sha(formal),
                    formal_result=dict(official_task_success=False),
                    replay_record=str(replay.relative_to(self.root)) if replay else None,
                    replay_record_sha256=file_sha(replay) if replay else None,
                    recording_receipt=str(receipt.relative_to(self.root)) if receipt else None,
                    recording_receipt_sha256=file_sha(receipt) if receipt else None,
                    replay_result=dict(official_task_success=True) if replay else None,
                    video='video.mp4',video_sha256=file_sha(video),video_validation=dict(decoded_frame_count=4)))
        old_demos=self.write('old_demo_index.json',dict(complete=True,expected=10,completed=10,entries=entries,
            formal_final_report_sha256=file_sha(old),original_formal_statistics_replaced=False,
            replay_results_in_formal_statistics=False,execution_inputs='oldlock.json',execution_inputs_sha256=file_sha(lock)))
        return new,old,old_demos,dict(decoder='decoder',decoder_sha256=file_sha(decoder))

    def test_combined80_preserves_old_replay_outcomes_and_separate_definitions(self):
        new,old,old_demos,decoded=self.published_blocks();out=self.base/'combined.json'
        with patch.object(report,'validate_video',return_value=decoded),\
             patch.object(report,'__file__',str(self.root/'our_mani/multitask/test_report.py')):
            result=report.combine_published_blocks(self.root,new,out,r09_report=old,r09_demos=old_demos)
        self.assertEqual(result['demo_completed'],80);self.assertEqual(result['formal_run_count_only'],3200)
        self.assertEqual(result['blocks']['R09']['formal_runs'],400);self.assertEqual(result['blocks']['R10']['formal_runs'],2800)
        self.assertFalse(result['pooled_cross_protocol_success_rate_reported'])
        replay=next(e for e in result['entries'] if e['protocol_block']=='R09' and e['method']=='cv_geometry')
        self.assertFalse(replay['formal_result']['official_task_success']);self.assertTrue(replay['replay_result']['official_task_success'])
        self.assertFalse(replay['formal_metrics_included']);self.assertFalse(result['old_formal_statistics_modified'])

    def test_combined_index_rechecks_accepted_evidence_after_a_raw_input_change(self):
        new,old,old_demos,decoded=self.published_blocks();(self.root/'artifact').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'proof input SHA differs'):
            report.combine_published_blocks(self.root,new,self.base/'combined.json',r09_report=old,r09_demos=old_demos)


if __name__=='__main__':unittest.main()
