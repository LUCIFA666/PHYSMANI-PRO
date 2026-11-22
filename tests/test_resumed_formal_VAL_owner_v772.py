"""User authorization and real old/new output provenance boundaries."""
import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from scripts import run_resumed_formal_VAL_owner_v772_r10 as owner


class ResumeOwnerTests(unittest.TestCase):
    def test_imported_rows_keep_original_argv_and_never_mutate_old_plan(self):
        old={'jobs':[{'job_id':'done','argv':['--out','old/done'],'task':'a'},
                     {'job_id':'pending','argv':['--out','old/pending'],'task':'a'}]}
        before=deepcopy(old)
        jobs=[dict(old['jobs'][0],argv=['--out','fresh/done']),dict(old['jobs'][1],argv=['--out','fresh/pending'])]
        control={'completed_job_ids':['done'],'previous_runtime':'old_runtime'}
        result=owner.resumed_jobs(jobs,old,control,'new_runtime')
        self.assertEqual(result[0]['argv'],['--out','old/done'])
        self.assertEqual(result[0]['evidence_runtime'],'old_runtime')
        self.assertTrue(result[0]['imported_completed'])
        self.assertEqual(result[1]['argv'],['--out','fresh/pending'])
        self.assertFalse(result[1]['imported_completed'])
        result[0]['argv'].append('changed')
        self.assertEqual(old,before)

    def fixture(self,root):
        paths=['suite.json','suite-lock.json','sources.json','old/launch.json','old/plan.json','old/status.json','old/terminal.json',
               'stop.json','resume.json','owner.py','auditor.py']
        for name in paths:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        (root/'resume.json').write_text(json.dumps({'user_authorized_resume':True,'revokes_previous_stop':True}))
        c=dict(schema='r10_user_authorized_formal_VAL_resume_control_v772',user_authorized_resume=True,split='validation',expected_cases=945,previous_runtime='old')
        mapping={'suite':'suite.json','suite_lock':'suite-lock.json','source_inputs':'sources.json','resume_owner':'owner.py','resume_auditor':'auditor.py',
                 'previous_launch':'old/launch.json','previous_plan':'old/plan.json','previous_status':'old/status.json','previous_terminal':'old/terminal.json',
                 'user_stop_record':'stop.json','user_resume_record':'resume.json'}
        c.update({k:owner.pointer(root/v) for k,v in mapping.items()})
        (root/'control.json').write_text(json.dumps(c))
        (root/'lock.json').write_text(json.dumps(dict(schema='r10_user_authorized_formal_VAL_resume_inputs_v772',approved_for_execution=True,files={})))
        return c

    def test_resume_needs_explicit_new_authorization_and_unchanged_source_scope(self):
        with (tempfile.TemporaryDirectory() as tmp,patch.object(owner,'ROOT',Path(tmp)),
              patch.object(owner,'__file__',str(Path(tmp)/'owner.py')),
              patch.object(owner,'verify_inputs',lambda *_:None),
              patch.object(owner,'locked_pointer',lambda r,f,p:r/p['path'])):
            root=Path(tmp);c=self.fixture(root)
            args=('control.json','lock.json','suite.json','suite-lock.json','sources.json')
            accepted,_=owner.load_resume_control(*args)
            self.assertTrue(accepted['user_authorized_resume'])
            c['user_authorized_resume']=False;(root/'control.json').write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'user-authorized'):
                owner.load_resume_control(*args)
            c['user_authorized_resume']=True;(root/'control.json').write_text(json.dumps(c))
            (root/'other-sources.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'source scope changed'):
                owner.load_resume_control(*args[:-1],'other-sources.json')
            (root/'resume.json').write_text(json.dumps({'user_authorized_resume':True,'revokes_previous_stop':False}))
            with self.assertRaisesRegex(ValueError,'explicit user resume'):
                owner.load_resume_control(*args)


if __name__=='__main__':unittest.main()

