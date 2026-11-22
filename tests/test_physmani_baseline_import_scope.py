from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from our_mani.multitask import baseline_import_scope as scope
from our_mani.multitask import baseline_dependencies as inventory
from our_mani.multitask import baseline_component as component
from our_mani.multitask.evaluation import file_sha


class ExactDelayedImportScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        project=Path(__file__).resolve().parents[1]
        for name in (*scope.SCOPES,'scripts/evaluate_multitask_case_r10.py',
                     'our_mani/multitask/baseline_import_scope.py'):
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/name,target)
        self.override=patch.object(scope,'__file__',str(self.root/'our_mani/multitask/baseline_import_scope.py'));self.override.start()
        for module in scope.PRUNED_MODULES:
            target=self.root.joinpath(*module.split('.')).with_suffix('.py')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('from .pipeline import Example\n' if module.endswith('factory') else
                'from .interaction import Example\n' if module.endswith(('pipeline','artifact_fitting')) else '# excluded actor/offline code\n')
        evidence=self.root/'ROOT_evidence.json';evidence.write_text(json.dumps(dict(
            schema='r10_ROOT_PhysMani_VAL_ours_only_import_edge_review_v1',
            reviewed_for=dict(split='validation',method='physmani',replica=0,record_demo=False,demo_only=False),
            required_bound_source_sha256={name:file_sha(self.root/name)
                for name in (*scope.SCOPES,'scripts/evaluate_multitask_case_r10.py')})))
        self.value=scope.draft_review(self.root,{'path':'ROOT_evidence.json','sha256':file_sha(evidence)})
        self.review=self.root/'review.json'

    def tearDown(self):self.override.stop();self.temp.cleanup()

    def pointer(self):
        self.review.write_text(json.dumps(self.value));return dict(path='review.json',sha256=file_sha(self.review))

    def test_exact_function_nodes_keep_owners_and_exclude_targets_without_module_top_pruning(self):
        value=scope.load_review(self.root,self.pointer(),require_reviewed=False)
        self.assertEqual(len(value['excluded_import_nodes']),3)
        self.assertTrue(all(row['scope']!='<module>' for row in value['excluded_import_nodes']))
        found,_=inventory.local_import_closure(self.root,[self.root/name for name in scope.SCOPES],
            excluded_import_nodes=value['excluded_import_nodes'])
        self.assertTrue(all(self.root/name in found for name in scope.SCOPES))
        for module in scope.PRUNED_MODULES:self.assertNotIn(self.root.joinpath(*module.split('.')).with_suffix('.py'),found)
        plain,_=inventory.local_import_closure(self.root,[self.root/name for name in scope.SCOPES])
        self.assertIn(self.root/'our_mani/multitask/pipeline.py',plain)
        self.assertIn(self.root/'our_mani/multitask/interaction.py',plain)

    def test_changes_to_excluded_live_implementation_do_not_invalidate_owner_proof(self):
        pointer=self.pointer()
        for module in scope.PRUNED_MODULES:self.root.joinpath(*module.split('.')).with_suffix('.py').write_text('# future ours-only training improvement\n')
        scope.load_review(self.root,pointer,require_reviewed=False)
        owner=self.root/'scripts/run_multitask_fresh_queue_r10.py';owner.write_text(owner.read_text()+'\n# changed owner\n')
        with self.assertRaises(ValueError):scope.load_review(self.root,pointer,require_reviewed=False)
        self.value=scope.draft_review(self.root,self.value['evidence'])
        with self.assertRaisesRegex(ValueError,'ROOT parent evidence is stale'):
            scope.load_review(self.root,self.pointer(),require_reviewed=False)

    def test_widened_node_module_scope_method_split_line_or_missing_node_rejected(self):
        original=deepcopy(self.value)
        for change in ('module','scope','split','method','line','missing'):
            self.value=deepcopy(original)
            if change=='module':self.value['excluded_modules'].append('our_mani.multitask.simulation')
            elif change=='scope':self.value['excluded_import_nodes'][0]['scope']='<module>'
            elif change in ('split','method'):self.value[change]='test' if change=='split' else 'context_value'
            elif change=='line':self.value['excluded_import_nodes'][0]['line']+=1
            else:self.value['excluded_import_nodes'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):scope.load_review(self.root,self.pointer(),require_reviewed=False)

    def test_draft_is_not_approved_and_unexpected_actual_import_is_rejected(self):
        pointer=self.pointer()
        with self.assertRaises(ValueError):scope.load_review(self.root,pointer)
        self.value['reviewed_before_execution']=True;scope.load_review(self.root,self.pointer())
        owner=self.root/'scripts/evaluate_multitask_case_r10.py'
        contract=self.root/'contract.json';contract.write_text(json.dumps(dict(dependency_groups={'local_sources':{str(owner):file_sha(owner)}})))
        backend=component.ProjectedBaselineBackend(root=self.root,execution_contract=dict(path='contract.json',sha256=file_sha(contract)),
            inner=SimpleNamespace());backend.witness={'imported_dependency_files_sha256':{},'loaded_native_files_sha256':{}}
        target=self.root/'our_mani/multitask/interaction.py'
        with patch.dict(sys.modules,{'unexpected_excluded_module':SimpleNamespace(__file__=str(target))},clear=True),\
             self.assertRaisesRegex(ValueError,'actual imported module is outside frozen closure'):
            backend._imports()
        with patch.dict(sys.modules,{'unknown_nonfile_import':SimpleNamespace(__file__=str(self.root/'missing.py'))},clear=True),\
             self.assertRaisesRegex(ValueError,'actual imported module file is missing'):
            backend._imports()


if __name__=='__main__':unittest.main()
