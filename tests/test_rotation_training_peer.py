"""Independent CPU numerical/source/scope controls for rotation experiments."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from our_mani.multitask import fitting,rotation_training_objective as objective
from tests import test_rotation_training_objective as original_tests

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'scripts/train_multitask_context_rotation_r10.py'
OLD=ROOT/'scripts/train_multitask_context_r10.py'
MANIFEST=ROOT/'artifacts/framework/multitask_r10/context_training_view_full_v6/manifest.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


class RotationExperimentIndependentPeerTests(unittest.TestCase):
    def test_unit_loss_all_parameter_gradients_and_actual_forward_are_exact(self):
        model,data,_=original_tests.RotationTrainingObjectiveTests().fixture()
        second=deepcopy(model)
        expected=fitting.loss_per_window(model,data,3).sum()
        actual=objective.loss_per_window(second,data,3).sum()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        first_grads=torch.autograd.grad(expected,tuple(model.parameters()),allow_unused=True)
        second_grads=torch.autograd.grad(actual,tuple(second.parameters()),allow_unused=True)
        self.assertEqual(len(first_grads),len(second_grads))
        for first,last in zip(first_grads,second_grads):
            if first is None:self.assertIsNone(last)
            else:torch.testing.assert_close(first,last,rtol=0,atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_weighted_sampler_retains_exact_original_replacement_rule_without_reading_history(self):
        rows=[]
        class Poison:
            def __array__(self,*args,**kwargs):raise AssertionError('model history was inspected by sampler')
        for parent,size in (('short',2),('long',8)):
            for i in range(size):
                rows.append(dict(source_key=('train','CPU',parent),rotation_mask=np.array([False,True]),
                    targets=dict(angular_velocities=np.array([[99.,99.,99.],[.1,0,0] if i==0 else [0,0,0.]])),
                    history=Poison(),state=Poison()))
        class RecordedRNG:
            def __init__(self):self.rng=np.random.default_rng(6);self.calls=[]
            def choice(self,values,n,*,replace,p):
                self.calls.append(dict(values=list(values),n=n,replace=replace,p=p.copy()))
                return self.rng.choice(values,n,replace=replace,p=p)
            def shuffle(self,values):return self.rng.shuffle(values)
        rng=RecordedRNG();indices=objective.balanced_parent_indices(rows,4,rng,rotation_sampling_multiplier=2.)
        self.assertEqual(len(indices),8)
        self.assertEqual({rows[i]['source_key'][2]:sum(rows[j]['source_key']==rows[i]['source_key'] for j in indices) for i in indices},dict(short=4,long=4))
        for call in rng.calls:
            self.assertEqual(call['replace'],len(call['values'])<4)
            self.assertEqual(call['n'],4)
            # Exactly one moving/observable window in each parent, all remaining
            # windows retain weight1; masked99rad/s is not counted.
            expected=np.ones(len(call['values']));expected[0]=2
            np.testing.assert_array_equal(call['p'],expected/expected.sum())
        for row in rows: self.assertIsInstance(row['history'],Poison)

    def test_numeric_training_controls_and_standard_VAL_calls_are_AST_identical(self):
        old=ast.parse(OLD.read_text());new=ast.parse(NEW.read_text())
        def calls(tree,qualified):
            return [ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Call)
                    and ast.unparse(node.func)==qualified]
        for name in ('SE3ContextNetwork','torch.optim.AdamW','torch.manual_seed','np.random.default_rng',
                     'train_normalization','examples','batch_tensors','evaluate','torch.nn.utils.clip_grad_norm_',
                     'torch.set_num_threads'):
            self.assertEqual(calls(old,name),calls(new,name),name)
        def assignment(tree,name):
            return [ast.dump(node.value,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id==name for target in node.targets)]
        self.assertEqual(assignment(old,'steps'),assignment(new,'steps'))
        self.assertEqual(assignment(old,'score'),assignment(new,'score'))
        self.assertEqual(assignment(old,'best_state'),assignment(new,'best_state'))
        formal=ast.parse((ROOT/'our_mani/multitask/release_preflight.py').read_text())
        context=next(n for n in formal.body if isinstance(n,ast.FunctionDef) and n.name=='_context')
        guard=next(n for n in context.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
                   and 'training_script_sha256' in ast.unparse(n.iter))
        metadata=json.loads((ROOT/'artifacts/framework/multitask_r10/context_se3_full_v6/report.json').read_text())['metadata']
        class SourceInputs:
            def file(self,name,expected):
                actual=sha(ROOT/name)
                if actual!=expected:raise ValueError('actual formal Context source SHA mismatch: '+name)
        def require(value,message):
            if not value:raise ValueError(message)
        code=compile(ast.Module(body=[guard],type_ignores=[]),'<actual formal Context source-guard AST>','exec')
        exec(code,dict(inputs=SourceInputs(),m=metadata,require=require))
        changed=dict(metadata,training_script_sha256=sha(NEW),rotation_objective_sha256=sha(Path(objective.__file__)))
        with self.assertRaisesRegex(ValueError,'train_multitask_context_r10.py'):
            exec(code,dict(inputs=SourceInputs(),m=changed,require=require))

    def test_actual_CPU_CLI_rejects_changed_manifest_and_fixed_budget_before_training(self):
        with tempfile.TemporaryDirectory(prefix='rotation_scope_peer_CPU_',dir=ROOT/'artifacts/framework/multitask_r10') as temp:
            temp=Path(temp);wrong=temp/'different_manifest.json';wrong.write_text('{}\n')
            options=[['--manifest',str(wrong)],['--epochs','1'],['--windows-per-parent','63'],
                     ['--seed','1019'],['--learning-rate','.001']]
            for i,option in enumerate(options):
                out=temp/f'out_{i}'
                command=[sys.executable,str(NEW),'--manifest',str(MANIFEST),'--out',str(out),'--device','cpu',*option]
                result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
                self.assertNotEqual(result.returncode,0,option)
                self.assertIn('same-data comparison requires',result.stderr,option)
                self.assertFalse(out.exists(),option)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
