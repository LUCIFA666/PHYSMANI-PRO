"""Model inputs cannot change when only private future supervision changes."""
import copy
import unittest
import numpy as np
from our_mani.cube_motion_residual_20261104_v1.data import compile_episode
from our_mani.cube_training_cycle_20261111_v1.common import plan,parents,heldout_parents

class SplitAndInputContracts(unittest.TestCase):
    def test_whole_parent_roles_are_disjoint_and_balanced(self):
        p=plan();groups=[set(parents(p,r))for r in('fit','selection','validation','test')]
        self.assertEqual([len(g)for g in groups],[12,4,4,4])
        for i,a in enumerate(groups):
            for b in groups[:i]:self.assertFalse(a&b)
        self.assertEqual(len(heldout_parents(p)),8)
        for role in('fit','selection','validation','test'):
            tasks=[c['task'].endswith('high_speed')for c in p['cases']if c['role']==role]
            self.assertEqual(sum(tasks),len(tasks)//2)
    def test_future_labels_never_change_motion_inputs(self):
        public=[dict(timestamp=i*.05,available=True,position_world=[.2,i*.002,.775],confidence=.9)for i in range(30)]
        labels=[dict(timestamp=i*.05,target_position_world=[.2,i*.003,.775],free_motion=True,segment_id=0)for i in range(30)]
        changed=copy.deepcopy(labels)
        for row in changed:row['target_position_world'][0]+=.1
        a,_=compile_episode(public,labels,parent_id='train')
        b,_=compile_episode(public,changed,parent_id='train')
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            for key in x['inputs']:np.testing.assert_array_equal(x['inputs'][key],y['inputs'][key])
        self.assertFalse(np.array_equal(a[0]['targets']['target_positions'],b[0]['targets']['target_positions']))

if __name__=='__main__':unittest.main()
