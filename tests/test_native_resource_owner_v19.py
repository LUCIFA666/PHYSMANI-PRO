from copy import deepcopy
import unittest
from scripts.run_native_resource_owner_v19_r10 import validate_request,ROOT
from our_mani.multitask.evaluation import file_sha


class Tests(unittest.TestCase):
    def request(self):
        return dict(schema='r10_owned_native_TRAIN_request_v19',job_id='fixture',kind='executor_smoke',
            argv=['bash','scripts/run_local_sim.sh','scripts/run_geometry_bin_v17_r10.py',
                '--split','train','--development-demo','--method','cv_geometry'],
            use_gpu=False,timeout_seconds=300,source_sha256=file_sha(ROOT/'scripts/run_geometry_bin_v17_r10.py'))

    def test_exact_geometry_TRAIN_request_is_accepted(self):
        r=self.request();self.assertIs(validate_request(r),r)

    def test_formal_split_and_prohibited_methods_are_rejected(self):
        for split,method in [('validation','cv_geometry'),('test','physmani'),('train','cv_value'),('train','NGFF'),('train','FASTER')]:
            r=self.request();r['argv'][r['argv'].index('--split')+1]=split;r['argv'][-1]=method
            with self.assertRaises(ValueError):validate_request(r)

    def test_stale_worker_and_false_GPU_allocation_are_rejected(self):
        r=self.request();r['source_sha256']='0'*64
        with self.assertRaises(ValueError):validate_request(r)
        r=self.request();r['argv'][-1]='physmani'
        with self.assertRaises(ValueError):validate_request(r)


if __name__=='__main__':unittest.main()
