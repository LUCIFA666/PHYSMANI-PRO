"""2 Hz visual decisions with continuous trajectories/MPC, geometry ranking only."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import evaluate_aligned_r05_v2 as runner
from our_mani.aligned.feedback import CachedVisualDynamics,CachedFeedbackInterception
from our_mani.context_dynamics import ContextLatentDynamics
from our_mani.dynamics import ConstantVelocityDynamics

if '--ranking' not in sys.argv or sys.argv[sys.argv.index('--ranking')+1]!='geometry':
    raise ValueError('this control-distribution ablation does not reuse an unmatched success Value')
runner.ContextLatentDynamics=lambda *args,**kwargs:CachedVisualDynamics(ContextLatentDynamics(*args,**kwargs))
runner.ConstantVelocityDynamics=lambda *args,**kwargs:CachedVisualDynamics(ConstantVelocityDynamics(*args,**kwargs))
runner.CandidateInterception=CachedFeedbackInterception
original_hash=runner.local_controller_hash
runner.local_controller_hash=lambda config,predictor:hashlib.sha256((original_hash(config,predictor)+
    hashlib.sha256((ROOT/'our_mani/aligned/feedback.py').read_bytes()).hexdigest()+':camera2Hz:robot20Hz').encode()).hexdigest()
runner.__file__=__file__
original_episode=runner.run_episode
def episode(*args,**kwargs):
    record,frames=original_episode(*args,**kwargs)
    record['ablation']='2Hz_visual_decisions_20Hz_proprioception_trajectory_MPC'
    record['camera_feedback_period_seconds']=.5
    return record,frames
runner.run_episode=episode

if __name__=='__main__':runner.main()
