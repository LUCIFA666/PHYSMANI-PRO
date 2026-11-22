"""RGB-D occupied-map ablation; preserve prediction, candidates and MPC."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import evaluate_aligned_r05_v2 as runner
from our_mani.aligned.observed_map import ObservedMapOfficialSimulation,ObservedMapLocalSimulation

if '--ranking' not in sys.argv or sys.argv[sys.argv.index('--ranking')+1]!='geometry':
    raise ValueError('map-source ablation uses geometry ranking, without an unmatched Value')
runner.AlignedOfficialSimulation=ObservedMapOfficialSimulation
runner.AlignedLocalSimulation=ObservedMapLocalSimulation
runner.__file__=__file__
original_episode=runner.run_episode
def episode(*args,**kwargs):
    record,frames=original_episode(*args,**kwargs)
    record['ablation']='RGBD_occupied_voxels_with_current_robot_masks'
    return record,frames
runner.run_episode=episode

if __name__=='__main__':runner.main()
