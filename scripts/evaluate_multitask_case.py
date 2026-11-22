"""Current legacy16-task entry: shared60second time limit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_multitask_case_60s_20261111_v1 import main

if __name__=='__main__':main()
