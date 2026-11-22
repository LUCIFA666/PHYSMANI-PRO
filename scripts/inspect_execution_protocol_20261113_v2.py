#!/usr/bin/env python3
"""Validate/print an execution envelope; performs no simulator or robot calls."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from our_mani.execution_protocol_20261113_v2.request import ExecutionRequest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path)
    parser.add_argument('--observation-time',type=float);parser.add_argument('--legacy-lookahead-seconds',type=float)
    args=parser.parse_args()
    request=ExecutionRequest.decode(json.loads(args.input.read_text()),observation_time=args.observation_time,
        legacy_lookahead_seconds=args.legacy_lookahead_seconds)
    print(json.dumps(request.to_wire(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
