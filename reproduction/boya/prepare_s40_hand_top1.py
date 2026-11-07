#!/usr/bin/env python3
"""CLI for frozen S40 hand-aware v3 top1 preparation.

The original v3 source and its completed runs remain unchanged. This wrapper
only supplies argument parsing and JSON summary printing around its run().
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

BOYA=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
HELPER_PATH=BOYA/'prepare_s40_port_top1_v3.py'
HELPER_SHA256='38c674f90d25fa4434ae243025c99967fc22f2e496960c7cb54606a96de11844'
if str(BOYA)not in sys.path:sys.path.insert(0,str(BOYA))
import prepare_s40_port_top1_v3 as helper

def validate_helper():
    if (Path(helper.__file__).resolve()!=HELPER_PATH.resolve()
        or hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()!=HELPER_SHA256):
        raise ValueError('Frozen v3 top1 preparation helper changed')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','backend-config','calibration','native-preflight','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=4)
    args=parser.parse_args(argv)
    validate_helper()
    result=helper.run(args)
    print(json.dumps(dict(complete=result['complete'],objects=len(result['objects']),
        top1=len(result['trials']),nulls=len(result['nulls']),physical_steps=0)))
    return result

if __name__=='__main__':main()
