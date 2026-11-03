"""Check every pair of existing dt grids at all common command-start times."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--series-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Retain prior evidence')
    receipt=json.loads(args.series_audit.read_text());assert receipt['status']=='all_saved_physics_traces_independently_verified'
    cubes=sorted((row for row in receipt['runs'] if row['mode']=='cube'),key=lambda r:r['declared_limits']['dt'],reverse=True)
    loaded=[]
    for row in cubes:
        path=Path(row['run_dir'])/'trace.npz';assert sha(path)==row['artifact_SHA']['trace.npz']
        with np.load(path,allow_pickle=False) as arrays:
            loaded.append({key:arrays[key].copy() for key in ('step','phase','register_target','joint_target')})
    results=[]
    for i,j in itertools.combinations(range(len(cubes)),2):
        coarse,fine=cubes[i],cubes[j];dtc=coarse['declared_limits']['dt'];dtf=fine['declared_limits']['dt']
        ratio=round(dtc/dtf);assert ratio*dtf==dtc
        indices=np.arange(len(loaded[i]['step']))*ratio
        assert (indices*dtf==np.arange(len(indices))*dtc).all()
        bits={key:loaded[i][key].dtype==loaded[j][key].dtype and loaded[i][key].tobytes()==loaded[j][key][indices].tobytes()
              for key in ('phase','register_target','joint_target')}
        assert all(bits.values())
        results.append(dict(coarse_Hz=round(1/dtc),fine_Hz=round(1/dtf),common_step_start_samples=len(indices),
            start_times_bit_identical=True,command_arrays_bit_identical=bits))
    out=dict(status='all_pairwise_common_physical_command_starts_bit_verified',
        source_series_audit=dict(path=str(args.series_audit.resolve()),sha256=sha(args.series_audit)),
        audit_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),pairwise_checks=results,
        audit_new_physics_steps=0,audit_new_forward_passes=0,
        boundary='Compare commands at (step-1)*dt, not state observations at step*dt. Different dynamics are retained.')
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],pairs=len(results),finest_pair=results[-1]),indent=2))


if __name__=='__main__':main()
