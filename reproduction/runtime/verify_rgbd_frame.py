"""Independently verify saved original-network/parser output consistency."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    result=json.loads((args.run_dir/'RESULTS.json').read_text())
    assert result['full_representation_forward'] and result['checkpoint']['strict_state_dict_load']
    with np.load(args.run_dir/'representation_outputs.npz',allow_pickle=False) as rep, np.load(args.run_dir/'raw_camera_candidates.npz',allow_pickle=False) as pred:
        candidates=pred['candidates'];features=pred['candidate_features']
        raw=rep['raw_CGR_480'][0]
        assert np.array_equal(raw,features[:,240:720])
        assert np.array_equal(candidates[:,12:15],rep['stage2_seed_xyz'][0])
        bilateral=np.minimum(rep['stage3_grasp_scores'][0,:,:24,:],rep['stage3_grasp_scores'][0,:,24:,:])
        reference=bilateral.max(axis=1).max(axis=1)
        assert np.array_equal(reference,candidates[:,0])
        rotations=candidates[:,3:12].reshape(-1,3,3)
        orthogonal_error=float(np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3))))
        determinant_error=float(np.max(np.abs(np.linalg.det(rotations)-1)))
        assert orthogonal_error<1e-5 and determinant_error<1e-5
        assert np.isfinite(candidates).all() and np.isfinite(features).all() and np.isfinite(raw).all()
        checks={'raw_CGR_equals_original_parser_feature_slice':True,'raw_candidates_equal_network_seed_positions':True,
            'candidate_scores_equal_independent_bilateral_score_reduction':True,
            'rotation_orthogonality_max_error':orthogonal_error,'rotation_determinant_max_error':determinant_error,
            'all_saved_arrays_finite':True,'candidate_count':len(candidates),
            'candidate_score_range':[float(candidates[:,0].min()),float(candidates[:,0].max())],
            'candidate_width_range_m':[float(candidates[:,1].min()),float(candidates[:,1].max())],
            'candidate_depth_values_m':np.unique(candidates[:,2]).tolist(),
            'raw_CGR_shape':list(raw.shape)}
    receipt={'scope':'saved single-frame software output consistency, not grasp feasibility or success',
        'run_dir':str(args.run_dir.resolve()),'all_checks_passed':True,'checks':checks,
        'artifact_sha256':{name:sha(args.run_dir/name) for name in ['RESULTS.json','representation_outputs.npz','raw_camera_candidates.npz']}}
    (args.run_dir/'INDEPENDENT_OUTPUT_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
