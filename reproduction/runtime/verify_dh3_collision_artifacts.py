"""Independently audit saved original collision decisions; never rerun a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args();report=json.loads((args.run_dir/'RESULTS.json').read_text())
    assert report['collision_check_executed'] and report['status']=='original_collision_admission_complete'
    assert not report['collision_gates_relaxed'] and not report['empty_pointcloud_substituted']
    for entry in report['output_files']:assert sha(entry['path'])==entry['sha256']
    records=[json.loads(line) for line in (args.run_dir/'candidate_collision_records.jsonl').read_text().splitlines()]
    with np.load(args.run_dir/'collision_outputs.npz',allow_pickle=False) as data:
        masks=np.unpackbits(data['scene_inclusion_masks_packed'],axis=1)[:,:report['input']['scene_count']]
        counts=masks.sum(axis=1)
        assert np.array_equal(counts,data['overlap_point_counts'])
        assert np.array_equal(counts==0,data['admission_mask'])
        indices=np.flatnonzero(data['min_width_index'])
        assert np.array_equal(indices,data['source_indices'])
        assert len(records)==len(data['min_width_index'])==report['stages']['pre_collision_candidates']
        assert [r['pre_collision_index'] for r in records]==list(range(len(records)))
        for j,i in enumerate(indices):
            assert records[i]['voxel_overlap_points']==int(counts[j])
            assert records[i]['admitted']==bool(counts[j]==0)
        assert np.array_equal(data['DH3_grasp_group_array'][counts==0],data['admitted_DH3_grasp_group_array'])
        assert np.array_equal(data['two_finger_grasp_group_array'][counts==0],data['admitted_two_finger_grasp_group_array'])
        assert int((counts==0).sum())==report['stages']['collision_admitted']
        assert int((counts>0).sum())==report['stages']['collision_rejected']
        assert np.isfinite(data['DH3_grasp_group_array']).all()
        summary={'candidates':len(records),'queried_scene_points_per_candidate':masks.shape[1],
                 'zero_overlap_admitted':int((counts==0).sum()),'positive_overlap_rejected':int((counts>0).sum()),
                 'overlap_count_min':int(counts.min()),'overlap_count_max':int(counts.max())}
    slots=report['assets']['slots'];sizes=[]
    for item in slots:
        path=Path(item['path']);assert sha(path)==item['sha256']
        receipt=Path(report['assets']['root'])/'slot_receipts'/(item['key']+'.json')
        assert sha(receipt)==item['slot_receipt_sha256']
        original=json.loads(receipt.read_text())
        assert original['physics_steps']==0 and original['all_five_original_voxel_scales_retained']
        for entry in [original['output_mesh'],*original['pointclouds']]:
            assert sha(entry['path'])==entry['sha256']
        sizes.append(item['point_count'])
    result={'all_passed':True,'verification':'Saved real Open3D point-in-voxel output bits independently reduced and cross-checked with every candidate decision, admitted arrays, source PCL and all original generated asset bytes',
            'source_RESULTS_sha256':sha(args.run_dir/'RESULTS.json'),'summary':summary,
            'asset_slots_verified':len(slots),'minimum_3mm_PCL_points':min(sizes),'maximum_3mm_PCL_points':max(sizes),
            'all_five_scales_and_STL_SHA_verified':True,'no_model_rerun':True,'physical_grasps_executed':0}
    (args.run_dir/'INDEPENDENT_ARTIFACT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
