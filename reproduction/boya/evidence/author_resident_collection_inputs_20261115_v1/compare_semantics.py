"""Read saved resident-vs-reference plans; metadata hashes are kept separate."""
from pathlib import Path
import argparse
import json
import hashlib
import sys

HERE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
import author_collection_batch as B

FIELDS=('source_row_index','source_top500_rank','collision_rank','gesture_id','source_reference_type',
    'score','source_if_flip','source_row15_camera','actual_row15_camera','actual_row17_camera',
    'source_feature2261','source_feature_identity','canonical_cgr480','canonical_cgr480_identity',
    'canonicalization','raw_angle_index','raw_depth_index','raw_width_m','expanded_width_m','raw_depth_m',
    'extra_depth_m','extra_depth_index','total_depth_m','slot_id','slot_sha256','calibrated_reference_width_m',
    'camera_T_palm_before_depth','camera_T_palm','world_T_palm','world_T_camera',
    'normalized_approach_camera','applied_depth_translation_camera_m','action_contract','collision_profile',
    'physical_semantics_validated','task')
METADATA=('collection_candidate_sha256','collection_execution_spec_sha256')

def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    snapshot=B.read(a.comparison);protocol=B.read(snapshot['protocol']['path']);root=Path(snapshot['protocol']['path']).parent
    reference=Path(snapshot['reference_protocol']['path']).parent
    rows=[]
    for scene in snapshot['scenes']:
        i=scene['source_queue_index'];name=f'{i:07d}_{scene["scene_id"]}'
        for g,record in scene['gestures'].items():
            current=root/'scenes'/name/f'g{g}/pool/selected_author_candidate.json'
            previous=reference/'scenes'/name/f'g{g}/pool/selected_author_candidate.json'
            x,y=B.read(current),B.read(previous)
            checks={k:x[k]==y[k]for k in FIELDS}
            arrays={key:all(value['exact']for name,value in content['arrays'].items()if name not in METADATA)
                for key,content in record.items()if isinstance(content,dict)and 'arrays'in content}
            differences={key:[name for name,value in content['arrays'].items()if not value['exact']]
                for key,content in record.items()if isinstance(content,dict)and 'arrays'in content}
            assert all(checks.values()),checks
            assert all(arrays.values()),arrays
            assert all(set(v)<=set(METADATA)for v in differences.values()),differences
            rows.append(dict(scene_id=scene['scene_id'],source_queue_index=i,gesture_id=int(g),
                semantic_fields_exact=len(FIELDS),semantic_checks=checks,numerical_command_and_pool_arrays_exact=arrays,
                nonexact_metadata_fields=differences,raw_index=x['source_row_index'],flip=x['source_if_flip'],
                extra_depth_m=x['extra_depth_m'],slot_id=x['slot_id'],current_candidate=B.binding(current),reference_candidate=B.binding(previous)))
    result=dict(status='pass',source=B.binding(__file__),comparison=B.binding(a.comparison),rows=rows,
        reason_for_hash_differences='Fresh capture/representation paths and source bindings change candidate and execution-spec SHA. Every selected semantic field and actual command array remains equal.',
        new_model_forwards=0,new_planning_calls=0,new_physics_steps=0)
    B.save(a.output,result,exclusive=True);print(json.dumps(dict(status='pass',comparisons=len(rows),semantic_fields_per_trial=len(FIELDS))))
if __name__=='__main__':main()
