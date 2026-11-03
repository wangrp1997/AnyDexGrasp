"""Apply the author's exact post-collision argsort/top10/top1 data selection.

No model, collision gate, hardware or physics executes. Equal-score ordering
uses the author's unchanged NumPy argsort expression, without stable-sort fixes.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from run_official_dh3_collision import ROOT,REV,sha,original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--collision-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--grasp-api-source',type=Path,default=ROOT/'outputs/native_deps/graspnetAPI_source/graspnetAPI/grasp.py')
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Use a fresh selection directory')
    args.output_dir.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    receipt=json.loads((args.collision_dir/'RESULTS.json').read_text())
    assert receipt['collision_check_executed'] and receipt['status']=='original_collision_admission_complete'
    data_path=args.collision_dir/'collision_outputs.npz'
    source=ROOT/'robot_dh3.py'
    api=original(args.grasp_api_source,{'GRASP_ARRAY_LEN','RECT_GRASP_ARRAY_LEN','EPS','Grasp','GraspGroup'},{'np':np,'copy':copy})
    hand_scope=original(ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py',
        {'grasp_types','MIN_GRASP_WIDTH','MAX_GRASP_WIDTH','DH3_DEFAULT_DEPTH','DH3_ARRAY_LEN','DH3Grasp','DH3GraspGroup'},
        {'np':np,'copy':copy,'os':os,'json':json,'math':math})
    with np.load(data_path,allow_pickle=False) as saved:
        hand=hand_scope['DH3GraspGroup'](saved['admitted_DH3_grasp_group_array'].copy())
        two=api['GraspGroup'](saved['admitted_two_finger_grasp_group_array'].copy())
        features=saved['admitted_grasp_features'].copy()
        admitted_source=saved['source_indices'][saved['admission_mask']]
    report={'scope':'Original post-collision DH3 top10 and final top1 prediction, no action',
        'source_revision':REV,'source':str(source),'source_sha256':sha(source),
        'collision_RESULTS_sha256':sha(args.collision_dir/'RESULTS.json'),'collision_outputs_sha256':sha(data_path),
        'collision_admitted_candidates':len(hand),'physical_grasps_executed':0,'simulation_run':False,'robot_connection':False,
        'network_rerun':False,'collision_rerun':False,'grasp_success_claim':False,'numpy_version':np.__version__}
    if len(hand)==0:
        report.update(status='no_candidate_after_original_collision',selected=False,top10_count=0)
    else:
        expected=[
            ('index_score','np.argsort(DH3_ggarray.scores)[::-1]'),
            ('two_fingers_ggarray_object_ids','two_fingers_ggarray_object_ids[index_score][0:10]'),
            ('DH3_ggarray','DH3_ggarray[index_score][0:10]'),
            ('two_fingers_ggarray','two_fingers_ggarray[index_score][0:10]'),
            ('grasp_features','grasp_features[index_score][0:10]'),
            ('DH3_grasp_used','DH3_ggarray[0]'),
            ('two_fingers_grasp_used','two_fingers_ggarray[0]'),
            ('grasp_features_used','grasp_features[0]')]
        assignments=[]
        for target,value in expected:
            matches=[node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node,ast.Assign)
                and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==target
                and ast.unparse(node.value)==value]
            assert len(matches)==1,(target,value,len(matches))
            assignments.append(matches[0])
        scope={'np':np,'DH3_ggarray':hand,'two_fingers_ggarray':two,'grasp_features':features,
               'two_fingers_ggarray_object_ids':two.object_ids.copy()}
        exec(compile(ast.Module(body=assignments,type_ignores=[]),str(source),'exec'),scope)
        selected_hand=scope['DH3_grasp_used'];selected_two=scope['two_fingers_grasp_used']
        top_indices=scope['index_score'][:10]
        assert np.array_equal(scope['DH3_ggarray'].grasp_group_array,hand.grasp_group_array[np.argsort(hand.scores)[::-1]][:10])
        assert np.array_equal(scope['grasp_features_used'],features[top_indices[0]])
        path=args.output_dir/'selected_original_DH3_pose.npz'
        np.savez_compressed(path,top10_DH3_grasp_group_array=scope['DH3_ggarray'].grasp_group_array,
            top10_two_finger_grasp_group_array=scope['two_fingers_ggarray'].grasp_group_array,
            top10_grasp_features=scope['grasp_features'],top10_object_ids=scope['two_fingers_ggarray_object_ids'],
            top10_admitted_array_indices=top_indices,top10_pre_collision_indices=admitted_source[top_indices],
            top1_DH3_grasp_array=selected_hand.grasp_array,top1_two_finger_grasp_array=selected_two.grasp_array,
            top1_grasp_features=scope['grasp_features_used'])
        report.update(status='original_final_hand_pose_selected',selected=True,top10_count=len(top_indices),
            original_selection_assignments=[{'line':node.lineno,'expression':ast.unparse(node)} for node in assignments],
            original_assignments_unchanged=True,
            selected_pre_collision_index=int(admitted_source[top_indices[0]]),
            selected_admitted_array_index=int(top_indices[0]),
            selected_hand={'score':float(selected_hand.score),'type':int(selected_hand.grasp_type),
                'width_m':float(selected_hand.width),'depth_m':float(selected_hand.depth),
                'actuator2':selected_hand.angle.tolist(),'rotation_matrix':selected_hand.rotation_matrix.tolist(),
                'translation_m':selected_hand.translation.tolist()},
            output={'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)},
            boundary='Predicted hand datum is in the input camera optical frame; depth is separately stored as in the author API. Robot calibration, wrist approach, controller, physics and lifting are not executed.')
    report['total_seconds']=time.monotonic()-start
    (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
