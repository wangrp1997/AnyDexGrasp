"""Independent read-only audit of fresh camera, inference binding and execution."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
import author_collection_batch as B
import author_mujoco_control as C
import author_fast_geometry_checks as F


def model_for(ready):
    C.T._backend_path()
    from canondex.planning.model import PlanningModel
    from canondex.robots.reference import load_reference
    plan=B.read(ready['plan']['path']);args=plan['arguments']
    return PlanningModel(Path(args['robot_urdf'])),load_reference(Path(args['reference_root']))


def check_states(rows,scene,planning,reference,legal,substeps=1):
    from author_mujoco_clock import validate
    vertices={o['object_id']:np.asarray(trimesh.load(o['source_mesh'],force='mesh',process=False).vertices)*o['scale'] for o in scene['objects']}
    robot=sorted({'/World/Robot/'+planning.fk.fixed_attachment(n,np.eye(4))[0] for n in planning.fk.links})
    metrics_error=loads_error=palm_error=0.
    for row in rows:
        validate(row['native_time_s'],row['world_step'],substeps)
        state=dict(zip(planning.joints,row['joint_position']))
        palm=planning.fk.transforms(state,reference.base_transform)['palm']
        palm_error=max(palm_error,float(np.max(abs(palm-row['palm_transform']))))
        views=C.object_views(row['contacts'],list(vertices))
        for key,saved in row['objects'].items():
            metrics=F.source_metrics(vertices[key],saved['pose_xyzw'])
            metrics_error=max(metrics_error,max(abs(v-saved[k]) for k,v in metrics.items()))
            for k,v in C.T.object_robot_loads('/World/'+key,views,robot,legal).items():
                loads_error=max(loads_error,float(np.max(abs(np.asarray(v)-saved[k]))))
    if metrics_error>1e-12 or loads_error>1e-12 or palm_error>1e-12:raise ValueError('Saved states disagree with independent source metrics/contact/FK')
    return dict(source_extrema_error_m=metrics_error,load_error_n=loads_error,palm_fk_matrix_error=palm_error)


def audit(root):
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');result=B.read(root/'RESULTS.json');B.verify(p['inputs'])
    for n,b in p['sources'].items():
        if B.sha(root/'source_archive'/n)!=b['sha256']:raise ValueError('Executed source archive changed: '+n)
    C.T._backend_path()
    from author_compiled_kinematics import CompiledKinematics
    from run_pilot_representation import read_frame,array_sha
    import author_collect as COLLECT
    import author_contact_pipeline as M
    from settle_s40_scenes import window_metrics
    from render_s40_observations import ray_diagnostic
    from canondex.grasp.template_gripper_adapter import TemplateFamily
    template=B.read(p['inputs']['template:0']['path']);planning,reference=model_for(template)
    substeps=result['world']['native_steps_per_control']
    with np.load(template['commands']['path']) as z:home=z['initial_position'].copy()
    bank_ids=sorted({o['object_id'] for r in p['rows'] for o in B.read(r['scene']['path'])['objects']})
    source_plan=B.read(template['plan']['path']);args=source_plan['arguments']
    family=TemplateFamily.load(Path(args['package']),Path(args['hand_urdf']),Path(args['hand_scene']),1)
    legal={planning.fk.fixed_attachment(n,np.eye(4))[0] for n in family.hand.links}
    audits=[]
    with CompiledKinematics().installed():
        for index in range(len(p['rows'])):
            directory=root/'scenes'/f'{index:02d}';capture=directory/'capture';scene=B.read(directory/'source_scene.json')
            journal=capture/'initialization.jsonl'
            if not journal.exists():continue
            rows=[json.loads(s) for s in journal.open()];ids=[o['object_id'] for o in scene['objects']]
            states=check_states(rows,scene,planning,reference,legal,substeps)
            windows=[window_metrics(np.array([r['objects'][key]['pose_xyzw'] for r in rows]),np.array([r['objects'][key]['velocity'] for r in rows])) for key in ids]
            saved=B.read(capture/'SETTLING.json')
            if windows!=saved['stability']:raise ValueError('Stability report differs from saved native states')
            item=dict(index=index,capture_steps=len(rows),capture_window_recomputed_exact=True,
                capture_stable=all(w['stable'] for w in windows),capture_states=states,initialization_trace=B.binding(journal))
            if (capture/'observation.json').exists():
                observation=B.read(capture/'observation.json');B.verify(observation['files'])
                points,coords,binding,calibration,files,_=read_frame(dict(observation_json=str(capture/'observation.json')),
                    SimpleNamespace(depth_min=.2,depth_max=.65,voxel_size=.005))
                observed=B.read(capture/'observed_scene.json');meshes=[]
                for o in observed['objects']:
                    mesh=trimesh.load(o['source_mesh'],force='mesh',process=False);mesh.apply_transform(o['world_T_object']);meshes.append(mesh)
                with np.load(capture/'native_rgbd.npz') as z:depth=z['depth_m'].copy()
                ray=ray_diagnostic(trimesh.util.concatenate(meshes),depth,np.asarray(observation['intrinsic_matrix']),np.asarray(observation['T_WC']))
                if not ray['diagnostic_pass']:raise ValueError('Source depth check failed')
                item['rgbd_ray_check_recomputed_pass']=True
                rep=directory/'representation';receipt=B.read(rep/'RESULTS.json')
                COLLECT.validate_cache_receipt(receipt,dict(observation=B.binding(capture/'observation.json'),
                    raw_candidates=B.binding(rep/'raw_camera_candidates.npz'),representation_outputs=B.binding(rep/'representation_outputs.npz')),
                    receipt['raw_candidates']['count'],p['domain']['sensor_adapter'])
                if array_sha(points)!=receipt['input_points_sha256'] or array_sha(coords)!=receipt['input_coords_sha256']:raise ValueError('CGR points differ from frozen camera')
                item['CGR_input_and_receipt_exact']=True
            execution=directory/'execution/TRACE.jsonl'
            if execution.exists():
                ready=B.read(directory/'live/READY.json');B.verify({k:ready[k] for k in ('plan','commands','command_binding','candidate')})
                observed=B.read(ready['scene']['path']);rows=[json.loads(s) for s in execution.open()]
                states=check_states(rows,observed,planning,reference,legal,substeps)
                with np.load(ready['commands']['path']) as z:arr={k:z[k].copy() for k in z.files}
                velocity=C.source_velocity_targets(arr['joint_target'],arr['joint_names'].tolist(),list(reference.arm_home))
                for i,r in enumerate(rows):
                    np.testing.assert_array_equal(r['joint_target'],arr['joint_target'][i]);np.testing.assert_array_equal(r['velocity_target'],velocity[i])
                    if r['phase']!=str(arr['phase'][i]) or r['step']!=i+1:raise ValueError('Original control order changed')
                maxima,first=C.numeric_maxima(rows,planning,list(planning.joints),arr['desired_palm'],reference,legal,ids)
                gates=C.source_gates(maxima,len(rows),len(arr['phase']));lift=C.LIFT.evaluate_lift(rows,ids,.004)
                saved=B.read(directory/'execution/RESULTS.json')
                if gates!=saved['gates'] or maxima!=saved['maxima'] or lift!=saved['evaluation']:raise ValueError('Execution numerical result differs')
                deepest=max((c for r in rows for c in r['contacts']),key=lambda c:-c['separation_m'])
                item.update(execution_steps=len(rows),execution_states=states,original_controls_exact=True,gates=gates,maxima=maxima,
                    diagnostic_valid=all(gates.values()),diagnostic_task_success=all(gates.values()) and lift['task_success'],
                    deepest_contact=deepest,trace=B.binding(execution),mapping_audit=M.audit_mapping(directory/'live'))
                if 'acceptance' in saved:
                    from author_mujoco_lift_policy import evaluate
                    acceptance=evaluate(rows,ids,maxima,len(arr['phase']),stop=saved['stop'])
                    if acceptance!=saved['acceptance']:raise ValueError('Applied lift policy differs from saved full contacts')
                    item.update(acceptance=acceptance,original_diagnostic_valid=all(gates.values()),
                        diagnostic_valid=acceptance['policy_admission_pass'],diagnostic_task_success=acceptance['diagnostic_task_success'])
                returned=directory/'return.jsonl'
                if returned.exists():
                    back=np.r_[arr['joint_target'][::-1],np.linspace(arr['joint_target'][0],home,101)[1:]]
                    speed=C.source_velocity_targets(back,arr['joint_names'].tolist(),list(reference.arm_home))
                    trace=[json.loads(s) for s in returned.open()]
                    wanted=[]
                    for i,r in enumerate(trace):
                        np.testing.assert_array_equal(r['joint_target'],back[i]);np.testing.assert_array_equal(r['velocity_target'],speed[i])
                        q=planning.command_state(dict(zip(arr['joint_names'].tolist(),back[i])))
                        wanted.append(planning.fk.transforms(q,reference.base_transform)['palm'])
                        from author_mujoco_clock import validate
                        validate(r['native_time_s'],r['world_step'],substeps)
                    maxima,first=C.numeric_maxima(trace,planning,list(planning.joints),wanted,reference,set(),bank_ids)
                    gates=C.source_gates(maxima,len(trace),len(back));saved=B.read(directory/'RETURN.json')
                    if gates!=saved['gates'] or maxima!=saved['maxima']:raise ValueError('Return result differs from saved states')
                    item['physical_return']=dict(steps=len(trace),gates=gates,controls_exact=True,trace=B.binding(returned))
            audits.append(item)
    world=result['world']
    if world['native_integration_steps']!=world['steps']*world['native_steps_per_control']:raise ValueError('Native integration cost mismatch')
    return dict(status='read_only_audit_pass',original_result=B.binding(root/'RESULTS.json'),rounds=audits,
        all_physical_admission_pass=len(audits)==len(p['rows']) and all(r['capture_stable'] and r.get('diagnostic_valid',False) for r in audits),additional_physics_steps=0,additional_CGR_forwards=0,
        recorded_control_steps=world['steps'],recorded_native_steps=world['native_integration_steps'],
        new_training_labels=0,production_ready=False,implementation=B.binding(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.root);B.save(a.output,result,exclusive=True)
    print(dict(status=result['status'],rounds=[dict(index=r['index'],stable=r['capture_stable'],gates=r.get('gates')) for r in result['rounds']]),flush=True)
