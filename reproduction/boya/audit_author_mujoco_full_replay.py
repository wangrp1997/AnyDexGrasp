"""Read-only raw-state/contact audit of one MuJoCo diagnostic, zero mj_step.

Reconstruct source bottoms, five-finger loads, preclosure displacement, actual
joint/mimic errors and palm FK from raw records. The original numeric gate
expressions and lift evaluator are reused; PhysX geometry parity is not implied.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import author_mujoco_control as C

B,T,L=C.B,C.T,C.LIFT


def audit(root):
    root=Path(root);p=B.read(root/'PROTOCOL.json');setup=B.read(root/'SETUP.json');result=B.read(root/'RESULTS.json')
    B.verify(p['inputs']);B.verify(p['sources'])
    C.T._backend_path()
    from canondex.planning.model import PlanningModel
    from canondex.robots.reference import load_reference
    from run_template_arm_positive import object_asset
    plan=B.read(p['inputs']['plan']['path']);args=plan['arguments']
    planning=PlanningModel(Path(args['robot_urdf']));ref=load_reference(Path(args['reference_root']))
    scene=B.read(p['inputs']['scene']['path']);ids=[r['object_id']for r in scene['objects']]
    assets={r['object_id']:object_asset(SimpleNamespace(object_root=Path(r['object_root'])),np.asarray(r['world_T_object']),r['scale'])for r in scene['objects']}
    with np.load(p['inputs']['commands']['path'],allow_pickle=False)as z:
        targets=z['joint_target'].copy();phases=z['phase'].tolist();desired=z['desired_palm'].copy();names=z['joint_names'].tolist()
    expected_velocity=C.source_velocity_targets(targets,names,setup['arm_names'])
    rows=[]
    with (root/'TRACE.jsonl').open()as f:
        for line in f:
            rows.append(json.loads(line))
    if len(rows)!=result['physics_steps_returned'] or len(rows)!=result['recorded_frames']:
        raise ValueError('Returned cost/recorded frame counts differ')
    robot_paths=sorted({v for v in setup['mapping'].values()if v.startswith('/World/Robot/')})
    legal=set(setup['legal_hand_owners']);max_bottom_error=0.;max_load_error=0.;max_palm_error=0.;settled={};stop=None;contact_points=0;max_clock_error=0.
    reconstructed=[]
    for i,row in enumerate(rows):
        if row['step']!=i+1 or row['phase']!=phases[i]or abs(row['time_s']-(i+1)*.004)>1e-10:
            raise ValueError('Time/phase/step discontinuity')
        max_clock_error=max(max_clock_error,abs(row['time_s']-(i+1)*.004))
        if not np.array_equal(row['joint_target'],targets[i])or not np.array_equal(row['velocity_target'],expected_velocity[i]):
            raise ValueError('Frozen position/velocity command changed')
        q=dict(zip(setup['measured_joint_names'],row['joint_position']))
        transforms=planning.fk.transforms(q,ref.base_transform)
        max_palm_error=max(max_palm_error,float(abs(transforms['palm']-row['palm_transform']).max()))
        for j,c in enumerate(row['contacts']):
            if c['native_index']!=j or len(c['geom_ids'])!=2 or len(c['body_ids'])!=2:
                raise ValueError('Contact provenance/cardinality mismatch')
            if c['bodies']!=[setup['mapping'][str(b)]for b in c['body_ids']]:
                raise ValueError('Canonical contact body map changed')
            values=np.r_[c['position_world_m'],c['contact_frame_world'],c['wrench_contact_frame'],c['separation_m']]
            if not np.isfinite(values).all()or c['normal_force_n']!=c['wrench_contact_frame'][0]:
                raise ValueError('Nonfinite/changed native contact field')
        contact_points+=len(row['contacts']);views=C.object_views(row['contacts'],ids);states={}
        for key in ids:
            state=row['objects'][key]
            metrics=L.source_metrics(assets[key].mesh.vertices,state['pose_xyzw'])
            loads=T.object_robot_loads('/World/'+key,views,robot_paths,legal)
            max_bottom_error=max(max_bottom_error,abs(metrics['source_bottom_m']-state['source_bottom_m']))
            max_load_error=max(max_load_error,float(abs(np.asarray(loads['finger_loads_n'])-state['finger_loads_n']).max()),
                abs(loads['illegal_robot_support_force_n']-state['illegal_robot_support_force_n']))
            states[key]=dict(state,**metrics,**loads)
        # The original evaluator uses integer step*dt, not accumulated engine
        # clock. Keep raw time untouched; adapt only the derived evaluator row.
        rebuilt=dict(row,objects=states,time_s=(i+1)*.004);reconstructed.append(rebuilt)
        if row['phase']=='settle':settled={k:np.asarray(s['pose_xyzw'][:2])for k,s in states.items()}
        depth=max([0.]+[-c['separation_m']for c in row['contacts']])
        if depth>.005:stop='multi_object_deep_collision_stop'
        pushed=T.preclosure_pushes(states,settled,row['phase'])
        if pushed:stop='source_object_pushed_before_closure:'+','.join(pushed)
        if stop and i!=len(rows)-1:raise ValueError('Execution continued past original physical stop')
    maxima,first=C.numeric_maxima(reconstructed,planning,setup['measured_joint_names'],desired,ref,legal,ids)
    gates=C.source_gates(maxima,len(rows),len(targets));lift=L.evaluate_lift(reconstructed,ids,.004)
    if max_bottom_error>1e-12 or max_load_error>1e-12 or max_palm_error>1e-10:
        raise ValueError('Source geometry/load/FK readback mismatch')
    if 'maxima'in result:
        if maxima!=result['maxima']or gates!=result['gates']or first!=result['first_numeric_gate_failure']:
            raise ValueError('Saved numeric gates differ from raw audit')
        if lift!=result['observed_lift']:
            raise ValueError('Lift audit differs')
    elif not(result['status']=='runtime_error' and 'Complete aligned all-body native trace required'in result.get('error','')):
        raise ValueError('Unexpected runtime failure cannot be recovered offline')
    if stop!=result['physical_stop']or contact_points!=result['contact_points']:
        raise ValueError('Contact/early-stop audit differs')
    if result['new_labels']!=0 or result['training_updates']!=0 or result['production_ready']:
        raise ValueError('Diagnostic must remain excluded from production')
    B.verify(p['inputs']);B.verify(p['sources'])
    out=dict(status='PASS',scope='Raw record integrity and original numeric rules; physical parity/production admission NOT certified',
        frames=len(rows),original_position_and_velocity_rows_exact=len(rows),contact_points=contact_points,
        source_bottom_max_recompute_error_m=max_bottom_error,finger_load_max_recompute_error_n=max_load_error,
        measured_palm_vs_independent_XML_FK_max_error=max_palm_error,gates=gates,maxima=maxima,
        observed_final_lift_success=lift['task_success'],physical_stop=stop,physics_steps_in_audit=0,
        accumulated_engine_clock_max_error_s=max_clock_error,
        evaluation_time_adapter='Integer step*0.004 only in derived evaluator rows; original raw time preserved',
        original_runtime_postprocessing_error_preserved=result.get('error'),
        bindings={n:B.binding(root/n)for n in ('PROTOCOL.json','SETUP.json','RESULTS.json','TRACE.jsonl','model.xml')})
    B.save(root/'RECOMPUTED_EVALUATION.json',dict(gates=gates,maxima=maxima,first_numeric_gate_failure=first,
        observed_lift=lift,physical_stop=stop,full_command_sequence_returned=len(rows)==len(targets),
        diagnostic_task_success=all(gates.values())and stop is None and lift['task_success'],
        exclude_training=True,label_eligible=False,new_labels=0,production_ready=False,
        evaluation_time_adapter=out['evaluation_time_adapter']),exclusive=True)
    B.save(root/'READONLY_AUDIT.json',out,exclusive=True);print(json.dumps(out,indent=2))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root)
