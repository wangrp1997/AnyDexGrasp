"""Saved state comparison; no simulator, model, or additional physical trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read(p):return json.loads(Path(p).read_text())
def binding(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def first(flags, phases):
    where=np.flatnonzero(flags)
    return dict(step=int(where[0]+1),phase=str(phases[where[0]])) if len(where) else None


def compare(root, reference):
    rows=[]
    for directory in sorted((Path(root)/'scenes').iterdir()):
        for gesture in (1,6,7,8):
            current=directory/f'g{gesture}/execution';old=Path(reference)/'scenes'/directory.name/f'g{gesture}/execution'
            if not (current/'trace.npz').exists():continue
            with np.load(current/'trace.npz',allow_pickle=False) as z:n={k:z[k].copy() for k in z.files}
            with np.load(old/'trace.npz',allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
            delta=abs(n['joint_pos']-o['joint_pos']);maximum=delta.max(axis=1)
            phases=n['phase'];blocks=list(dict.fromkeys(phases));mi=np.unravel_index(delta.argmax(),delta.shape)
            initial_n=read(current/'initial_state_audit.json');initial_o=read(old/'initial_state_audit.json')
            setup_n=read(current/'MULTI_OBJECT_SETUP.json')['entry'];setup_o=read(old/'MULTI_OBJECT_SETUP.json')['entry']
            body_n=read(current/'MULTI_OBJECT_TRACE.json')['rows'];body_o=read(old/'MULTI_OBJECT_TRACE.json')['rows']
            bodies={}
            for key in body_n[0]['objects']:
                a=np.asarray([r['objects'][key]['pose_xyzw'] for r in body_n]);b=np.asarray([r['objects'][key]['pose_xyzw'] for r in body_o])
                translation=np.linalg.norm(a[:,:3]-b[:,:3],axis=1)
                angle=(Rotation.from_quat(a[:,3:])*Rotation.from_quat(b[:,3:]).inv()).magnitude()
                bottom=np.asarray([r['objects'][key]['source_bottom_m'] for r in body_n])-np.asarray([r['objects'][key]['source_bottom_m'] for r in body_o])
                bodies[key]=dict(max_translation_euclidean_m=float(translation.max()),
                    max_translation_component_m=float(abs(a[:,:3]-b[:,:3]).max()),
                    max_rotation_angle_rad=float(angle.max()),max_rotation_angle_deg=float(np.rad2deg(angle.max())),
                    max_source_bottom_delta_m=float(abs(bottom).max()),
                    first_translation_over_10um=first(translation>1e-5,phases),
                    phase_max_translation_euclidean_m={str(p):float(translation[phases==p].max()) for p in blocks},
                    phase_max_rotation_angle_rad={str(p):float(angle[phases==p].max()) for p in blocks})
            label=read(current.parent/'LIFT_PIPELINE_EXECUTION.json')['label'];old_label=read(old.parent/'LIFT_PIPELINE_EXECUTION.json')['label']
            rows.append(dict(scene_id=directory.name,gesture_id=gesture,reused=read(current/'evaluation.json')['resident_scene']['reused'],
                current_trace=binding(current/'trace.npz'),original_trace=binding(old/'trace.npz'),
                current_allbody_trace=binding(current/'MULTI_OBJECT_TRACE.json'),original_allbody_trace=binding(old/'MULTI_OBJECT_TRACE.json'),
                current_allbody_initial=binding(current/'MULTI_OBJECT_SETUP.json'),original_allbody_initial=binding(old/'MULTI_OBJECT_SETUP.json'),
                measured_initial_all_body_equal={key:{field:np.array_equal(setup_n[key][field],setup_o[key][field])
                    for field in ('pose_xyzw','velocity')} for key in setup_n},
                measured_initial_q_exact=np.array_equal(initial_n['measured_joint_positions'],initial_o['measured_joint_positions']),
                measured_initial_qdot_exact=np.array_equal(initial_n['measured_joint_velocities'],initial_o['measured_joint_velocities']),
                first_joint_difference={str(v):first(maximum>v,phases) for v in (1e-7,1e-5,.001,.01,.1)},
                phase_max_joint_delta_rad={str(p):float(delta[phases==p].max()) for p in blocks},
                max_joint_delta=dict(step=int(mi[0]+1),phase=str(phases[mi[0]]),name=str(n['joint_names'][mi[1]]),delta_rad=float(delta[mi])),
                objects=bodies,label=label,original_label=old_label,label_equal=label==old_label,
                original_gates=read(current/'LIFT_EVALUATION.json')['gates']))
    return dict(status='complete_readonly_comparison',rows=rows,reused_job_count=sum(r['reused'] for r in rows),new_physics_steps=0,
        scope='Measured original initial states and all source-body saved trajectories; labels/gates compared, no new labels generated',
        claim='Only saved state comparisons are reported; label agreement alone does not prove unbiased collection or trajectory equivalence',
        solver_cache_or_other_cause_proven=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    v=compare(a.run_dir,a.reference);a.output.write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps([dict(scene=r['scene_id'],g=r['gesture_id'],first_joint_difference=r['first_joint_difference']['1e-07'],
        max_joint_delta=r['max_joint_delta'],max_body_translation_m=max(o['max_translation_euclidean_m'] for o in r['objects'].values()),
        max_body_rotation_rad=max(o['max_rotation_angle_rad'] for o in r['objects'].values()),label=r['label'],label_equal=r['label_equal']) for r in v['rows']],indent=2))
