"""Independent zero-integration review of the complete MJX collection journal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import author_collection_batch as B


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def audit_rays(mesh,depth,K,T,saved):
    """Independent full-triangle Moller scan, same 96 pixel-center rays."""
    import author_fast_geometry_checks as F
    vertices=(mesh.vertices-T[:3,3])@T[:3,:3];uv=vertices[:,:2]/vertices[:,2:3]@K[:2,:2].T+K[:2,2]
    lo=np.maximum(np.floor(uv.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(uv.max(0)).astype(int),[depth.shape[1]-1,depth.shape[0]-1])
    rng=np.random.default_rng(20261105);pixels=np.column_stack([rng.integers(lo[0],hi[0]+1,96),rng.integers(lo[1],hi[1]+1,96)])
    np.testing.assert_array_equal(pixels,saved['pixels_xy'])
    rays=np.c_[(pixels+.5-K[:2,2])/np.diag(K)[:2],np.ones(96)]@T[:3,:3].T
    triangles=np.asarray(mesh.triangles);answer=[F.first_depth(triangles,T[:3,3],ray)for ray in rays]
    expected=np.array([a[0]for a in answer]);hit=np.array([a[1]for a in answer]);observed=depth[pixels[:,1],pixels[:,0]]
    previous=np.array([np.inf if x is None else x for x in saved['expected_source_first_depth_m']])
    np.testing.assert_allclose(expected,previous,rtol=0,atol=1e-9);np.testing.assert_array_equal(hit,saved['source_first_hit_is_object'])
    valid=np.isfinite(expected)&np.isfinite(observed)&(expected>.1)&(expected<1.)&(observed>.1)&(observed<1.)
    difference=np.abs(expected[valid]-observed[valid])
    passed=bool(hit.sum()>=2 and np.median(difference)<.002 and np.quantile(difference,.9)<.004)
    return dict(diagnostic_pass=passed,full_source_triangles=len(triangles),all_original_rays=96,
        max_difference_from_runtime_ray_m=float(np.max(np.abs(expected[np.isfinite(expected)]-previous[np.isfinite(expected)]))))


def audit(root):
    import author_resident_prepare as P
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');result=B.read(root/'RESULTS.json')
    B.verify(p['inputs']);B.verify(p['sources'])
    for name,source in p['sources'].items():
        if B.sha(root/'source_archive'/name)!=source['sha256']:raise ValueError('Executed archive changed')
    P.bind_backend(dict(backend=p['inputs']['backend']['path']))
    import author_mujoco_control as C
    import author_fast_geometry_checks as F
    import author_mujoco_sample as SAMPLE
    from author_mjx_live_batch import CONTACT_FIELDS,clock_table
    from author_compiled_kinematics import CompiledKinematics
    from audit_author_mujoco_collection import model_for
    from settle_s40_scenes import window_metrics
    from run_pilot_representation import read_frame,array_sha
    from render_s40_observations import ray_diagnostic
    import author_collect as COLLECT
    import author_contact_pipeline as M
    import trimesh
    template=B.read(p['inputs']['template:0']['path']);planning,reference=model_for(template)
    with np.load(template['commands']['path'])as z:home=z['initial_position'].copy();names=z['joint_names'].tolist()
    import mujoco
    receipt=B.read(root/'MODEL_RECEIPT.json');B.verify({'model':receipt['model']})
    model=mujoco.MjModel.from_xml_path(receipt['model']['path'])
    qids=C.R.joint_addresses(model,list(planning.joints));vids=model.jnt_dofadr[[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)for n in planning.joints]]
    palm_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'palm');world=result['world'];count=world['worlds']
    bank=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.jnt_bodyid[j]))for j in range(model.njnt)if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
    mapping=C.body_mapping(model,planning,bank)
    source_plan=B.read(template['plan']['path'])
    from canondex.grasp.template_gripper_adapter import TemplateFamily
    a=source_plan['arguments'];family=TemplateFamily.load(Path(a['package']),Path(a['hand_urdf']),Path(a['hand_scene']),1)
    legal={planning.fk.fixed_attachment(n,np.eye(4))[0]for n in family.hand.links}
    robot=sorted({v for v in mapping.values()if v.startswith('/World/Robot/')})
    wanted={};audits=[];clocks=clock_table(p['max_global_control_ticks']);vertices={}
    def states(rows,scene,lane):
        ids=[o['object_id']for o in scene['objects']];metrics_error=load_error=palm_error=0.
        for o in scene['objects']:
            if o['object_id']not in vertices:vertices[o['object_id']]=np.asarray(trimesh.load(o['source_mesh'],force='mesh',process=False).vertices)*o['scale']
        for row in rows:
            step=row['world_step'];np.testing.assert_array_equal(np.float32(row['native_time_s']),clocks[step])
            fk=planning.fk.transforms(dict(zip(planning.joints,row['joint_position'])),reference.base_transform)['palm']
            palm_error=max(palm_error,float(np.max(np.abs(fk-np.asarray(row['palm_transform'])))))
            views=C.object_views(row['contacts'],ids)
            for key,saved in row['objects'].items():
                metrics=F.source_metrics(vertices[key],saved['pose_xyzw'])
                metrics_error=max(metrics_error,max(abs(value-saved[k])for k,value in metrics.items()))
                for key_,value in C.T.object_robot_loads('/World/'+key,views,robot,legal).items():load_error=max(load_error,float(np.max(np.abs(np.asarray(value)-saved[key_]))))
            core=[row['joint_position'],row['joint_velocity'],row['contacts'],row['native_post_step_palm_transform'],row['native_time_s']]
            key=(step,lane)
            if key in wanted:raise ValueError('Duplicate phase record for one world tick')
            wanted[key]=digest(core)
        if max(metrics_error,load_error,palm_error)>1e-12:raise ValueError('Source extrema/contact loads/XML FK disagree')
        return dict(rows=len(rows),source_extrema_error_m=metrics_error,load_error_n=load_error,CPU_kinematic_palm_error=palm_error)
    with CompiledKinematics().installed():
        for spec in p['rows']:
            index=spec['index'];directory=root/'scenes'/f'{index:02d}';record=B.read(directory/'RECORD.json')
            item=dict(index=index,status=record['status'],label=record['label'])
            if not spec.get('scene'):audits.append(item);continue
            scene=B.read(spec['scene']['path']);ids=[o['object_id']for o in scene['objects']];lane=index%count
            path=directory/'capture/initialization.jsonl'
            if not path.exists():audits.append(item);continue
            rows=[json.loads(line)for line in path.open()];item['capture_states']=states(rows,scene,lane)
            stable=[window_metrics(np.array([r['objects'][k]['pose_xyzw']for r in rows]),np.array([r['objects'][k]['velocity']for r in rows]))for k in ids]
            saved=B.read(directory/'capture/SETTLING.json')
            if stable!=saved['stability']:raise ValueError('Stable window differs from native trace')
            support={k:sum(any('/World/'+k in c['bodies']and '/World/ground'in c['bodies']and c['normal_force_n']>.001 for c in r['contacts'])for r in rows[-250:])for k in ids}
            if support!=saved['supported_ticks']:raise ValueError('Support window differs')
            item['capture_stable']=all(s['stable']for s in stable)and all(support.values())
            capture=directory/'capture';observation=capture/'observation.json'
            if observation.exists():
                obs=B.read(observation);B.verify(obs['files'])
                points,coords,_,_,_,_=read_frame(dict(observation_json=str(observation)),SimpleNamespace(depth_min=.2,depth_max=.65,voxel_size=.005))
                measured=B.read(capture/'observed_scene.json');meshes=[]
                for o in measured['objects']:
                    mesh=trimesh.load(o['source_mesh'],force='mesh',process=False);mesh.apply_transform(o['world_T_object']);meshes.append(mesh)
                with np.load(capture/'native_rgbd.npz')as z:depth=z['depth_m']
                ray=audit_rays(trimesh.util.concatenate(meshes),depth,np.asarray(obs['intrinsic_matrix']),np.asarray(obs['T_WC']),B.read(capture/'RAY_CHECK.json'))
                if not ray['diagnostic_pass']:raise ValueError('Independent source ray check failed')
                rep=directory/'representation';r=B.read(rep/'RESULTS.json')
                COLLECT.validate_cache_receipt(r,dict(observation=B.binding(observation),raw_candidates=B.binding(rep/'raw_camera_candidates.npz'),representation_outputs=B.binding(rep/'representation_outputs.npz')),r['raw_candidates']['count'],p['domain']['sensor_adapter'])
                if array_sha(points)!=r['input_points_sha256']or array_sha(coords)!=r['input_coords_sha256']:raise ValueError('CGR input differs from actual camera')
                item.update(rgbd_ray_check_recomputed_pass=True,CGR_input_and_receipt_exact=True)
            execution=directory/'execution/RESULTS.json'
            if execution.exists():
                ready=B.read(directory/'live/READY.json');B.verify(record['bindings'])
                with np.load(ready['commands']['path'])as z:arr={k:z[k].copy()for k in z.files}
                rows=[json.loads(line)for line in (directory/'execution/TRACE.jsonl').open()]
                item['execution_states']=states(rows,B.read(ready['scene']['path']),lane)
                velocity=C.source_velocity_targets(arr['joint_target'],names,list(reference.arm_home))
                for i,row in enumerate(rows):
                    np.testing.assert_array_equal(row['joint_target'],arr['joint_target'][i]);np.testing.assert_array_equal(row['velocity_target'],velocity[i])
                    np.testing.assert_array_equal(row['applied_position_target_float32'],arr['joint_target'][i].astype(np.float32))
                    if row['step']!=i+1 or row['phase']!=str(arr['phase'][i]):raise ValueError('Original control order changed')
                maxima,first=C.numeric_maxima(rows,planning,list(planning.joints),arr['desired_palm'],reference,legal,ids)
                gates=C.source_gates(maxima,len(rows),len(arr['phase']));saved=B.read(execution)
                complete=len(rows)==len(arr['phase'])and saved['stop']is None
                lift=C.LIFT.evaluate_lift(rows,ids,.004)if complete else dict(task_success=False,incomplete=True)
                if gates!=saved['gates']or maxima!=saved['maxima']or lift!=saved['evaluation']:raise ValueError('Numerical gates/lift disagree with raw states')
                decision=SAMPLE.classify(saved)
                for key,value in decision.items():
                    if record[key]!=value:raise ValueError('0/1/null label differs from original classification')
                item.update(original_controls_exact=True,trace=B.binding(directory/'execution/TRACE.jsonl'),gates=gates,eligible=decision['eligible'],mapping_audit=M.audit_mapping(directory/'live'))
                returned=directory/'RETURN.json'
                if returned.exists():
                    previous=arr['joint_target'][:len(rows)];targets=np.r_[previous[::-1],np.linspace(previous[0],home,101)[1:]]
                    speed=C.source_velocity_targets(targets,names,list(reference.arm_home));rows=[json.loads(s)for s in (directory/'return.jsonl').open()]
                    states(rows,dict(objects=[]),lane);desired=[]
                    for i,row in enumerate(rows):
                        np.testing.assert_array_equal(row['joint_target'],targets[i]);np.testing.assert_array_equal(row['velocity_target'],speed[i])
                        desired.append(planning.fk.transforms(planning.command_state(dict(zip(names,targets[i]))),reference.base_transform)['palm'])
                    maxima,_=C.numeric_maxima(rows,planning,list(planning.joints),desired,reference,set(),bank)
                    gates=C.source_gates(maxima,len(rows),len(targets));saved=B.read(returned)
                    if gates!=saved['gates']or maxima!=saved['maxima']:raise ValueError('Return differs from actual original controls/states')
                    item['return_pass']=saved['pass_']
            audits.append(item)
        matched=0;contacts=0;max_native_palm_error=0.;ticks=0
        journal=root/'GPU_JOURNAL.jsonl'
        for line in journal.open():
            row=json.loads(line);ticks+=1
            if row['tick']!=ticks or np.any(row['overflow'])or max(row['nefc'])>8192:raise ValueError('Native global journal/capacity discontinuity')
            np.testing.assert_array_equal(np.asarray(row['native_time_s'],dtype=np.float32),np.full(count,clocks[ticks],dtype=np.float32))
            q=np.asarray(row['qpos']);v=np.asarray(row['qvel']);indices=[]
            for lane in range(count):
                native=np.eye(4);native[:3,:3]=np.asarray(row['native_xmat'][lane][palm_id]).reshape(3,3);native[:3,3]=row['native_xpos'][lane][palm_id]
                fk=planning.fk.transforms(dict(zip(planning.joints,q[lane,qids])),reference.base_transform)['palm']
                max_native_palm_error=max(max_native_palm_error,float(np.max(np.abs(native-fk))))
                key=(ticks,lane)
                if key in wanted:
                    core=[q[lane,qids].tolist(),v[lane,vids].tolist(),row['contacts'][lane],native.tolist(),row['native_time_s'][lane]]
                    if digest(core)!=wanted.pop(key):raise ValueError('Per-trial state/contact differs from complete GPU journal')
                    matched+=1
                for c in row['contacts'][lane]:
                    indices.append(c['native_index']);raw=c['native_fields'];g=c['geom_ids']
                    if set(raw)!=set(CONTACT_FIELDS)or raw['worldid']!=lane or c['native_contact_fields_dropped']:raise ValueError('Native contact fields/world identity lost')
                    if c['body_ids']!=[int(model.geom_bodyid[x])for x in g]or c['bodies']!=[mapping[b]for b in c['body_ids']]:raise ValueError('Native point identity changed')
                    if c['separation_m']!=raw['dist']or c['normal_force_n']!=c['wrench_contact_frame'][0]:raise ValueError('Native point scalar changed')
            if sorted(indices)!=list(range(len(indices))):raise ValueError('Global native contacts were dropped/duplicated')
            contacts+=len(indices)
        if wanted or ticks!=world['global_control_ticks']or ticks*count*8!=world['native_environment_integrations']:raise ValueError('Missing trace rows or native cost mismatch')
        if max_native_palm_error>1e-5:raise ValueError('Native GPU body pose differs from independent source FK')
    output=dict(status='read_only_audit_pass',original_result=B.binding(root/'RESULTS.json'),rounds=audits,
        global_journal=B.binding(journal),global_ticks=ticks,native_environment_integrations=ticks*count*8,
        all_world_contacts_including_idle=contacts,matched_trial_rows=matched,max_native_palm_FK_error=max_native_palm_error,
        source_archive_and_inputs_verified=True,additional_physics_steps=0,additional_CGR_forwards=0,
        all_complete_grasps_pass=all(r.get('eligible',False)for r in audits if r.get('trace')),production_ready=False,
        implementation=B.binding(__file__))
    B.save(root/'READ_ONLY_AUDIT.json',output,exclusive=True);return output


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args()
    p=B.read(args.root/'PROTOCOL.json');marker=B.sha(args.root/'PROTOCOL.json')
    if os.environ.get('ANYDEX_MJX_AUDIT_ENV')!=marker:
        env=B._environment(p);env['ANYDEX_MJX_AUDIT_ENV']=marker
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(args.root.resolve())],env)
    out=audit(args.root);print({'status':out['status'],'rounds':len(out['rounds']),'native_integrations':out['native_environment_integrations']},flush=True)
