"""Freeze a 40/10 Boya collection split from the active public DGN mesh groups.

Selection uses only source identity, metadata geometry, and a fixed seed. The
50 IDs are written before asset preflight. Missing/problematic assets remain
in the selected split and are reported, never replaced using grasp outcomes.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,re,time,xml.etree.ElementTree as ET
from pathlib import Path

SHAPES=('sphere','box','cylinder','bottle','mug','bowl','banana','irregular')
SOURCES=('core','ddg','mujoco','sem')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rank(seed,*parts):return hashlib.sha256((':'.join(map(str,(seed,*parts)))).encode()).hexdigest()


def shape_label(identity):
    """Experimental reporting strata only; not a perception/grasp classifier."""
    s=identity.lower()
    # Positive source noun/category match: a ball-ended tool is not a sphere.
    if s.startswith('sem_ball_') or re.match(r'^(ddg_gd_|ddg_ycb_[0-9]+_|mujoco_)(soccer_ball|tennis_ball|golf_ball|baseball|softball|racquetball)(_|$)',s):return 'sphere'
    if any(x in s for x in ('_box_','_foam_brick','_cerealbox','_milkcarton','_rubikscube','_tissuebox','_book_','_filecabinet')):return 'box'
    if '_mug_' in s or 'coffee_mug' in s:return 'mug'
    if '_bowl_' in s or '_ramekin_' in s:return 'bowl'
    if '_bottle_' in s:return 'bottle'
    if any(x in s for x in ('_can_','_sodacan_','_jar_')):return 'cylinder'
    if 'banana' in s:return 'banana'
    return 'irregular'


def quotas(count,names,seed,tag,weights=None):
    # Coverage first, then the largest source-availability proportions.
    base=count//len(names);out={n:base for n in names};left=count-base*len(names)
    order=sorted(names,key=lambda n:(-(weights or {}).get(n,0),rank(seed,tag,n)))
    for n in order[:left]:out[n]+=1
    return out


def allocate(groups,count,seed,split):
    shape_count=collections.Counter(g['shape'] for g in groups)
    sq=quotas(count,SHAPES,seed,split+':shape',shape_count)
    oq=quotas(count,SOURCES,seed,split+':source')
    capacities=collections.Counter((g['source'],g['shape']) for g in groups)
    capacity={};adj=collections.defaultdict(list)
    def edge(a,b,c):
        capacity[a,b]=c;capacity[b,a]=0;adj[a].append(b);adj[b].append(a)
    source_order=sorted(SOURCES,key=lambda n:rank(seed,split,'source_order',n))
    shape_order=sorted(SHAPES,key=lambda n:rank(seed,split,'shape_order',n))
    for s in source_order:edge('start','source:'+s,oq[s])
    for s in source_order:
        for h in shape_order:edge('source:'+s,'shape:'+h,capacities[s,h])
    for h in shape_order:edge('shape:'+h,'end',sq[h])
    residual=capacity.copy();flow=0
    while True:
        previous={'start':None};queue=collections.deque(['start'])
        while queue and 'end' not in previous:
            n=queue.popleft()
            for other in adj[n]:
                if other not in previous and residual[n,other]>0:previous[other]=n;queue.append(other)
        if 'end' not in previous:break
        n='end';delta=count
        while previous[n] is not None:delta=min(delta,residual[previous[n],n]);n=previous[n]
        n='end'
        while previous[n] is not None:
            p=previous[n];residual[p,n]-=delta;residual[n,p]+=delta;n=p
        flow+=delta
    if flow!=count:raise ValueError('Declared source/shape quotas infeasible; no objects replaced: '+json.dumps(dict(split=split,allocated=flow,requested=count,shape_quotas=sq,source_quotas=oq)))
    selected=[];cells=[]
    for s in SOURCES:
        for h in SHAPES:
            n=capacity['source:'+s,'shape:'+h]-residual['source:'+s,'shape:'+h]
            candidates=sorted((g for g in groups if g['source']==s and g['shape']==h),key=lambda g:rank(seed,split,'mesh',g['mesh_sha256']))
            selected.extend(candidates[:n]);cells.append(dict(source=s,shape=h,available=len(candidates),selected=n))
    selected.sort(key=lambda g:rank(seed,split,'output_order',g['mesh_sha256']))
    assert len(selected)==count and len({g['mesh_sha256'] for g in selected})==count
    return selected,dict(shape_quotas=sq,source_quotas=oq,cells=cells)


def preflight(group,dataset_root,split,run_dir):
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    row=group['row'];oid=row['object_id'];root=dataset_root/row['path']
    record=dict(id=oid,object_id=oid,pilot_id=oid,split=split,source_split=row['split'],source_family=group['source'],shape_selection_label=group['shape'],selection_stratum=group['source']+':'+group['shape'],mesh_group_sha256=group['mesh_sha256'],mesh_group_alias_ids=group['aliases'],object_root=str(root),mesh_path=str(root/'mesh/simplified.obj'),urdf_path=str(root/'urdf/coacd.urdf'),pose_path=str(root/'info/tabletop_pose.json'),metadata_path=str(root/'info/simplified.json'),scale=.1,geometry_scaled_on_disk=False,obb_extents_sorted_m=row['obb_extents_sorted_m'],source_active_list=str(dataset_root/'objects.json'),active_member=True,object_mass_kg=.1,object_friction=.6,grasp_outcome_consulted=False,prior_grasp_pose_consulted=False,observation_json=str(run_dir/'observations'/oid/'observation.json'),observation=str(run_dir/'observations'/oid/'observation.json'),representation_dir=str(run_dir/'representations'/oid),selection_reason='Fixed seed and exact source/shape coverage quotas, sampled by source mesh SHA group before asset preflight; no outcome-based filtering.')
    issues=[]
    for path_key,hash_key in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')]:
        path=Path(record[path_key])
        if not path.is_file():issues.append(dict(kind='missing_asset',path=str(path)));continue
        record[hash_key]=sha(path)
    if record.get('mesh_sha256')!=group['mesh_sha256']:issues.append(dict(kind='active_mesh_sha_mismatch',expected=group['mesh_sha256'],actual=record.get('mesh_sha256')))
    try:
        urdf=Path(record['urdf_path']);parts=[]
        for node in ET.parse(urdf).getroot().findall('.//collision/geometry/mesh'):
            path=(urdf.parent/node.get('filename','')).resolve()
            if not path.is_file():issues.append(dict(kind='missing_convex_part',path=str(path)));continue
            parts.append(dict(path=str(path),sha256=sha(path),scale=node.get('scale','1 1 1')))
        record['convex_parts']=parts
        if not parts:issues.append(dict(kind='no_collision_mesh_parts'))
    except Exception as e:issues.append(dict(kind='urdf_preflight',error=repr(e)))
    try:
        metadata=json.loads(Path(record['metadata_path']).read_text());obb=np.sort(np.asarray(metadata['obb'],float))*.1
        if not np.allclose(obb,record['obb_extents_sorted_m'],atol=1e-10,rtol=1e-9):issues.append(dict(kind='metadata_obb_mismatch',active=record['obb_extents_sorted_m'],file=obb.tolist()))
        mesh=trimesh.load_mesh(record['mesh_path'],process=False)
        if not isinstance(mesh,trimesh.Trimesh)or not len(mesh.faces):raise ValueError('Expected source triangle mesh')
        vertices=np.asarray(mesh.vertices,float)*.1
        if not np.isfinite(vertices).all():raise ValueError('Nonfinite source geometry')
        poses=json.loads(Path(record['pose_path']).read_text());measure=[]
        for i,pose in enumerate(poses):
            a=np.asarray(pose,float)
            if a.shape!=(7,)or not np.isfinite(a).all():raise ValueError('Invalid source support pose')
            rot=Rotation.from_quat(np.r_[a[4:7],a[3]]).as_matrix();w=vertices@rot.T
            measure.append(dict(index=i,height_m=float(np.ptp(w[:,2])),source_Y_up_dot=float(rot[2,1]),R=rot,raw=a.tolist(),horizontal_short_width_m=float(min(np.ptp(w[:,0]),np.ptp(w[:,1])))))
        if not measure:raise ValueError('No provided source support pose')
        h=group['shape'];policy='source_Y_up'if h in ('bottle','cylinder','mug','bowl')else'first_source_pose'if h=='sphere'else'minimum_source_supported_height'
        chosen=max(measure,key=lambda x:(x['source_Y_up_dot'],-x['index']))if policy=='source_Y_up'else measure[0]if policy=='first_source_pose'else min(measure,key=lambda x:(x['height_m'],x['index']))
        R=chosen['R'];w=vertices@R.T;T=np.eye(4);T[:3,:3]=R;T[:2,3]=np.array([-.75,.10])-(w[:,:2].min(0)+w[:,:2].max(0))/2;T[2,3]=-w[:,2].min();world=w+T[:3,3]
        record.update(source_mesh_AABB_m_unscaled=mesh.bounds.tolist(),selected_source_pose_index=chosen['index'],selected_source_pose_xyz_wxyz=chosen['raw'],source_pose_quaternion_order='wxyz',source_pose_policy=policy,source_pose_y_up_dot=chosen['source_Y_up_dot'],all_source_pose_summaries=[{k:x[k] for k in ('index','height_m','source_Y_up_dot','horizontal_short_width_m')}for x in measure],world_T_object=T.tolist(),world_pose_xyzw=np.r_[T[:3,3],Rotation.from_matrix(R).as_quat()].tolist(),world_AABB_m=[world.min(0).tolist(),world.max(0).tolist()],world_extents_m=np.ptp(world,axis=0).tolist(),source_placement_rule='Same predeclared pilot setup: source supported rotation; container source+Y upright, sphere first supported pose, other shapes minimum supported height. Preserve scale .1; actual horizontal AABB center(-.75,.10), shift Z to original vertex minimum at table z=0. This is scene setup, not an inference input or result-conditioned pose search.')
        assert abs(world[:,2].min())<1e-12
        if float(obb.min())<.025-1e-10:issues.append(dict(kind='thickness_scope_violation',minimum_obb_m=float(obb.min())))
        if h=='sphere' and float(obb.max())>=.13:issues.append(dict(kind='sphere_scope_review',maximum_obb_m=float(obb.max()),boundary='Original active ball-size policy uses its named audited ball set; label here is only a sampling proxy. Selected object retained for review.'))
        if h=='box':
            flat=min(measure,key=lambda x:(x['height_m'],x['index']))
            record['scope_flat_box_width_m']=flat['horizontal_short_width_m']
            if flat['horizontal_short_width_m']>=.106:issues.append(dict(kind='box_scope_review',flat_short_width_m=flat['horizontal_short_width_m'],boundary='Selected object retained; no silent replacement, scope must be resolved before physics.'))
    except Exception as e:issues.append(dict(kind='geometry_pose_preflight',error=repr(e)))
    record['preflight_status']='ready'if not issues else'asset_review_required';record['preflight_issues']=issues
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',type=Path,default=Path('/mnt/public/datasets/CanonDex_DGN_tabletop_25mm_v1'));p.add_argument('--exclude-selection',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp/outputs/boya_pilot8_20261105_v1/object_selection.json'));p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=202611054010);p.add_argument('--gesture-ids',nargs='+',type=int,default=[1,6,8]);p.add_argument('--additional-depth-mm',nargs='+',type=float,default=[0.]);a=p.parse_args()
    a.dataset_root=a.dataset_root.resolve();a.exclude_selection=a.exclude_selection.resolve();a.output_dir=a.output_dir.resolve();run_dir=a.output_dir.parent
    if any(g not in (1,2,3,5,6,7,8,9,10)for g in a.gesture_ids)or len(set(a.gesture_ids))!=len(a.gesture_ids):p.error('Use existing calibrated hand types; type4 has no opposing datum')
    if any(not math.isfinite(d)or not 0<=d<=40 for d in a.additional_depth_mm)or len(set(a.additional_depth_mm))!=len(a.additional_depth_mm):p.error('Explicit unique extra depths in0..40mm required')
    start=time.monotonic();active_path=a.dataset_root/'objects.json';active=json.loads(active_path.read_text())
    if active['active_object_count']!=4330 or len(active['objects'])!=4330 or active['scale']!=.1 or active['geometry_scaled_on_disk']:raise ValueError('Expected unchanged active4330 scale.1 public selection')
    excluded=json.loads(a.exclude_selection.read_text())['objects'];exclude_ids={r['object_id']for r in excluded};exclude_hash={r['mesh_sha256']for r in excluded};assert len(exclude_ids)==8
    grouped=collections.defaultdict(list)
    for row in active['objects']:
        if row['object_id']in exclude_ids or row['mesh_sha256']in exclude_hash:continue
        grouped[row['mesh_sha256']].append(row)
    groups=[]
    for digest,aliases in grouped.items():
        row=min(aliases,key=lambda r:rank(a.seed,'canonical_mesh_alias',r['object_id']));groups.append(dict(mesh_sha256=digest,row=row,aliases=sorted(x['object_id']for x in aliases),source=row['object_id'].split('_')[0],shape=shape_label(row['object_id'])))
    train,train_quota=allocate(groups,40,a.seed,'train');selected_hash={g['mesh_sha256']for g in train};validation,val_quota=allocate([g for g in groups if g['mesh_sha256']not in selected_hash],10,a.seed,'validation')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    plan=dict(schema='anydex_boya_collection40_selection_plan_v1',seed=a.seed,source_dataset_root=str(a.dataset_root),source_active_list=str(active_path),source_active_list_sha256=sha(active_path),exclude_selection=str(a.exclude_selection),exclude_selection_sha256=sha(a.exclude_selection),excluded_development_ids=sorted(exclude_ids),excluded_development_mesh_sha256=sorted(exclude_hash),eligible_mesh_groups=len(groups),source_shape_group_counts={s:{h:sum(g['source']==s and g['shape']==h for g in groups)for h in SHAPES}for s in SOURCES},selection_method='Exact balanced source and shape coverage quotas solved with deterministic integer max-flow; groups sampled by SHA256(seed,split,meshSHA). Full mesh SHA is the grouping identity; canonical alias is fixed by seed. Plan written before asset preflight. No grasp logs are read.',train_allocation=train_quota,validation_allocation=val_quota,selected_before_preflight=[dict(object_id=g['row']['object_id'],mesh_sha256=g['mesh_sha256'],source=g['source'],shape=g['shape'],split=split)for split,items in [('train',train),('validation',validation)]for g in items],source_script_sha256=sha(__file__),grasp_outcomes_consulted=False,physics_steps=0)
    save(a.output_dir/'selection_plan.json',plan)
    objects=[preflight(g,a.dataset_root,split,run_dir)for split,items in [('train',train),('validation',validation)]for g in items]
    for r in objects:r['pools']={str(g):str(run_dir/'pools'/r['object_id']/('g'+str(g)))for g in a.gesture_ids}
    camera=dict(width=1280,height=720,fx=640.,fy=640.,eye_world_m=[-.75,-.20,.50],target_world_m=[-.75,.10,.06],up_world=[0.,0.,1.],depth_range_m=[.20,.65],focal_length=24.,horizontal_aperture=48.,clipping_range_m=[.1,1.])
    manifest=dict(schema='anydex_boya_collection40_frozen_objects_v1',seed=a.seed,source_dataset_root=str(a.dataset_root),source_active_list_sha256=sha(active_path),active_object_count=4330,scale=.1,objects=objects,camera=camera,observation_api=dict(subdirectories=[r['id']for r in objects],network_files=['color.png','depth.png','meta.mat'],receipt='observation.json',robot_absent=True,control_steps=0,grasp_attempts=0),pool_contract=dict(gesture_ids=a.gesture_ids,additional_depth_mm=a.additional_depth_mm,additional_depth_m=[d/1000 for d in a.additional_depth_mm],status='Declaration only; selector does not build pools or alter control. Existing pilot builder supports only extra depth0 until a separately frozen builder implements expansion.',candidate_order='Original CGR score order; no grasp outcomes in selector'),selection_boundary='40 adaptation-train and10 development-held-out-validation mesh groups; pilot8 IDs and all exact mesh aliases excluded from both. Shape names are sampling/placement-report strata only, never network/grasp features. Exact mesh byte hashes guarantee no identical source mesh across splits, not semantic/physical-instance independence and not unseen-in-CGR-pretraining. Balanced coverage sample is not a prevalence estimate for4330.',selection_plan_sha256=sha(a.output_dir/'selection_plan.json'),physics_steps=0,grasp_attempts=0)
    save(a.output_dir/'object_selection.json',manifest)
    issues=[dict(object_id=r['object_id'],split=r['split'],issues=r['preflight_issues'])for r in objects if r['preflight_issues']]
    save(a.output_dir/'preflight.json',dict(status='ready'if not issues else'asset_review_required',selected=50,ready=50-len(issues),asset_problem_objects=issues,replacement_count=0,physics_steps=0))
    train_set={r['mesh_group_sha256']for r in objects if r['split']=='train'};val_set={r['mesh_group_sha256']for r in objects if r['split']=='validation'}
    assert len(train_set)==40 and len(val_set)==10 and not train_set&val_set and not(train_set|val_set)&exclude_hash
    summary=dict(schema='anydex_boya_collection40_selection_summary_v1',status='ready'if not issues else'asset_review_required',seed=a.seed,train_objects=40,validation_objects=10,unique_selected_mesh_groups=50,mesh_group_overlap=0,pilot_mesh_overlap=0,source_counts={split:dict(collections.Counter(r['source_family']for r in objects if r['split']==split))for split in ('train','validation')},shape_counts={split:dict(collections.Counter(r['shape_selection_label']for r in objects if r['split']==split))for split in ('train','validation')},inherited_source_split_counts={split:dict(collections.Counter(r['source_split']for r in objects if r['split']==split))for split in ('train','validation')},asset_review_required=len(issues),object_selection=str(a.output_dir/'object_selection.json'),object_selection_sha256=sha(a.output_dir/'object_selection.json'),selection_plan_sha256=sha(a.output_dir/'selection_plan.json'),source_script_sha256=sha(__file__),elapsed_wall_s=time.monotonic()-start,physics_steps=0,training_labels=0,public_mutated=False)
    save(a.output_dir/'summary.json',summary)
    for split in ('train','validation'):save(a.output_dir/(split+'_ids.json'),[r['object_id']for r in objects if r['split']==split])
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
