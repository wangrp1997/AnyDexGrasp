#!/usr/bin/env python3
"""Generic bounded offline V-HACD preparation for a not-ready source mesh.

Uses a separate immutable recipe/output. Never stops or edits the original
CoACD campaign. No Bullet client is connected and no physics step is allowed.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
PARAMETERS=dict(resolution=100000,concavity=.0025,alpha=.05,beta=.05,gamma=.0005,
    minVolumePerCH=.0001,maxNumVerticesPerCH=64,depth=20,planeDownsampling=4,
    convexhullDownsampling=4,pca=0,mode=0,convexhullApproximation=1)
DEFAULT_HELPER=Path('/home/wangrenpeng/AnyDexGrasp/reproduction/simulation/prepare_real_mesh_assets.py')
DEFAULT_OVERLAY=Path('/home/wangrenpeng/AnyDexGrasp/outputs/native_deps/python_overlay')

def helper(path):
    spec=importlib.util.spec_from_file_location('frozen_mesh_geometry_helper',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def original_cost_snapshot(directory,h):
    directory=Path(directory);plan=directory/'PLAN.json';rows=[]
    if not plan.exists():return dict(directory=str(directory),available=False)
    data=json.loads(plan.read_text())
    for job in data['jobs']:
        p=Path(job['directory'])/'RESULTS.json'
        if p.exists():
            j=json.loads(p.read_text());rows.append(dict(id=job['id'],status=j['status'],cost_seconds=j.get('cost_seconds'),RESULTS=h.record(p)))
        else:
            log=Path(job['directory'])/'coacd.log'
            rows.append(dict(id=job['id'],status='original_campaign_not_ready',log=str(log),
                log_bytes=log.stat().st_size if log.exists() else 0,
                latest_log_lines=log.read_text(errors='replace').splitlines()[-3:] if log.exists() else []))
    return dict(directory=str(directory),PLAN=h.record(plan),observed_at_unix_s=time.time(),
        observed_elapsed_since_original_PLAN_publication_s=time.time()-plan.stat().st_mtime,
        elapsed_boundary='Observed original campaign wall since PLAN mtime, not a sum of parallel workers; original running job continues',objects=rows)

def split_obj(path):
    import numpy as np
    vertices=[];groups=[];faces=[]
    def flush():
        nonlocal faces
        if faces:groups.append(faces);faces=[]
    for line in Path(path).open():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] in ['o','g']:flush()
        elif fields[0]=='f':
            nums=[int(x.split('/')[0]) for x in fields[1:]]
            idx=[x-1 if x>0 else len(vertices)+x for x in nums]
            faces.extend([[idx[0],idx[j],idx[j+1]] for j in range(1,len(idx)-1)])
    flush();all_v=np.asarray(vertices,float);parts=[]
    for block in groups:
        f=np.asarray(block,int);used,remap=np.unique(f,return_inverse=True)
        parts.append((all_v[used],remap.reshape(f.shape).astype(np.int32)))
    if not parts:raise ValueError('V-HACD output contains no OBJ objects/groups with triangles')
    return parts

def worker(args):
    import numpy as np,trimesh
    h=helper(args.helper);root=args.output_dir;plan=json.loads((root/'PLAN.json').read_text());item=plan['item'];source=item['preferred_source_mesh']
    receipt=dict(schema='AnyDex-bounded-offline-VHACD-v1',id=item['category'],method='PyBullet offline V-HACD',
        recipe_sha256=plan['recipe_sha256'],parameters=PARAMETERS,source=source,new_physics_steps=0,new_renders=0,new_network_forwards=0)
    start=time.monotonic()
    try:
        sys.path.insert(0,str(args.overlay));import pybullet as p
        # No client exists or is required by the standalone V-HACD entry point.
        def forbidden(*a,**kw):raise RuntimeError('Offline preparation forbids Bullet connect/step')
        p.connect=forbidden;p.stepSimulation=forbidden
        receipt['pybullet_native']=h.record(p.__file__);receipt['pybullet_API_version']=p.getAPIVersion();receipt['vhacd_doc']=p.vhacd.__doc__
        if h.sha(source['path'])!=source['sha256']:raise ValueError('Source mesh SHA differs')
        vertices,faces=h.load_obj(source['path']);lo=vertices.min(0);hi=vertices.max(0)
        needs_cavity=bool(item.get('requires_cavity','source_open_cavity_vertical_ray_probe' in item))
        cavity=h.source_cavity(vertices,faces) if needs_cavity else None
        receipt['source_geometry']=dict(AABB_m=[lo.tolist(),hi.tolist()],vertices=len(vertices),triangles=len(faces),scale=[1,1,1])
        receipt['cavity_before_decomposition']=cavity
        raw=root/'vhacd_components.obj';native_log=root/'vhacd_native.log';begin=time.monotonic()
        p.vhacd(source['path'],str(raw),str(native_log),**PARAMETERS)
        receipt['cost_seconds']=dict(native_vhacd=time.monotonic()-begin)
        if not raw.exists() or raw.stat().st_size==0:raise RuntimeError('V-HACD produced no OBJ')
        components=split_obj(raw);parts=[];all_valid=True
        if needs_cavity and len(components)<=1:raise ValueError('Required cavity cannot be substituted by one convex hull')
        for i,(v,f) in enumerate(components):
            mesh=trimesh.Trimesh(vertices=v,faces=f,process=False);valid=bool(mesh.is_convex and mesh.is_watertight and abs(mesh.volume)>1e-14 and np.isfinite(v).all())
            all_valid=all_valid and valid;path=root/('convex_part_%03d.obj'%i);h.write_obj(path,v,f)
            rv,rf=h.load_obj(path)
            if not np.array_equal(rv,v) or not np.array_equal(rf,f):raise ValueError('Saved OBJ changed component bits')
            parts.append(dict(**h.record(path),scale=[1,1,1],vertices=len(v),triangles=len(f),is_convex=bool(mesh.is_convex),
                is_watertight=bool(mesh.is_watertight),volume_m3=float(abs(mesh.volume)),geometry_valid=valid,AABB_m=[v.min(0).tolist(),v.max(0).tolist()]))
        lower=np.min([v.min(0) for v,f in components],axis=0);upper=np.max([v.max(0) for v,f in components],axis=0)
        bound_error=float(np.max(np.abs(np.array([lower-lo,upper-hi]))));limit=max(.001,.02*float(np.max(hi-lo)))
        receipt['compound_geometry']=dict(part_count=len(parts),all_components_convex_watertight_positive_volume=all_valid,
            AABB_m=[lower.tolist(),upper.tolist()],lower_bound_delta_from_source_m=(lower-lo).tolist(),upper_bound_delta_from_source_m=(upper-hi).tolist(),
            max_absolute_AABB_bound_error_m=bound_error,predeclared_AABB_bound_limit_m=limit,AABB_check_passed=bound_error<=limit,
            full_Hausdorff_or_volume_union_accuracy_certified=False)
        cavity_pass=True
        if needs_cavity:
            per_part=[h.top_intersections(v,f,cavity['xy_source_m']) for v,f in components]
            hits=[max([row[i] for row in per_part if row[i] is not None],default=None) for i in range(len(cavity['xy_source_m']))]
            observations=[]
            for i in cavity['qualifying_indices']:
                old=cavity['source_rim_upper_z_m']-cavity['source_first_hit_z_m'][i];new=None if hits[i] is None else cavity['source_rim_upper_z_m']-hits[i]
                observations.append(dict(index=i,source_recess_m=old,compound_recess_m=new,retained_fraction=None if new is None else new/old))
            cavity_pass=len(observations)>=h.CAVITY_POLICY['minimum_qualifying_rays'] and all(x['retained_fraction'] is not None and x['retained_fraction']>=h.CAVITY_POLICY['minimum_retained_recess_fraction'] for x in observations)
            receipt['cavity_check']=dict(**cavity,compound_first_hit_z_m=hits,qualifying_ray_results=observations,passed=cavity_pass,
                one_convex_hull_substitution=False,whole_cavity_or_handle_hole_certified=False)
        else:receipt['cavity_check']=dict(required=False,passed=True)
        # Deterministic surface samples are descriptive, not a Hausdorff proof.
        src=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        out=trimesh.util.concatenate([trimesh.Trimesh(vertices=v,faces=f,process=False) for v,f in components])
        src_points=vertices[np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int)]
        out_v=np.concatenate([v for v,f in components]);out_points=out_v[np.linspace(0,len(out_v)-1,min(512,len(out_v)),dtype=int)]
        d1=trimesh.proximity.closest_point(out,src_points)[1];d2=trimesh.proximity.closest_point(src,out_points)[1]
        receipt['sampled_surface_disagreement']=dict(source_vertex_samples=len(src_points),compound_vertex_samples=len(out_points),
            source_to_component_triangles_max_m=float(d1.max()),source_to_component_triangles_p95_m=float(np.quantile(d1,.95)),
            component_to_source_triangles_max_m=float(d2.max()),component_to_source_triangles_p95_m=float(np.quantile(d2,.95)),
            boundary='Unsigned distances of deterministic sampled vertices, including internal component surfaces; not signed union error or certified Hausdorff distance')
        receipt['parts']=parts;receipt['raw_VHACD_OBJ']=h.record(raw)
        receipt['geometry_checks_pass']=bool(all_valid and bound_error<=limit and cavity_pass)
        receipt['status']='bounded_VHACD_geometry_checks_passed' if receipt['geometry_checks_pass'] else 'bounded_VHACD_geometry_checks_failed'
        receipt['asset']=dict(id=item['category'],visual_mesh=dict(path=source['path'],sha256=source['sha256'],scale=[1,1,1]),
            convex_parts=[dict(path=x['path'],sha256=x['sha256'],scale=[1,1,1]) for x in parts],requires_cavity=needs_cavity,
            upright_orientation_xyzw=[0,0,0,1],decomposition_receipt=str((root/'RESULTS.json').resolve()),
            decomposition_method='independent_bounded_VHACD',decomposition_recipe_sha256=plan['recipe_sha256'])
    except Exception as exc:
        receipt.update(status='bounded_VHACD_failed',geometry_checks_pass=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    receipt.setdefault('cost_seconds',{})['worker_total']=time.monotonic()-start
    h.save(root/'RESULTS.json',receipt)
    return 0 if receipt.get('geometry_checks_pass') else 2

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=Path('/tmp/anydex_real_mesh_assets.json'))
    parser.add_argument('--id',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--original-decomposition',type=Path,required=True)
    parser.add_argument('--timeout-seconds',type=float,default=120)
    parser.add_argument('--helper',type=Path,default=DEFAULT_HELPER)
    parser.add_argument('--overlay',type=Path,default=DEFAULT_OVERLAY)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:return worker(args)
    if args.timeout_seconds<=0:parser.error('Timeout must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Use a fresh independent output directory')
    h=helper(args.helper);inventory=json.loads(args.inventory.read_text());item=next(x for x in inventory['objects'] if x['category']==args.id)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    recipe=dict(method='PyBullet offline V-HACD',parameters=PARAMETERS,seed='API exposes no seed parameter',
        timeout_seconds=args.timeout_seconds,threads=1,source=item['preferred_source_mesh'],inventory=h.record(args.inventory),
        script=h.record(Path(__file__)),geometry_helper=h.record(args.helper),cavity_policy=h.CAVITY_POLICY,
        AABB_policy='All six bounds within max(1mm,2% longest source AABB dimension); approximation check only, not physical gate',
        relation_to_original='Separate generic preparation fallback for any not-ready asset. Original CoACD source/recipe/process untouched, not reported completed')
    plan=dict(recipe=recipe,recipe_sha256=h.canonical_sha(recipe),item=item,
        original_cost_snapshot_before=original_cost_snapshot(args.original_decomposition,h),frozen_before_execution=True,new_physics_steps=0)
    h.save(args.output_dir/'PLAN.json',plan)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--id',args.id,'--inventory',str(args.inventory),
        '--output-dir',str(args.output_dir.resolve()),'--original-decomposition',str(args.original_decomposition.resolve()),
        '--timeout-seconds',str(args.timeout_seconds),'--helper',str(args.helper.resolve()),'--overlay',str(args.overlay.resolve())]
    start=time.monotonic();timed_out=False
    with (args.output_dir/'worker.log').open('wb') as log:
        try:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout_seconds,check=False);code=completed.returncode
        except subprocess.TimeoutExpired:timed_out=True;code=None
    path=args.output_dir/'RESULTS.json'
    if path.exists():receipt=json.loads(path.read_text())
    else:receipt=dict(id=args.id,status='bounded_VHACD_timeout' if timed_out else 'bounded_VHACD_worker_failed',geometry_checks_pass=False,new_physics_steps=0,new_renders=0,new_network_forwards=0)
    receipt.update(wrapper_wall_seconds=time.monotonic()-start,timeout_seconds=args.timeout_seconds,timed_out=timed_out,worker_returncode=code,
        original_cost_snapshot_before=plan['original_cost_snapshot_before'],original_cost_snapshot_after=original_cost_snapshot(args.original_decomposition,h),
        original_CoACD_job_modified_or_terminated=False,not_original_CoACD_completion=True,recipe_sha256=plan['recipe_sha256'])
    h.save(path,receipt)
    ready=receipt.get('geometry_checks_pass',False) and not timed_out and code==0
    assets=dict(assets=[receipt['asset']] if ready else [],status=receipt['status'],recipe_sha256=plan['recipe_sha256'],
        no_parameter_search_or_single_hull_cavity_substitution=True)
    h.save(args.output_dir/'ASSETS.json',assets)
    print(json.dumps(dict(status=receipt['status'],geometry_checks_pass=ready,parts=receipt.get('compound_geometry',{}).get('part_count'),
        wall_seconds=receipt['wrapper_wall_seconds'],RESULTS=str(path),ASSETS=str(args.output_dir/'ASSETS.json')),indent=2),flush=True)
    return 0 if ready else 2

if __name__=='__main__':sys.exit(main())
