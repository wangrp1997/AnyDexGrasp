#!/usr/bin/env python3
"""Freeze and run six real-YCB CoACD jobs; no simulator or network code.

Usage: python -B /tmp/anydex_prepare_real_mesh_assets.py --output-root NEW_DIR
Output ASSETS.json is compatible with anydex_mesh_scene.prepare_asset. Only
geometrically valid compounds passing the predeclared cavity probes are exposed.
All other outcomes remain in per-object RESULTS.json and batch RESULTS.json.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, inspect, json, multiprocessing, os
from pathlib import Path
import sys, time, traceback

# Set before NumPy/native libraries load in main AND spawned workers. No global
# host setting is changed; this environment belongs only to this task process.
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]='1'
IDS=['apple','banana','foam_brick','bottle','mug','bowl']
PARAMETERS=dict(threshold=.01,resolution=2000,mcts_iterations=75,mcts_nodes=20,
    mcts_max_depth=3,max_convex_hull=128,preprocess_mode='auto',preprocess_resolution=50,
    pca=False,merge=True,decimate=False,max_ch_vertex=256,extrude=False,
    extrude_margin=.01,apx_mode='ch',seed=20261103)
CAVITY_POLICY=dict(probe_fractions=[-.1,0,.1],source_min_recess_height_fraction=.25,
    minimum_retained_recess_fraction=.75,minimum_qualifying_rays=3,
    scope='Nine predeclared central vertical rays; verifies a recessed open cavity, not whole-surface Hausdorff accuracy or the entire mug handle hole')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def record(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def load_obj(path):
    import numpy as np
    vertices=[];faces=[]
    for line in Path(path).open():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
        elif fields[0]=='f':
            numbers=[int(x.split('/')[0]) for x in fields[1:]]
            idx=[x-1 if x>0 else len(vertices)+x for x in numbers]
            faces.extend([[idx[0],idx[j],idx[j+1]] for j in range(1,len(idx)-1)])
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.int32)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3:raise ValueError('Expected nonempty triangle OBJ')
    if not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):raise ValueError('Invalid mesh coordinates/indices')
    return v,f

def write_obj(path,vertices,faces):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    with temporary.open('w') as f:
        f.write('# Original source coordinate frame, meters, scale1; generated CoACD convex component\no convex_part\n')
        for v in vertices:f.write('v '+' '.join(format(float(x),'.17g') for x in v)+'\n')
        for row in faces:f.write('f '+' '.join(str(int(x)+1) for x in row)+'\n')
    temporary.replace(path)

def top_intersections(vertices,faces,xy):
    """Pure triangle geometry; first downward hit of each vertical ray."""
    import numpy as np
    tri=vertices[faces];a=tri[:,0];b=tri[:,1]-a;c=tri[:,2]-a
    den=b[:,0]*c[:,1]-b[:,1]*c[:,0];safe=np.abs(den)>1e-14
    hits=[]
    for point in xy:
        q=np.asarray(point)-a[:,:2]
        u=np.divide(q[:,0]*c[:,1]-q[:,1]*c[:,0],den,out=np.zeros(len(den)),where=safe)
        w=np.divide(b[:,0]*q[:,1]-b[:,1]*q[:,0],den,out=np.zeros(len(den)),where=safe)
        inside=safe&(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)
        z=a[:,2]+u*b[:,2]+w*c[:,2]
        hits.append(float(z[inside].max()) if inside.any() else None)
    return hits

def source_cavity(vertices,faces):
    import numpy as np
    lo=vertices.min(0);hi=vertices.max(0);center=(lo+hi)/2;extent=hi-lo
    xy=[(center[:2]+np.array([x,y])*extent[:2]).tolist() for x in CAVITY_POLICY['probe_fractions'] for y in CAVITY_POLICY['probe_fractions']]
    hits=top_intersections(vertices,faces,xy)
    qualifying=[i for i,z in enumerate(hits) if z is not None and hi[2]-z>=CAVITY_POLICY['source_min_recess_height_fraction']*extent[2]]
    return dict(policy=CAVITY_POLICY,xy_source_m=xy,source_first_hit_z_m=hits,
        source_rim_upper_z_m=float(hi[2]),source_height_m=float(extent[2]),qualifying_indices=qualifying,
        frozen_before_decomposition=True)

def worker(job):
    directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=False)
    # Capture only this worker's native verbose output, without changing parent IO.
    log=(directory/'coacd.log').open('a',buffering=1)
    os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
    start=time.monotonic();receipt=dict(schema='AnyDex-source-mesh-CoACD-v1',id=job['id'],status='starting',
        source=job['source'],recipe_sha256=job['recipe_sha256'],parameters=PARAMETERS,
        task_threads={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
        new_physics_steps=0,new_renders=0,new_network_forwards=0)
    try:
        import numpy as np,coacd,trimesh
        receipt['coacd_API_signature']=str(inspect.signature(coacd.run_coacd))
        receipt['coacd_python']=record(coacd.__file__)
        native=Path(coacd._lib._name)
        receipt['coacd_native_library']=record(native)
        if any(k not in inspect.signature(coacd.run_coacd).parameters for k in PARAMETERS):raise TypeError('Frozen parameter unsupported by installed CoACD API')
        path=Path(job['source']['path'])
        if sha(path)!=job['source']['sha256']:raise ValueError('Source mesh bytes changed')
        vertices,faces=load_obj(path);source_lo=vertices.min(0);source_hi=vertices.max(0)
        requires_cavity=job['id'] in ['mug','bowl']
        cavity=source_cavity(vertices,faces) if requires_cavity else None
        save(directory/'PLAN.json',dict(id=job['id'],source=job['source'],parameters=PARAMETERS,
            cavity_policy=CAVITY_POLICY if requires_cavity else None,source_cavity_before_decomposition=cavity,
            recipe_sha256=job['recipe_sha256'],threshold_interpretation='CoACD normalized concavity setting; not a fixed metric surface-error certificate'))
        receipt['source_geometry']=dict(vertices=len(vertices),triangles=len(faces),AABB_m=[source_lo.tolist(),source_hi.tolist()],scale=[1,1,1])
        begin=time.monotonic()
        components=coacd.run_coacd(coacd.Mesh(vertices,faces),**PARAMETERS)
        receipt['cost_seconds']={'coacd_decomposition':time.monotonic()-begin}
        if not components:raise RuntimeError('CoACD returned no components')
        if requires_cavity and len(components)<=1:raise RuntimeError('A single convex hull cannot represent required open cavity')
        parts=[];output_components=[];all_convex=True
        for i,(part_v,part_f) in enumerate(components):
            part_v=np.asarray(part_v,dtype=np.float64);part_f=np.asarray(part_f,dtype=np.int32)
            if not np.isfinite(part_v).all():raise ValueError('Nonfinite convex component')
            mesh=trimesh.Trimesh(vertices=part_v,faces=part_f,process=False)
            convex=bool(mesh.is_convex);watertight=bool(mesh.is_watertight);volume=float(abs(mesh.volume))
            valid=convex and watertight and volume>1e-14
            all_convex=all_convex and valid
            path=directory/('convex_part_%03d.obj'%i);write_obj(path,part_v,part_f)
            # Validate serialized doubles and topology without mutation or re-optimization.
            check_v,check_f=load_obj(path)
            if not np.array_equal(check_v,part_v) or not np.array_equal(check_f,part_f):raise ValueError('OBJ roundtrip changed component geometry')
            parts.append(dict(**record(path),scale=[1,1,1],vertices=len(part_v),triangles=len(part_f),is_convex=convex,
                is_watertight=watertight,volume_m3=volume,geometry_valid=valid,AABB_m=[part_v.min(0).tolist(),part_v.max(0).tolist()]))
            output_components.append((part_v,part_f))
        lower=np.min([x[0].min(0) for x in output_components],axis=0);upper=np.max([x[0].max(0) for x in output_components],axis=0)
        receipt['compound_geometry']=dict(part_count=len(parts),all_components_convex_watertight_positive_volume=all_convex,
            AABB_m=[lower.tolist(),upper.tolist()],AABB_extent_m=(upper-lower).tolist(),
            lower_bound_delta_from_source_m=(lower-source_lo).tolist(),upper_bound_delta_from_source_m=(upper-source_hi).tolist(),
            max_outward_AABB_expansion_m=float(max(np.max(source_lo-lower),np.max(upper-source_hi),0)),
            full_Hausdorff_or_volume_union_accuracy_certified=False,
            boundary='Convexity/AABB/source binding and sparse cavity checks; not an exact shape-preserving surface or collision certificate')
        cavity_pass=True
        if requires_cavity:
            per_part=[top_intersections(v,f,cavity['xy_source_m']) for v,f in output_components]
            hits=[max([row[i] for row in per_part if row[i] is not None],default=None) for i in range(len(cavity['xy_source_m']))]
            retained=[];passes=[]
            for i in cavity['qualifying_indices']:
                old_depth=cavity['source_rim_upper_z_m']-cavity['source_first_hit_z_m'][i]
                new_depth=None if hits[i] is None else cavity['source_rim_upper_z_m']-hits[i]
                ratio=None if new_depth is None else new_depth/old_depth
                retained.append(dict(index=i,source_recess_m=old_depth,compound_recess_m=new_depth,retained_fraction=ratio))
                passes.append(ratio is not None and ratio>=CAVITY_POLICY['minimum_retained_recess_fraction'])
            cavity_pass=len(passes)>=CAVITY_POLICY['minimum_qualifying_rays'] and all(passes)
            receipt['cavity_check']=dict(**cavity,compound_first_hit_z_m=hits,qualifying_ray_results=retained,passed=cavity_pass,
                one_convex_hull_substitution=False,whole_handle_hole_or_whole_cavity_certified=False)
        else:receipt['cavity_check']=dict(required=False,passed=True)
        receipt['parts']=parts
        receipt['geometry_checks_pass']=bool(all_convex and cavity_pass)
        receipt['status']='convex_components_and_predeclared_cavity_checks_passed' if receipt['geometry_checks_pass'] else 'decomposition_geometry_or_cavity_check_failed'
        receipt['asset']=dict(id=job['id'],visual_mesh=dict(path=job['source']['path'],sha256=job['source']['sha256'],scale=[1,1,1]),
            convex_parts=[dict(path=x['path'],sha256=x['sha256'],scale=x['scale']) for x in parts],
            requires_cavity=requires_cavity,upright_orientation_xyzw=[0,0,0,1],
            decomposition_receipt=str((directory/'RESULTS.json').resolve()),decomposition_recipe_sha256=job['recipe_sha256'])
    except Exception as exc:
        receipt.update(status='failed',geometry_checks_pass=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    receipt.setdefault('cost_seconds',{})['worker_total']=time.monotonic()-start
    save(directory/'RESULTS.json',receipt)
    return dict(id=job['id'],status=receipt['status'],geometry_checks_pass=receipt.get('geometry_checks_pass',False),
        RESULTS=record(directory/'RESULTS.json'),part_count=receipt.get('compound_geometry',{}).get('part_count'),
        asset=receipt.get('asset'),cost_seconds=receipt['cost_seconds'],error=receipt.get('error'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,default=Path('/tmp/anydex_real_mesh_assets.json'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=6:parser.error('--workers must be 1..6')
    if args.output_root.exists() and any(args.output_root.iterdir()):parser.error('Use a fresh output directory; existing evidence is never overwritten')
    args.output_root.mkdir(parents=True,exist_ok=True)
    inv=json.loads(args.inventory.read_text());by_id={x['category']:x for x in inv['objects']}
    if set(IDS)!=set(by_id):raise ValueError('Expected the frozen six original object categories')
    recipe=dict(parameters=PARAMETERS,cavity_policy=CAVITY_POLICY,source_inventory=record(args.inventory),
        script=record(Path(__file__)),ids=IDS,process_count=args.workers,OMP_NUM_THREADS=1)
    recipe_sha=canonical_sha(recipe)
    jobs=[dict(id=name,source=by_id[name]['preferred_source_mesh'],directory=str((args.output_root/name).resolve()),recipe_sha256=recipe_sha) for name in IDS]
    plan=dict(schema='AnyDex-real-YCB-six-CoACD-campaign-v1',status='frozen_before_workers',recipe=recipe,recipe_sha256=recipe_sha,
        jobs=jobs,new_physics_steps=0,new_renders=0,new_network_forwards=0)
    save(args.output_root/'PLAN.json',plan);print(json.dumps(dict(status='started',output_root=str(args.output_root.resolve()),workers=args.workers,recipe_sha256=recipe_sha)),flush=True)
    start=time.monotonic();completed=[]
    # spawn avoids inheriting partially initialized numerical/native runtime.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            job=futures[future]
            try:result=future.result()
            except Exception as exc:result=dict(id=job['id'],status='worker_process_failed',geometry_checks_pass=False,error_type=type(exc).__name__,error=str(exc))
            completed.append(result)
            state=dict(status='running',recipe_sha256=recipe_sha,elapsed_wall_seconds=time.monotonic()-start,completed=completed)
            save(args.output_root/'PROGRESS.json',state)
            print(json.dumps({k:v for k,v in result.items() if k!='asset'}),flush=True)
    ordered=[next(x for x in completed if x['id']==name) for name in IDS]
    ready=[x['asset'] for x in ordered if x['geometry_checks_pass']]
    assets=dict(assets=ready,recipe_sha256=recipe_sha,all_six_geometry_checks_pass=len(ready)==len(IDS),
        excluded_not_silently_substituted=[x['id'] for x in ordered if not x['geometry_checks_pass']])
    save(args.output_root/'ASSETS.json',assets)
    final=dict(status='all_six_original_mesh_compounds_ready' if len(ready)==len(IDS) else 'partial_or_failed_compounds_preserved',
        recipe_sha256=recipe_sha,assets_manifest=record(args.output_root/'ASSETS.json'),objects=ordered,
        ready_count=len(ready),expected_count=len(IDS),wall_seconds=time.monotonic()-start,
        new_physics_steps=0,new_renders=0,new_network_forwards=0,
        boundary='No parameter retries or single-hull cavity substitution; simulation physics and source mesh accuracy require their own validation')
    save(args.output_root/'RESULTS.json',final)
    print(json.dumps(dict(status=final['status'],ready_count=len(ready),wall_seconds=final['wall_seconds'],ASSETS=str((args.output_root/'ASSETS.json').resolve()))),flush=True)
    return 0 if len(ready)==len(IDS) else 2

if __name__=='__main__':sys.exit(main())
