#!/usr/bin/env python3
"""Uniform bounded V-HACD EGAD12 assets; no network, scoring or grasp.

New asset protocol before any external forward. All12 retain the same native
visual triangles/unit/placement. V-HACD is used for ALL12, never chosen by outcome.
Original CoACD source/results/timeouts remain in their independent directories.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import trimesh
import prepare_egad_external_scenes as base

ROOT=Path(__file__).resolve().parents[2]
FALLBACK=ROOT/'reproduction/simulation/prepare_mesh_fallback.py'
GEOMETRY=ROOT/'reproduction/simulation/prepare_real_mesh_assets.py'
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
fallback=load_module(FALLBACK,'frozen_vhacd_fallback')
geometry=load_module(GEOMETRY,'frozen_mesh_ray_geometry')
PARAMETERS=dict(fallback.PARAMETERS)
TIMEOUT_S=120
RAY_FRACTIONS=(-.25,0.,.25)
SURFACE_SAMPLES=256


def approximation(visual,components):
    vertices=np.asarray(visual.vertices);bounds=np.asarray(visual.bounds);parts=[trimesh.Trimesh(v,f,process=False)for v,f in components]
    compound=trimesh.util.concatenate(parts);all_v=np.asarray(compound.vertices)
    cb=np.array([all_v.min(0),all_v.max(0)])
    error=float(np.abs(cb-bounds).max());limit=max(.001,.02*float(np.ptp(vertices,axis=0).max()))
    sv=vertices[np.linspace(0,len(vertices)-1,min(SURFACE_SAMPLES,len(vertices)),dtype=int)]
    cv=all_v[np.linspace(0,len(all_v)-1,min(SURFACE_SAMPLES,len(all_v)),dtype=int)]
    d1=trimesh.proximity.closest_point(compound,sv)[1];d2=trimesh.proximity.closest_point(visual,cv)[1]
    rays=[]
    for axis in range(3):
        axes=[i for i in range(3)if i!=axis]+[axis]
        for sign in (-1,1):
            transformed=vertices[:,axes].copy();transformed[:,2]*=sign
            lo=transformed.min(0);hi=transformed.max(0);ext=hi-lo;center=(lo+hi)/2
            xy=[(center[:2]+np.array([a,b])*ext[:2]).tolist()for a in RAY_FRACTIONS for b in RAY_FRACTIONS]
            src=geometry.top_intersections(transformed,visual.faces,xy)
            out=[]
            for v,f in components:
                pv=np.asarray(v)[:,axes].copy();pv[:,2]*=sign
                out.append(geometry.top_intersections(pv,f,xy))
            hit=[max((row[i]for row in out if row[i]is not None),default=None)for i in range(len(xy))]
            for i,(s,c)in enumerate(zip(src,hit)):
                depth=None if s is None else float(hi[2]-s)
                ratio=None if depth is None or depth<.25*ext[2]or c is None else float((hi[2]-c)/depth)
                rays.append(dict(axis=axis,sign=sign,index=i,xy_m=xy[i],source_hit_z_m=s,compound_hit_z_m=c,
                    source_recess_m=depth,source_qualifying_recess=ratio is not None,
                    retained_recess_fraction=ratio,source_empty_ray_filled=bool(s is None and c is not None)))
    ratios=[r['retained_recess_fraction']for r in rays if r['retained_recess_fraction']is not None]
    return dict(AABB_error_m=error,AABB_limit_m=limit,AABB_pass=error<=limit,
        sampled_surface_disagreement=dict(source_samples=len(sv),compound_samples=len(cv),
            source_to_component_triangles_max_m=float(d1.max()),source_to_component_triangles_p95_m=float(np.quantile(d1,.95)),
            component_to_source_triangles_max_m=float(d2.max()),component_to_source_triangles_p95_m=float(np.quantile(d2,.95)),
            boundary='Deterministic sampled unsigned vertex distances; includes internal component surfaces. Not signed union error or Hausdorff certificate.'),
        concavity_observation=dict(policy='All3 axes/both directions, nine central rays each, source recess>=25% axis extent reported; descriptive only, no shape/class branch',
            ray_count=len(rays),qualifying_recess_rays=len(ratios),minimum_retained_recess_fraction=min(ratios)if ratios else None,
            filled_source_empty_rays=sum(r['source_empty_ray_filled']for r in rays),rays=rays,
            boundary='Sparse first-hit rays cannot certify all cavities or through-holes; not an asset acceptance or grasp-success gate'),
        one_convex_hull_substitution=False,native_result_part_count=len(components),
        asset_acceptance='Finite convex watertight positive-volume parts and all6 AABB bounds within max(1mm,2% longest source dimension); no grasp results used')


def worker(jobfile):
    job=json.loads(Path(jobfile).read_text());directory=Path(jobfile).parent;start=time.monotonic()
    result=dict(schema='anydex_boya_egad_external_asset_vhacd_v2',object_id=job['object_id'],complete=False,
        source=job['source'],physics_steps=0,network_forwards=0,training_steps=0)
    try:
        sys.path.insert(0,str(ROOT/'outputs/native_deps/python_overlay'))
        import pybullet as p
        def forbidden(*a,**kw):raise RuntimeError('Offline V-HACD forbids connect/physics')
        p.connect=forbidden;p.stepSimulation=forbidden
        if base.sha(job['source']['path'])!=job['source']['sha256']:raise ValueError('Source changed')
        _,visual=base.scaled_visual(job['source']['path'])
        source=directory/'scaled_source.obj'
        source.write_text(trimesh.exchange.obj.export_obj(visual,include_normals=False,include_color=False,include_texture=False,digits=16))
        output=directory/'vhacd_components.obj';native_log=directory/'vhacd_native.log'
        native_start=time.monotonic();p.vhacd(str(source),str(output),str(native_log),**PARAMETERS)
        native_s=time.monotonic()-native_start
        components=fallback.split_obj(output)
        measured=approximation(visual,components)
        runtime=dict(method='PyBullet standalone V-HACD; no client connection',parameters=PARAMETERS,
            pybullet_native=str(Path(p.__file__).resolve()),pybullet_native_sha256=base.sha(p.__file__),
            API_version=p.getAPIVersion(),native_wall_s=native_s,
            raw_components=str(output),raw_components_sha256=base.sha(output),source_scaled_obj_sha256=base.sha(source))
        result.update(approximation=measured,decomposition=runtime)
        if not measured['AABB_pass']:raise ValueError('V-HACD exceeds fixed AABB approximation bound')
        record=base.package(job,components,runtime)
        geo=record['geometry_approximation'];geo.pop('coacd_parameters',None);geo.pop('coacd_runtime',None)
        geo.update(decomposition_method='PyBullet standalone V-HACD',decomposition_parameters=PARAMETERS,
            decomposition_runtime=runtime,approximation_checks=measured,
            geometry_boundary='V-HACD approximate collision geometry. Fixed AABB gate plus descriptive sampled surface/concavity records; not Hausdorff, topology or native20um source-mesh accuracy certificate.')
        base.save(record['metadata_path'],geo);record['metadata_sha256']=base.sha(record['metadata_path'])
        base.save(Path(record['object_root'])/'PROVENANCE.json',record)
        result.update(complete=True,status='ready',record=record)
    except Exception as exc:
        result.update(status='asset_preparation_failed',error=repr(exc),traceback=traceback.format_exc(),partial_assets_retained=True)
    result['wall_s']=time.monotonic()-start
    base.save(directory/'RESULT.json',result)
    return 0 if result['complete']else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('inventory','asset-root','output-dir','run-dir','previous-stop'):p.add_argument('--'+key,type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--worker-job',type=Path,help=argparse.SUPPRESS);args=p.parse_args()
    if args.worker_job:return worker(args.worker_job)
    if any(getattr(args,k)is None for k in ('inventory','asset_root','output_dir','run_dir','previous_stop')):p.error('All paths required')
    for k in ('inventory','asset_root','output_dir','run_dir','previous_stop'):setattr(args,k,getattr(args,k).resolve())
    if args.asset_root.exists()or args.output_dir.exists():raise ValueError('Fresh v2 assets/output required')
    rows=base.validate_inventory(json.loads(args.inventory.read_text()));digest=base.sha(args.inventory)
    stop=json.loads(args.previous_stop.read_text())
    if stop['processes_after']:raise ValueError('Original preparation processes remain alive')
    # Reuse original pre-forward exclusion hashes only after original manifests match.
    prior_selection=json.loads((args.previous_stop.parent/'SELECTION.json').read_text())
    refs=prior_selection['exclusion_manifest_refs']
    for ref in refs:
        if base.sha(ref['path'])!=ref['sha256']:raise ValueError('Exclusion manifest changed')
    old_job=json.loads((args.previous_stop.parent/'egad_A0/JOB.json').read_text());exclusion=old_job['exclusion']
    args.output_dir.mkdir(parents=True);args.asset_root.mkdir(parents=True);args.run_dir.mkdir(parents=True,exist_ok=True)
    selection=dict(schema='anydex_boya_egad_external_selection_v2',method='Uniform standalone V-HACD for all12',
        fixed_before_scores_or_outcomes=True,source_object_ids=list(base.FIXED_IDS),approved_object_ids=['egad_'+x for x in base.FIXED_IDS],
        source_inventory=str(args.inventory),source_inventory_sha256=digest,source_native_units_to_m=base.UNIT_M,
        parameters=PARAMETERS,worker_timeout_s=TIMEOUT_S,workers=args.workers,
        approximation_acceptance='Convex/watertight/positive-volume components and AABB bound max(1mm,2% longest source dimension)',
        surface_and_concavity_observations_descriptive_only=True,exclusion_manifest_refs=refs,
        original_preparation_stop=dict(path=str(args.previous_stop),sha256=base.sha(args.previous_stop)),
        helpers={str(path):base.sha(path)for path in (Path(__file__),Path(base.__file__),FALLBACK,GEOMETRY,Path(base.__file__).with_name('prepare_s40_scenes.py'))},
        old_CoACD_success_E0_not_reused=True,training_or_scores_or_outcomes_read=False)
    base.save(args.output_dir/'SELECTION.json',selection);started=time.monotonic()
    def one(row):
        oid='egad_'+row['object_id'];directory=args.output_dir/oid;directory.mkdir()
        job=dict(object_id=oid,source=row,inventory_sha256=digest,object_root=str(args.asset_root/oid),run_dir=str(args.run_dir),exclusion=exclusion)
        path=directory/'JOB.json';base.save(path,job)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        try:
            with(directory/'worker.log').open('xb')as log:
                done=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker-job',str(path)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=TIMEOUT_S)
            receipt=json.loads((directory/'RESULT.json').read_text())
            if done.returncode and receipt.get('complete'):raise ValueError('Ready receipt/exit mismatch')
        except Exception as exc:
            receipt=dict(schema='anydex_boya_egad_external_asset_vhacd_v2',object_id=oid,complete=False,status='asset_preparation_failed',
                error=repr(exc),source=row,physics_steps=0,network_forwards=0,training_steps=0,partial_assets_retained=True)
            base.save(directory/'RESULT.json',receipt)
        return receipt
    completed={}
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        for future in as_completed({pool.submit(one,row):row for row in rows}):
            result=future.result();completed[result['object_id']]=result
            base.save(args.output_dir/'PROGRESS.json',dict(status='preparing',completed=len(completed),total=12,
                ready=sum(r['complete']for r in completed.values()),objects=list(completed.values()),wall_s=time.monotonic()-started))
            print(json.dumps({k:result[k]for k in ('object_id','status','complete')}),flush=True)
    records=[]
    for row in rows:
        oid='egad_'+row['object_id'];result=completed[oid]
        if result['complete']:records.append(result['record'])
        else:records.append(dict(id=oid,object_id=oid,pilot_id=oid,egad_object_id=row['object_id'],split='external_validation',
            source_family='EGAD_official_eval',object_root=str(args.asset_root/oid),scale=1.,source_native_units_to_m=base.UNIT_M,
            source_mesh_sha256=row['sha256'],mesh_sha256=None,mesh_group_sha256=row['sha256'],source_inventory_sha256=digest,
            preflight_status='asset_preparation_failed',preflight_issues=[result['error']],
            observation_json=str(args.run_dir/'observations'/oid/'observation.json'),representation_dir=str(args.run_dir/'representations'/oid)))
    for row in rows:
        if base.sha(row['path'])!=row['sha256']:raise ValueError('Original EGAD source changed')
    for path,h in selection['helpers'].items():
        if base.sha(path)!=h:raise ValueError('Frozen v2 source changed')
    manifest=dict(schema='anydex_boya_external_scene_manifest_v1',source_inventory=str(args.inventory),source_inventory_sha256=digest,
        source_active_list_sha256=digest,source_active_list_boundary='Official49 EGAD inventory; not S40 or DGN list',source_dataset_root=str(args.asset_root),
        active_object_count=12,scale=1.,objects=records,camera=base.CAMERA,approved_object_ids=selection['approved_object_ids'],
        exclusion_manifest_refs=refs,source_native_units_to_m=base.UNIT_M,selection=str(args.output_dir/'SELECTION.json'),selection_sha256=base.sha(args.output_dir/'SELECTION.json'),
        split_boundary='Fixed external EGAD12; all12 identities including asset failures retained; not training or policy-tuning input',
        scene_boundary='Uniform V-HACD approximation, native axes, units .001m/native unit, mass.1kg/friction.6; not original AnyDex real printed-object protocol',
        original_AnyDex_CAD_identity_verified=False,original_scene_pose_reproduced=False,physics_steps=0,grasp_attempts=0,training_steps=0,network_forwards=0)
    base.save(args.output_dir/'objects.json',manifest)
    summary=dict(schema='anydex_boya_egad_external_preparation_v2',complete=True,status='complete',total_objects=12,
        ready_objects=sum(r['preflight_status']=='ready'for r in records),asset_failed_objects=[r['object_id']for r in records if r['preflight_status']!='ready'],
        scene_manifest=str(args.output_dir/'objects.json'),scene_manifest_sha256=base.sha(args.output_dir/'objects.json'),
        selection_sha256=manifest['selection_sha256'],asset_root=str(args.asset_root),previous_stop=selection['original_preparation_stop'],
        worker_wall_s_sum=sum(r.get('wall_s',0)for r in completed.values()),wall_s=time.monotonic()-started,
        physics_steps=0,grasp_attempts=0,training_steps=0,network_forwards=0)
    base.save(args.output_dir/'SCENE_PREPARATION.json',summary);print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['ready_objects']==12 else 2


if __name__=='__main__':raise SystemExit(main())
