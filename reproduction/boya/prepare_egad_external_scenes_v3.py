#!/usr/bin/env python3
"""Normalize ALL fixed EGAD12 V-HACD convex envelopes without rerunning V-HACD.

Only re-triangulate each original component's convex hull with its exact source
vertex subset. No added point, scale, inflation, component removal or visual edit.
The v2 raw output and all failed receipts remain immutable and SHA-bound.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial import ConvexHull
import trimesh
import prepare_egad_external_scenes as base
import prepare_egad_external_scenes_v2 as previous


def normalize_component(vertices,faces):
    v=np.asarray(vertices,dtype=float);f=np.asarray(faces,dtype=np.int32)
    original=trimesh.Trimesh(v,f,process=False)
    hull=ConvexHull(v)  # No QJ, new points, offsets, resampling or perturbation.
    oriented=np.asarray(hull.simplices).copy()
    for i,face in enumerate(oriented):
        cross=np.cross(v[face[1]]-v[face[0]],v[face[2]]-v[face[0]])
        if cross@hull.equations[i,:3]<0:oriented[i,[1,2]]=oriented[i,[2,1]]
    kept=np.unique(oriented);lookup=np.full(len(v),-1,dtype=int);lookup[kept]=np.arange(len(kept))
    out_v=v[kept].copy();out_f=lookup[oriented].astype(np.int32)
    out=trimesh.Trimesh(out_v,out_f,process=False)
    point_plane=v@hull.equations[:,:3].T+hull.equations[:,3]
    maximum=float(point_plane.max())
    if not(out.is_convex and out.is_watertight and out.is_winding_consistent and abs(out.volume)>1e-14):
        raise ValueError('Normalized original convex envelope invalid')
    if maximum>1e-12 or not np.array_equal(out_v,v[kept]):
        raise ValueError('Hull no longer encloses original source vertices')
    old_bounds=np.array([v.min(0),v.max(0)]);new_bounds=np.array([out_v.min(0),out_v.max(0)])
    if not np.array_equal(old_bounds,new_bounds):raise ValueError('Original component AABB changed')
    rng=np.random.default_rng(20261107);directions=np.vstack([np.eye(3),-np.eye(3),rng.normal(size=(64,3))])
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    support_error=float(np.abs((v@directions.T).max(0)-(out_v@directions.T).max(0)).max())
    if support_error>1e-14:raise ValueError('Convex support changed')
    normals=np.asarray(original.face_normals);tri=np.asarray(original.triangles)
    original_plane=(v@normals.T-np.einsum('ij,ij->i',tri[:,0],normals)).max()
    return (out_v,out_f),dict(original_vertex_count=len(v),original_triangle_count=len(f),
        output_vertex_count=len(out_v),output_triangle_count=len(out_f),
        original_is_convex=bool(original.is_convex),original_is_watertight=bool(original.is_watertight),
        original_winding_consistent=bool(original.is_winding_consistent),
        original_volume_m3=float(abs(original.volume)),convex_envelope_volume_m3=float(abs(out.volume)),
        original_face_plane_max_vertex_violation_m=float(original_plane),
        original_point_to_hull_max_signed_plane_m=maximum,
        original_vertices_affine_rank=int(np.linalg.matrix_rank(v-v.mean(0))),
        exact_source_vertex_indices=kept.tolist(),output_vertices_exact_original_subset=True,
        AABB_bitwise_equal=True,sampled_support_directions=len(directions),sampled_support_max_error_m=support_error,
        convex_envelope_preservation='Output vertices are original subset; every original point lies in same convex hull within reported floating point residual. Thus support sets coincide; sampled directions are supplementary, not sole certificate.',
        normalization='Re-triangulate convex envelope; no added points/QJ/inflation/scale/component deletion')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source-scenes','asset-root','output-dir','run-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4);args=p.parse_args()
    for k in ('source_scenes','asset_root','output_dir','run_dir'):setattr(args,k,getattr(args,k).resolve())
    if args.asset_root.exists()or args.output_dir.exists():raise ValueError('Fresh v3 output required')
    source_selection=args.source_scenes/'SELECTION.json';old=json.loads(source_selection.read_text())
    approved=['egad_'+x for x in base.FIXED_IDS]
    if old['approved_object_ids']!=approved:raise ValueError('Require same fixed EGAD12 before any forward')
    inventory=Path(old['source_inventory']);base.validate_inventory(json.loads(inventory.read_text()))
    if base.sha(inventory)!=old['source_inventory_sha256']:raise ValueError('Source inventory changed')
    for ref in old['exclusion_manifest_refs']:
        if base.sha(ref['path'])!=ref['sha256']:raise ValueError('Exclusion source changed')
    jobs=[];input_sha={str(source_selection):base.sha(source_selection),str(inventory):base.sha(inventory)}
    for oid in approved:
        jobfile=args.source_scenes/oid/'JOB.json';receipt=args.source_scenes/oid/'RESULT.json'
        job=json.loads(jobfile.read_text());result=json.loads(receipt.read_text())
        raw=Path(result['decomposition']['raw_components'])
        if base.sha(raw)!=result['decomposition']['raw_components_sha256']:raise ValueError('Original V-HACD output changed')
        if job['object_id']!=oid or base.sha(job['source']['path'])!=job['source']['sha256']:raise ValueError('Original mesh changed')
        input_sha.update({str(jobfile):base.sha(jobfile),str(receipt):base.sha(receipt),str(raw):base.sha(raw)})
        jobs.append(dict(job=job,source_receipt=result,receipt_path=str(receipt),raw=str(raw)))
    args.output_dir.mkdir(parents=True);args.asset_root.mkdir(parents=True)
    selection=dict(schema='anydex_boya_egad_external_selection_v3',approved_object_ids=approved,
        fixed_before_scores_or_outcomes=True,source_native_units_to_m=base.UNIT_M,
        decomposition_method='Same fixed v2 V-HACD raw outputs; uniform convex-envelope triangulation normalization',
        original_vhacd_parameters=previous.PARAMETERS,decomposition_rerun=False,
        source_selection=str(source_selection),source_selection_sha256=base.sha(source_selection),
        source_inventory=str(inventory),source_inventory_sha256=base.sha(inventory),
        exclusion_manifest_refs=old['exclusion_manifest_refs'],input_sha256=input_sha,
        helper_sha256={str(Path(path).resolve()):base.sha(path)for path in (Path(__file__),Path(base.__file__),Path(previous.__file__),previous.FALLBACK,previous.GEOMETRY)},
        all_components_normalized=True,component_removal_allowed=False,visual_changes_allowed=False,
        original_failure_receipts_preserved=True,physics_steps=0,network_forwards=0,training_steps=0)
    base.save(args.output_dir/'SELECTION.json',selection);start=time.monotonic()
    def one(item):
        oldjob=item['job'];oid=oldjob['object_id'];directory=args.output_dir/oid;directory.mkdir()
        started=time.monotonic();result=dict(schema='anydex_boya_egad_external_asset_vhacd_v3',object_id=oid,
            complete=False,physics_steps=0,network_forwards=0,training_steps=0,source_v2_result=item['receipt_path'])
        try:
            raw=previous.fallback.split_obj(item['raw']);normalized=[];checks=[]
            for v,f in raw:
                component,check=normalize_component(v,f);normalized.append(component);checks.append(check)
            if len(normalized)!=len(raw):raise ValueError('Component count changed')
            job=dict(oldjob,object_root=str(args.asset_root/oid),run_dir=str(args.run_dir))
            _,visual=base.scaled_visual(job['source']['path']);measured=previous.approximation(visual,normalized)
            runtime=dict(item['source_receipt']['decomposition'],decomposition_rerun=False,
                source_raw_obj_sha256=base.sha(item['raw']),normalization='Original vertex-set convex envelopes, outward hull triangles; no added vertices or inflation')
            if not measured['AABB_pass']:raise ValueError('Normalized V-HACD exceeds same fixed AABB bound')
            record=base.package(job,normalized,runtime);geo=record['geometry_approximation']
            geo.pop('coacd_parameters',None);geo.pop('coacd_runtime',None)
            geo.update(decomposition_method='PyBullet standalone V-HACD plus uniform source-vertex convex-envelope triangulation',
                decomposition_parameters=previous.PARAMETERS,decomposition_runtime=runtime,
                approximation_checks=measured,component_normalization=checks,source_v2_result_sha256=base.sha(item['receipt_path']),
                geometry_boundary='Unchanged V-HACD convex envelopes; numerical face normalization only. V-HACD source-mesh approximation remains and is reported separately from native20um import audit.')
            base.save(record['metadata_path'],geo);record['metadata_sha256']=base.sha(record['metadata_path'])
            base.save(Path(record['object_root'])/'PROVENANCE.json',record)
            result.update(complete=True,status='ready',record=record,normalization_checks=checks,approximation=measured)
        except Exception as exc:result.update(status='asset_preparation_failed',error=repr(exc),traceback=traceback.format_exc())
        result['wall_s']=time.monotonic()-started;base.save(directory/'RESULT.json',result);return result
    completed={}
    with ThreadPoolExecutor(max_workers=args.workers)as pool:
        for future in as_completed({pool.submit(one,item):item for item in jobs}):
            result=future.result();completed[result['object_id']]=result
            base.save(args.output_dir/'PROGRESS.json',dict(status='preparing',completed=len(completed),total=12,
                ready=sum(x['complete']for x in completed.values()),wall_s=time.monotonic()-start))
            print(json.dumps({k:result[k]for k in ('object_id','status','complete')}),flush=True)
    records=[]
    for item in jobs:
        oid=item['job']['object_id'];result=completed[oid]
        if result['complete']:records.append(result['record'])
        else:records.append(dict(object_id=oid,id=oid,pilot_id=oid,split='external_validation',egad_object_id=item['job']['source']['object_id'],
            source_mesh_sha256=item['job']['source']['sha256'],mesh_sha256=None,scale=1.,object_root=str(args.asset_root/oid),
            preflight_status='asset_preparation_failed',preflight_issues=[result['error']],observation_json=str(args.run_dir/'observations'/oid/'observation.json'),
            representation_dir=str(args.run_dir/'representations'/oid)))
    for path,h in {**selection['input_sha256'],**selection['helper_sha256']}.items():
        if base.sha(path)!=h:raise ValueError('Frozen input/helper changed')
    manifest=dict(schema='anydex_boya_external_scene_manifest_v1',source_inventory=str(inventory),source_inventory_sha256=base.sha(inventory),
        source_active_list_sha256=base.sha(inventory),source_active_list_boundary='Official49 EGAD inventory, not S40/DGN list',source_dataset_root=str(args.asset_root),
        active_object_count=12,scale=1.,objects=records,camera=base.CAMERA,approved_object_ids=approved,exclusion_manifest_refs=old['exclusion_manifest_refs'],
        source_native_units_to_m=base.UNIT_M,selection=str(args.output_dir/'SELECTION.json'),selection_sha256=base.sha(args.output_dir/'SELECTION.json'),
        split_boundary='Fixed external12; no train/model/policy selection by these objects; all failures retained',
        scene_boundary='Native-axis .001m/native unit, uniform V-HACD envelopes and fixed mass.1kg/friction.6 simulation approximation, not original paper real EGAD protocol',
        original_AnyDex_CAD_identity_verified=False,original_scene_pose_reproduced=False,physics_steps=0,grasp_attempts=0,training_steps=0,network_forwards=0)
    base.save(args.output_dir/'objects.json',manifest)
    allchecks=[c for result in completed.values()for c in result.get('normalization_checks',[])]
    summary=dict(schema='anydex_boya_egad_external_preparation_v3',complete=True,status='complete',total_objects=12,
        ready_objects=sum(x['preflight_status']=='ready'for x in records),asset_failed_objects=[x['object_id']for x in records if x['preflight_status']!='ready'],
        scene_manifest=str(args.output_dir/'objects.json'),scene_manifest_sha256=base.sha(args.output_dir/'objects.json'),selection_sha256=manifest['selection_sha256'],
        asset_root=str(args.asset_root),normalized_components=len(allchecks),original_nonconvex_triangle_meshes=sum(not c['original_is_convex']for c in allchecks),
        max_original_point_to_hull_signed_plane_m=max((c['original_point_to_hull_max_signed_plane_m']for c in allchecks),default=None),
        max_supplementary_support_error_m=max((c['sampled_support_max_error_m']for c in allchecks),default=None),
        all_AABB_bitwise_equal=all(c['AABB_bitwise_equal']for c in allchecks),all_output_vertices_original_subset=all(c['output_vertices_exact_original_subset']for c in allchecks),
        worker_wall_s_sum=sum(x['wall_s']for x in completed.values()),wall_s=time.monotonic()-start,
        source_v2_costs_retained=True,new_decomposition_calls=0,physics_steps=0,grasp_attempts=0,training_steps=0,network_forwards=0)
    base.save(args.output_dir/'SCENE_PREPARATION.json',summary);print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['ready_objects']==12 else 2


if __name__=='__main__':raise SystemExit(main())
