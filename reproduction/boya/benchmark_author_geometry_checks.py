"""Zero-physics comparison against original saved full-source measurements."""
import argparse
from pathlib import Path
import time
import numpy as np
import trimesh
import author_collection_batch as B
import author_lift as L
import author_fast_geometry_checks as F
import audit_author_scene as A


def benchmark(root,output):
    root=Path(root).resolve();started=time.monotonic();F.precompile();rounds=[];sources={}
    for index in ('00','01'):
        d=root/'scenes'/index
        ready=B.read(d/'live/READY.json');scene=B.read(ready['scene']['path'])
        rows=B.read(d/'live/execution/MULTI_OBJECT_TRACE.json')['rows'];objects=[]
        for obj in scene['objects']:
            key=obj['object_id'];path=Path(obj['source_mesh']);B.verify({'mesh':dict(path=str(path),sha256=obj['source_mesh_sha256'])})
            mesh=trimesh.load_mesh(path,process=False);vertices=np.asarray(mesh.vertices)*obj['scale'];poses=np.asarray([r['objects'][key]['pose_xyzw']for r in rows])
            choose=np.unique(np.linspace(0,len(rows)-1,32).astype(int));sample=poses[choose]
            start=time.perf_counter();original=[L.source_metrics(vertices,p)for p in sample];old=time.perf_counter()-start
            start=time.perf_counter();fast=[F.source_metrics(vertices,p)for p in sample];new=time.perf_counter()-start
            error=max(abs(a[k]-b[k])for a,b in zip(original,fast)for k in a)
            if error>1e-12:raise ValueError('Original metric differs from optimized full-source scan')
            start=time.perf_counter();maximum=0.
            for row,p in zip(rows,poses):
                value=F.source_metrics(vertices,p);maximum=max(maximum,*[abs(value[k]-row['objects'][key][k])for k in value])
            full=time.perf_counter()-start
            if maximum>1e-12:raise ValueError('A saved full execution extremum changed')
            result=dict(object_id=key,source_vertices=len(vertices),source_mesh=B.binding(path),poses_checked=len(poses),
                paired_32_full_original_wall_s=old,paired_32_full_fast_wall_s=new,paired_component_ratio=old/new,
                maximum_original_sample_difference_m=error,maximum_saved_all_pose_difference_m=maximum,
                all_saved_poses_full_vertex_scan_s=full)
            objects.append(result);print(dict(round=index,object=key,ratio=old/new,all_poses_max_error=maximum),flush=True)
        obs=B.read(d/'capture/observation.json');observed=B.read(d/'capture/observed_scene.json');ray=B.read(d/'capture/render_geometry_check.json')
        tri=[]
        for obj in observed['objects']:
            m=trimesh.load_mesh(obj['source_mesh'],process=False);m.apply_scale(obj['scale']);m.apply_transform(obj['world_T_object']);tri.append(m.triangles)
        tri=np.concatenate(tri);K=np.asarray(obs['intrinsic_matrix']);T=np.asarray(obs['T_WC']);expected=[];sample_original=[];sample_fast=[];orig_s=0.
        begin=time.perf_counter()
        for pixel in np.asarray(ray['pixels_xy']):
            direction=T[:3,:3]@np.r_[np.linalg.solve(K[:2,:2],pixel+.5-K[:2,2]),1.]
            expected.append(F.first_depth(tri,T[:3,3],direction))
        fast_s=time.perf_counter()-begin
        for i in (0,31,63,95):
            pixel=np.asarray(ray['pixels_xy'][i]);direction=T[:3,:3]@np.r_[np.linalg.solve(K[:2,:2],pixel+.5-K[:2,2]),1.]
            begin=time.perf_counter();a=A.first_depth(tri,T[:3,3],direction);orig_s+=time.perf_counter()-begin
            sample_original.append(a);sample_fast.append(expected[i])
        np.testing.assert_allclose(np.asarray(sample_original)[:,0],np.asarray(sample_fast)[:,0],atol=1e-10,rtol=0)
        if any(a[1:]!=b[1:]for a,b in zip(sample_original,sample_fast)):raise ValueError('Ray first-hit identity/tie changed')
        saved=np.asarray([np.inf if x is None else x for x in ray['expected_source_first_depth_m']]);got=np.asarray([x[0]for x in expected])
        np.testing.assert_allclose(got,saved,atol=1e-10,rtol=0)
        rounds.append(dict(index=index,objects=objects,full_source_triangles=len(tri),all_saved_rays=len(expected),
            full_fast_96_ray_wall_s=fast_s,original_4_ray_wall_s=orig_s,
            ray_maximum_saved_difference_m=float(np.max(abs(got-saved))),ray_original_subset_differences_below_1e_minus10_m=True))
        for p in (d/'live/execution/MULTI_OBJECT_TRACE.json',d/'capture/render_geometry_check.json'):sources[str(p)]=B.binding(p)
    result=dict(status='pass',rounds=rounds,source_bindings=sources,implementation=B.binding(F.__file__),benchmark_source=B.binding(__file__),
        wall_s=time.monotonic()-started,new_Kit_starts=0,physics_steps=0,new_labels=0,training_updates=0,
        all_source_vertices_and_triangles_retained=True,scope='Component timing and saved-data equivalence; not a live collection speedup')
    B.verify(sources);B.save(output,result,exclusive=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=benchmark(a.root,a.output);print({'status':r['status'],'wall_s':r['wall_s']})
