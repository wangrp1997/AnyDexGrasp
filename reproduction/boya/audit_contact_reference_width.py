"""Read-only actual pad/mesh/reference distances; not a contact certificate.

Retain degenerate source triangles as segments for finite nearest-point
measurement. No geometry writes, pose fitting, candidate search or physics.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh
import author_collection_batch as B
import render_author_boya_candidate_comparison as R


def nearest_surface(triangles,point):
    t=np.asarray(triangles,dtype=float);p=np.asarray(point,dtype=float)
    a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a
    normal=np.cross(ab,ac);area2=np.einsum('ij,ij->i',normal,normal)
    projected=p-np.divide(np.einsum('ij,ij->i',p-a,normal),area2,out=np.zeros(len(t)),where=area2>0)[:,None]*normal
    v=projected-a;d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
    d20=np.einsum('ij,ij->i',v,ab);d21=np.einsum('ij,ij->i',v,ac)
    u=np.divide(d11*d20-d01*d21,area2,out=np.full(len(t),np.inf),where=area2>0)
    w=np.divide(d00*d21-d01*d20,area2,out=np.full(len(t),np.inf),where=area2>0)
    inside=(area2>0)&(u>=0)&(w>=0)&(u+w<=1)
    best=np.full((len(t),3),np.nan);distance2=np.full(len(t),np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        delta=end-start;length2=np.einsum('ij,ij->i',delta,delta)
        f=np.clip(np.divide(np.einsum('ij,ij->i',p-start,delta),length2,out=np.zeros(len(t)),where=length2>0),0,1)
        closest=start+f[:,None]*delta;d2=np.einsum('ij,ij->i',closest-p,closest-p)
        replace=d2<distance2;best[replace]=closest[replace];distance2[replace]=d2[replace]
    d2=np.einsum('ij,ij->i',projected-p,projected-p);replace=inside&(d2<distance2)
    best[replace]=projected[replace];distance2[replace]=d2[replace]
    i=int(np.argmin(distance2))
    if not np.isfinite(best[i]).all()or not np.isfinite(distance2[i]):raise ValueError('Finite nearest triangle/segment required')
    return best[i],float(np.sqrt(max(distance2[i],0.))),i,int((area2==0).sum())


def winding_number(triangles,point):
    vectors=np.asarray(triangles)-point;length=np.linalg.norm(vectors,axis=-1)
    a,b,c=vectors[:,0],vectors[:,1],vectors[:,2]
    determinant=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=length.prod(axis=1)+np.einsum('ij,ij->i',a,b)*length[:,2]+np.einsum('ij,ij->i',b,c)*length[:,0]+np.einsum('ij,ij->i',c,a)*length[:,1]
    return float(np.sum(2*np.arctan2(determinant,denominator))/(4*np.pi))


def run(benchmark,output):
    root=Path(benchmark).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=B.read(root/'PROTOCOL.json');B.verify(protocol['inputs']);B.verify(protocol['sources'])
    candidate=B.read(protocol['inputs']['candidate']['path']);scene=B.read(candidate['scene_binding']['scene_json'])
    objects=[];bindings={}
    for row in scene['objects']:
        mesh=trimesh.load_mesh(row['source_mesh'],process=False)
        mesh.vertices=R.transform(mesh.vertices*row['scale'],np.asarray(row['world_T_object']))
        objects.append((row['object_id'],mesh,np.asarray(mesh.triangles)))
        bindings[row['object_id']]=B.binding(row['source_mesh'])
    assets=R.BoyaAssets();branches=[]
    with np.load(root/'EXAMPLE_MAPPINGS.npz',allow_pickle=False)as z:
        for j,opponent in enumerate(('ff','mf','rf','lf')):
            if not z['valid'][0,j]:continue
            palm=z['world_T_palm'][0,j];q=z['q_contact'][0,j]
            points=R.transform(z['pad_points_palm'][0,j],palm)[[0,j+1]]
            hand=R.hand_meshes(assets,q,palm);values=[]
            for k,group in enumerate(('th',opponent)):
                closest=[]
                for oid,mesh,triangles in objects:
                    point,dist,index,degenerate=nearest_surface(triangles,points[k])
                    closest.append(dict(object_id=oid,distance_mm=dist*1000,nearest_surface_world_m=point.tolist(),
                        triangle_index=index,zero_area_triangles=degenerate,watertight=bool(mesh.is_watertight),
                        winding_consistent=bool(mesh.is_winding_consistent),winding_number=winding_number(triangles,points[k])))
                nearest=min(closest,key=lambda a:a['distance_mm'])
                m=next(m for m in hand if m['name']==group+'distal')
                _,own_distance,_,_=nearest_surface(m['vertices'][m['faces']],points[k])
                values.append(dict(group=group,world_point_m=points[k].tolist(),point_to_own_rendered_link_surface_mm=own_distance*1000,
                    nearest_object_surface=nearest,all_scene_distances=closest))
            branches.append(dict(opponent=opponent,closure_progress=float(z['progress'][0,j]),raw_CGR_width_mm=candidate['raw_width_m']*1000,
                measured_reference_point_distance_mm=float(np.linalg.norm(points[1]-points[0])*1000),points=values))
    result=dict(status='finite_source_mesh_distance_measurement_complete',source=B.binding(__file__),source_candidate=protocol['inputs']['candidate'],
        mappings=B.binding(root/'EXAMPLE_MAPPINGS.npz'),scene=B.binding(candidate['scene_binding']['scene_json']),object_bindings=bindings,branches=branches,
        width_semantics='Distance of two selected source-surface reference points; not the whole-hand envelope, closest inner-surface gap or actual loaded contact width',
        contact_or_bearing_certified=False,physical_grasp_complete=False,production_ready=False,
        mesh_scope='Complete original source triangles; zero-area triangles retained as edges; source assets not repaired or changed',
        winding_scope='Diagnostic winding only; no inside/outside certificate for nonwatertight or inconsistent meshes',
        physics_steps=0,IK_calls=0,new_labels=0,training_updates=0)
    B.verify(bindings);B.verify(protocol['inputs']);B.verify(protocol['sources']);B.save(out/'MEASUREMENTS.json',result,exclusive=True)
    print([{k:b[k]for k in ('opponent','closure_progress','measured_reference_point_distance_mm')}|{'surface_distances_mm':[(p['group'],p['nearest_object_surface']['distance_mm'],p['nearest_object_surface']['winding_number'])for p in b['points']]}for b in branches])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.benchmark_dir,a.output_dir)
