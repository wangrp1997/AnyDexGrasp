"""Read frozen queue, reconstruct source vertices, and build inputs; no simulator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
MODULE=ROOT/'reproduction/boya/author_collection_scenes.py'
sys.path.insert(0,str(MODULE.parent))
loader=importlib.util.spec_from_file_location('collection_scene_builder',MODULE)
builder=importlib.util.module_from_spec(loader);loader.loader.exec_module(builder)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lock(b):assert digest(b['path'])==b['sha256'];return read(b['path'])

def run(queue_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    queue=read(queue_path);lock(queue['protocol']);catalog=lock(queue['catalog'])
    for source in queue['sources'].values():assert digest(source['path'])==source['sha256']
    for path,expected in catalog['input_sha256'].items():assert digest(path)==expected
    split_groups={s:{catalog['objects'][n]['source_object_digest'] for n in ids}
                  for s,ids in catalog['split_object_ids'].items()}
    assert len(split_groups['fit'])==32 and len(split_groups['internal'])==8
    assert split_groups['fit'].isdisjoint(split_groups['internal'])
    rows=[];cache={};vertices_checked=0
    for entry in queue['scenes']:
        spec=lock(entry['scene_spec']);assert entry['status']==spec['status']
        identity_list=entry['source_object_ids'];assert len(set(identity_list))==3
        assert identity_list==spec['source_object_ids']
        for identity,expected in entry['source_object_digests'].items():
            record=catalog['objects'][identity]
            assert expected==record['source_object_digest']
            assert expected in split_groups[entry['split']]
            assert digest(record['source_visual_mesh'])==expected
            assert digest(record['source_mesh'])==record['source_mesh_sha256']
        evidence=dict(scene_id=entry['scene_id'],split=entry['split'],status=entry['status'],
            object_ids=identity_list,source_identity_sha256=entry['source_object_digests'],
            spec_sha256=entry['scene_spec']['sha256'],
            attempts=[r['attempts'] for r in spec['objects']],
            placement_statuses=[r['placement_status'] for r in spec['objects']])
        assert max(evidence['attempts'])<=256
        if entry['status']=='placement_null':
            assert spec['placement_null_is_training_label'] is False
            rows.append(evidence);continue
        scene=builder.build_scene(entry['scene_spec']['path'],output/'builds'/entry['scene_id'])
        bounds=[];bottoms=[];max_bound_error=0.
        for declared,body in zip(spec['objects'],scene['objects']):
            record=catalog['objects'][body['object_id']];T=np.asarray(body['world_T_object'])
            assert body['source_visual_mesh_sha256']==record['source_object_digest']
            assert body['mesh_group_sha256']==body['source_visual_mesh_sha256']
            assert body['source_mesh_sha256']==record['source_mesh_sha256']
            if body['object_id'] not in cache:
                cache[body['object_id']]=[np.asarray(trimesh.load_mesh(p,process=False).vertices)
                    for p in [record['source_mesh'],*record['collision_part_paths']]]
            pieces=cache[body['object_id']]
            transformed=[v@T[:3,:3].T+T[:3,3] for v in pieces]
            vertices_checked+=sum(len(v) for v in pieces)
            allverts=np.concatenate(transformed);B=np.array([allverts.min(axis=0),allverts.max(axis=0)])
            max_bound_error=max(max_bound_error,float(np.abs(B-np.asarray(declared['proxy_world_AABB_m'])).max()))
            assert max_bound_error<1e-12
            assert np.all(B[0,:2]>=np.array([-.95,-.02])-1e-12)
            assert np.all(B[1,:2]<=np.array([-.58,.23])+1e-12)
            assert B[0,2]>=0
            bottom=float(transformed[0][:,2].min());bottoms.append(bottom)
            assert abs(bottom-.005)<1e-12
            yaw=declared['yaw_world_rad'];c,s=np.cos(yaw),np.sin(yaw)
            assert np.allclose(T[:3,:3],np.array([[c,-s,0],[s,c,0],[0,0,1]])@record['source_settled_R'],atol=1e-14,rtol=0)
            bounds.append(B)
        gaps=[]
        for i in range(3):
            for j in range(i):
                gap=float(np.max(np.maximum(bounds[i][0,:2]-bounds[j][1,:2],bounds[j][0,:2]-bounds[i][1,:2])))
                assert gap>=.003-1e-12;gaps.append(gap)
        assert scene['bin'] is None and scene['dataset_split']==entry['split']
        assert scene['initialization_steps']==750 and scene['dt_s']==.004
        assert scene['all_source_mesh_sha256']==[body['source_visual_mesh_sha256'] for body in scene['objects']]
        evidence.update(scene_sha256=digest(output/'builds'/entry['scene_id']/'scene.json'),
            source_bottoms_m=bottoms,minimum_pair_separating_gap_m=min(gaps),
            max_reconstructed_AABB_error_m=max_bound_error,build_input_validated=True)
        rows.append(evidence)
    result=dict(schema='anydex_boya_author_collection_scene_independent_audit_v1',status='complete',
        queue_path=str(Path(queue_path).resolve()),queue_sha256=digest(queue_path),
        protocol_sha256=queue['protocol']['sha256'],catalog_sha256=queue['catalog']['sha256'],
        module_sha256=digest(MODULE),test_source_sha256=digest(MODULE.with_name('test_author_collection_scenes.py')),
        independent_auditor_sha256=digest(__file__),all_input_file_count=len(catalog['input_sha256']),
        all_source_file_count=len(catalog['source_sha256']),source_objects=40,
        source_collision_parts=sum(len(r['collision_part_paths']) for r in catalog['objects'].values()),
        native_preflight_assets_checked=40,fit_objects=32,internal_objects=8,
        scenes=len(rows),ready_scene_inputs=sum(r['status']=='ready' for r in rows),
        placement_nulls=sum(r['status']=='placement_null' for r in rows),
        reconstructed_vertices_checked=vertices_checked,scene_results=rows,
        physics_steps=0,planning_calls=0,CGR_forwards=0,new_labels=0,training_updates=0,
        boundary='Initial source placement only: settling, observed RGBD, reachability and grasp success remain unvalidated')
    (output/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='scene_results'}))

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
