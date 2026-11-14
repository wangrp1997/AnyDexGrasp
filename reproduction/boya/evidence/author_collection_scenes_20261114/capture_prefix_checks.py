"""Independent read-only streamed initialization/observation audit; never Kit."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())

def audit(directory,output):
    directory=Path(directory).resolve();capture=directory/'capture'
    artifacts={}
    def bound(path):
        path=Path(path);artifacts[str(path)]=sha(path);return read(path)
    journal=bound(capture/'CONTACT_JOURNAL.json');path=Path(journal['path'])
    assert journal['status']=='complete' and journal['frames']==750
    assert sha(path)==journal['sha256'];artifacts[str(path)]=journal['sha256']
    roots=journal['original_object_paths'];floors=set(journal['floor_collision_paths'])
    counts={r:0 for r in roots};last_support={r:0 for r in roots};all_support={r:0 for r in roots}
    frames=contacts=0;logical=hashlib.sha256();pairs={}
    with gzip.open(path,'rb') as stream:
        for line in stream:
            logical.update(line);row=json.loads(line);frames+=1
            assert row['step']==frames
            supported=set()
            for c in row['contacts']:
                a,b=c['bodies'];assert a in counts
                assert np.isfinite(c['normal_force_n']) and np.isfinite(c['separation_m'])
                counts[a]+=1;contacts+=1
                pair=a+' -> '+b;pairs[pair]=pairs.get(pair,0)+1
                if b in floors and c['normal_force_n']>.001:supported.add(a)
            for root in supported:
                all_support[root]+=1
                if frames>500:last_support[root]+=1
    assert frames==750 and contacts==journal['contacts']
    assert logical.hexdigest()==journal['logical_jsonl_sha256']
    check=bound(capture/'CONTACT_SENSOR_CHECK.json')
    evidence={r:dict(final_window_supported_ticks=last_support[r],contact_rows=counts[r]) for r in roots}
    assert evidence==journal['evidence']==check['evidence']
    assert all(v>0 for v in last_support.values())
    native=bound(capture/'native/native_audit.json')
    assert native['complete'] and native['within_20um'] and native['physics_steps']==0
    trace_path=capture/'initialization_trace.npz';artifacts[str(trace_path)]=sha(trace_path)
    with np.load(trace_path,allow_pickle=False) as z:
        poses=np.asarray(z['pose_xyzw'],float);velocities=np.asarray(z['velocity'],float)
    assert poses.shape==(751,3,7) and velocities.shape==(751,3,6)
    assert np.isfinite(poses).all() and np.isfinite(velocities).all()
    original=bound(directory/'build/scene.json');observed=bound(capture/'observed_scene.json')
    assert sha(directory/'build/scene.json')==observed['source_scene_sha256']
    assert [r['object_id'] for r in original['objects']]==[r['object_id'] for r in observed['objects']]
    assert original['all_source_mesh_sha256']==observed['all_source_mesh_sha256']
    bodies=[]
    for i,(initial,obj) in enumerate(zip(original['objects'],observed['objects'])):
        for key in ('source_mesh','source_visual_mesh'):
            expected=obj[key+'_sha256'];assert sha(obj[key])==expected;artifacts[obj[key]]=expected
        R=Rotation.from_quat(poses[:,i,3:]).as_matrix();T=np.asarray(obj['world_T_object'])
        assert np.max(abs(T[:3,:3]-R[-1]))<1e-12 and np.array_equal(T[:3,3],poses[-1,i,:3])
        Ti=np.asarray(initial['world_T_object'])
        assert np.max(abs(Ti[:3,:3]-R[0]))<2e-7 and np.max(abs(Ti[:3,3]-poses[0,i,:3]))<1e-7
        vertices=np.asarray(trimesh.load_mesh(obj['source_mesh'],process=False).vertices)
        bottoms=[]
        for start in range(0,751,16):
            projections=vertices@R[start:start+16,2,:].T
            bottoms.extend((projections.min(axis=0)+poses[start:start+16,i,2]).tolist())
        assert abs(bottoms[0]-.005)<1e-7
        p=poses[-250:,i];v=velocities[-250:,i];q=p[:,3:]/np.linalg.norm(p[:,3:],axis=1)[:,None]
        metrics=dict(max_linear_speed_m_s=float(np.linalg.norm(v[:,:3],axis=1).max()),
            max_angular_speed_rad_s=float(np.linalg.norm(v[:,3:],axis=1).max()),
            max_window_translation_m=float(np.linalg.norm(p[:,:3]-p[-1,:3],axis=1).max()),
            max_window_rotation_rad=float((2*np.arccos(np.clip(abs(q@q[-1]),0,1))).max()))
        assert metrics['max_linear_speed_m_s']<=.02 and metrics['max_angular_speed_rad_s']<=.5
        assert metrics['max_window_translation_m']<=.0002 and metrics['max_window_rotation_rad']<=.005
        for key,value in metrics.items():assert abs(value-obj['initialization_stability'][key])<1e-12
        bodies.append(dict(object_id=obj['object_id'],vertices=len(vertices),source_initial_bottom_m=bottoms[0],
            source_final_bottom_m=bottoms[-1],source_minimum_bottom_m=min(bottoms),
            all_750_supported_ticks=all_support[roots[i]],last250_supported_ticks=last_support[roots[i]],
            contact_rows=counts[roots[i]],independent_stability=metrics))
    log_path=directory/'stages/capture/process.log';text=log_path.read_text(errors='replace')
    active=[line for line in text.splitlines() if re.search(r'^\|\s*\d+\s*\|.*Yes:',line)]
    assert len(active)==1 and re.search(r'^\|\s*0\s*\|.*Yes:\s*0',active[0])
    stage=bound(directory/'stages/capture/STAGE.json')
    assert stage['status']=='complete'
    for entry in stage['outputs'].values():assert sha(entry['path'])==entry['sha256']
    assert sha(log_path)==stage['log']['sha256'];artifacts[str(log_path)]=stage['log']['sha256']
    result=dict(schema='anydex_boya_collection_capture_independent_audit_v1',status='initialization_pass',
        auditor_sha256=sha(__file__),scene_directory=str(directory),frames=frames,contact_rows=contacts,
        compressed_contact_sha256=journal['sha256'],logical_jsonl_sha256=logical.hexdigest(),
        gzip_bytes=path.stat().st_size,all_body_support=evidence,contact_pair_counts=pairs,bodies=bodies,
        native_hulls=native['hulls'],native_max_source_outside_cooked_bound_m=native['max_source_outside_cooked_bound_m'],
        actual_gpu_active_log_lines=active,physics_ticks=750,initialization_object_steps=2250,
        grasp_attempts=0,physical_steps_added_by_audit=0,artifacts=artifacts)
    if (capture/'RESULTS.json').exists():
        receipt=bound(capture/'RESULTS.json');observation=bound(capture/'observation.json')
        assert receipt['status']=='complete' and receipt['physics_ticks']==750 and receipt['initialization_object_steps']==2250
        assert receipt['render_physics_steps']==0 and receipt['grasp_attempts']==0
        assert receipt['observation_sha256']==sha(capture/'observation.json')
        assert observation['scene_sha256']==sha(capture/'observed_scene.json')
        for b in observation['files'].values():assert sha(b['path'])==b['sha256'];artifacts[b['path']]=b['sha256']
        ray=bound(capture/'render_geometry_check.json');assert ray['diagnostic_pass']
        assert receipt['geometry_check_sha256']==sha(capture/'render_geometry_check.json')
        result.update(status='complete',observation_sha256=sha(capture/'observation.json'),
            observed_scene_sha256=sha(capture/'observed_scene.json'),render_geometry_check=ray,render_physics_steps=0)
    for path,digest in artifacts.items():assert sha(path)==digest
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','contact_pair_counts')}))

if __name__=='__main__':audit(sys.argv[1],sys.argv[2])
