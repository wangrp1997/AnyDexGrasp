"""Read existing frozen prefix only; XML/NumPy FK, no planner or simulator."""
from pathlib import Path
import collections
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

ROOT=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/batch_v1/scenes/0000000_s40_lift_fit_00000_s2026111401')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def origin(node):
    out=np.eye(4)
    if node is not None:
        out[:3,3]=np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        out[:3,:3]=Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return out


def main():
    locks={};results=[]
    def load(path):locks[str(path)]=sha(path);return read(path)
    scene=load(ROOT/'capture/observed_scene.json')
    for gesture in (1,6,7,8):
        d=ROOT/f'g{gesture}';plan=load(d/'arm/native/plan.json');candidate=load(d/'pool/selected_author_candidate.json')
        fixed=load(d/'home/rrt/fixed_segments_geometry.json');direct=load(d/'arm/native/path_geometry.json')
        home=load(d/'home/HOME_ROUTE_PREPARATION.json');rrt=load(d/'home/rrt/HOME_ROUTE_PREPARATION.json')
        preparation=load(d/'LIFT_PIPELINE_RESULTS.json');commands=d/'arm/native/commands.npz';locks[str(commands)]=sha(commands)
        with np.load(commands,allow_pickle=False) as z:arrays={k:z[k].copy()for k in z.files}
        phases=arrays['phase'];n=len(phases);approach=np.flatnonzero(phases=='approach');endpoint=approach[0]-1
        # Compare the *same* reported states, with original zero-based indexes.
        canonical=lambda row,index:(int(index),row['kind'],row['link'],str(row.get('other')),row['distance_m'])
        old=[canonical(v,v['sample'])for v in direct['violations']]
        new=[canonical(v,v['command_index'])for v in fixed['violations']]
        assert old==new
        assert fixed['samples']==50+1+500 and len(approach)==500 and endpoint==799
        assert all(v['phase']=='approach' and v['kind']=='floor'for v in fixed['violations'])
        urdf=Path(plan['arguments']['robot_urdf']);locks[str(urdf)]=sha(urdf);robot=ET.parse(urdf).getroot()
        joints={n.find('child').get('link'):n for n in robot.findall('joint')}
        definitions={n.get('name'):n for n in robot.findall('joint')}
        joint_origins={link:origin(node.find('origin'))for link,node in joints.items()}
        meshes={}
        for link in ('ffdistal','mfdistal'):
            node=robot.find(f"link[@name='{link}']/collision");m=node.find('geometry/mesh');path=Path(m.get('filename').removeprefix('file://'))
            if not path.is_absolute():path=urdf.parent/path
            locks[str(path)]=sha(path);vertices=np.asarray(trimesh.load_mesh(path,process=False).vertices)*np.fromstring(m.get('scale','1 1 1'),sep=' ')
            frame=origin(node.find('origin'));meshes[link]=vertices@frame[:3,:3].T+frame[:3,3]
        def source_bottoms(values):
            q=dict(zip(map(str,arrays['joint_names']),map(float,values)));frames={}
            def coordinate(name):
                if name not in q:
                    m=definitions[name].find('mimic');q[name]=coordinate(m.get('joint'))*float(m.get('multiplier','1'))+float(m.get('offset','0'))
                return q[name]
            def fk(link):
                if link not in frames:
                    if link not in joints:frames[link]=np.eye(4)
                    else:
                        node=joints[link];motion=np.eye(4)
                        if node.get('type')!='fixed':
                            axis=np.fromstring(node.find('axis').get('xyz'),sep=' ')
                            if node.get('type') in ('revolute','continuous'):motion[:3,:3]=Rotation.from_rotvec(axis*coordinate(node.get('name'))).as_matrix()
                            elif node.get('type')=='prismatic':motion[:3,3]=axis*coordinate(node.get('name'))
                            else:raise ValueError('Unsupported source joint')
                        frames[link]=fk(node.find('parent').get('link'))@joint_origins[link]@motion
                return frames[link]
            return {link:float((vertices@fk(link)[:3,:3].T+fk(link)[:3,3])[:,2].min())for link,vertices in meshes.items()}
        indexes=[endpoint]+approach.tolist();bottom={i:source_bottoms(arrays['joint_target'][i])for i in indexes}
        maximum_error=max(abs(bottom[v['command_index']][v['link']]-v['distance_m'])for v in fixed['violations'])
        assert maximum_error<1e-12
        for link in meshes:
            computed=[i for i in indexes if bottom[i][link]<-fixed['tolerance_m']]
            reported=[v['command_index']for v in fixed['violations']if v['link']==link]
            assert computed==reported
        groups=collections.defaultdict(list)
        for v in fixed['violations']:groups[(v['phase'],v['kind'],v['link'])].append(v)
        result=dict(gesture_id=gesture,status=preparation['status'],stage=preparation['stage'],candidate_sha256=candidate['candidate_sha256'],
            source_row_index=candidate['source_row_index'],slot_id=candidate['slot_id'],source4cm=plan['arguments']['pregrasp_distance_m'],
            phase_rows=dict(collections.Counter(phases.tolist())),frozen_sample_count=fixed['samples'],
            fixed_pregrasp_endpoint_index_zero_based=int(endpoint),fixed_pregrasp_finger_source_bottom_m=bottom[endpoint],
            fixed_pregrasp_endpoint_violations=[v for v in fixed['violations']if v['command_index']==endpoint],
            settle_violations=[v for v in fixed['violations']if v['phase']=='settle'],
            original_direct_report_and_fixed_report_exact_same_violations=True,
            original_direct_all_path_violations=len(direct['violations']),fixed_violations=len(fixed['violations']),
            violations=[dict(phase=k[0],kind=k[1],link=k[2],count=len(v),first_command_index_zero_based=min(x['command_index']for x in v),
                last_command_index_zero_based=max(x['command_index']for x in v),first_approach_step_one_based=min(x['command_index']for x in v)-int(approach[0])+1,
                min_distance_m=min(x['distance_m']for x in v))for k,v in groups.items()],
            final_finger_source_bottom_m=bottom[n-1],independent_XML_NumPy_FK_floor_max_error_m=maximum_error,
            independent_all_approach_finger_source_vertex_flags_exact=True,
            RRT_reason=rrt['reason'],RRT_search_skipped=rrt['search_skipped'],home_cost=home['cost'],
            source_palm_start_xyz_m=arrays['desired_palm'][endpoint,:3,3].tolist(),source_palm_final_xyz_m=arrays['desired_palm'][-1,:3,3].tolist(),
            source_palm_endpoint_displacement_m=float(np.linalg.norm(arrays['desired_palm'][-1,:3,3]-arrays['desired_palm'][endpoint,:3,3])),
            interpretation='Only unchanged approach finger/floor collisions, also present at the final grasp endpoint; HOME endpoint clear and no RRT search was executed. No global reachability or physical-success conclusion.')
        results.append(result)
    for path,digest in locks.items():assert sha(path)==digest
    return dict(schema='anydex_boya_author_collection_first_prefix_null_diagnosis_v1',status='pass',scene_id=scene['scene_id'],
        scene_objects=[o['object_id']for o in scene['objects']],bin=None,results=results,
        input_sha256=locks,diagnostic_source_sha256=sha(__file__),index_semantics='command_index zero-based; first_approach_step_one_based explicitly one-based',
        costs=dict(Kit_starts=0,physics_steps=0,IK_calls=0,planning_calls=0,CGR_forwards=0,training_updates=0,new_labels=0),
        source_or_frozen_output_modified=False)


if __name__=='__main__':
    path=Path(__file__).parent/'NULL_DIAGNOSIS.json'
    with path.open('x') as output:output.write(json.dumps(main(),indent=2)+'\n')
    print(path)
