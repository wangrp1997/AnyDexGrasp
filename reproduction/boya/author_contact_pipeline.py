"""Opt-in preclosure-reference integration with the unchanged lift executor.

Fresh candidate/calibration/manifest bindings carry the new mapping through
source collision, arm preparation and command validation. Original modules and
historical files are never edited. Continued closure remains the original
250-step move to the company wrap target, not a stop at the virtual reference.
"""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import author_collection_batch as B
import author_action_semantics as A
import author_collision as C
import author_collect as COLLECT
import author_collection_bridge as BRIDGE
import author_lift_pipeline as LIFT
import author_pipeline as COMMON
from author_fast_contact_alignment import FiveFingerWidthMapper,OPPONENTS
from author_hand_clouds import HandCloudLibrary
from asset_adapter import CGR_R_REFERENCE,rigid
from trial_record import json_sha

SCHEMA='anydex_boya_preclosure_contact_mapping_v1'
BASE_CONTRACT=A.ActionContract
_CACHE={}


def reference_from_pads(contract,slot,q):
    coordinates=dict(zip(contract.names,q));coordinates.update(slot['wrist'])
    frames=contract.fk.evaluate(coordinates);points={};normals={}
    for group in ('th',slot['opposing_group']):
        pad=contract.pads[group];t=frames[pad['collision_link']]
        points[group]=t[:3,:3]@pad['point_link_m']+t[:3,3]
        normals[group]=A.unit(t[:3,:3]@pad['normal_link'])
    delta=points[slot['opposing_group']]-points['th'];x=A.unit(delta)
    tangent=np.array([0.,1.,0.]);tangent-=(tangent@normals['th'])*normals['th']
    if np.linalg.norm(tangent)<1e-6:
        tangent=np.eye(3)[np.argmin(abs(normals['th']))];tangent-=(tangent@normals['th'])*normals['th']
    z=A.unit(tangent-(tangent@x)*x);ref=np.eye(4)
    ref[:3,:3]=np.column_stack([x,np.cross(z,x),z]);ref[:3,3]=(points['th']+points[slot['opposing_group']])/2
    return ref,float(np.linalg.norm(delta))


class PreclosureContract(BASE_CONTRACT):
    def _validate_slot(self,slot):
        if slot.get('mapping_variant')!=SCHEMA:return super()._validate_slot(slot)
        opening=copy.deepcopy(slot)
        opening['palm_T_reference']=slot['source_opening_palm_T_reference']
        opening['reference_width_m']=slot['source_opening_reference_width_m']
        opening['opposing_group']='ff'
        super()._validate_slot(opening)
        s=float(slot['global_preclosure_progress']);base=self.slots['g1:distal_000:ff']
        if not slot['fraction']-1e-12<=s<=1 or slot['gesture_id']!=1 or slot['opposing_group']not in OPPONENTS:
            raise ValueError('Preclosure reference outside original wrap path')
        q=(1-s)*np.asarray(base['independent_q'])+s*np.asarray(base['close_q'])
        if not np.allclose(q,slot['preclosure_q'],atol=2e-14,rtol=0):raise ValueError('Reference hand state changed')
        ref,width=reference_from_pads(self,slot,q)
        if abs(width-slot['reference_width_m'])>1e-6 or np.max(abs(ref-slot['palm_T_reference']))>2e-10:
            raise ValueError('Preclosure reference does not bind original pad FK')

    def for_slot(self,slot_id):
        result=super().for_slot(slot_id);slot=self.slots[slot_id]
        if slot.get('mapping_variant')==SCHEMA:
            result.pop('contract_sha256')
            result.update(mapping_variant=SCHEMA,reference_semantics='Virtual preclosure pad reference, not the open pose or an actual contact certificate',
                global_preclosure_progress=slot['global_preclosure_progress'],preclosure_q=slot['preclosure_q'],
                actual_opening_reference_width_m=slot['source_opening_reference_width_m'],
                reference_is_execution_stop=False,thumb_required=True,other_fingers_optional=True)
            result['contract_sha256']=A.digest_json(result)
        return result


def contract_factory(calibration=A.DEFAULT_CALIBRATION,urdf=None):
    data=calibration if isinstance(calibration,dict)else B.read(calibration)
    if data.get('contact_mapping',{}).get('schema')!=SCHEMA:return BASE_CONTRACT(calibration,urdf)
    key=json_sha(data),str(urdf)
    if key not in _CACHE:_CACHE[key]=PreclosureContract(calibration,urdf)
    return _CACHE[key]


def create_mapping_bundle(rep,observation,calibration,manifest,seed,domain,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rep=Path(rep);original=B.read(calibration);surface=B.read(manifest);obs=B.read(observation)
    with np.load(rep/'raw_camera_candidates.npz',allow_pickle=False)as z:raw=z['candidates'].copy();features=z['candidate_features'].copy()
    filtered=COLLECT.source_domain(raw,obs['T_WC'],domain);keep=filtered['keep_indices']
    source=COLLECT.author.collection_candidates(COLLECT._row17(raw[keep]),features[keep],reference='dh3',reference_type=1,seed=seed)
    actual15=np.c_[source['rows'][:,:2],source['rows'][:,3:16]]
    mapper=FiveFingerWidthMapper(BASE_CONTRACT(calibration))
    mapping=mapper.align(actual15,np.eye(4))if len(actual15)else None
    opening_slots=[s for s in original['slots']if s['gesture_id']==1 and s['opposing_group']=='ff'and s['self_collision_clear']]
    cal=copy.deepcopy(original);new_slots=[];branch_slots=[]
    if mapping is not None:
        for i,row in enumerate(actual15):
            opening=C.lookup_slot(row[1]*1.7,opening_slots)
            if opening is None:continue
            for j,opponent in enumerate(OPPONENTS):
                if not mapping.valid[i,j]or mapping.progress[i,j]+1e-12<opening['fraction']:continue
                slot=copy.deepcopy(opening);slot.update(slot_id=f'g1:preclose_r{int(keep[source["source_indices"][i]])}:{opponent}',
                    mapping_variant=SCHEMA,source_local_index=i,source_surface_slot_id=opening['slot_id'],
                    source_opening_palm_T_reference=opening['palm_T_reference'],source_opening_reference_width_m=opening['reference_width_m'],
                    global_preclosure_progress=float(mapping.progress[i,j]),preclosure_q=mapping.q_contact[i,j].tolist(),
                    palm_T_reference=mapping.palm_T_contact_reference[i,j].tolist(),reference_width_m=float(mapping.width_m[i,j]),opposing_group=opponent)
                new_slots.append(slot);branch_slots.append(slot['slot_id'])
    cal['slots'].extend(new_slots)
    cal['contact_mapping']=dict(schema=SCHEMA,seed=seed,rows_identity=COLLECT.array_ref(source['rows']),
        raw_source_indices=keep[source['source_indices']].tolist(),branch_slots=branch_slots,
        original_calibration=B.binding(calibration),original_surface_manifest=B.binding(manifest),
        mapper=B.binding(B.HERE/'author_fast_contact_alignment.py'),implementation=B.binding(__file__),
        target_width='raw CGR; 1.7 only selects original preopening configuration',reference_is_execution_stop=False)
    B.save(out/'calibration.json',cal,exclusive=True)
    surface['calibration']=B.binding(out/'calibration.json')
    surface['source_and_input_sha256'][str((out/'calibration.json').resolve())]=B.sha(out/'calibration.json')
    for slot in new_slots:surface['slot_to_state'][slot['slot_id']]=surface['slot_to_state'][slot['source_surface_slot_id']]
    surface['preclosure_mapping']=dict(schema=SCHEMA,original_manifest=B.binding(manifest),points_changed=False)
    B.save(out/'HAND_CLOUD_MANIFEST.json',surface,exclusive=True)
    contract_factory(out/'calibration.json')
    B.save(out/'BUNDLE.json',dict(calibration=B.binding(out/'calibration.json'),surface_manifest=B.binding(out/'HAND_CLOUD_MANIFEST.json'),
        source_candidates=len(actual15),mapping_branches=len(new_slots),gesture=1,physics_steps=0,labels=0),exclusive=True)
    return out/'calibration.json',out/'HAND_CLOUD_MANIFEST.json'


class BranchGroup:
    def __init__(self,ids,rows,extra,slots):
        self.ids=np.atleast_1d(ids);self.rows=rows;self.extra=extra;self.slots=slots;self.mapped=[]
    def __len__(self):return len(self.ids)
    def __iter__(self):return iter(self.mapped)
    def __getitem__(self,index):return BranchGroup(self.ids[index],self.rows,self.extra,self.slots)
    def graspgroupTR_2_TR(self,two,unused):
        self.mapped=[]
        for branch,row in zip(self.ids,two.rows):
            slot=self.slots[int(branch)];cgr=np.eye(4);cgr[:3,:3]=row[4:13].reshape(3,3);cgr[:3,3]=row[13:16]
            ref=np.eye(4);ref[:3,:3]=CGR_R_REFERENCE
            palm=rigid(rigid(cgr)@ref@np.linalg.inv(rigid(slot['palm_T_reference'])))
            self.mapped.append(SimpleNamespace(get_grasp_type_with_finger_name=lambda name=slot['slot_id']:name,
                width=slot['source_opening_reference_width_m'],translation=palm[:3,3],rotation_matrix=palm[:3,:3],
                depth=float(row[3])+float(self.extra[int(branch)]),slot=slot,palm_before_depth=palm,
                source_index=slot['source_local_index']))


def contact_detect(rows17,gesture_ids,extra_depth_m,scene_points,*,calibration,library,profile):
    import open3d as o3d
    if profile!='dh3_collection'or not np.all(np.asarray(gesture_ids)==1):raise ValueError('New mapping is g1 collection only')
    meta=calibration['contact_mapping'];rows=np.asarray(rows17)
    if COLLECT.array_ref(rows)!=meta['rows_identity']:raise ValueError('Frozen source sampling differs from mapping bundle')
    by_id={s['slot_id']:s for s in calibration['slots']};slots=[by_id[k]for k in meta['branch_slots']]
    available=np.zeros(len(rows),dtype=bool)
    for slot in slots:available[slot['source_local_index']]=True
    if not slots:return dict(accepted_source_indices=np.array([],dtype=int),hand=SimpleNamespace(ids=np.array([],dtype=int),mapped=[]),
        mapping_available_mask=available,source_width_mask=np.zeros(len(rows),dtype=bool))
    source_indices=np.array([s['source_local_index']for s in slots]);branch_rows=rows[source_indices].copy();extra=np.asarray(extra_depth_m)[source_indices]
    meshes={};cached={}
    for slot in slots:
        source_slot=slot['source_surface_slot_id']
        if source_slot not in cached:
            cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(library.world_points(slot,np.eye(4),voxel_mm=3));cached[source_slot]=cloud
        meshes[slot['slot_id']+'_'+str(round(slot['source_opening_reference_width_m']*100,1))]=cached[source_slot]
    cls=COLLECT.author.definitions('utils/collision_detector.py',{'ModelFreeCollisionDetectorMultifinger'},
        dict(np=np,o3d=o3d,copy=copy,time=COLLECT.author.time,math=C.math))['ModelFreeCollisionDetectorMultifinger']
    group=BranchGroup(np.arange(len(slots)),branch_rows,extra,slots)
    detector=cls(np.asarray(scene_points),voxel_size=.001);cfg=C.PROFILES[profile]
    hand,two,admitted,width_mask=detector.detect(group,C.TwoFingerGroup(branch_rows),'',meshes,
        min_grasp_width=cfg['min_width'],VoxelGrid=cfg['voxel'],approach_dist=cfg['approach'],collision_thresh=cfg['threshold'],adjust_gripper_centers=False,DEBUG=False)
    width_available=np.zeros(len(rows),dtype=bool)
    for k in np.flatnonzero(width_mask):width_available[source_indices[k]]=True
    chosen={};first={};accepted=[];details=[]
    for item,passed in zip(hand.mapped,admitted):
        idx=item.source_index;first.setdefault(idx,item)
        details.append(dict(source_local_index=idx,slot_id=item.slot['slot_id'],opponent=item.slot['opposing_group'],collision_clear=bool(passed)))
        if passed and idx not in chosen:chosen[idx]=item;accepted.append(idx)
    all_ids=sorted(first);mapped=[chosen.get(i,first[i])for i in all_ids]
    return dict(accepted_source_indices=np.asarray(accepted,dtype=int),hand=SimpleNamespace(ids=np.asarray(all_ids),mapped=mapped),
        mapping_available_mask=available,source_width_mask=width_available,branch_checks=details,
        original_width_expansion_and_depth_and_voxel_functions=True,physics_steps=0)


@contextmanager
def installed(collision_log=None):
    changes=[]
    def swap(module,name,value):changes.append((module,name,getattr(module,name)));setattr(module,name,value)
    old_sources=LIFT._sources;old_bridge=BRIDGE.bridge_candidate
    def sources():
        value=old_sources()
        for name in ('author_contact_pipeline.py','author_fast_contact_alignment.py'):value[name]=B.binding(B.HERE/name)
        return value
    def bridge(*args,**kwargs):
        result=old_bridge(*args,**kwargs);candidate=args[0];out=Path(args[1]);contract=contract_factory(kwargs['calibration'])
        slot=contract.slots[candidate['slot_id']]
        result['records'][0]['jaw_assignment']['opponent']=slot['opposing_group']
        result['mapping']['mapping_variant']=SCHEMA;result['input_sha256'][str(Path(__file__).resolve())]=B.sha(__file__)
        B.save(out/'generation.json',result);return result
    def detect(*args,**kwargs):
        result=contact_detect(*args,**kwargs)
        if collision_log is not None:
            saved={k:v for k,v in result.items()if k!='hand'}
            saved={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in saved.items()}
            B.save(collision_log,saved,exclusive=True)
        return result
    try:
        for module in (A,BRIDGE,COMMON,LIFT):swap(module,'ActionContract',contract_factory)
        swap(C,'detect',detect);swap(LIFT,'_sources',sources);swap(BRIDGE,'bridge_candidate',bridge)
        yield
    finally:
        for module,name,value in reversed(changes):setattr(module,name,value)


def prepare(scene,representation_dir,observation,seed,calibration,cloud_manifest,backend,output,*,domain=None,archive_sources=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);domain=domain or COMMON.DOMAIN
    B.save(output/'CONTACT_INTEGRATION_PROTOCOL.json',dict(schema=SCHEMA,gesture=1,seed=seed,
        source=B.binding(__file__),mapping=B.binding(B.HERE/'author_fast_contact_alignment.py'),
        scene=B.binding(scene),observation=B.binding(observation),calibration=B.binding(calibration),manifest=B.binding(cloud_manifest),
        closure='Original open-to-company-wrap 250 steps; preclosure reference is NOT the command endpoint',
        candidate_fallback_after_planning=False,physics_started=False),exclusive=True)
    cal,manifest=create_mapping_bundle(representation_dir,observation,calibration,cloud_manifest,seed,domain,output/'mapping')
    with installed(output/'COLLISION_BRANCHES.json'):
        return LIFT.prepare(scene,representation_dir,observation,1,1,seed,cal,manifest,backend,output/'pipeline',archive_sources=archive_sources,domain=domain)


def audit_mapping(prepared):
    """Independent scalar XML FK and actual commands; no scene GT or physics."""
    directory=Path(prepared);ready=B.read(directory/'READY.json')
    B.verify({k:ready[k]for k in ('candidate','commands','plan')})
    candidate=B.read(ready['candidate']['path']);contract=contract_factory(candidate['input_bindings']['calibration']['path'])
    slot=contract.slots[candidate['slot_id']]
    if slot.get('mapping_variant')!=SCHEMA:raise ValueError('New mapping required')
    reference,width=reference_from_pads(contract,slot,slot['preclosure_q'])
    if abs(width-candidate['actual_row15_camera'][1])>1e-6:raise ValueError('Raw CGR width not matched')
    palm=contract.mapped_palm(candidate['slot_id'],candidate['actual_row15_camera'],candidate['world_T_camera'],extra_depth_m=candidate['extra_depth_m'])
    if np.max(abs(palm-candidate['world_T_palm']))>2e-6:raise ValueError('Candidate lost virtual-reference mapping')
    with np.load(ready['commands']['path'],allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
    names=list(map(str,arrays['joint_names']));motors=arrays['joint_target'][:,[names.index(n)for n in contract.names]]
    expanded,_=COLLECT.expanded_planned_hand(arrays,contract)
    contract.validate_command_arrays(expanded,slot_id=candidate['slot_id'],motor_targets=motors)
    close=np.flatnonzero(arrays['phase']=='close');hold=np.flatnonzero(arrays['phase']=='hold')
    if len(close)!=250 or len(hold)!=50:raise ValueError('Original closure duration changed')
    if np.max(abs(motors[close[-1]]-slot['close_q']))>1e-6:raise ValueError('Closure stopped before company wrap endpoint')
    approach=np.flatnonzero(arrays['phase']=='approach')
    error=float(np.max(abs(arrays['desired_palm'][approach[-1]]-palm)))
    if error>2e-6:raise ValueError('Actual planned palm differs from new reference')
    return dict(status='pass',mapping_variant=SCHEMA,slot=candidate['slot_id'],opponent=slot['opposing_group'],
        raw_width_m=candidate['actual_row15_camera'][1],independent_XML_width_m=width,
        planned_palm_matrix_error=error,close_steps=250,hold_steps=50,continues_to_company_wrap=True,
        reference_is_execution_stop=False,physics_steps=0)


def execute(prepared):
    import author_verified_execution as verified
    with installed(),verified.install():return verified.E.execute(prepared)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='stage',required=True)
    q=sub.add_parser('prepare')
    for name in ('scene','representation-dir','observation','calibration','cloud-manifest','backend','output-dir'):q.add_argument('--'+name,type=Path,required=True)
    q.add_argument('--seed',type=int,required=True);q.add_argument('--domain',type=Path);q.add_argument('--archive-sources',type=Path)
    q=sub.add_parser('execute');q.add_argument('--prepared-dir',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='execute':r=execute(a.prepared_dir)
    else:r=prepare(a.scene,a.representation_dir,a.observation,a.seed,a.calibration,a.cloud_manifest,a.backend,a.output_dir,
        domain=B.read(a.domain)if a.domain else None,archive_sources=B.read(a.archive_sources)if a.archive_sources else None)
    print(json.dumps(r,indent=2))
