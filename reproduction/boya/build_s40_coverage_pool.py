"""Source-bound S40 supplement: depth strata, novel CGR, unchanged geometry.

This tool only builds a finite candidate pool. Geometry rejection is null, never
an unexecuted success/failure label. It imports no decision head and runs no Kit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
BOYA = ROOT/'reproduction/boya'
GESTURES = (1,6,7,8)
S40_IDS = {0,2,5,7,8,9,11,14,15,17,18,20,21,22,26,27,29,30,34,36,37,38,40,41,43,44,46,48,51,52,56,57,58,60,61,62,63,66,69,70}
OFFICIAL_SHA = '7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def fingerprint(feature):
    x=np.ascontiguousarray(feature,dtype=np.float32)
    if x.shape!=(480,)or not np.isfinite(x).all():raise ValueError('Finite canonical float32[480] required')
    return hashlib.sha256(x.tobytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class Reader:
    def __init__(self):self.files={}
    def freeze(self,path,expected=None):
        path=str(Path(path).resolve())
        if path not in self.files:self.files[path]=sha(path)
        if expected is not None and self.files[path]!=expected:raise ValueError('Frozen input SHA mismatch: '+path)
        return self.files[path]
    def verify(self):
        for path,digest in self.files.items():
            if sha(path)!=digest:raise ValueError('Input changed: '+path)


class History:
    def __init__(self,counts,features,gestures,extras,roots=None):
        self.counts=np.asarray(counts,dtype=np.int64)
        self.features=np.asarray(features,dtype=np.float32)
        self.gestures=np.asarray(gestures,dtype=np.int64)
        self.extras=np.asarray(extras,dtype=np.int64)
        if (self.counts.shape!=(4,20)or np.any(self.counts<0)or self.features.ndim!=2
            or self.features.shape[1:]!=(480,)or not np.isfinite(self.features).all()
            or self.gestures.shape!=(len(self.features),)or self.extras.shape!=(len(self.features),)
            or not np.isin(self.gestures,GESTURES).all()or not np.isin(self.extras,range(4)).all()):
            raise ValueError('Invalid source-bound history arrays')
        self.excluded={(g,d):set()for g in GESTURES for d in range(4)}
        unique={}
        for feature,g,d in zip(self.features,self.gestures,self.extras):
            fp=fingerprint(feature);self.excluded[int(g),int(d)].add(fp);unique.setdefault(fp,feature)
        self.feature_ids=set(unique)
        self.reference=np.stack(list(unique.values()))if unique else np.empty((0,480),np.float32)
        self.roots=dict(roots or {})


def prepare_history(dataset_path,results_path,output):
    reader=Reader();dataset_path=Path(dataset_path).resolve();results_path=Path(results_path).resolve()
    reader.freeze(dataset_path);dataset=read(dataset_path)
    if (dataset.get('schema')!='anydex_s40_decision_dataset_v1'or dataset.get('complete')is not True
        or dataset.get('collection_complete')is not True or dataset.get('benchmark_only')is not False):
        raise ValueError('Final source S40 dataset required; no external/benchmark data')
    source=dataset['source_collection']
    if Path(source['final_results']).resolve()!=results_path:raise ValueError('History results are not this dataset source')
    reader.freeze(results_path,source['final_results_sha256'])
    protocol_path=Path(source['protocol']).resolve();reader.freeze(protocol_path,source['protocol_sha256']);protocol=read(protocol_path)
    if protocol.get('schema')!='anydex_boya_s40_collection_v1'or protocol.get('no_test_objects')is not True:
        raise ValueError('Only the frozen S40 outer-training collection is allowed')
    objects_path=Path(source['objects_manifest']).resolve();reader.freeze(objects_path,source['objects_manifest_sha256']);objects=read(objects_path)['objects']
    if (len(objects)!=40 or {x.get('graspnet_object_id')for x in objects}!=S40_IDS
        or any(x.get('split')!='train'for x in objects)):raise ValueError('Original S40 identity union, all outer train, required')
    object_ids={x['object_id']for x in objects}
    counts=np.zeros((4,20),np.int64);eligible=set()
    # Labels themselves are intentionally never read or used for acquisition.
    for gi,g in enumerate(GESTURES):
        entry=dataset['gesture_files'][str(g)];path=Path(entry['path']);reader.freeze(path,entry['sha256'])
        with np.load(path,allow_pickle=False)as z:
            features=z['features'];targets=z['target_index'];ids=z['trial_id'];oids=z['object_id'];gs=z['gesture_ids']
            if (features.shape!=(len(targets),480)or features.dtype!=np.float32 or not np.isfinite(features).all()
                or targets.dtype!=np.int64 or not np.isin(targets,range(20)).all()or not (gs==g).all()
                or len(set(map(str,ids)))!=len(ids)or not set(map(str,oids))<=object_ids):raise ValueError('Malformed source feature coverage')
            if eligible&set(map(str,ids)):raise ValueError('Repeated source actual identity')
            eligible.update(map(str,ids));counts[gi]=np.bincount(targets,minlength=20)
    result=read(results_path)
    if result.get('complete')is not True or result.get('unknown_execution_trial_ids')or result.get('unexecuted_ready_trial_ids'):
        raise ValueError('Only fully reconciled previous actual attempts can be excluded')
    actual=[];actual_ids=set();excluded_features=[];gs=[];ds=[];terminal_count=0
    for row in result['trials']:
        if type(row.get('physics_steps'))is not int or row['physics_steps']<=0:raise ValueError('History contains non-actual or ambiguous rows')
        if row['trial_id']in actual_ids or row['object_id']not in object_ids:raise ValueError('Foreign/repeated actual history')
        actual_ids.add(row['trial_id']);path=Path(row['binding_json']);reader.freeze(path);b=read(path)
        if (b['trial_binding_sha256']!=row['trial_binding_sha256']
            or json_sha({k:v for k,v in b.items()if k!='trial_binding_sha256'})!=b['trial_binding_sha256']):raise ValueError('Edited previous actual binding')
        f=np.asarray(b['canonical_CGR480'],dtype=np.float32)
        if (fingerprint(f)!=b['canonical_CGR480_identity']['c_order_bytes_sha256']or b.get('canonicalization_applied_once')is not True
            or b['multifinger_pose_finger_type']!=row['gesture_id']or b['multifinger_pose_depth_type']!=row['extra_depth_index']):raise ValueError('Previous actual feature/gesture/depth differs')
        reader.freeze(b['source_npz'],b['source_npz_sha256']);reader.freeze(b['official_canonical_source'],b['official_canonical_source_sha256'])
        excluded_features.append(f);gs.append(row['gesture_id']);ds.append(row['extra_depth_index']);actual.append(row['trial_id'])
        terminal_count+=row.get('label_eligible')is not True
    if not eligible<=actual_ids or len(eligible)!=dataset['sample_count']:raise ValueError('Feature coverage is not an eligible subset of source actuals')
    reader.verify();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    npz=output/'history.npz'
    np.savez_compressed(npz,counts=counts,features=np.stack(excluded_features),gesture_ids=np.asarray(gs,np.int64),extra_depth_indices=np.asarray(ds,np.int64),trial_ids=np.asarray(actual,dtype='<U160'))
    roots={str(dataset_path):reader.files[str(dataset_path)],str(results_path):reader.files[str(results_path)],str(protocol_path):reader.files[str(protocol_path)],str(objects_path):reader.files[str(objects_path)]}
    receipt=dict(schema='anydex_s40_coverage_history_v1',complete=True,history_npz=dict(path=str(npz),sha256=sha(npz)),
        input_sha256=reader.files,root_input_sha256=roots,source_builder_sha256=sha(__file__),
        actual_attempts=len(actual),eligible_label_presence=len(eligible),terminal_null_attempts_excluded=terminal_count,
        exclusion_key='canonical480_float32_bytes_SHA256 x gesture x extra; all actual attempts including terminal null',
        coverage_basis='Presence of valid executed labels only; success/failure values and model predictions never read',
        new_physics_steps=0,new_network_forwards=0,new_training_steps=0)
    save(output/'history.json',receipt)
    return receipt


def load_history(path,reader):
    path=Path(path).resolve();reader.freeze(path);doc=read(path)
    if doc.get('schema')!='anydex_s40_coverage_history_v1'or doc.get('complete')is not True or doc['source_builder_sha256']!=sha(__file__):raise ValueError('Foreign/incomplete history or changed builder')
    # Deep sources were checked once by prepare_history. Every child checks small
    # frozen binding roots and the compiled arrays, rather than rereading40 meshes.
    for p,digest in doc['root_input_sha256'].items():reader.freeze(p,digest)
    entry=doc['history_npz'];reader.freeze(entry['path'],entry['sha256'])
    with np.load(entry['path'],allow_pickle=False)as z:
        history=History(z['counts'],z['features'],z['gesture_ids'],z['extra_depth_indices'],reader.files)
    if len(history.features)!=doc['actual_attempts']:raise ValueError('History attempt count differs')
    return history,doc


def unit_rows(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),np.finfo(np.float64).tiny)


def diverse_variants(candidates,history):
    """Novel canonical vectors first, then deterministic greedy max-min cosine.

    Both original flip poses remain alternatives for a feature until it is admitted;
    no pose is dropped merely because it shares its feature with another flip.
    """
    grouped={}
    for row in sorted(candidates,key=lambda x:(-x['cgr_score'],x['source_candidate_index'],x['if_flip'])):
        grouped.setdefault(row['feature_sha256'],[]).append(row)
    if not grouped:return []
    groups=list(grouped.values());matrix=unit_rows(np.stack([x[0]['feature']for x in groups]))
    if len(history.reference):distance=np.clip(1.-(matrix@unit_rows(history.reference).T).max(1),0.,2.)
    else:distance=np.full(len(groups),2.)
    novelty=np.array([g[0]['feature_sha256']not in history.feature_ids for g in groups],bool)
    active=np.ones(len(groups),bool);output=[]
    for _ in range(len(groups)):
        available=np.flatnonzero(active)
        best=max(available,key=lambda i:(int(novelty[i]),float(distance[i]),groups[i][0]['cgr_score'],-groups[i][0]['source_candidate_index'],-int(groups[i][0]['if_flip'])))
        gap=float(distance[best]);active[best]=False
        for row in groups[best]:output.append(dict(row,novel_to_any_previous_actual=bool(novelty[best]),max_min_cosine_distance=gap))
        distance=np.minimum(distance,np.clip(1.-matrix@matrix[best],0.,2.))
    return output


def candidate_catalog(raw,features,canonical,history,gesture,extra,candidate_limit=1000):
    if (raw.ndim!=2 or raw.shape[1:]!=(15,)or features.shape!=(len(raw),2261)
        or not np.isfinite(raw).all()or not np.isfinite(features).all()or type(candidate_limit)is not int or not 5<=candidate_limit<=1000):raise ValueError('Finite paired raw arrays and source-row budget5..1000 required')
    # Exactly robot_dh3.get_graspgroup_features:183-186. Every selected row is
    # independently cross-checked through the actual official canonical function.
    depths=(features[:,-1]*100.+.1).astype(int)
    if not np.isin(depths,range(5)).all():raise ValueError('Original raw depth outside0..4')
    quota=[candidate_limit//5+int(d<candidate_limit%5)for d in range(5)]
    chosen=[[]for _ in range(5)];excluded=[0]*5;seen=[0]*5
    order=np.argsort(-raw[:,0],kind='stable')
    for depth in range(5):
        indices=[int(i)for i in order if depths[i]==depth][:quota[depth]]
        for index in indices:
            seen[depth]+=1
            for flip in [False,True]:
                info=canonical(raw[index],features[index],if_flip=flip)
                if info['two_fingers_pose_depth_type']!=depth:raise ValueError('Raw stratification disagrees with official parser')
                feature=np.asarray(info['canonical_CGR480'],dtype=np.float32);fp=fingerprint(feature)
                if fp in history.excluded[gesture,extra]:excluded[depth]+=1;continue
                chosen[depth].append(dict(source_candidate_index=index,if_flip=bool(info['if_flip']),raw_depth_index=depth,
                    cgr_score=float(raw[index,0]),feature=feature,feature_sha256=fp))
    chosen=[diverse_variants(x,history)for x in chosen]
    return chosen,dict(original_source_rows_by_raw_depth=[int(np.sum(depths==d))for d in range(5)],
        source_rows_considered_by_raw_depth=seen,source_row_quota=quota,excluded_previous_actual_variants_by_raw_depth=excluded,
        remaining_variants_by_raw_depth=[len(x)for x in chosen])


def depth_priority(history,gesture,extra,observation_sha,target_per_output):
    counts=history.counts[GESTURES.index(gesture)]
    def key(depth):
        count=int(counts[depth*4+extra]);tie=hashlib.sha256(f'{observation_sha}:{gesture}:{extra}:{depth}'.encode()).hexdigest()
        return count>=target_per_output,count,tie
    return sorted(range(5),key=key)


def collect_balanced(catalog,depth_order,assess,keep=8,checks_per_depth=64):
    if not 1<=keep<=8 or not 1<=checks_per_depth<=64:raise ValueError('Fixed bounded keep/check budget exceeded')
    per_depth_limit=2;buckets=[[]for _ in range(5)];rejected=[];checks=[];calls=[0]*5;considered=[0]*5
    for depth in depth_order:
        accepted_features=set()
        for row in catalog[depth]:
            if calls[depth]>=checks_per_depth or len(buckets[depth])>=per_depth_limit:break
            considered[depth]+=1
            if row['feature_sha256']in accepted_features:continue
            outcome=assess(row)
            if outcome.get('geometry_called',True):calls[depth]+=1
            meta={k:v for k,v in row.items()if k!='feature'};meta.update(outcome['meta'])
            meta['training_label']=None;meta['physics_steps']=0;checks.append(meta)
            if not outcome['ok']:rejected.append(meta);continue
            accepted_features.add(row['feature_sha256']);buckets[depth].append(dict(meta=meta,**outcome['arrays']))
    selected=[]
    for rank in range(per_depth_limit):
        for depth in depth_order:
            if rank<len(buckets[depth])and len(selected)<keep:selected.append(buckets[depth][rank])
    return selected,rejected,dict(checks=checks,geometry_calls_by_raw_depth=calls,variants_considered_by_raw_depth=considered,
        admitted_by_raw_depth=[len(x)for x in buckets],selected_by_raw_depth=[sum(r['meta']['raw_depth_index']==d for r in selected)for d in range(5)],
        total_geometry_calls=sum(calls),per_raw_depth_admitted_cap=per_depth_limit)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-history',action='store_true');p.add_argument('--previous-dataset-manifest',type=Path);p.add_argument('--previous-collection-results',type=Path)
    p.add_argument('--coverage-history',type=Path);p.add_argument('--observation',type=Path);p.add_argument('--frame-dir',type=Path);p.add_argument('--calibration',type=Path)
    p.add_argument('--gesture-id',type=int,choices=GESTURES);p.add_argument('--extra-depth-index',type=int,choices=range(4));p.add_argument('--candidate-limit',type=int,default=1000)
    p.add_argument('--keep',type=int,default=8);p.add_argument('--width-expansion',type=float,default=1.7);p.add_argument('--opponent',choices=['ff'],default='ff')
    p.add_argument('--geometry-checks-per-depth',type=int,default=64);p.add_argument('--coverage-target-per-output',type=int,default=4)
    p.add_argument('--flip-policy',choices=['both'],default='both');p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    names=['previous_dataset_manifest','previous_collection_results']if a.prepare_history else ['coverage_history','observation','frame_dir','calibration','gesture_id','extra_depth_index']
    if any(getattr(a,n)is None for n in names):p.error('Missing explicit inputs: '+','.join(names))
    if (not 5<=a.candidate_limit<=1000 or not 1<=a.keep<=8 or not 1<=a.geometry_checks_per_depth<=64
        or not 1<=a.coverage_target_per_output<=100 or not np.isfinite(a.width_expansion)or a.width_expansion!=1.7):p.error('Frozen coverage budget or original width expansion changed')
    return a


def main():
    a=parse_args()
    if a.prepare_history:
        doc=prepare_history(a.previous_dataset_manifest,a.previous_collection_results,a.output_dir)
        print(json.dumps({k:doc[k]for k in ['complete','actual_attempts','eligible_label_presence','terminal_null_attempts_excluded']},indent=2));return
    if str(BOYA)not in sys.path:sys.path.insert(0,str(BOYA))
    from trial_record import canonical_candidate,official_functions
    from asset_adapter import BoyaAssets,map_slot,cgr_reference_pose
    from build_collection_pool import OpeningGeometryCache,geometry
    started=time.monotonic();reader=Reader();history,hdoc=load_history(a.coverage_history,reader)
    a.observation=a.observation.resolve();a.frame_dir=a.frame_dir.resolve();a.calibration=a.calibration.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('Existing pool retained')
    reader.freeze(a.observation);obs=read(a.observation);reader.freeze(a.calibration);cal=read(a.calibration)
    frame=a.frame_dir/'RESULTS.json';reader.freeze(frame);receipt=read(frame);scene=receipt.get('scene_binding',{})
    if (receipt.get('full_representation_forward')is not True or Path(scene.get('observation_json','')).resolve()!=a.observation
        or scene.get('observation_sha256')!=reader.files[str(a.observation)]or scene.get('T_WC')!=obs['T_WC']):raise ValueError('CGR/observation binding mismatch')
    expected={str(Path(v['path']).resolve()):v['sha256']for v in obs['files'].values()};actual={str(Path(k).resolve()):v['sha256']for k,v in receipt['input_files'].items()}
    if expected!=actual:raise ValueError('CGR image inputs mismatch')
    for path,digest in expected.items():reader.freeze(path,digest)
    for name in ['raw_camera_candidates.npz','representation_outputs.npz']:
        item=receipt['output_files'][name]
        if Path(item['path']).resolve()!=a.frame_dir/name:raise ValueError('CGR output path mismatch')
        reader.freeze(item['path'],item['sha256'])
    source=a.frame_dir/'raw_camera_candidates.npz'
    with np.load(source,allow_pickle=False)as z:raw=z['candidates'].copy();features=z['candidate_features'].copy()
    with np.load(a.frame_dir/'representation_outputs.npz',allow_pickle=False)as z:cloud=z['sparse_input_XYZ_features'].copy()
    camera=np.asarray(obs['T_WC']);cloud=cloud@camera[:3,:3].T+camera[:3,3]
    official=ROOT/'robot_dh3.py';reader.freeze(official,OFFICIAL_SHA);functions=official_functions(official)
    canonical=lambda row,feature,if_flip:canonical_candidate(row,feature,functions,if_flip=if_flip)
    catalog,audit=candidate_catalog(raw,features,canonical,history,a.gesture_id,a.extra_depth_index,a.candidate_limit)
    priority=depth_priority(history,a.gesture_id,a.extra_depth_index,reader.files[str(a.observation)],a.coverage_target_per_output)
    assets=BoyaAssets();family=assets.families[a.gesture_id];lock=cal['source_lock']
    if lock['independent_motor_names']!=list(assets.hand.names):raise ValueError('Calibration motor order mismatch')
    for path,item in lock['source_files'].items():reader.freeze(path,item['sha256'])
    for path,digest in lock['readonly_fk_dependencies'].items():reader.freeze(path,digest)
    slots=[s for s in cal['slots']if s['gesture_id']==a.gesture_id and s['self_collision_clear']and s['opposing_group']==a.opponent]
    if not slots:raise ValueError('No original calibrated opening slots')
    for slot in slots:
        if slot['wrist']!=family.wrist or not np.array_equal(slot['close_q'],family.nominal)or not np.array_equal(slot['independent_q'],family.motor_state(slot['fraction'],slot['fraction'])):raise ValueError('Original calibrated opening/closure differs')
    lower=min(s['reference_width_m']for s in slots);upper=max(s['reference_width_m']for s in slots);cache=OpeningGeometryCache()
    def assess(item):
        row=raw[item['source_candidate_index']].copy();expanded=float(row[1])*a.width_expansion;requested=float(np.clip(expanded,lower,upper))
        matches=[s for s in slots if abs(s['reference_width_m']-requested)<=.002]
        if not matches:return dict(ok=False,geometry_called=False,meta=dict(geometry=dict(ok=False,reason='outside_calibrated_width_tolerance')))
        slot=min(matches,key=lambda s:(abs(s['reference_width_m']-requested),s['slot_id']))
        if item['if_flip']:row[3:12].reshape(3,3)[:,1:3]*=-1
        palm=map_slot(row,camera,slot,insertion_m=.01*a.extra_depth_index);approach=cgr_reference_pose(row,camera)[:3,2]
        check=geometry(assets,np.asarray(slot['independent_q']),palm,cloud,approach,cache)
        meta=dict(source_slot=slot['slot_id'],actual_reference_width_m=slot['reference_width_m'],raw_width_m=float(row[1]),requested_opening_m=requested,
            width_expansion=a.width_expansion,width_clamped=bool(expanded<lower or expanded>upper),raw_depth_m=float(row[2]),additional_depth_index=a.extra_depth_index,
            additional_depth_m=.01*a.extra_depth_index,target_index=item['raw_depth_index']*4+a.extra_depth_index,
            score_source='Original CGR quality tie break only; no decision head',jaw_assignment=dict(thumb='th',opponent='ff'),geometry=check)
        return dict(ok=check['ok'],geometry_called=True,meta=meta,arrays=dict(palm=palm,q=slot['independent_q'],q_close=slot['close_q'],approach=approach))
    selected,rejected,checked=collect_balanced(catalog,priority,assess,a.keep,a.geometry_checks_per_depth)
    for path in [Path(__file__).resolve(),BOYA/'asset_adapter.py',BOYA/'build_collection_pool.py',BOYA/'trial_record.py',assets.urdf,assets.scene,assets.package,assets.reference_path]:reader.freeze(path)
    root=Path(obs['object_root']);reader.freeze(root/'mesh/simplified.obj');reader.freeze(root/'urdf/coacd.urdf')
    import xml.etree.ElementTree as ET
    for node in ET.parse(root/'urdf/coacd.urdf').findall('.//mesh'):reader.freeze((root/'urdf'/node.get('filename')).resolve())
    reader.verify();out.mkdir(parents=True,exist_ok=False)
    coverage=dict(schema='anydex_s40_coverage_selection_v1',complete=True,input_sha256=reader.files,
        history_json=str(a.coverage_history.resolve()),history_json_sha256=reader.files[str(a.coverage_history.resolve())],previous_actual_attempts=hdoc['actual_attempts'],
        excluded_key=hdoc['exclusion_key'],gesture_id=a.gesture_id,extra_depth_index=a.extra_depth_index,
        previous_valid_counts20=history.counts[GESTURES.index(a.gesture_id)].tolist(),raw_depth_priority=priority,
        source_row_budget=a.candidate_limit,geometry_checks_per_depth=a.geometry_checks_per_depth,coverage_target_per_output=a.coverage_target_per_output,
        both_flips=True,max_admitted_per_raw_depth=2,keep=a.keep,catalog=audit,geometry=checked,
        selected_targets=[x['meta']['target_index']for x in selected],labels_read_for_selection=False,training_label=None,physics_steps=0,
        boundary='Coverage sampling variant, not original published selection. No synthesis of raw depths; unchanged opening/map/hand-table-cloud checks. Rejections and unexecuted outputs remain unlabelled.',elapsed_wall_s=time.monotonic()-started)
    save(out/'coverage_selection.json',coverage)
    if not selected:
        save(out/'no_candidate.json',dict(records=rejected,physics_steps=0,training_label=None,gesture_id=a.gesture_id,
            status='no_hand_geometry_in_declared_budget',source_candidates_sha256=reader.files[str(source)],source_observation_sha256=reader.files[str(a.observation)],
            calibration_sha256=reader.files[str(a.calibration)],additional_depth_index=a.extra_depth_index,coverage_selection_sha256=sha(out/'coverage_selection.json'),
            empty_raw_depths=[d for d,x in enumerate(catalog)if not x],elapsed_wall_s=time.monotonic()-started))
        print('No geometry-admitted unexecuted coverage candidate',flush=True);raise SystemExit(1)
    target=out/'candidates.npz'
    np.savez_compressed(target,independent_motor_names=assets.hand.names,proposal_q=[x['q']for x in selected],proposal_q_close=[x['q_close']for x in selected],
        proposal_palm=[x['palm']for x in selected],proposal_approach_direction=[x['approach']for x in selected],world_T_object=obs['world_T_object'])
    save(out/'generation.json',dict(complete=True,kind='anydex_boya_collection_geometry_depth_v1',gesture_id=a.gesture_id,additional_depth_index=a.extra_depth_index,
        scale=obs['object_scale'],selected_indices=list(range(len(selected))),candidates_sha256=sha(target),input_sha256=dict(reader.files,**{str(out/'coverage_selection.json'):sha(out/'coverage_selection.json')}),
        records=[x['meta']for x in selected],rejected=rejected,source_frame=str(a.frame_dir),source_observation=str(a.observation),
        hand_head_loaded=False,original_CGR_parser=receipt['candidate_parser_overlay'],physics_steps=0,training_labels=0,
        collection_policy='Depth-stratified coverage and max-min novel canonical features; fixed original geometry; no actual outcome used',
        object_pose_as_network_input=False,coverage_selection_json=str(out/'coverage_selection.json'),coverage_selection_sha256=sha(out/'coverage_selection.json'),elapsed_wall_s=time.monotonic()-started))
    save(out/'world_object_pose.json',obs['world_T_object'])
    print(json.dumps(dict(admitted=len(selected),selected_targets=coverage['selected_targets'],geometry_calls=checked['total_geometry_calls'],new_physics_steps=0)),flush=True)


if __name__=='__main__':main()
