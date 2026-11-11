"""Read-only435 data/provenance inspection; writes only this audit directory."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,datetime
import numpy as np

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
OUT=ROOT/'outputs/boya_shared_score_audit_20261111_v1'
DATA=ROOT/'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar(v):return v.item() if isinstance(v,np.generic) else v
def tiny(x):
    if isinstance(x,dict):return {k:tiny(v) for k,v in x.items()}
    if isinstance(x,list):return x if len(x)<12 else dict(type='list',length=len(x),first=small(x[0]))
    return x
def small(x):
    if isinstance(x,dict):return {k:tiny(v) for k,v in x.items()}
    return x
def walk(x,prefix=''):
    if isinstance(x,dict):
        for k,v in x.items():
            key=prefix+'.'+k if prefix else k
            yield key,v
            if isinstance(v,(dict,list)):yield from walk(v,key)
    elif isinstance(x,list):
        for i,v in enumerate(x):
            if isinstance(v,(dict,list)):yield from walk(v,prefix+f'[{i}]')

m=read(DATA/'manifest.json');frozen={str(DATA/'manifest.json'):sha(DATA/'manifest.json')}
rows=[];field_schemas={}
for path in sorted(DATA.glob('gesture_*.npz')):
    frozen[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:
        field_schemas[path.name]={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
        assert np.isfinite(z['features']).all() and z['features'].shape[1]==480
        assert set(z['labels'])<={0,1} and set(z['target_index'])<=set(range(20))
        for i in range(len(z['labels'])):
            row={k:scalar(z[k][i]) for k in z.files if k!='features'}
            row['feature_bytes_sha256']=hashlib.sha256(np.ascontiguousarray(z['features'][i]).tobytes()).hexdigest()
            row['npz_file']=str(path);row['npz_index']=i;rows.append(row)
assert len(rows)==435 and sum(r['labels'] for r in rows)==302
assert len({r['trial_id'] for r in rows})==435
mesh_objects=defaultdict(set);object_meshes=defaultdict(set);mesh_splits=defaultdict(set)
for r in rows:
    mesh_objects[r['mesh_sha256']].add(r['object_id']);object_meshes[r['object_id']].add(r['mesh_sha256']);mesh_splits[r['mesh_sha256']].add(r['split'])
assert all(len(x)==1 for x in mesh_objects.values()) and all(len(x)==1 for x in object_meshes.values()) and all(len(x)==1 for x in mesh_splits.values())
def summarize(groups):
    answer=[]
    for key,values in sorted(groups.items()):
        pos=sum(x['labels'] for x in values);n=len(values)
        answer.append(dict(group=key,n=n,positive=pos,negative=n-pos,mixed=0<pos<n,positive_negative_pairs=pos*(n-pos),gesture_counts=dict(Counter(x['gesture_ids'] for x in values)),object_ids=sorted({x['object_id'] for x in values}),split=values[0]['split']))
    return answer
object_groups=defaultdict(list)
for r in rows:object_groups[r['object_id']].append(r)
objects=summarize(object_groups)
split_stats={}
for split in ['train','internal_check']:
    rr=[r for r in rows if r['split']==split];oo=[o for o in objects if o['split']==split]
    split_stats[split]=dict(samples=len(rr),positive=sum(x['labels'] for x in rr),negative=sum(1-x['labels'] for x in rr),manifest_objects=len(m['mesh_groups'][split]),labelled_objects=len(oo),mixed_objects=sum(x['mixed'] for x in oo),all_positive_objects=sum(x['negative']==0 for x in oo),all_negative_objects=sum(x['positive']==0 for x in oo),absent_mesh_groups=sorted(set(m['mesh_groups'][split])-{x['mesh_sha256'] for x in rr}))
duplicates={}
for name,fields in [('feature',['feature_bytes_sha256']),('feature_gesture',['feature_bytes_sha256','gesture_ids']),('feature_gesture_target',['feature_bytes_sha256','gesture_ids','target_index'])]:
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in fields)].append(r)
    repeated=[v for v in groups.values() if len(v)>1]
    conflicts=[v for v in repeated if len({r['labels'] for r in v})>1]
    duplicates[name]=dict(unique_keys=len(groups),repeated_keys=len(repeated),rows_in_repeated_keys=sum(map(len,repeated)),cross_object_repeated_keys=sum(len({r['object_id'] for r in v})>1 for v in repeated),conflicting_label_keys=len(conflicts),conflicting_examples=[[{k:r[k] for k in ['trial_id','object_id','gesture_ids','target_index','labels']} for r in v] for v in conflicts[:3]])

samples=m['samples']
sample_list=list(samples.values()) if isinstance(samples,dict) else samples
samples_by_trial={}
for sample in sample_list:
    for key,value in walk(sample):
        if key.split('.')[-1] in ('trial_id','merged_trial_id','namespaced_trial_id') and isinstance(value,str):samples_by_trial[value]=sample
provenance=[];unmatched=[];obs_groups=defaultdict(list)
for r in rows:
    sample=samples_by_trial.get(r['trial_id'])
    if sample is None:
        unmatched.append(r['trial_id']);continue
    docs=[('sample',sample)];seen=set();bindings=[];observation_hashes=set();observation_paths=set()
    # Follow only explicit JSON record/binding refs, never physical or network helpers.
    for depth in range(4):
        fresh=[]
        for label,doc in docs:
            for key,value in walk(doc):
                lk=key.lower()
                if isinstance(value,str) and 'observation' in lk:
                    if len(value)==64 and all(c in '0123456789abcdef' for c in value):observation_hashes.add(value)
                    elif value.endswith('.json') and Path(value).is_file():
                        observation_paths.add(value);observation_hashes.add(sha(value));frozen[value]=sha(value)
                if isinstance(value,str) and value.endswith('.json') and Path(value).is_file() and any(word in lk for word in ('record','binding')) and value not in seen:
                    seen.add(value);frozen[value]=sha(value);loaded=read(value);fresh.append((value,loaded))
                    if 'binding' in lk:bindings.append((value,loaded))
        if not fresh:break
        docs=fresh
    # Save compact evidence for root to resolve schema variations without guesswork.
    entry=dict(trial_id=r['trial_id'],sample_found=True,observation_hashes=sorted(observation_hashes),observation_paths=sorted(observation_paths),followed_record_binding_files=sorted(seen))
    if bindings:
        entry['binding_field_examples']={key:value for key,value in walk(bindings[0][1]) if isinstance(value,(str,int,float,bool)) and any(w in key.lower() for w in ('flip','width','depth','observation','aperture'))}
    if len(observation_hashes)==1:
        r['source_observation_sha256']=next(iter(observation_hashes));obs_groups[r['object_id']+'|'+r['source_observation_sha256']].append(r)
    provenance.append(entry)
obs_summary=summarize(obs_groups)

models=[]
for variant,dirname in [('original_MSE','s40_code_recipe_20261107_v1'),('preactivation','s40_preactivation_control_20261107_v1')]:
    report_path=ROOT/'outputs'/dirname/'TRAINING_RESULTS.json';report=read(report_path);frozen[str(report_path)]=sha(report_path)
    for gesture in (1,6,7,8):
        path=ROOT/'outputs'/dirname/'internal_check'/f'g{gesture:02d}.pt';digest=sha(path);frozen[str(path)]=digest
        declared=report['internal_heads'][str(gesture)]
        assert declared['sha256']==digest and Path(declared['path'])==path
        models.append(dict(variant=variant,gesture=gesture,path=str(path),sha256=digest,declared_metadata={k:v for k,v in declared.items() if k not in ('path','sha256')}))
for path,digest in frozen.items():assert sha(path)==digest
inspection=dict(schema='anydex_boya_shared_scorer_data_audit_v1',complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest=str(DATA/'manifest.json'),input_sha256=frozen,field_schemas=field_schemas,split_stats=split_stats,objects=objects,source_collection_counts=dict(Counter(r['source_collection_manifest'] for r in rows)),source_protocol_counts=dict(Counter(r['source_protocol_sha'] for r in rows)),target_index_counts=dict(Counter(r['target_index'] for r in rows)),duplicate_feature_keys=duplicates,models=models,source_manifest_metadata={k:m[k] for k in ('labelled_mesh_groups','split_counts','supervision','duplicate_policy','boundary')},manifest_sample_count=len(sample_list),first_manifest_sample=small(sample_list[0]),sample_binding_unmatched=unmatched,observation_grouping_complete=len(obs_summary)>0 and not unmatched and all(len(x['observation_hashes'])==1 for x in provenance),observation_group_stats={split:dict(groups=sum(x['split']==split for x in obs_summary),mixed_groups=sum(x['split']==split and x['mixed'] for x in obs_summary),samples=sum(x['n'] for x in obs_summary if x['split']==split)) for split in ('train','internal_check')},observation_groups=obs_summary,provenance=provenance,protocol_recommendations=['Group4fold by full mesh SHA across all gestures/depths/flips and both source collections; only32original-fit groups participate.','Fit preprocessing and class weights only on eachfold train24groups; select architecture on4fold mean across3fixed seeds; do not choose bestseed.','Use fixed epochs or innergroupCV; no checkpoint selection or tuning on8oldinternal objects.','Old8 are repeatedly inspected development data, not independent test. Compare both existing4head baselines on those8 only; original32seen baseline predictions are not OOF.','Primary development ranking: macro within-object mixed-label pairwiseAUC, equal scores count0.5; list eligible mixed-group count and labels.','Report same-observation×object mixed-label AUC and top-tie expected label separately; cross-observation candidates do not constitute one deployable candidate pool.','For pool-top1 report deterministic tie rule plus expected positive label among all tiedmax candidates; constant baseline top-tie expectation is group positive fraction.','Report Brier/logloss for probability-output sharedBCE heads and immutable raw MSE for original heads; any clipping/calibration must be separately named and not tuned oninternal8.','No EGAD diagnostic/external12/new6 outcomes, worldobject pose, trace, measured load, learned rank or previous ready queue as features.'],new_training_steps=0,new_physics_steps=0,new_network_forwards=0)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'DATA_INSPECTION.json').write_text(json.dumps(inspection,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(complete=True,split_stats=split_stats,observation_grouping_complete=inspection['observation_grouping_complete'],observation_group_stats=inspection['observation_group_stats'],sample_unmatched=len(unmatched),first_manifest_sample=inspection['first_manifest_sample'],duplicates=duplicates),indent=2))
