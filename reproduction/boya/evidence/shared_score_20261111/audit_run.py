"""Independent saved-checkpoint CPU audit; never trains or invokes CGR/physics.

Root runs this with --run-dir after the bounded experiment completes. Model
forward math is rebuilt from checkpoint tensors, not production model classes.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,datetime,hashlib,json
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
OUT=ROOT/'outputs/boya_shared_score_audit_20261111_v1'
DATA=ROOT/'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b,tol=1e-7):
    assert np.allclose(a,b,atol=tol,rtol=0),(np.max(np.abs(np.asarray(a)-np.asarray(b))),tol)
def grouped(rows,scores,field):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r[field]].append(i)
    values=[]
    for key,idx in sorted(groups.items()):
        y=np.array([rows[i]['label'] for i in idx]);s=np.asarray(scores)[idx]
        top=s==s.max();pos=s[y==1];neg=s[y==0]
        auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))) if len(pos) and len(neg) else None
        values.append(dict(group=key,n=len(idx),positive=int(y.sum()),negative=int(len(y)-y.sum()),auc=auc,top_tie_expected_label=float(y[top].mean()),top_tie_count=int(top.sum()),brier=float(np.mean((s-y)**2))))
    mixed=[r for r in values if r['auc'] is not None]
    return dict(groups=len(values),mixed_groups=len(mixed),macro_object_auc=float(np.mean([r['auc'] for r in mixed])) if mixed else None,macro_top_tie_expected_label=float(np.mean([r['top_tie_expected_label'] for r in values])),macro_mixed_top_tie_expected_label=float(np.mean([r['top_tie_expected_label'] for r in mixed])) if mixed else None,macro_brier=float(np.mean([r['brier'] for r in values])),rows=values)
def metrics(rows,scores):return dict(object=grouped(rows,scores,'object_id'),same_observation=grouped(rows,scores,'observation_group'))
def resolve_ref(ref,base):
    path=Path(ref['path'] if isinstance(ref,dict) else ref)
    path=path if path.is_absolute() else base/path
    if isinstance(ref,dict) and ref.get('sha256'):assert sha(path)==ref['sha256']
    return path
def predict(family,state,x):
    state={k:v.detach().cpu() for k,v in state.items()}
    with torch.no_grad():
        if family=='condition_linear':z=F.linear(torch.from_numpy(x[:,480:]),state['layers.weight'],state['layers.bias'])
        elif family=='shared_linear':z=F.linear(torch.from_numpy(x),state['layers.weight'],state['layers.bias'])
        elif family=='shared_mlp64':
            hidden=F.relu(F.linear(torch.from_numpy(x),state['layers.0.weight'],state['layers.0.bias']))
            z=F.linear(hidden,state['layers.2.weight'],state['layers.2.bias'])
        else:raise ValueError('Unexpected family '+family)
        return torch.sigmoid(z).reshape(-1).numpy()

def run(run_dir):
    torch.set_num_threads(1)
    inspection=read(OUT/'DATA_INSPECTION.json')
    assert inspection['observation_grouping_complete']
    for path,digest in inspection['input_sha256'].items():assert sha(path)==digest,path
    manifest=read(DATA/'manifest.json')
    provenance={r['trial_id']:r for r in inspection['provenance']}
    records=[];byid={};binding_files={}
    for p in sorted(DATA.glob('gesture_*.npz')):
        z=load(p)
        for i in range(len(z['labels'])):
            tid=str(z['trial_id'][i]);prov=provenance[tid];bindings=[]
            for path in prov['followed_record_binding_files']:
                doc=read(path);doc=doc.get('binding',doc)
                if {'canonical_CGR480','if_flip','raw_width_m','aperture'}<=set(doc):bindings.append((path,doc))
            assert len(bindings)==1,(tid,len(bindings))
            path,b=bindings[0];binding_files[path]=sha(path)
            canonical=np.asarray(b['canonical_CGR480'],dtype=np.float32)
            assert np.array_equal(canonical,z['features'][i]),tid
            gesture=int(z['gesture_ids'][i]);target=int(z['target_index'][i]);rawdepth,extra=divmod(target,4)
            assert gesture==int(b['multifinger_pose_finger_type'])
            assert rawdepth==int(b['two_fingers_pose_depth_type']) and extra==int(b['multifinger_pose_depth_type'])
            metadata=np.zeros(16,dtype=np.float32);metadata[(1,6,7,8).index(gesture)]=1
            metadata[4+rawdepth]=1;metadata[9+extra]=1;metadata[13]=float(b['if_flip']);metadata[14]=float(b['raw_width_m']);metadata[15]=float(b['aperture']['actual_ref_width_m'])
            features=np.concatenate((canonical,metadata)).astype(np.float32)
            assert features.shape==(496,) and np.isfinite(features).all()
            obs=prov['observation_hashes'];assert len(obs)==1
            row=dict(trial_id=tid,object_id=str(z['object_id'][i]),mesh_sha256=str(z['mesh_sha256'][i]),split=str(z['split'][i]),label=int(z['labels'][i]),target_index=target,gesture_id=gesture,observation_sha256=obs[0],observation_group=str(z['object_id'][i])+'::'+obs[0],features=features)
            records.append(row);byid[tid]=row
    assert len(records)==len(byid)==435
    protocol=read(run_dir/'PROTOCOL.json')
    split_paths=[run_dir/f'cv_fold{i}'/'SPLIT.json' for i in range(4)]+[run_dir/'internal8/SPLIT.json']
    all_cv_checks=set();checkpoint_rows=[];oof=defaultdict(dict);internal={};split_rows=[]
    train_meshes=set(manifest['mesh_groups']['train']);internal_meshes=set(manifest['mesh_groups']['internal_check'])
    assert len(train_meshes)==32 and len(internal_meshes)==8 and not train_meshes&internal_meshes
    for sp in split_paths:
        split=read(sp);fit_ids=split['fit_trial_ids'];check_ids=split['check_trial_ids'];fit=[byid[x] for x in fit_ids];check=[byid[x] for x in check_ids]
        assert not set(fit_ids)&set(check_ids)
        assert not set(split['fit_meshes'])&set(split['check_meshes'])
        assert {r['mesh_sha256'] for r in fit}==set(split['fit_meshes']) and {r['mesh_sha256'] for r in check}==set(split['check_meshes'])
        if sp.parent.name=='internal8':
            declared_fit,declared_check=train_meshes,internal_meshes
            assert set(split['fit_meshes'])<=train_meshes and set(split['check_meshes'])<=internal_meshes
            assert len(fit)==364 and len(check)==71
        else:
            fold_index=int(sp.parent.name.removeprefix('cv_fold'))
            fold=protocol['folds'][fold_index]
            declared_fit,declared_check=set(fold['fit_meshes']),set(fold['check_meshes'])
            assert len(declared_fit)==24 and len(declared_check)==8 and not declared_fit&declared_check
            assert declared_fit|declared_check==train_meshes
            assert set(split['fit_meshes'])<=declared_fit and set(split['check_meshes'])<=declared_check
            assert not all_cv_checks&set(check_ids);all_cv_checks.update(check_ids)
            assert all(r['split']=='train' for r in fit+check)
        xfit=np.stack([r['features'] for r in fit]);xcheck=np.stack([r['features'] for r in check])
        standard=load(sp.parent/'STANDARDIZER.npz')
        mean=standard['mean'];scale=standard['scale'];continuous=np.array(list(range(480))+[494,495]);binary=np.arange(480,494)
        assert mean.shape==scale.shape==(496,) and np.isfinite(mean).all() and np.isfinite(scale).all() and np.all(scale>0)
        assert np.array_equal(mean[binary],np.zeros(14)) and np.array_equal(scale[binary],np.ones(14))
        mm=xfit[:,continuous].astype(np.float64).mean(0)
        ss=xfit[:,continuous].astype(np.float64).std(0,ddof=0)
        expected_scale=np.where(ss>1e-12,ss,1.)
        same(mean[continuous],mm,2e-12)
        same(scale[continuous],expected_scale,2e-12)
        assert np.array_equal(standard['continuous_columns'],continuous)
        assert int(standard['fit_count'])==len(fit)
        xscaled=((xcheck-mean)/scale).astype(np.float32)
        counts=Counter(r['mesh_sha256'] for r in fit);weights=np.array([1/counts[r['mesh_sha256']] for r in fit]);weights/=weights.sum()
        assert all(abs(weights[[i for i,r in enumerate(fit) if r['mesh_sha256']==mesh]].sum()-1/len(counts))<1e-12 for mesh in counts)
        families=defaultdict(list)
        assert len(split['models'])==5
        for name,item in split['models'].items():
            ckpath=resolve_ref(item['checkpoint'],sp.parent);predpath=resolve_ref(item['predictions'],sp.parent)
            ck=torch.load(ckpath,map_location='cpu',weights_only=True);pred=load(predpath)
            assert ck['fit_trial_ids']==fit_ids and set(ck['fit_meshes'])==set(split['fit_meshes']) and ck['input_dim']==496
            family=ck['family'];seed=int(ck['seed']);state=ck['model_state_dict']
            assert family in ('condition_linear','shared_linear','shared_mlp64')
            assert list(pred['trial_id'])==check_ids
            assert list(pred['object_id'])==[r['object_id'] for r in check]
            assert list(pred['mesh_sha256'])==[r['mesh_sha256'] for r in check]
            assert np.array_equal(pred['labels'],[r['label'] for r in check])
            assert np.array_equal(pred['indices'],split['check_indices'])
            # Observation separators may be cosmetic; bind exact final hash and
            # object ID rather than accepting a foreign observation group.
            for value,row in zip(pred['observation_group'],check):assert row['object_id'] in str(value) and row['observation_sha256'] in str(value)
            score=predict(family,state,xscaled)
            same(score,pred['scores'],2e-6)
            updates=300 if family=='shared_mlp64' else 1000
            assert item['optimizer_steps']==updates and len(item['history'])==updates
            assert item['parameter_count']==sum(v.numel() for v in state.values())=={'condition_linear':17,'shared_linear':497,'shared_mlp64':31873}[family]
            assert all(np.isfinite(h['weighted_bce']) and np.isfinite(h['learning_rate']) for h in item['history'])
            assert seed in ((20261111,20261112,20261113) if family=='shared_mlp64' else (20261111,))
            families[family].append((seed,score));checkpoint_rows.append(dict(split=sp.parent.name,model=name,family=family,seed=seed,checkpoint=str(ckpath),checkpoint_sha256=sha(ckpath),prediction_sha256=sha(predpath),heldout_scores_recomputed=True,max_probability_error=float(np.max(np.abs(score-pred['scores']))),optimizer_steps=updates,parameter_count=item['parameter_count'],metrics=metrics(check,score)))
        for family,values in families.items():
            assert len(values)==(3 if family=='shared_mlp64' else 1)
            ensemble=np.mean(np.stack([x[1].astype(np.float64) for x in sorted(values)]),axis=0)
            if sp.parent.name=='internal8':internal[family]=dict(scores=ensemble,rows=check,metrics=metrics(check,ensemble))
            else:
                for row,value in zip(check,ensemble):assert row['trial_id'] not in oof[family];oof[family][row['trial_id']]=float(value)
        split_rows.append(dict(split=sp.parent.name,fit_labelled_meshes=len(counts),fit_manifest_meshes=len(declared_fit),check_manifest_meshes=len(declared_check),check_labelled_meshes=len(split['check_meshes']),fit_records=len(fit),check_records=len(check),fit_only_mean_and_std=True,all_object_weights_equal=True,standardizer_sha256=sha(sp.parent/'STANDARDIZER.npz'),split_sha256=sha(sp)))
    assert len(all_cv_checks)==364 and len(checkpoint_rows)==25 and sum(x['optimizer_steps'] for x in checkpoint_rows)==14500
    trainrows=[r for r in records if r['split']=='train'];cv={}
    for family,values in oof.items():
        assert set(values)=={r['trial_id'] for r in trainrows}
        cv[family]=metrics(trainrows,np.array([values[r['trial_id']] for r in trainrows]))
    declared_cv=read(run_dir/'CV_RESULTS.json');selection=read(run_dir/'SELECTION.json')
    for family,value in cv.items():same(value['object']['macro_object_auc'],declared_cv['metrics'][family]['object']['macro_object_auc'])
    chosen='shared_mlp64' if cv['shared_mlp64']['object']['macro_object_auc']>cv['shared_linear']['object']['macro_object_auc'] else 'shared_linear'
    assert selection['selected']==chosen and selection['internal_predictions_computed'] is False
    assert any(sha(run_dir/'CV_RESULTS.json')==v for v in selection.values() if isinstance(v,str)) or any(isinstance(v,dict) and v.get('sha256')==sha(run_dir/'CV_RESULTS.json') for v in selection.values())
    for p,h in inspection['input_sha256'].items():assert sha(p)==h
    report=dict(schema='anydex_boya_shared_scorer_independent_checkpoint_audit_v1',complete=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(run_dir/'PROTOCOL.json'),data_inspection_sha256=sha(OUT/'DATA_INSPECTION.json'),input_data_and_old8_heads_unchanged=True,canonical480_and16_metadata_reconstructed_from_original_bindings=True,metadata_whitelist=['gesture_onehot4','rawdepth_onehot5','extra_depth_onehot4','if_flip','raw_width_m','actual_ref_width_m'],split_audits=split_rows,checkpoint_audits=checkpoint_rows,CV_metrics=cv,internal_metrics={k:v['metrics'] for k,v in internal.items()},selected_from_CV_only=chosen,models=25,training_updates_from_frozen_records=14500,all_364_train_records_exactly_once_OOF=True,internal_labelled_records=71,source_binding_sha256=binding_files,auditor_training_updates=0,auditor_CGR_forwards=0,auditor_physics_steps=0,auditor_CPU_checkpoint_prediction_passes=25,limits=['Offline fixed435 executed-candidate support only; no new grasp or external validation.','Manifest32train/8internal include3/2objects without labels; actual29/6labelled-group coverage kept explicit.','Optimizer updates/history are bound to saved records; no retraining performed to replicate optimization.','Original8internal objects have been repeatedly used in development and are not a blinded test.'])
    (OUT/'RUN_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,models=25,optimizer_updates=14500,selected=chosen,output=str(OUT/'RUN_AUDIT.json'))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);run(parser.parse_args().run_dir.resolve())
