import sys,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,torch
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
import train_boya_decision as b
from trial_record import canonical_candidate,official_functions
read=lambda p:json.loads(Path(p).read_text())
tr=read(ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json');manifest=read(tr['dataset_manifest']['path']);data={}
for g,e in manifest['gesture_files'].items():
 with np.load(e['path'],allow_pickle=False)as z:data[int(g)]={k:z[k].copy()for k in z.files}
cls=b.official_components()[0];torch.set_num_threads(4)
R={'scope':'Read-only original435 support and fixed code100 forward; no retraining, checkpoint mutation, EGAD or physics','per_head':{},'selected':[],'domain_support':{}}
def fh(x):return hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()
def csupport(a,t,idx):
 ids=idx[a['target_index'][idx]==t]
 return {'count':len(ids),'positive':int(np.sum(a['labels'][ids]==1)),'negative':int(np.sum(a['labels'][ids]==0))}
for g,a in data.items():
 fit=np.flatnonzero(a['split']=='train');check=np.flatnonzero(a['split']=='internal_check');target=a['target_index'][fit];label=a['labels'][fit]
 ck=torch.load(tr['internal_heads'][str(g)]['path'],map_location='cpu',weights_only=True);model=cls(**ck['architecture_kwargs']);model.load_state_dict(ck['model_state_dict']);model.eval()
 with torch.inference_mode():pred=model(torch.from_numpy(a['features'][fit]))[0][:,0].numpy();s=pred[np.arange(len(fit)),target]
 groups=defaultdict(list)
 for i in range(len(a['labels'])):groups[(fh(a['features'][i]),int(a['target_index'][i]))].append(i)
 conflict=[dict(feature_sha256=k[0],target=k[1],ids=[str(a['trial_id'][i])for i in ids],labels=[int(a['labels'][i])for i in ids])for k,ids in groups.items()if len(set(map(int,a['labels'][ids])))>1]
 prior=float(label.mean());internal_label=a['labels'][check]
 pos=label==1;zero=s==0;tz=(a['target_index'][fit,None]==np.arange(20)).T
 R['per_head'][str(g)]={'total_records':len(a['labels']),'fit_count':len(fit),'fit_positive':int(pos.sum()),'fit_negative':int((~pos).sum()),'fit_positive_fraction':prior,'fit_prior_constant_fit_MSE':float(np.mean((label-prior)**2)),'fit_prior_constant_internal_MSE':float(np.mean((internal_label-prior)**2)),'fit_eval_selected_MSE':float(np.mean((s-label)**2)),'fit_positive_selected_exact_zero':int(np.sum(pos&zero)),'fit_positive_selected_exact_zero_fraction':float(np.mean(zero[pos])),'fit_negative_selected_exact_zero':int(np.sum((~pos)&zero)),'fit_positive_zero_MSE_contribution':float(np.sum(pos&zero)/len(fit)),'target_fit_support':{str(t):csupport(a,t,fit)for t in range(20)},'exact_feature_target_conflict_groups_all435':conflict,'exact_feature_target_duplicate_groups':sum(len(ids)>1 for ids in groups.values()),'fit_output_unobserved_channels_gt1':int(np.sum((pred>1)&(~tz.T))), 'fit_selected_score_max':float(s.max()),'fit_selected_score_min':float(s.min())}
# Every eligible current-domain combination retains source raw depth; no reranking.
lookup={};count=Counter();byhead={str(g):Counter()for g in data};byobj={};targets=Counter()
for out in sorted((ROOT/'outputs/s40_hand_domain_20261107_v1').glob('graspnet_*')):
 with np.load(out/'head_outputs.npz',allow_pickle=False)as z:z={k:z[k].copy()for k in z.files}
 look={(int(i),int(g),bool(f)):(int(d),pred)for i,g,f,d,pred in zip(z['source_indices'],z['gesture_ids'],z['if_flip'],z['raw_depth_indices'],z['outputs20'])}
 domain=read(out/'DOMAIN.json');objcounts=Counter()
 for row in domain['opening_support_records']:
  key=(row['source_candidate_index'],row['gesture_id'],row['if_flip'])
  if key not in look:continue
  d,pred=look[key];a=data[key[1]];fit=np.flatnonzero(a['split']=='train')
  for extra in row['allowed_extra_depth_indices']:
   t=d*4+extra;n=csupport(a,t,fit)['count'];cat='0'if n==0 else'1-2'if n<=2 else'3-5'if n<=5 else'>5'
   count[cat]+=1;objcounts[cat]+=1;byhead[str(key[1])][cat]+=1;targets[(key[1],t)]+=1
 byobj[out.name]=dict(objcounts)
R['domain_support']={'total':sum(count.values()),'fit_count_buckets':dict(count),'per_head':{g:dict(v)for g,v in byhead.items()},'per_object':byobj,'combination_counts_by_head_target':{f'g{g}_t{t}':n for (g,t),n in sorted(targets.items())}}
# Existing actual final choices plus the exact previous success044.
queries=[]
for name,queue in [('new',ROOT/'outputs/s40_hand_top1_20261107_v1/queue_manifest.json'),('old',ROOT/'outputs/s40_port_top1_20261107_v2/queue_manifest.json')]:
 for obj in read(queue)['objects']:
  if obj['selected'] and(name=='new'or obj['object_id']=='graspnet_044'):
   s=obj['selected'];row=s['selected_candidate'];queries.append((name,obj['object_id'],row,s))
func=official_functions(ROOT/'robot_dh3.py')
for name,oid,row,selected in queries:
 g=int(row['gesture_id']);t=int(row['target_index']);a=data[g];fit=np.flatnonzero(a['split']=='train');X=a['features'][fit].astype(np.float64)
 with np.load(ROOT/f'outputs/s40_stable_supplement_20261107_v2/collection/representations/{oid}/raw_camera_candidates.npz',allow_pickle=False)as z:
  i=row['source_candidate_index'];feature=np.asarray(canonical_candidate(z['candidates'][i],z['candidate_features'][i],func,if_flip=row['if_flip'])['canonical_CGR480'],np.float32)
 h=fh(feature);v=feature.astype(np.float64);dist=np.linalg.norm(X-v[None,:],axis=1);cos=1-(X@v)/(np.linalg.norm(X,axis=1)*np.linalg.norm(v));ids=np.flatnonzero(a['target_index'][fit]==t)
 result={'protocol':name,'object_id':oid,'gesture_id':g,'target':t,'raw_index':int(i),'flip':row['if_flip'],'learned_score':row['learned_score'],'feature_sha256':h,'fit_support':csupport(a,t,fit),'exact_feature_target_matching_fit_records':[],'nearest_fit_same_target':{},'nearest_fit_any_target':{}}
 for j in fit:
  if a['target_index'][j]==t and fh(a['features'][j])==h:result['exact_feature_target_matching_fit_records'].append({'trial_id':str(a['trial_id'][j]),'label':int(a['labels'][j])})
 def nearest(candidates,metric):
  if not len(candidates):return None
  j=int(candidates[np.argmin(metric[candidates])]);actual=int(fit[j]);return {'distance':float(metric[j]),'object_id':str(a['object_id'][actual]),'trial_id':str(a['trial_id'][actual]),'label':int(a['labels'][actual]),'target':int(a['target_index'][actual])}
 for metric,values in [('L2',dist),('cosine_distance',cos)]:
  result['nearest_fit_same_target'][metric]=nearest(ids,values);result['nearest_fit_any_target'][metric]=nearest(np.arange(len(fit)),values)
  result.setdefault('nearest_fit_same_target_positive',{})[metric]=nearest(ids[a['labels'][fit[ids]]==1],values)
  result.setdefault('nearest_fit_same_target_negative',{})[metric]=nearest(ids[a['labels'][fit[ids]]==0],values)
 R['selected'].append(result)
json.dump(R,open('/tmp/CODE100_SCORE_SUPPORT_AUDIT_20261107.json','w'),indent=2,allow_nan=False)
print(json.dumps({'per_head':{g:{k:v for k,v in a.items()if k not in ['target_fit_support','exact_feature_target_conflict_groups_all435']}for g,a in R['per_head'].items()},'domain_support':R['domain_support'],'selected':R['selected']},indent=2))
