"""Promote audited quota rows and merge only the matching formal MJX cohort."""
from pathlib import Path
import numpy as np
import author_collection_batch as B
from author_mjx_quota_progress import UNSTARTED
from finish_author_expansion_metadata import coverage


def promote(root):
    """Create new formal records only after the complete original raw audit/export."""
    directory = root/'collection'
    p = B.read(directory/'PROTOCOL.json')
    B.verify(p['inputs']);B.verify(p['sources'])
    audit = B.read(directory/'READ_ONLY_AUDIT.json')
    manifest = B.read(directory/'export/MANIFEST.json')
    if (audit['status']!='read_only_audit_pass'
        or audit['original_result']!=B.binding(directory/'RESULTS.json')
        or manifest['source_audit']!=B.binding(directory/'READ_ONLY_AUDIT.json')):
        raise ValueError('Formal eligibility requires a bound complete audit')
    B.verify({'dataset':manifest['dataset']})
    checked = {r['index']:r for r in audit['rounds']}
    out = root/'formal'
    out.mkdir(exist_ok=False)
    with np.load(manifest['dataset']['path'],allow_pickle=False) as data:
        arrays = {k:data[k].copy() for k in data.files}
    refs,hashes = [],[]
    for old_path in arrays['record_json']:
        path = Path(str(old_path));record = B.read(path);index = int(path.parent.name)
        if not record['eligible'] or not checked[index].get('return_pass'):
            raise ValueError('No promotion of nulls or failed returns')
        result = dict(record,schema='anydex_mjx_audited_expansion_record_v1',
            exclude_training=False,formal_training_eligible=True,source_staged_record=B.binding(path),
            independent_audit=B.binding(directory/'READ_ONLY_AUDIT.json'),
            cohort='cgr_fit_quota_20261119_v1',training_updates=0)
        dest = out/'records'/f'{index:03d}.json'
        B.save(dest,result,exclusive=True);refs.append(str(dest));hashes.append(B.sha(dest))
    arrays['record_json'] = np.asarray(refs,str)
    arrays['record_sha256'] = np.asarray(hashes,str)
    target = out/'gesture_1.npz'
    np.savez_compressed(target,**arrays)
    with np.load(target,allow_pickle=False) as data:
        for k,v in arrays.items():
            np.testing.assert_array_equal(data[k],v)
    B.save(out/'OBJECT_COVERAGE.json',coverage(directory),exclusive=True)
    result = dict(manifest,status='audited_quota_campaign_rows_complete',dataset=B.binding(target),
        exclude_training=False,formal_training_eligible=True,source_staging_export=B.binding(directory/'export/MANIFEST.json'),
        object_coverage=B.binding(out/'OBJECT_COVERAGE.json'),scene_budget=len(p['rows']),new_independent_scene_draws=len(p['rows']),
        processed_scenes=sum(r['status']not in UNSTARTED for r in B.read(directory/'RESULTS.json')['records']),
        unstarted_slots=sum(r['status']in UNSTARTED for r in B.read(directory/'RESULTS.json')['records']),
        training_updates=0,old_data_merged=False,
        note='Eligibility is for this separate MJX functional domain; no training is dispatched. Legacy target mesh ID is not actual grasped-object attribution.')
    B.save(out/'MANIFEST.json',result,exclusive=True)
    return result


SCIENTIFIC_SOURCES=('author_mjx_native_batch.py','author_mjx_native_graph.py','author_mjx_live_batch_v4.py',
    'author_mjx_compatibility.py','author_mujoco_control.py','author_mujoco_sample.py','author_contact_pipeline.py',
    'author_lift_pipeline.py','author_mujoco_joint_limits.py','author_mujoco_object_dynamics.py',
    'author_mjx_measured_planning.py','author_mujoco_representation.py')
PHYSICS_FIELDS=('source_dt_s','control_dt_s','GPU_noslip_iterations','GPU_ccd_tolerance_m',
    'domain','min_settle','max_settle','sample_policy')


def same_domain(a,b):
    for name in SCIENTIFIC_SOURCES:
        if a['sources'][name]['sha256']!=b['sources'][name]['sha256']:
            raise ValueError('Different scientific implementation: '+name)
    for name in PHYSICS_FIELDS:
        if a[name]!=b[name]:raise ValueError('Different physics or label domain: '+name)


def read_formal(manifest_path):
    m=B.read(manifest_path)
    if not m.get('formal_training_eligible') or m.get('exclude_training',True):
        raise ValueError('Only explicitly audited formal cohorts may be merged')
    B.verify({'dataset':m['dataset']})
    with np.load(m['dataset']['path'],allow_pickle=False)as z:data={k:z[k].copy()for k in z.files}
    for i,(path,digest) in enumerate(zip(data['record_json'],data['record_sha256'])):
        if B.sha(str(path))!=str(digest):raise ValueError('Formal source record changed')
        r=B.read(str(path))
        if not r['eligible'] or not r['formal_training_eligible'] or r['exclude_training'] or r['label']!=int(data['labels'][i]):
            raise ValueError('Invalid source record in formal cohort')
    if len(data['labels'])!=m['records']:raise ValueError('Manifest row count differs')
    return m,data


def merge(root):
    import author_training as T
    root=Path(root).resolve();p=B.read(root/'collection/PROTOCOL.json')
    paths=[Path(p['inputs']['existing_formal_manifest']['path']),root/'formal/MANIFEST.json']
    source_data=[];manifests=[]
    for path in paths:
        m,data=read_formal(path);source_data.append(data);manifests.append(m)
    old=B.read(manifests[0]['source_protocol']['path']);same_domain(old,p)
    if any(set(d)!=set(source_data[0])for d in source_data):raise ValueError('Cohort array schemas differ')
    data={key:np.concatenate([d[key]for d in source_data],axis=0)for key in source_data[0]}
    fit,internal=T.validate_arrays(data)
    # validate_arrays rejects duplicate trials and whole-scene object leakage.
    out=root/'merged';out.mkdir(exist_ok=False);dataset=out/'gesture_1.npz';np.savez_compressed(dataset,**data)
    with np.load(dataset,allow_pickle=False)as z:
        for key,value in data.items():np.testing.assert_array_equal(z[key],value)
    result=dict(status='audited_same_domain_merged_dataset_complete',dataset=B.binding(dataset),
        source_manifests=[B.binding(path)for path in paths],records=len(data['labels']),fit=len(fit),internal=len(internal),
        positive=int(data['labels'].sum()),negative=int((data['labels']==0).sum()),
        target_fit_records=p['target_fit_records'],target_met=len(fit)>=p['target_fit_records'],
        first100_fit_trial_ids=data['trial_id'][fit[:100]].tolist(),
        original_trial_record_references_preserved=True,diagnostic_and_comparison_rows_merged=False,
        formal_training_eligible=True,exclude_training=False,training_updates=0,
        source_protocol=B.binding(root/'collection/PROTOCOL.json'),
        source_audit=B.binding(root/'collection/READ_ONLY_AUDIT.json'))
    B.save(out/'MANIFEST.json',result,exclusive=True);return result

