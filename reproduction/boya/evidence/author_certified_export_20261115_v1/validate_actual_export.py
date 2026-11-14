"""Independent saved-array comparison; reads old candidate/record, no audit replay."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
SOURCE = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development/batch_v1')
OUTPUT = ROOT/'outputs/boya_author_certified_export_20261115_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def main():
    rows=[read(p) for p in sorted((SOURCE/'ledger').glob('*.json')) if read(p).get('label') is not None]
    assert len(rows)==14
    manifest=read(OUTPUT/'MANIFEST.json')
    fields={'features','labels','target_index','gesture_ids','trial_id','split','mesh_sha256',
        'scene_object_ids','scene_mesh_sha256','scene_id','candidate_sha256','commands_sha256',
        'evaluation_sha256','record_json','record_sha256'}
    result=dict(schema='anydex_boya_certified_export_independent_array_check_v1',status='pass',
        exported_rows=0,positive=0,negative=0,gestures={},inputs_sha256={str(OUTPUT/'MANIFEST.json'):sha(OUTPUT/'MANIFEST.json')},
        new_numeric_audits=0,new_physics_steps=0,new_training_updates=0)
    for gesture in (1,6,7,8):
        expected=[row for row in rows if row['gesture_id']==gesture]
        path=OUTPUT/f'gesture_{gesture}.npz';result['inputs_sha256'][str(path)]=sha(path)
        assert manifest['datasets'][str(gesture)]['binding']==dict(path=str(path),sha256=sha(path))
        with np.load(path,allow_pickle=False) as data:
            assert set(data.files)==fields and data['features'].shape==(len(expected),480)
            assert all(not data[k].dtype.hasobject for k in data.files)
            for i,row in enumerate(expected):
                trial=row['trial'];record=read(trial['record']['path']);candidate=read(trial['candidate']['path'])
                for key in ('record','candidate'):
                    p=trial[key]['path'];assert sha(p)==trial[key]['sha256'];result['inputs_sha256'][p]=sha(p)
                features=np.asarray(candidate['canonical_cgr480'],dtype=np.float32)
                assert features.shape==(480,) and np.isfinite(features).all()
                np.testing.assert_array_equal(data['features'][i],features)
                assert hashlib.sha256(features.tobytes()).hexdigest()==candidate['canonical_cgr480_identity']['c_order_bytes_sha256']
                assert int(data['labels'][i])==record['label']==row['label']
                assert int(data['target_index'][i])==candidate['raw_depth_index']*4+candidate['extra_depth_index']
                assert int(data['gesture_ids'][i])==gesture
                assert str(data['trial_id'][i])==row['trial_id']
                assert str(data['split'][i])==('train' if row['split']=='fit' else 'internal_check')
                ids=sorted(row['source_object_ids'])
                assert list(data['scene_object_ids'][i])==ids
                assert list(data['scene_mesh_sha256'][i])==[row['source_object_digests'][key] for key in ids]
                for field,key in [('candidate_sha256','candidate_sha256'),('mesh_sha256','mesh_sha256')]:
                    assert str(data[field][i])==trial[key]
                for field,key in [('commands_sha256','commands'),('evaluation_sha256','evaluation'),('record_sha256','record')]:
                    assert str(data[field][i])==trial[key]['sha256']
                assert str(data['record_json'][i])==trial['record']['path'] and str(data['scene_id'][i])==row['scene_id']
                cert=row['artifact_bindings']['runtime_audit'];reuse=manifest['certificate_reuse'][row['trial_id']]
                assert reuse['original_certificate']==cert and sha(cert['path'])==cert['sha256']
                copied=manifest['audits'][row['trial_id']]
                assert sha(copied['path'])==copied['sha256']==cert['sha256']
            positive=int(data['labels'].sum());n=len(expected)
            result['gestures'][str(gesture)]=dict(rows=n,positive=positive,negative=n-positive)
            result['exported_rows']+=n;result['positive']+=positive;result['negative']+=n-positive
    assert result['positive']==10 and result['negative']==4
    assert manifest['verification_cost']['numeric_replays']==0
    result['source']=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
    target=OUTPUT/'INDEPENDENT_ARRAY_CHECK.json'
    with target.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':main()
