"""Certificate integrity and unchanged export arrays; all stages synthetic."""
import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_export as E
import author_collection_waves as W
import author_collection_certified_export as C
from test_author_collection_batch import fixture, FakeStages, fake_trial


def enrich(batch_dir):
    """Synthetic first-audit certificates for integrity tests, never real labels."""
    protocol = B.read(batch_dir/'BATCH_PROTOCOL.json')
    for ledger_path in sorted((batch_dir/'ledger').glob('*.json')):
        row = B.read(ledger_path)
        if row.get('label') is None:
            continue
        trial = row['trial']; directory = Path(trial['record']['path']).parent
        scene_path = directory.parent/'capture/observed_scene.json'
        trial['scene'] = B.binding(scene_path)
        row['artifact_bindings']['scene'] = trial['scene']
        scene = B.read(scene_path)
        files = set()
        for name in ('LIFT_PIPELINE_PROTOCOL.json', 'LIFT_PIPELINE_RESULTS.json', 'READY.json',
                     'COLLECTION_RECORD.json', 'EXECUTION_STARTED.json', 'LIFT_PIPELINE_EXECUTION.json'):
            p = directory/name
            if not p.exists(): B.save(p, dict(synthetic=True))
            files.add(p)
        execution = directory/'execution'; execution.mkdir()
        data = execution/'contacts.jsonl.gz'; data.write_bytes(b'synthetic compressed contacts; never decoded')
        pointer = execution/'MULTI_OBJECT_CONTACTS.json'
        B.save(pointer, dict(schema='anydex_boya_contact_journal_pointer_v1', data=B.binding(data)))
        trace = execution/'MULTI_OBJECT_TRACE.json'; B.save(trace, dict(synthetic=True))
        auxiliary = directory/'auxiliary_source.py'; auxiliary.write_text('# synthetic source dependency\n')
        files.update((data, pointer, trace, auxiliary))
        files.update(Path(trial[key]['path']) for key in ('record', 'candidate', 'evaluation', 'commands', 'command_binding', 'scene'))
        files.update(Path(obj[name]) for obj in scene['objects'] for name in ('source_mesh', 'source_visual_mesh'))
        files.update((Path(protocol['sources']['audit_entry']['path']), Path(protocol['queue']['split_manifest']['path'])))
        cert = directory/'RUNTIME_AUDIT.json'
        old = B.read(cert)
        old.update(schema='anydex_boya_author_collection_runtime_audit_v1', prepared_dir=str(directory),
            audit_source=protocol['sources']['audit_entry'], checks=sorted(C.CHECKS), scene_id=row['scene_id'],
            candidate_sha256=trial['candidate_sha256'], commands_sha256=trial['commands']['sha256'],
            runtime=dict(physics_steps=trial['actual_physics_steps']),
            audit_inputs_sha256={str(p.resolve()): B.sha(p) for p in files}, wall_s=1.)
        B.save(cert, old)
        stage_path = Path(row['artifact_bindings']['audit_stage']['path'])
        stage = B.read(stage_path)
        stage.update(returncode=0, command=[sys.executable,'-B',protocol['sources']['audit_entry']['path'],
            '--prepared-dir',str(directory),'--split-manifest',protocol['queue']['split_manifest']['path'],
            '--output',str(cert)], outputs={str(cert):B.binding(cert)})
        B.save(stage_path, stage)
        row['artifact_bindings'].update(runtime_audit=B.binding(cert), audit_stage=B.binding(stage_path))
        B.save(ledger_path, row)


def completed(root, splits=('fit','internal'), wave=False):
    batch_dir, protocol = fixture(root, splits)
    protocol['sources']['audit_entry'] = B.binding(B.HERE/'audit_author_collection_runtime.py')
    B.save(batch_dir/'BATCH_PROTOCOL.json', protocol)
    if not wave:
        with patch.object(B,'_stage',side_effect=FakeStages()), patch.object(B,'validate_trial',side_effect=fake_trial), patch.object(B,'_lock',side_effect=lambda p:nullcontext()):
            B.run(batch_dir)
        enrich(batch_dir)
        return batch_dir
    parent = root/'waves'
    W.freeze(protocol['inputs']['queue']['path'], parent, fit_per_gesture=1, internal_per_gesture=1,
        calibration=protocol['inputs']['calibration']['path'],cloud_manifest=protocol['inputs']['cloud_manifest']['path'],
        backend=protocol['inputs']['backend']['path'],checkpoint=protocol['inputs']['checkpoint']['path'],
        scene_entry=protocol['sources']['scene_entry']['path'],capture_entry=protocol['sources']['capture_entry']['path'],
        execution_entry=protocol['sources']['execution_entry']['path'],audit_entry=protocol['sources']['audit_entry']['path'])
    wp, plan, index = W.verify_protocol(parent); admitted=[]
    for child in index['children']:
        directory = Path(child['protocol']['path']).parent.parent
        with patch.object(B,'_stage',side_effect=FakeStages()), patch.object(B,'validate_trial',side_effect=fake_trial), patch.object(B,'_lock',side_effect=lambda p:nullcontext()):
            B.run(directory/'batch')
        enrich(directory/'batch')
        result = W._child_result(directory,child,'complete')
        B.save(directory/'WORKER_RESULT.json',result,exclusive=True)
        child_trials=trial_lookup(directory/'batch')
        with patch.object(B,'validate_trial',side_effect=lambda p,s:child_trials[Path(p).resolve()]):
            W._admit(parent,wp,plan,child,result,admitted)
    return parent


def trial_lookup(source, wave=False):
    if wave:
        p,plan,index = W.verify_protocol(source)
        rows = [raw for _,raw,_,_ in C.X._ordered_rows(W.ledger(source,p),p,plan,index)]
    else:
        rows = B.ledger(source,B.read(source/'BATCH_PROTOCOL.json'))
    return {Path(row['trial']['record']['path']).parent.resolve():row['trial'] for row in rows if row.get('label') is not None}


def rebind_certificate(ledger_path, mutate):
    row=B.read(ledger_path);cert_path=Path(row['artifact_bindings']['runtime_audit']['path'])
    cert=B.read(cert_path);mutate(cert);B.save(cert_path,cert)
    stage_path=Path(row['artifact_bindings']['audit_stage']['path']);stage=B.read(stage_path)
    stage['outputs'][str(cert_path)]=B.binding(cert_path);B.save(stage_path,stage)
    row['artifact_bindings'].update(runtime_audit=B.binding(cert_path),audit_stage=B.binding(stage_path))
    B.save(ledger_path,row)


class CertifiedExportTests(unittest.TestCase):
    def test_same_npz_fields_values_and_no_numeric_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=completed(root);trials=trial_lookup(source)
            original=E._audit_trial
            with patch.object(B,'validate_trial',side_effect=lambda p,s:trials[Path(p).resolve()]):
                with patch.object(E,'_audit_trial',side_effect=lambda p,protocol:B.read(Path(p)/'RUNTIME_AUDIT.json')):
                    old=E.export(source,root/'reference')
                result=C.export(source,root/'certified')
            self.assertIs(E._audit_trial,original)
            self.assertEqual(result['verification_cost']['numeric_replays'],0)
            self.assertEqual(len(result['certificate_reuse']),8)
            for g in B.GESTURES:
                with np.load(root/'reference'/f'gesture_{g}.npz',allow_pickle=False) as a, np.load(root/'certified'/f'gesture_{g}.npz',allow_pickle=False) as b:
                    self.assertEqual(a.files,b.files)
                    for name in a.files:np.testing.assert_array_equal(a[name],b[name])
                B.verify({'npz':result['datasets'][str(g)]['binding']})
            self.assertEqual(result['wrapped_export_schema'],old['schema'])

    def test_tampered_contact_mesh_auxiliary_source_and_certificate_are_rejected(self):
        for kind in ('contact','mesh','source','certificate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                root=Path(td);source=completed(root,('fit',));trials=trial_lookup(source)
                directory=next(iter(trials));cert=directory/'RUNTIME_AUDIT.json'
                target={'contact':directory/'execution/contacts.jsonl.gz',
                    'mesh':Path(B.read(trials[directory]['scene']['path'])['objects'][0]['source_mesh']),
                    'source':directory/'auxiliary_source.py','certificate':cert}[kind]
                data=target.read_bytes();target.write_bytes(bytes([data[0]^1])+data[1:])
                with patch.object(B,'validate_trial',side_effect=lambda p,s:trials[Path(p).resolve()]):
                    with self.assertRaises((ValueError,FileNotFoundError)):C.export(source,root/'out')
                self.assertFalse((root/'out').exists())

    def test_missing_certificate_never_falls_back_to_numeric_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=completed(root,('fit',));trials=trial_lookup(source)
            (next(iter(trials))/'RUNTIME_AUDIT.json').unlink()
            with patch.object(E,'_audit_trial') as replay, self.assertRaises(FileNotFoundError):
                C.export(source,root/'out')
            replay.assert_not_called();self.assertFalse((root/'out').exists())

    def test_rebound_failed_wrong_trial_source_and_incomplete_input_map_are_rejected(self):
        mutations=[lambda a:a.update(status='failed'),lambda a:a.update(prepared_dir='/wrong/trial'),
            lambda a:a['audit_source'].update(sha256='f'*64),
            lambda a:a['audit_inputs_sha256'].pop(next(p for p in a['audit_inputs_sha256'] if p.endswith('contacts.jsonl.gz')))]
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i),tempfile.TemporaryDirectory() as td:
                root=Path(td);source=completed(root,('fit',));trials=trial_lookup(source)
                rebind_certificate(next((source/'ledger').glob('*.json')),mutate)
                with patch.object(B,'validate_trial',side_effect=lambda p,s:trials[Path(p).resolve()]):
                    with self.assertRaises(ValueError):C.export(source,root/'out')
                self.assertFalse((root/'out').exists())

    def test_input_changed_after_reuse_fails_final_full_rehash(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=completed(root,('fit',));trials=trial_lookup(source)
            target=next(iter(trials))/'execution/contacts.jsonl.gz';original=C._FullHashes.verify_again
            def change_then_verify(instance):
                target.write_bytes(b'changed after certificate reuse')
                return original(instance)
            old_audit=E._audit_trial
            with patch.object(B,'validate_trial',side_effect=lambda p,s:trials[Path(p).resolve()]), patch.object(C._FullHashes,'verify_again',change_then_verify):
                with self.assertRaises(ValueError):C.export(source,root/'out')
            self.assertIs(E._audit_trial,old_audit);self.assertFalse((root/'out').exists())

    def test_original_nonprimary_fit_internal_isolation_is_retained(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=completed(root);trials=trial_lookup(source)
            fit=next(t for p,t in trials.items() if 'fit_scene' in str(p))
            for trial in trials.values():
                if 'internal_scene' in trial['record']['path']:
                    trial['scene_object_ids'][-1]=fit['scene_object_ids'][-1]
                    trial['scene_mesh_sha256'][-1]=fit['scene_mesh_sha256'][-1]
                    p=source/'ledger'/f'0000001_g{trial["gesture_id"]}.json';row=B.read(p);row['trial']=trial;B.save(p,row)
            with patch.object(B,'validate_trial',side_effect=lambda p,s:trials[Path(p).resolve()]):
                with self.assertRaisesRegex(ValueError,'leakage'):C.export(source,root/'out')
            self.assertFalse((root/'out').exists())

    def test_wave_uses_same_admitted_prefix_and_certificates(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=completed(root,('fit',),wave=True);trials=trial_lookup(source,wave=True)
            with patch.object(B,'validate_trial',side_effect=lambda p,s:trials[Path(p).resolve()]):
                result=C.export(source,root/'out',mode='wave')
            self.assertEqual(result['exported_labels'],4);self.assertEqual(len(result['certificate_reuse']),4)
            self.assertEqual(result['wrapped_export_schema'],'anydex_boya_author_collection_wave_export_v1')
            self.assertFalse(result['size_ready']);self.assertEqual(result['verification_cost']['verification_passes'],2)


if __name__=='__main__':unittest.main()
