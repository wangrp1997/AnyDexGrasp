"""Wave admission/export integrity; synthetic stages, no Kit or model loads."""
from pathlib import Path
from contextlib import nullcontext
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import author_collection_batch as B
import author_collection_export as E
import author_collection_waves as W
import author_collection_wave_export as X
from test_author_collection_batch import fixture, FakeStages, fake_trial


def completed_parent(root, splits=('fit', 'fit', 'fit', 'internal'), skip_indices=()):
    legacy, old = fixture(root, splits)
    parent = root/'waves'
    W.freeze(old['inputs']['queue']['path'], parent, fit_per_gesture=2, internal_per_gesture=1,
        calibration=old['inputs']['calibration']['path'], cloud_manifest=old['inputs']['cloud_manifest']['path'],
        backend=old['inputs']['backend']['path'], checkpoint=old['inputs']['checkpoint']['path'],
        scene_entry=old['sources']['scene_entry']['path'], capture_entry=old['sources']['capture_entry']['path'],
        execution_entry=old['sources']['execution_entry']['path'], audit_entry=old['sources']['audit_entry']['path'])
    protocol, plan, index = W.verify_protocol(parent)
    rows = []
    with patch.object(B, '_stage', side_effect=FakeStages()), patch.object(B, 'validate_trial', side_effect=fake_trial), patch.object(B, '_lock', side_effect=lambda path: nullcontext()):
        for child in index['children']:
            if child['queue_index'] in skip_indices:
                W._quota_skip(parent, protocol, child, rows)
                continue
            directory = Path(child['protocol']['path']).parent.parent
            B.run(directory/'batch')
            result = W._child_result(directory, child, 'complete')
            B.save(directory/'WORKER_RESULT.json', result, exclusive=True)
            W._admit(parent, protocol, plan, child, result, rows)
    return parent


def stored_audit(prepared, protocol):
    return B.read(Path(prepared)/'RUNTIME_AUDIT.json')


class WaveExportTests(unittest.TestCase):
    def test_only_admitted_prefix_exported_and_all_three_source_columns_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = completed_parent(root)
            with patch.object(B, 'validate_trial', side_effect=fake_trial), patch.object(E, '_audit_trial', side_effect=stored_audit):
                result = X.export(parent, root/'export')
            self.assertTrue(result['size_ready']); self.assertEqual(result['exported_labels'], 12)
            self.assertEqual(result['quota_excess_not_exported'], 4)
            for g in B.GESTURES:
                with np.load(root/'export'/f'gesture_{g}.npz', allow_pickle=False) as data:
                    self.assertEqual(data['features'].shape, (3, 480))
                    self.assertEqual(data['scene_object_ids'].shape, (3, 3))
                    self.assertEqual(data['scene_mesh_sha256'].shape, (3, 3))
                    self.assertEqual(list(data['labels']), [0, 0, 0])
                    self.assertEqual(list(data['scene_id']), ['fit_scene_0', 'fit_scene_1', 'internal_scene_3'])
                    self.assertFalse(any(data[k].dtype.hasobject for k in data.files))
            self.assertFalse((parent/'BATCH_PROTOCOL.json').exists())

    def test_tampered_raw_record_source_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = completed_parent(root)
            row = W.ledger(parent)[0]
            Path(row['raw_ledger']['path']).write_text('{}\n')
            with self.assertRaises(ValueError):
                X.export(parent, root/'export')
            self.assertFalse((root/'export').exists())

    def test_later_label_cannot_replace_an_earlier_eligible_label(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = completed_parent(root)
            path = parent/'ledger/0000000_g1.json'; row = B.read(path)
            row['label'] = None; row['status'] = 'quota_excess_not_admitted'; row.pop('trial')
            B.save(path, row)
            with self.assertRaisesRegex(ValueError, 'first eligible'):
                X.export(parent, root/'export')

    def test_failed_independent_reaudit_cannot_produce_training_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = completed_parent(root)
            with patch.object(B, 'validate_trial', side_effect=fake_trial), patch.object(E, '_audit_trial', return_value=dict(status='failed')):
                with self.assertRaises(ValueError):
                    X.export(parent, root/'export')
            self.assertFalse((root/'export').exists())

    def test_partial_quantity_is_explicit_and_does_not_dispatch_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = completed_parent(root, ('fit',))
            with patch.object(B, 'validate_trial', side_effect=fake_trial), patch.object(E, '_audit_trial', side_effect=stored_audit), patch.object(W.subprocess, 'Popen') as process:
                result = X.export(parent, root/'export')
            process.assert_not_called()
            self.assertFalse(result['size_ready']); self.assertEqual(result['exported_labels'], 4)
            self.assertEqual(result['datasets']['1']['counts'], {'fit': 1, 'internal': 0})
            self.assertFalse(result['old435_merged']); self.assertEqual(result['training_updates'], 0)

    def test_unstarted_quota_skip_exports_no_raw_label_or_unknown_cost(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); parent = completed_parent(root, skip_indices=(2,))
            skipped = [r for r in W.ledger(parent) if r['queue_index'] == 2]
            self.assertTrue(all(r['status'] == 'quota_reached' and r['label'] is None
                                and r['actual_physics_steps'] == 0 and not r['execution_steps_unknown'] for r in skipped))
            with patch.object(B, 'validate_trial', side_effect=fake_trial), patch.object(E, '_audit_trial', side_effect=stored_audit):
                result = X.export(parent, root/'export')
            self.assertTrue(result['size_ready']); self.assertEqual(result['exported_labels'], 12)
            self.assertEqual(result['quota_excess_not_exported'], 0)


if __name__ == '__main__':
    unittest.main()
