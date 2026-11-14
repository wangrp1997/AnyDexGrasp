"""Complete thin-loop contracts with synthetic stages; no model or simulator."""
from contextlib import nullcontext
import copy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import author_collection_batch as B
import author_collection_wave_resources as R
import author_resident_collection as C
from test_author_collection_batch import fixture, FakeStages, fake_trial


class Stages(FakeStages):
    def __init__(self, *, states=None, audit_status='pass', group_failure=False, unknown_group=False):
        super().__init__(states=states, audit_status=audit_status)
        self.model_loads = 0; self.forwards = 0; self.groups = []; self.exec_active = 0; self.max_exec = 0
        self.group_failure = group_failure; self.unknown_group = unknown_group

    def __call__(self, directory, command, expected, protocol):
        directory = Path(directory)
        if directory.name == 'representation':
            directory.mkdir(parents=True, exist_ok=True)
            manifest = B.read(command[command.index('--manifest')+1])
            self.model_loads += 1; self.forwards += len(manifest['objects'])
            B.save(expected[0], dict(model_load_count=1, network_forward_count=len(manifest['objects']),
                hand_head_load_count=0, completed_forward_objects=len(manifest['objects'])))
            for path in expected[1:]: path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
            result = dict(status='complete', outputs={str(p): B.binding(p) for p in expected})
            B.save(directory/'STAGE.json', result)
            self.calls.append('representation'); return result
        if directory.name == 'resident_execute':
            directory.mkdir(parents=True, exist_ok=True)
            targets = [Path(p) for p in command[command.index('--prepared-dirs')+1:command.index('--output-dir')]]
            self.groups.append(targets)
            with self.lock: self.exec_active += 1; self.max_exec = max(self.max_exec, self.exec_active)
            time.sleep(.02)
            jobs = []
            for index, target in enumerate(targets):
                if self.group_failure and index > 0:
                    jobs.append(dict(prepared_dir=str(target), status='not_started')); continue
                if not self.unknown_group:
                    B.save(target/'LIFT_PIPELINE_EXECUTION.json', dict(status='complete', complete=True, physics_steps=23))
                    B.save(target/'RESIDENT_EXECUTION_ENTRY.json', dict(status='complete'))
                    B.save(target/'COLLECTION_RECORD.json', dict(label=0))
                    jobs.append(dict(prepared_dir=str(target), status='complete'))
            if not self.unknown_group:
                B.save(expected[0], dict(jobs=jobs, status='failed' if self.group_failure else 'complete'))
                B.save(expected[1], dict(status='complete', real_close_pending=True))
            with self.lock: self.exec_active -= 1
            result = dict(status='infrastructure_failure' if self.group_failure or self.unknown_group else 'complete',
                outputs={} if self.group_failure or self.unknown_group else {str(p): B.binding(p) for p in expected})
            B.save(directory/'STAGE.json', result); self.calls.append('resident_execute'); return result
        return super().__call__(directory, command, expected, protocol)


class ResidentCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def frozen(self, splits=('fit', 'fit'), placement_null=(), indices=None, workers=1, compact=False):
        batch, base = fixture(self.root, splits=splits, placement_null=placement_null,
            capture_entry=B.HERE/'author_collection_capture.py' if compact else None)
        output = self.root/'resident'
        protocol = C.freeze_from_batch(batch, list(range(len(splits))) if indices is None else indices,
            output, resident_workers=workers)
        return output, protocol

    def execute(self, root, stages):
        with patch.object(B, '_stage', stages), patch.object(B, 'validate_trial', fake_trial), \
                patch.object(R, 'install_worker_adapters', lambda *_: nullcontext()), \
                patch.object(C, 'LEGACY_LOCK', self.root/'gpu.lock'), patch.object(C, 'REGISTRY', self.root/'active.json'):
            return C.run(root)

    def test_freeze_reference_exact_and_no_runtime_or_legacy_parent(self):
        with patch.object(B.subprocess, 'Popen', side_effect=AssertionError('Freeze dispatched')):
            root, protocol = self.frozen(splits=('fit', 'fit', 'internal'), indices=[1], workers=2)
        self.assertEqual(protocol['source_scene_indices'], [1]); self.assertEqual(protocol['queue']['seed'], 123)
        self.assertEqual(protocol['total_gesture_attempt_budget'], 4)
        self.assertTrue(protocol['exclude_training']); self.assertFalse((root/'BATCH_PROTOCOL.json').exists())
        self.assertEqual(C.verify_protocol(root), protocol)
        with self.assertRaises(FileNotFoundError): B.run(root)
        with self.assertRaises(ValueError): C.freeze_from_batch(self.root/'batch', [1, 0], self.root/'invalid')

    def test_one_model_for_two_scenes_prepare_three_and_resident_groups_two(self):
        root, protocol = self.frozen(workers=2)
        stages = Stages()
        result = self.execute(root, stages)
        self.assertEqual(stages.model_loads, 1); self.assertEqual(stages.forwards, 2)
        self.assertEqual(stages.maximum, 3); self.assertEqual(stages.max_exec, 2)
        self.assertEqual([[p.name for p in group] for group in stages.groups], [['g1','g6','g7','g8']]*2)
        rows = C.ledger(root)
        self.assertEqual(len(rows), 8); self.assertTrue(all(r['raw_label'] == 0 for r in rows))
        self.assertTrue(all(r['label'] is None and r['exclude_training'] and not r['formal_training_eligible'] for r in rows))
        self.assertEqual(result['known_final_execution_physics_steps'], 184)
        self.assertEqual(result['captured_initialization_ticks'], 1500)
        self.assertEqual(result['captured_initialization_object_steps'], 4500)
        self.assertGreater(result['finished_unix_s'], result['started_unix_s'])
        self.assertGreater(result['timing_seconds']['total_wall'], 0)
        self.assertFalse(result['parent_launches_kit'])
        self.assertTrue(all('runtime_audit' in row['artifact_bindings'] for row in rows))

    def test_nulls_are_not_negative_and_only_ready_executes(self):
        root, protocol = self.frozen(placement_null=(1,))
        stages = Stages(states={1:'ready',6:'null_geometry',7:'no_candidate',8:'infrastructure_failure'})
        result = self.execute(root, stages)
        rows = C.ledger(root)
        self.assertEqual([p.name for p in stages.groups[0]], ['g1'])
        self.assertEqual(sum(r['raw_label'] is not None for r in rows), 1)
        self.assertEqual(sum(r['status'] == 'placement_null' for r in rows), 4)
        self.assertEqual(result['known_final_execution_physics_steps'], 23)
        self.assertEqual(stages.forwards, 1)

    def test_audit_failure_preserves_cost_and_rejects_label(self):
        root, _ = self.frozen(splits=('fit',))
        result = self.execute(root, Stages(audit_status='failed'))
        rows = C.ledger(root)
        self.assertTrue(all(r['status'] == 'infrastructure_failure' and r['raw_label'] is None for r in rows))
        self.assertEqual(result['known_final_execution_physics_steps'], 92)
        self.assertTrue(all('runtime_audit' in r['artifact_bindings'] for r in rows))

    def test_later_group_failure_retains_complete_job_and_explicit_not_started(self):
        root, _ = self.frozen(splits=('fit',))
        result = self.execute(root, Stages(group_failure=True))
        rows = C.ledger(root)
        self.assertEqual(rows[0]['raw_label'], 0)
        self.assertTrue(all(r['raw_label'] is None and not r['execution_steps_unknown'] for r in rows[1:]))
        self.assertTrue(all(r['status'] == 'execution_not_started' for r in rows[1:]))
        self.assertEqual(result['known_final_execution_physics_steps'], 23)

    def test_missing_group_finalization_does_not_claim_zero_physics(self):
        root, _ = self.frozen(splits=('fit',))
        result = self.execute(root, Stages(unknown_group=True))
        self.assertEqual(result['execution_starts_with_unknown_steps'], 4)
        self.assertTrue(all(r['raw_label'] is None for r in C.ledger(root)))

    def test_started_run_is_never_restarted(self):
        root, _ = self.frozen(splits=('fit',))
        stages = Stages(); self.execute(root, stages)
        before = len(stages.calls)
        with self.assertRaises(FileExistsError): self.execute(root, stages)
        self.assertEqual(len(stages.calls), before)

    def test_bound_input_change_prevents_any_dispatch(self):
        root, protocol = self.frozen(splits=('fit',))
        Path(protocol['inputs']['backend']['path']).write_text('{"changed":true}')
        stages = Stages()
        with self.assertRaises(ValueError): self.execute(root, stages)
        self.assertEqual(stages.calls, [])
        self.assertFalse((root/'RUN_STARTED.json').exists())

    def test_capture_binds_full_initialization_and_lossless_contacts(self):
        root, protocol = self.frozen(splits=('fit',), compact=True)
        names = {p.name for p in C._capture_outputs(root/'capture', protocol)}
        self.assertTrue({'CONTACT_JOURNAL.json','initialization_contacts.jsonl.gz','initialization_trace.npz',
            'native_rgbd.npz','render_geometry_check.json'}.issubset(names))

    def test_postexecution_integrity_failure_cannot_create_label_and_retains_cost(self):
        root, protocol = self.frozen(splits=('fit',))
        class Changed(Stages):
            def __call__(self, directory, command, expected, source_protocol):
                result = super().__call__(directory, command, expected, source_protocol)
                if Path(directory).name == 'resident_execute':
                    Path(protocol['inputs']['backend']['path']).write_text('{"changed_after_execution":true}')
                return result
        with self.assertRaises(ValueError): self.execute(root, Changed())
        self.assertFalse(any((root/'ledger').glob('*.json')))
        final = B.read(root/'RESIDENT_COLLECTION_RESULTS.json')
        self.assertEqual(final['known_final_execution_physics_steps'], 92)
        self.assertEqual(len(final['unadmitted_execution_costs']), 4)
        self.assertIsNotNone(final['error'])

    def test_new_execute_entry_uses_original_kit_slot_classifier(self):
        root, protocol = self.frozen(splits=('fit',))
        stage_protocol = copy.deepcopy(protocol)
        stage_protocol['sources']['execution_entry'] = protocol['sources']['author_resident_execution.py']
        self.assertEqual(R.classify([protocol['inputs']['python']['path'], '-B',
            stage_protocol['sources']['execution_entry']['path'], 'execute-many'], stage_protocol), ('kit',))

    def test_new_native_resident_dependency_is_frozen_before_execution(self):
        import author_resident_execution as resident
        native = self.root/'resident_native.py'; native.write_text('# frozen existing helper\n')
        with patch.object(resident, 'NATIVE_RESIDENT', native):
            root, protocol = self.frozen(splits=('fit',))
        self.assertEqual(protocol['sources']['frozen_native_resident'], B.binding(native))
        native.write_text('# changed after freeze\n')
        with self.assertRaises(ValueError): C.verify_protocol(root)


if __name__ == '__main__': unittest.main()
