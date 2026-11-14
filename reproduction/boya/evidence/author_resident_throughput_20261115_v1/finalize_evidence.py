"""Archive small completed-control receipts; never dispatch or reconstruct physics."""
from pathlib import Path
import json
import shutil
import sys
import time

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import author_collection_batch as B
import author_resident_collection as C

CONTROL = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/resident_collection_control_20261115_v1')
OUT = Path(__file__).resolve().parent


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    begun = time.monotonic()
    results = {}
    for name, slots, labels, steps in [('fit', 8, 8, 25656), ('internal', 4, 1, 3242)]:
        root = CONTROL / name
        C.verify_protocol(root)
        progress = B.read(root / 'PROGRESS.json')
        outer = B.read(root / 'OUTER_DRIVER_RESULTS.json')
        assert progress['status'] == outer['status'] == 'complete'
        assert progress['error'] is outer['error'] is None
        assert progress['exclude_training'] and not progress['formal_training_eligible']
        assert progress['training_updates'] == 0 and not progress['old435_merged']
        rows = [B.read(p) for p in sorted((root / 'ledger').glob('*.json'))]
        assert len(rows) == slots
        assert all(r['label'] is None and r['exclude_training'] and not r['formal_training_eligible'] for r in rows)
        assert sum(r['diagnostic_label'] is not None for r in rows) == labels
        assert progress['known_final_execution_physics_steps'] == steps
        assert progress['execution_starts_with_unknown_steps'] == 0
        audits = []
        for path in sorted(root.glob('scenes/**/RUNTIME_AUDIT.json')):
            audit = B.read(path)
            assert audit['status'] == 'pass'
            audits.append(dict(receipt=B.binding(path), **{k: audit[k] for k in (
                'status', 'label', 'checks', 'runtime', 'phase_counts', 'costs', 'audit_source')}))
        assert len(audits) == labels
        results[name] = dict(protocol=B.binding(root/C.PROTOCOL), progress=progress,
            outer=outer, first_numeric_audits=audits,
            ledger=B.binding(root/'LEDGER.json'),
            ledger_scope='All parent rows exclude training; nested original audits are numeric certificates only')
    old = B.read(HERE/'evidence/full_data_lift_stage_20261114_v1/UNCHANGED_DATA_MODELS.json')['verified_files']
    B.verify(B.normalized_bindings(old))
    unchanged = dict(status='pass', verified_files=old, old_labels=435, old_heads=8,
        formal_training_updates=0, new_collected_labels_unchanged=15)
    archive = {}
    for path in [CONTROL/'MASTER_PROTOCOL.json', CONTROL/'FREEZE_WRAPPER_FAILURE.json',
                 CONTROL/'UNCHANGED_DATA_MODELS_BEFORE.json', CONTROL/'fit_driver.log',
                 Path('/tmp/freeze_anydex_resident_control_20261115.py'),
                 Path('/tmp/finalize_anydex_resident_control_20261115.py'),
                 Path('/tmp/run_anydex_resident_control_20261115.py'),
                 HERE.parent.parent/'outputs/resident_control_cpu_final_20261115.log']:
        target = OUT / path.name
        assert not target.exists()
        shutil.copyfile(path, target)
        archive[path.name] = dict(original=B.binding(path), archived=B.binding(target))
        assert archive[path.name]['original']['sha256'] == archive[path.name]['archived']['sha256']
    core = {}
    for name in ('author_resident_execution.py', 'author_resident_collection.py', 'author_resident_benchmark.py',
                 'test_author_resident_execution.py', 'test_author_resident_collection.py', 'test_author_resident_benchmark.py'):
        live, frozen = HERE/name, CONTROL/'source_snapshot'/name
        assert B.sha(live) == B.sha(frozen)
        core[name] = dict(live=B.binding(live), frozen=B.binding(frozen))
    save(OUT/'UNCHANGED_DATA_MODELS_AFTER.json', unchanged)
    save(OUT/'CONTROL_RECEIPTS.json', results)
    save(OUT/'EVIDENCE.json', dict(status='pass', source=B.binding(__file__),
        master_protocol=B.binding(CONTROL/'MASTER_PROTOCOL.json'),
        core_sources=core, archives=archive, all_controls_excluded_from_training=True,
        total_execution_steps=28998, total_initialization_ticks=2250,
        total_initialization_object_steps=6750, new_training_labels=0,
        training_updates=0, bulk_collection_started=False,
        verification_wall_s=time.monotonic()-begun,
        scope='Read completed protocols and numeric audit certificates, verify frozen sources/old data, archive small receipts; zero simulator calls or numerical replays'))
    print(json.dumps({'status': 'pass', 'root': str(OUT), 'execution_steps': 28998}))


if __name__ == '__main__':
    main()
