"""Read existing collection receipts; never import a simulator or run a stage.

A stage's wall time includes startup, checks, simulation and I/O. This tool does
not infer an internal bottleneck from that aggregate. Active/incomplete stages
are reported separately and excluded from completed interval measurements.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics

import author_collection_batch as B


def _intervals(rows):
    events = []
    for row in rows:
        events.extend(((row['start'], 1), (row['end'], -1)))
    active = maximum = 0
    for _, delta in sorted(events):
        active += delta; maximum = max(maximum, active)
    return maximum


def profile(run_dir):
    root = Path(run_dir).resolve(); inputs = {}; sources = {'profile': B.binding(__file__), 'batch_helpers': B.binding(B.__file__)}
    def read(path):
        value = B.read(path); inputs[str(path)] = B.binding(path); return value
    parent = root/'WAVE_PROTOCOL.json'
    if parent.exists():
        protocol = read(parent); index = read(Path(protocol['child_index']['path']))
        batches = [(c['queue_index'], Path(c['protocol']['path']).parent) for c in index['children']]
        ledger_paths = sorted((root/'ledger').glob('*.json'))
    else:
        protocol = read(root/'BATCH_PROTOCOL.json'); batches = [(None, root)]
        ledger_paths = sorted((root/'ledger').glob('*.json'))
    tasks = []; incomplete = []; scene_tasks = collections.defaultdict(list)
    audit_sizes = []; capture_sizes = []; scenes = {}
    for parent_index, batch in batches:
        for directory in sorted((batch/'scenes').glob('*')):
            index = int(directory.name.split('_')[0]) if parent_index is None else parent_index
            scenes[index] = directory.name
            for path in sorted((directory/'stages').glob('*/STAGE.json')):
                item = read(path); name = path.parent.name; kind = name.split('_g')[0]
                if item.get('status') == 'running' or 'wall_s' not in item:
                    incomplete.append(dict(path=str(path), status=item.get('status'))); continue
                value = dict(queue_index=index, stage=name, kind=kind, status=item['status'],
                    wall_s=item['wall_s'], start=item['started_unix_s'], end=item['started_unix_s']+item['wall_s'])
                tasks.append(value); scene_tasks[index].append(value)
            for path in directory.glob('g*/RUNTIME_AUDIT.json'):
                item = read(path)
                audit_sizes.append(dict(path=str(path), status=item.get('status'), wall_s=item.get('wall_s'),
                    contact_points=item.get('runtime', {}).get('contact_points'), physics_steps=item.get('runtime', {}).get('physics_steps')))
            for path in directory.glob('capture/CONTACT_JOURNAL.json'):
                item = read(path)
                capture_sizes.append(dict(path=str(path), records=item.get('record_count'), data_bytes=item.get('data', {}).get('bytes')))
    rows = [read(p) for p in ledger_paths]
    counts = collections.Counter(r['status'] for r in rows)
    eligible = [r for r in rows if r.get('label') is not None]
    if tasks:
        span = max(t['end'] for t in tasks)-min(t['start'] for t in tasks)
    else: span = 0.
    kinds = {}
    for kind in sorted({t['kind'] for t in tasks}):
        values = [t for t in tasks if t['kind'] == kind]; walls = [t['wall_s'] for t in values]
        kinds[kind] = dict(count=len(values), summed_stage_wall_s=sum(walls), mean_stage_wall_s=statistics.mean(walls),
            min_stage_wall_s=min(walls), max_stage_wall_s=max(walls), max_observed_completed_stage_overlap=_intervals(values))
    details = []
    for index, values in sorted(scene_tasks.items()):
        prep = [t for t in values if t['kind'] == 'prepare']
        details.append(dict(queue_index=index, scene=scenes[index],
            observed_scene_span_s=max(t['end'] for t in values)-min(t['start'] for t in values),
            prepare_span_s=max(t['end'] for t in prep)-min(t['start'] for t in prep) if prep else 0.))
    result = dict(schema='anydex_boya_collection_receipt_profile_v1', run_dir=str(root),
        protocol_schema=protocol['schema'], receipt_span_s=span, ledger_rows=len(rows), statuses=dict(counts),
        eligible_labels=len(eligible), labels_per_wall_hour=len(eligible)*3600/span if span else None,
        stage_timings=kinds, scenes=details, running_or_unfinished_stages=incomplete,
        contacts_from_existing_audits=audit_sizes, capture_journal_metadata=capture_sizes,
        max_observed_completed_kit_stage_overlap=_intervals([t for t in tasks if t['kind'] in ('capture', 'execute')]),
        max_observed_completed_observation_stage_overlap=_intervals([t for t in tasks if t['kind'] in ('capture','representation')]),
        scope='Receipt-only snapshot; no data eligibility certification, no model loads, no planning/physics; completed intervals only',
        internal_phase_timing='Unavailable in old receipts; stage wall includes startup, simulation, contact serialization, hashing and shutdown',
        sources=sources, input_receipts=inputs)
    if protocol['schema'] == B.SCHEMA and details:
        costs = {s['queue_index']: s['observed_scene_span_s'] for s in details}
        waves = [max((costs.get(i, 0.) for i in range(k,min(k+3,protocol['total_scene_budget']))), default=0.)
                 for k in range(0,protocol['total_scene_budget'],3)]
        result['hypothetical_frozen_wave_replay'] = dict(ideal_wave_spans_s=waves, ideal_total_s=sum(waves),
            upper_speedup_against_observed_span=span/sum(waves),
            assumptions='Same per-scene task durations, no slowdown or resource contention, no new overhead; theoretical replay, not a measured speedup')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True); p.add_argument('--output', type=Path)
    a = p.parse_args(); result = profile(a.run_dir)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    else: print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
