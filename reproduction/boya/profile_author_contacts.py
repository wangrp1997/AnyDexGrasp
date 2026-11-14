"""Offline, lossless contact-storage benchmark; never imports a simulator.

The experimental column format is a diagnostic artifact, not an execution
entry point. It retains every contact in order, including zero-force proximity
reports, reversed pairs, binary64 values and the absence of robot normals.
"""
from __future__ import annotations

import argparse
import collections
import cProfile
import gc
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import time

import numpy as np

SCHEMA = 'anydex_boya_offline_contact_profile_v1'
COLUMN_SCHEMA = 'anydex_boya_contact_columns_experiment_v1'
GROUPS = ('object_contacts', 'robot_contacts')
CONTACT_KEYS = {'bodies', 'position_world_m', 'separation_m', 'normal_force_n'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pack_columns(rows):
    """Dictionary-code paths and store all numbers as binary64, without casts."""
    paths, path_ids, pairs, values, normals, ends = [], {}, [], [], [], [0]
    steps = []
    for row in rows:
        if set(row) != {'step', *GROUPS} or type(row['step']) is not int:
            raise ValueError('Unknown contact-row fields or noninteger step')
        steps.append(row['step'])
        for group in GROUPS:
            for contact in row[group]:
                has_normal = 'raw_normal_world' in contact
                if set(contact) != CONTACT_KEYS | ({'raw_normal_world'} if has_normal else set()):
                    raise ValueError('Unknown contact fields cannot be silently dropped')
                pair = contact['bodies']
                if len(pair) != 2 or any(type(p) is not str for p in pair):
                    raise ValueError('Two string body paths required')
                encoded_pair = []
                for path in pair:
                    if path not in path_ids:
                        path_ids[path] = len(paths); paths.append(path)
                    encoded_pair.append(path_ids[path])
                position = contact['position_world_m']
                normal = contact.get('raw_normal_world', [0., 0., 0.])
                if len(position) != 3 or len(normal) != 3:
                    raise ValueError('Three coordinates required')
                numbers = [*position, *normal, contact['separation_m'], contact['normal_force_n']]
                if any(type(v) is not float or not np.isfinite(v) for v in numbers):
                    raise ValueError('Finite original float values required; no implicit conversion')
                pairs.append(encoded_pair); values.append(numbers); normals.append(has_normal)
            ends.append(len(pairs))
    metadata = dict(schema=COLUMN_SCHEMA, groups=list(GROUPS), paths=paths,
        value_columns=['position_x', 'position_y', 'position_z', 'normal_x', 'normal_y',
                       'normal_z', 'separation_m', 'normal_force_n'],
        contact_omission_count=0, numeric_storage='little-endian binary64',
        contact_order='tick, object_contacts, robot_contacts; original within-group order')
    return dict(metadata=np.frombuffer(json.dumps(metadata, separators=(',', ':')).encode(), dtype=np.uint8),
        steps=np.asarray(steps, dtype='<i8'), ends=np.asarray(ends, dtype='<i8'),
        pairs=np.asarray(pairs, dtype='<u4').reshape(-1, 2),
        values=np.asarray(values, dtype='<f8').reshape(-1, 8), has_normal=np.asarray(normals, dtype=np.uint8))


def unpack_columns(arrays):
    metadata = json.loads(arrays['metadata'].tobytes())
    if metadata['schema'] != COLUMN_SCHEMA or metadata['groups'] != list(GROUPS):
        raise ValueError('Unsupported experimental column schema')
    paths, ends, values, pairs, mask = metadata['paths'], arrays['ends'], arrays['values'], arrays['pairs'], arrays['has_normal']
    if len(ends) != len(arrays['steps']) * 2 + 1 or int(ends[-1]) != len(values):
        raise ValueError('Experimental column counts differ')
    rows = []
    for i, step in enumerate(arrays['steps']):
        row = dict(step=int(step))
        for j, group in enumerate(GROUPS):
            contacts = []
            for k in range(int(ends[2*i+j]), int(ends[2*i+j+1])):
                c = dict(bodies=[paths[int(v)] for v in pairs[k]], position_world_m=values[k, :3].tolist(),
                         separation_m=float(values[k, 6]), normal_force_n=float(values[k, 7]))
                if mask[k]:
                    c['raw_normal_world'] = values[k, 3:6].tolist()
                contacts.append(c)
            row[group] = contacts
        rows.append(row)
    return rows


def exact_rows(left, right):
    """Independent full-field comparison includes signed zero via JSON spelling."""
    if len(left) != len(right):
        raise AssertionError('Row count differs')
    digest = hashlib.sha256()
    for a, b in zip(left, right):
        x = json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        y = json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if x != y:
            raise AssertionError('At least one field, path, value, optional normal or order differs')
        digest.update(x); digest.update(b'\n')
    return digest.hexdigest()


def measure(name, operation, timings, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        gc.collect(); wall = time.perf_counter(); cpu = time.process_time()
        result = operation()
        samples.append(dict(wall_s=time.perf_counter()-wall, process_cpu_s=time.process_time()-cpu))
    timings[name] = dict(samples=samples,
        minimum_wall_s=min(s['wall_s'] for s in samples),
        median_wall_s=float(np.median([s['wall_s'] for s in samples])),
        median_process_cpu_s=float(np.median([s['process_cpu_s'] for s in samples])))
    return result


def classify(rows):
    counts = collections.Counter(); pairs = collections.Counter()
    for row in rows:
        for group in GROUPS:
            for c in row[group]:
                a, b = c['bodies']; force = c['normal_force_n']; separation = c['separation_m']
                counts[group] += 1
                counts['zero_force'] += force == 0.
                counts['positive_separation'] += separation > 0.
                kind = ('object_ground' if group == 'object_contacts' and '/ground/' in b else
                        'robot_robot' if all('/Robot/' in p for p in (a, b)) else
                        'object_robot' if group == 'object_contacts' and '/Robot/' in b else
                        'other')
                counts[kind] += 1; pairs[(group, a, b)] += 1
    return dict(counts=dict(counts), pairs=[dict(group=g, bodies=[a, b], count=n)
        for (g, a, b), n in pairs.most_common()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ticks', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.ticks not in (100, 200) or args.repeats < 1:
        raise ValueError('Predeclared prefix length 100 or 200 and positive repeats required')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    prepared = args.prepared_dir.resolve(); execution = prepared/'execution'
    start = time.perf_counter(); timings = {}; bindings = {}
    def bind(path):
        path = Path(path).resolve(); bindings[str(path)] = sha(path); return path
    pointer_path = bind(execution/'MULTI_OBJECT_CONTACTS.json'); pointer = read(pointer_path)
    data_path = bind(pointer['data']['path'])
    if bindings[str(data_path)] != pointer['data']['sha256']:
        raise ValueError('Original contact gzip does not match its frozen pointer')
    scene_path = bind(read(execution/'LIFT_EVALUATION.json')['scene_path'])
    trace_path = bind(execution/'MULTI_OBJECT_TRACE.json'); setup_path = bind(execution/'MULTI_OBJECT_SETUP.json')
    scene = read(scene_path); trace = read(trace_path)['rows'][:args.ticks]; setup = read(setup_path)
    def read_prefix():
        with gzip.open(data_path, 'rb') as stream:
            lines = [stream.readline() for _ in range(args.ticks)]
        if any(not line for line in lines):
            raise ValueError('Original execution has fewer than declared prefix ticks')
        return lines
    lines = measure('read_and_gzip_decompress_prefix', read_prefix, timings, args.repeats)
    rows = measure('json_decode', lambda: [json.loads(line) for line in lines], timings, args.repeats)
    if [r['step'] for r in rows] != list(range(1, args.ticks+1)):
        raise ValueError('Original prefix is not exactly the declared ticks')
    classification = classify(rows)
    encoded = measure('json_encode_original_runtime_options',
        lambda: [(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode() for row in rows],
        timings, args.repeats)
    def gzip_json():
        with (out/'sample.jsonl.gz').open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
                for line in encoded: stream.write(line)
    measure('gzip1_json_and_buffered_file_write', gzip_json, timings, args.repeats)
    arrays = measure('experimental_pack_from_original_dicts', lambda: pack_columns(rows), timings, args.repeats)
    def write_npz():
        with (out/'sample.columns.npz').open('wb') as stream: np.savez(stream, **arrays)
    measure('experimental_npz_uncompressed_write', write_npz, timings, args.repeats)
    raw_columns = (out/'sample.columns.npz').read_bytes()
    def gzip_columns():
        with (out/'sample.columns.npz.gz').open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
                stream.write(raw_columns)
    measure('experimental_gzip1_columns_and_buffered_file_write', gzip_columns, timings, args.repeats)
    def read_columns():
        with gzip.open(out/'sample.columns.npz.gz', 'rb') as stream:
            content = stream.read()
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    reloaded = measure('experimental_gzip_decompress_and_npz_load', read_columns, timings, args.repeats)
    restored = measure('experimental_restore_original_dicts', lambda: unpack_columns(reloaded), timings, args.repeats)
    exact_digest = exact_rows(rows, restored)
    try:
        from . import author_lift as lift, author_transport as transport
    except ImportError:
        import author_lift as lift, author_transport as transport
    import trimesh
    bind(lift.__file__); bind(transport.__file__)
    vertices = {}
    for obj in scene['objects']:
        mesh = trimesh.load_mesh(bind(obj['source_mesh']), process=False)
        vertices[obj['object_id']] = np.asarray(mesh.vertices, dtype=float)*obj['scale']
    robot = [p for p in setup['contact_targets'] if '/Robot/' in p]
    legal = set(setup['legal_hand_rigid_owners'])
    def aggregates():
        values = []
        for contacts, state in zip(rows, trace):
            result = {}
            for key, path in setup['body_paths'].items():
                result[key] = dict(**transport.object_robot_loads(path, contacts['object_contacts'], robot, legal),
                    **lift.source_metrics(vertices[key], state['objects'][key]['pose_xyzw']))
            result['max_penetration_m'] = max([0.]+[-c['separation_m'] for g in GROUPS for c in contacts[g]])
            values.append(result)
        return values
    aggregated = measure('original_three_body_loads_full_source_vertices_and_depth', aggregates, timings, args.repeats)
    bottom_error = max(abs(values[key]['source_bottom_m']-state['objects'][key]['source_bottom_m'])
        for values, state in zip(aggregated, trace) for key in setup['body_paths'])
    if bottom_error > 1e-9:
        raise ValueError('Loaded source-coordinate geometry does not reproduce actual runtime bottom')
    prof = cProfile.Profile(); prof.runcall(aggregates)
    prof.dump_stats(out/'aggregate.cprofile')
    with (out/'aggregate_profile.txt').open('w') as stream:
        pstats.Stats(prof, stream=stream).sort_stats('cumulative').print_stats(35)
    changed = [p for p, digest in bindings.items() if sha(p) != digest]
    if changed:
        raise ValueError('Read-only benchmark inputs changed: '+repr(changed))
    artifacts = {p.name: dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in out.iterdir()}
    report = dict(schema=SCHEMA, status='complete', sample_rule='Exactly first predeclared 100 or 200 native contact ticks; no success selection',
        prepared_dir=str(prepared), ticks=args.ticks, repeats=args.repeats, timings=timings,
        classification=classification, source_vertex_counts={k:len(v) for k,v in vertices.items()},
        max_recomputed_source_bottom_error_m=bottom_error,
        exact_roundtrip=dict(all_fields_values_optional_normals_contact_order_equal=True,
                             signed_zero_preserved=True, canonical_sha256=exact_digest),
        json_uncompressed_bytes=sum(map(len, encoded)), original_inputs_sha256=bindings,
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)), artifacts=artifacts,
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        physics_steps=0, original_input_writes=0, simulator_launches=0, label_updates=0,
        wall_s=time.perf_counter()-start,
        limitations=['Offline prefix CPU and buffered-file-write measurements; not an end-to-end speedup claim',
            'First ticks contain resting contacts; loaded grasp phases are not represented',
            'Dictionary-to-columns packing measured; direct native-array ingestion is not implemented or timed',
            'Experimental npz/gzip is not integrated with any execution entry or dataset audit',
            'Cold filesystem/cache, native physics, Kit startup and durable fsync were not benchmarked'])
    (out/'PROFILE.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out/'PROFILE.json'),timings=timings,
        classification=classification['counts'],bytes={k:v['bytes'] for k,v in artifacts.items()}),indent=2))


if __name__ == '__main__':
    main()
