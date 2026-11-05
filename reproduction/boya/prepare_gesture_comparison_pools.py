"""Prepare the six additional calibrated Boya gestures on frozen pilot inputs.

CPU hand geometry only. Old pilot pools and all physical trials remain read only.
A normal no-candidate result is retained; an infrastructure error stops dispatch.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

GESTURES = [2, 3, 5, 7, 9, 10]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify(mapping):
    for path, expected in mapping.items():
        if sha(path) != expected:
            raise ValueError('Frozen geometry input changed: ' + str(path))


def freeze(args, repo):
    old = read(args.pilot_protocol)
    cal_path = Path(old['calibration'])
    cal = read(cal_path)
    cfg = dict(candidate_limit=500, keep=8, width_expansion=1.7,
               opposing_group='ff', additional_depth_index=0,
               additional_depth_m=0., flips=[False, True])
    for k in ['candidate_limit', 'keep', 'width_expansion', 'opposing_group', 'additional_depth_index']:
        if old['geometry'][k] != cfg[k]:
            raise ValueError('Requested setting differs from original pilot: ' + k)
    if old['gesture_ids'] != [1, 6, 8] or len(old['objects']) != 8:
        raise ValueError('Expected the unchanged eight-object, three-gesture pilot')
    original = old['input_sha256']
    files = {str(args.pilot_protocol): sha(args.pilot_protocol)}
    def bind(path, expected=None):
        p = str(Path(path).resolve())
        value = sha(p)
        if expected is not None and value != expected:
            raise ValueError('Original pilot bytes differ: ' + p)
        files[p] = value
    bind(cal_path, original.get(str(cal_path)))
    bind(old['object_selection'], old['object_selection_sha256'])
    for path, record in cal['source_lock']['source_files'].items():
        bind(path, record['sha256'])
    for path, digest in cal['source_lock']['readonly_fk_dependencies'].items():
        bind(path, digest)
    for path in [Path(__file__), repo/'reproduction/boya/build_collection_pool.py',
                 repo/'reproduction/boya/asset_adapter.py']:
        bind(path)
    clear = {g: sum(s['gesture_id'] == g and s['self_collision_clear'] and
                   s['opposing_group'] == 'ff' for s in cal['slots']) for g in GESTURES}
    if any(n != 33 for n in clear.values()):
        raise ValueError('Expected all 33 self-clear ff openings for each added gesture')
    objects = []
    legacy = []
    for obj in old['objects']:
        row = dict(obj)
        row['pools'] = {str(g): str(args.run_dir/'pools'/obj['object_id']/('g'+str(g))) for g in GESTURES}
        objects.append(row)
        obs_path = Path(obj['observation_json']); obs = read(obs_path)
        if obs['object_id'] != obj['object_id'] or obs['explicit_physics_steps'] != 0:
            raise ValueError('Observation identity/zero-step boundary differs')
        bind(obs_path, original.get(str(obs_path)))
        for item in obs['files'].values():
            bind(item['path'], item['sha256'])
        frame = Path(obj['representation_dir']); receipt = read(frame/'RESULTS.json')
        if receipt['full_representation_forward'] is not True:
            raise ValueError('Expected an actual saved original CGR forward')
        for name in ['RESULTS.json', 'raw_camera_candidates.npz', 'representation_outputs.npz']:
            p = frame/name
            bind(p, original.get(str(p)))
        objroot = Path(obs['object_root'])
        for p in [objroot/'mesh/simplified.obj', objroot/'urdf/coacd.urdf']:
            bind(p, original.get(str(p)))
        for node in ET.parse(objroot/'urdf/coacd.urdf').findall('.//mesh'):
            bind((objroot/'urdf'/node.get('filename')).resolve())
        for g, path in obj['pools'].items():
            pool = Path(path)
            result = pool/'generation.json' if (pool/'generation.json').exists() else pool/'no_candidate.json'
            if not result.is_file():
                raise ValueError('Missing historical pilot geometry result: ' + str(pool))
            bind(result)
            if (pool/'candidates.npz').exists():
                bind(pool/'candidates.npz')
            legacy.append(dict(object_id=obj['object_id'], gesture_id=int(g), pool=str(pool), result=str(result)))
    return dict(schema='anydex_boya_gesture_comparison_geometry_v1',
        pilot_protocol=str(args.pilot_protocol), pilot_protocol_sha256=sha(args.pilot_protocol),
        run_dir=str(args.run_dir), objects=objects, object_ids=[o['object_id'] for o in objects],
        gesture_ids=GESTURES, previous_gesture_ids=[1, 6, 8],
        excluded_gesture_4=cal['excluded_templates'], calibration=str(cal_path),
        calibration_clear_ff_slots=clear, geometry=cfg, workers=args.workers,
        total_jobs=48, legacy_pools=legacy, input_sha256=files,
        new_network_forwards=0, new_physics_steps=0, new_training_steps=0,
        scope='Only six new geometric pools per original pilot object. No learned Boya head, no arm planning or physical trial; old pools/outcomes are not changed or ranked by success.')


def pool_result(obj, gesture, protocol, repo):
    pool = Path(obj['pools'][str(gesture)])
    cfg = protocol['geometry']
    command = [sys.executable, '-B', str(repo/'reproduction/boya/build_collection_pool.py'),
        '--observation', obj['observation_json'], '--frame-dir', obj['representation_dir'],
        '--calibration', protocol['calibration'], '--gesture-id', str(gesture),
        '--extra-depth-index', '0', '--candidate-limit', str(cfg['candidate_limit']),
        '--keep', str(cfg['keep']), '--width-expansion', str(cfg['width_expansion']),
        '--opponent', 'ff', '--output-dir', str(pool)]
    began = time.monotonic(); returncode = None
    reused = pool.exists()
    if not reused:
        pool.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with pool.with_suffix('.log').open('xb') as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=600)
        returncode = result.returncode
    generation, no_candidate = pool/'generation.json', pool/'no_candidate.json'
    if generation.exists() and not no_candidate.exists():
        d = read(generation)
        if returncode not in [None, 0] or d.get('complete') is not True or d['physics_steps'] != 0:
            raise ValueError('Invalid complete geometry result: ' + str(pool))
        if d['gesture_id'] != gesture or d['additional_depth_index'] != 0:
            raise ValueError('Wrong gesture/depth in output: ' + str(pool))
        if sha(pool/'candidates.npz') != d['candidates_sha256']:
            raise ValueError('Candidate bytes differ: ' + str(pool))
        verify(d['input_sha256'])
        return dict(status='ready', pool=str(pool), result_json=str(generation),
                    result_sha256=sha(generation), candidates_sha256=d['candidates_sha256'],
                    admitted=len(d['records']), rejected=len(d['rejected']),
                    reused=reused, child_returncode=returncode,
                    elapsed_wall_s=time.monotonic()-began, argv=command)
    if no_candidate.exists() and not generation.exists():
        d = read(no_candidate)
        if d['gesture_id'] != gesture or d['physics_steps'] != 0 or d['training_label'] is not None:
            raise ValueError('Invalid no-candidate boundary: ' + str(pool))
        if d['source_candidates_sha256'] != sha(Path(obj['representation_dir'])/'raw_camera_candidates.npz'):
            raise ValueError('No-candidate CGR binding differs')
        if d['source_observation_sha256'] != sha(obj['observation_json']) or d['calibration_sha256'] != sha(protocol['calibration']):
            raise ValueError('No-candidate input binding differs')
        return dict(status='no_candidate', pool=str(pool), result_json=str(no_candidate),
                    result_sha256=sha(no_candidate), admitted=0, rejected=len(d['records']),
                    reused=reused, child_returncode=returncode,
                    elapsed_wall_s=time.monotonic()-began, argv=command)
    raise RuntimeError('Missing final geometry result; partial output retained: ' + str(pool))


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-protocol', type=Path, default=repo/'outputs/boya_pilot8_20261105_v1/protocol_v2.json')
    parser.add_argument('--run-dir', type=Path, default=repo/'outputs/boya_gesture_comparison_20261105_v1')
    parser.add_argument('--workers', type=int, default=2, choices=[1, 2])
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args(); args.pilot_protocol = args.pilot_protocol.resolve(); args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.run_dir/'geometry.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = args.run_dir/'GEOMETRY_PROTOCOL.json'
        proposed = freeze(args, repo)
        if protocol_path.exists():
            protocol = read(protocol_path)
            if protocol != proposed:
                raise ValueError('Existing frozen protocol differs; use another output directory')
        else:
            protocol = proposed; atomic(protocol_path, protocol)
        verify(protocol['input_sha256'])
        if args.freeze_only:
            print(json.dumps(dict(status='frozen', protocol=str(protocol_path), jobs=48)), flush=True)
            return
        start = time.monotonic()
        tasks = [(o, g) for o in protocol['objects'] for g in GESTURES]
        jobs = {o['object_id']+'__g'+str(g): dict(object_id=o['object_id'], gesture_id=g,
                pool=o['pools'][str(g)], status='queued') for o, g in tasks}
        report = dict(schema='anydex_boya_gesture_comparison_preparation_v1', status='running',
            pid=os.getpid(), protocol=str(protocol_path), protocol_sha256=sha(protocol_path),
            started_at_unix=time.time(), geometry_workers=args.workers, total_jobs=48, jobs=jobs,
            errors=[], new_network_forwards=0, new_physics_steps=0, new_training_steps=0)
        target = args.run_dir/'POOL_PREPARATION.json'
        def publish():
            report.update(elapsed_wall_s=time.monotonic()-start,
                ready_jobs=sum(v['status']=='ready' for v in jobs.values()),
                no_candidate_jobs=sum(v['status']=='no_candidate' for v in jobs.values()),
                admitted_hand_candidates=sum(v.get('admitted', 0) for v in jobs.values()))
            atomic(target, report)
        publish(); iterator = iter(tasks); futures = {}; stopped = False
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            def dispatch():
                try: obj, gesture = next(iterator)
                except StopIteration: return False
                key = obj['object_id']+'__g'+str(gesture)
                jobs[key]['status'] = 'running'
                futures[executor.submit(pool_result, obj, gesture, protocol, repo)] = key
                publish(); return True
            for _ in range(args.workers): dispatch()
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    key = futures.pop(future)
                    try:
                        jobs[key].update(future.result())
                        print(json.dumps(dict(job=key, status=jobs[key]['status'], pool=jobs[key]['pool'], admitted=jobs[key]['admitted'])), flush=True)
                    except Exception as error:
                        stopped = True
                        jobs[key].update(status='infrastructure_error', error=repr(error))
                        report['errors'].append(dict(job=key, error=repr(error)))
                        print(json.dumps(report['errors'][-1]), flush=True)
                    publish()
                if not stopped:
                    while len(futures) < args.workers and dispatch(): pass
            if stopped:
                for value in jobs.values():
                    if value['status'] == 'queued': value['status'] = 'not_dispatched_after_infrastructure_error'
        try:
            verify(protocol['input_sha256'])
            if sha(protocol_path) != report['protocol_sha256']:
                raise ValueError('Frozen protocol changed during geometry preparation')
            report['inputs_unchanged'] = True
        except Exception as error:
            stopped = True; report['inputs_unchanged'] = False
            report['errors'].append(dict(stage='final_input_integrity', error=repr(error)))
        report.update(status='infrastructure_stopped' if stopped else 'complete', completed_at_unix=time.time())
        publish()
        print(json.dumps({k:v for k,v in report.items() if k not in ['jobs']}), flush=True)
        if stopped: raise SystemExit(1)


if __name__ == '__main__':
    main()
