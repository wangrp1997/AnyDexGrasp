"""Independent, read-only host observation of the fixed fresh-six wave run.

Only /proc, existing stage records/logs and nvidia-smi are read. This program
never acquires a runtime lock or imports/starts a planner, simulator or model.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
RUN = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/development_waves/waves_v1')
OUT = ROOT/'outputs/boya_parallel_runtime_audit_20261115_v1'
PROTOCOL = RUN/'WAVE_PROTOCOL.json'
GLOBAL = ROOT/'outputs/author_collection_gpu0.lock'
REGISTRY = GLOBAL.with_suffix('.wave_active.json')
INTERVAL_S = 2.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
        tail = raw[raw.rfind(')')+2:].split()
        return dict(pid=int(pid), pgid=int(tail[2]), start_ticks=int(tail[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def lock_fds(pid, paths):
    expected = {}
    for path in paths:
        try:
            stat = path.stat(); expected[(stat.st_dev, stat.st_ino)] = str(path)
        except FileNotFoundError:
            pass
    matches = []
    for p in Path('/proc', str(pid), 'fd').glob('*'):
        try:
            stat = p.stat(); target = expected.get((stat.st_dev, stat.st_ino))
            if target is None:
                continue
            lines = Path('/proc', str(pid), 'fdinfo', p.name).read_text().splitlines()
            locks = [line for line in lines if line.startswith('lock:')]
            matches.append(dict(fd=int(p.name), path=target,
                                inode=[stat.st_dev, stat.st_ino], lock_lines=locks,
                                exclusive_flock_observed=any('FLOCK' in x and 'WRITE' in x for x in locks)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return matches


def process_sample(state, slot_path, cache):
    expected = state.get('process')
    if expected is None:
        return None
    pid = expected['pid']; actual = identity(pid)
    if actual != expected:
        return dict(expected_process=expected, observed_process=actual, alive_identity_match=False)
    proc = Path('/proc', str(pid))
    try:
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        argv = [a.decode(errors='replace') for a in argv if a]
        libraries = sorted({line.split()[-1] for line in proc.joinpath('maps').read_text().splitlines()
                            if any(name in line.lower() for name in ('libcarb.so', 'libomni.kit', 'libomni.usd'))})
        env = {}
        for item in proc.joinpath('environ').read_bytes().split(b'\0'):
            if b'=' in item:
                k, v = item.split(b'=', 1)
                if k in (b'CUDA_VISIBLE_DEVICES', b'NVIDIA_VISIBLE_DEVICES'):
                    env[k.decode()] = v.decode(errors='replace')
        status = proc.joinpath('status').read_text().splitlines()
        memory = [line for line in status if line.startswith(('VmRSS:', 'VmHWM:'))]
        key = (pid, actual['start_ticks'], str(slot_path))
        if key not in cache:
            targets = [GLOBAL, slot_path.with_suffix('.lock')]
            targets += list((RUN/'resource_slots/observation').glob('*.lock'))
            cache[key] = lock_fds(pid, targets)
        fds = []
        for old in cache[key]:
            p = proc/'fd'/str(old['fd'])
            try:
                stat = p.stat()
                lines = (proc/'fdinfo'/str(old['fd'])).read_text().splitlines()
                locks = [line for line in lines if line.startswith('lock:')]
                fds.append(dict(old, current_inode=[stat.st_dev, stat.st_ino],
                                current_lock_lines=locks,
                                still_same_inode=[stat.st_dev, stat.st_ino] == old['inode'],
                                current_exclusive_flock=any('FLOCK' in x and 'WRITE' in x for x in locks)))
            except FileNotFoundError:
                fds.append(dict(old, descriptor_closed=True))
        stage = Path(state['stage_dir'])
        text = (stage/'process.log').read_text(errors='replace') if (stage/'process.log').exists() else ''
        active = [line for line in text.splitlines() if re.search(r'^\|\s*\d+\s*\|.*Yes:', line)]
        scene = stage.parents[1]
        if stage.name == 'capture':
            progress_path = scene/'capture/PROGRESS.json'
            journal = scene/'capture/initialization_contacts.jsonl.gz'
        else:
            gesture = stage.name.removeprefix('execute_')
            progress_path = scene/gesture/'execution/execution_progress.json'
            journal = scene/gesture/'execution/MULTI_OBJECT_CONTACTS.jsonl.gz'
        try:
            progress = read(progress_path)
            progress = {k:progress[k] for k in ('status', 'phase', 'physics_ticks', 'total_physics_steps', 'wall_s') if k in progress}
        except (FileNotFoundError, json.JSONDecodeError):
            progress = None
        journal_bytes = journal.stat().st_size if journal.exists() else 0
        return dict(process=actual, alive_identity_match=True, argv=argv,
                    kit_libraries=libraries, kit_loaded=bool(libraries), gpu_environment=env,
                    memory=memory, inherited_lock_fds=fds,
                    gpu_active_log_lines=active, gpu0_active_log_confirmed=bool(active) and
                    all(re.search(r'^\|\s*0\s*\|.*Yes:\s*0', line) for line in active),
                    stage_progress=progress, journal_path=str(journal), journal_bytes=journal_bytes)
    except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
        return dict(expected_process=expected, observed_process=identity(pid),
                    alive_identity_match=False, read_error=repr(error))


def gpu_snapshot():
    try:
        p = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as error:
        return dict(error=repr(error))


def main():
    if len(list(Path('/proc').glob('[0-9]*'))) < 8:
        raise RuntimeError('Host process namespace is required; sandbox-only /proc cannot witness Kit')
    protocol = read(PROTOCOL); protocol_sha = sha(PROTOCOL)
    pins = {x['path']:x['sha256'] for x in protocol['sources'].values()}
    for path, digest in pins.items():
        assert sha(path) == digest
    assert not (OUT/'FINAL.json').exists()
    events = OUT/'OBSERVATIONS.jsonl'
    stream = events.open('x')
    started = time.time(); cache = {}; history = {}; stages = {}; witnesses = []
    max_reserved = max_actual = max_loaded = samples = 0
    previous_live = {}; source_changed = []; parse_errors = []
    deadline = started + 6 * 3600
    while time.time() < deadline:
        now = time.time(); states = []; loaded = []; live = []
        for path in sorted((RUN/'resource_slots/kit').glob('*.json')):
            try:
                state = read(path)
            except (FileNotFoundError, json.JSONDecodeError) as error:
                parse_errors.append(dict(unix_s=now, path=str(path), error=repr(error)))
                continue
            key = state['reservation_id']; stage_record = stages.setdefault(key, {})
            stage_record.update(slot=state['slot'], stage_dir=state['stage_dir'],
                                reservation_id=key, reserved_unix_s=state['reserved_unix_s'],
                                last_observed_state=state['state'])
            if state.get('process'):
                stage_record['process'] = state['process']
            if 'finished_unix_s' in state:
                stage_record['finished_unix_s'] = state['finished_unix_s']
            fingerprint = json.dumps(state, sort_keys=True)
            if history.get(str(path)) != fingerprint:
                history[str(path)] = fingerprint
                event = dict(unix_s=now, event='resource_state_observed', resource_file=str(path), state=state)
                stream.write(json.dumps(event)+'\n'); stream.flush()
            if state['state'] not in ('running', 'reserved'):
                continue
            sample = process_sample(state, path, cache)
            value = dict(slot=state['slot'], reservation_id=key, stage_dir=state['stage_dir'], sample=sample)
            states.append(value)
            if sample and sample.get('alive_identity_match'):
                live.append(value)
                stage_record.setdefault('first_process_observed_unix_s', now)
                stage_record['last_process_observed_unix_s'] = now
                if sample['kit_loaded']:
                    loaded.append(value)
                    stage_record.setdefault('first_kit_loaded_observed_unix_s', now)
                    stage_record['last_kit_loaded_observed_unix_s'] = now
                    stage_record['gpu0_active_log_confirmed'] = sample['gpu0_active_log_confirmed']
                    stage_record['inherited_lock_fds'] = sample['inherited_lock_fds']
        samples += 1
        max_reserved = max(max_reserved, len(states)); max_actual = max(max_actual, len(live))
        growing = []
        for value in loaded:
            pid = value['sample']['process']['pid']; size = value['sample']['journal_bytes']
            if pid in previous_live and size > previous_live[pid] > 0:
                growing.append(pid)
        previous_live = {v['sample']['process']['pid']:v['sample']['journal_bytes'] for v in loaded}
        evidence_kind = None
        if len(loaded) > max_loaded:
            max_loaded = len(loaded); evidence_kind = 'new_maximum_simultaneously_loaded_kit'
        if len(growing) >= 2 and not any(w['kind'] == 'two_concurrently_growing_original_contact_journals' for w in witnesses):
            evidence_kind = 'two_concurrently_growing_original_contact_journals'
        if evidence_kind:
            witness = dict(unix_s=now, kind=evidence_kind, kits=loaded,
                           concurrently_growing_journal_pids=growing, gpu=gpu_snapshot())
            witnesses.append(witness)
            stream.write(json.dumps(dict(event='runtime_witness', **witness))+'\n'); stream.flush()
            print(json.dumps(dict(event=evidence_kind, loaded_kit_count=len(loaded),
                                 unix_s=now, pids=[v['sample']['process']['pid'] for v in loaded])), flush=True)
        summary = dict(schema='anydex_boya_independent_live_wave_runtime_audit_v1', status='monitoring',
                       protocol=dict(path=str(PROTOCOL), sha256=protocol_sha),
                       observer=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                       host_process_count=len(list(Path('/proc').glob('[0-9]*'))),
                       sample_interval_s=INTERVAL_S, samples=samples, first_sample_unix_s=started,
                       last_sample_unix_s=now, maximum_reserved_kit_slots=max_reserved,
                       maximum_identity_verified_stage_processes=max_actual,
                       maximum_library_verified_kit_processes=max_loaded,
                       simultaneous_two_kit_proven=max_loaded>=2,
                       concurrent_journal_growth_proven=any(w['kind']=='two_concurrently_growing_original_contact_journals' for w in witnesses),
                       witnesses=witnesses, stages=list(stages.values()), transient_read_errors=parse_errors,
                       costs_added=dict(Kit_starts=0, physics_steps=0, planning_calls=0, CGR_forwards=0, labels=0))
        save(OUT/'LIVE.json', summary)
        registry = read(REGISTRY) if REGISTRY.exists() else {}
        if (registry.get('run_dir') == str(RUN) and registry.get('state') == 'released'
                and len(list((RUN/'ledger').glob('*.json'))) == 24 and not live):
            summary.update(status='complete', registry=registry, observations=dict(path=str(events), sha256=sha(events)))
            source_changed = [p for p,h in pins.items() if sha(p)!=h]
            summary['changed_runtime_sources'] = source_changed
            summary['protocol_unchanged'] = sha(PROTOCOL)==protocol_sha
            save(OUT/'FINAL.json', summary)
            break
        if registry.get('run_dir') == str(RUN) and registry.get('state') == 'active_review_required' and not live:
            summary.update(status='runtime_quarantine', registry=registry)
            save(OUT/'FINAL.json', summary)
            break
        time.sleep(INTERVAL_S)
    else:
        summary.update(status='observer_deadline_reached')
        save(OUT/'FINAL.json', summary)
    stream.close()
    print(json.dumps(dict(status=summary['status'], final=str(OUT/'FINAL.json'), sha256=sha(OUT/'FINAL.json'),
                         simultaneous_two_kit_proven=summary['simultaneous_two_kit_proven'])), flush=True)


if __name__ == '__main__':
    main()
