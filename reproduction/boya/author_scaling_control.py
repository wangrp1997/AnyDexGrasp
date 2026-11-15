"""Bounded GPU-batch versus CPU-process controls; never a collection launcher.

One unchanged g6 template, GPU 1/4/8 then CPU 1/4/8, 100 ticks per
environment. Each group is exclusive; CPU workers within a group run together.
All prefixes are incomplete, unlabeled, and excluded from training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import author_collection_batch as B
import author_collection_wave_resources as R
import author_gpu_control as G

SCHEMA = 'anydex_boya_author_scaling_control_v1'
PROTOCOL = 'SCALING_CONTROL_PROTOCOL.json'
RESULTS = 'SCALING_CONTROL_RESULTS.json'
WORKER_RECEIPT = 'SCALING_EXECUTION.json'
COUNTS = (1, 4, 8)
PREFIX_TICKS = 100
MAX_WORKERS = 8
WORKER = B.HERE/'author_scaling_execution.py'
REFERENCE = G.BASE/'gpu_vector_control_20261115_v2'/G.PROTOCOL
COST_FIELDS = ('returned_physical_steps', 'actual_simulation_step_counter',
    'observations_completed_steps_per_lane', 'physical_steps_unknown',
    'inflight_step_unconfirmed', 'physical_steps_lower_bound', 'physical_steps_upper_bound',
    'integrated_environment_steps_lower_bound', 'integrated_environment_steps_upper_bound')


def need(condition, message):
    if not condition:
        raise ValueError(message)


class ScalingPool(R.ResourcePool):
    """Existing reservations/FD inheritance, independent eight-Kit ceiling.

    Only _acquire differs. The frozen shared R.LIMITS is never mutated.
    """
    def _acquire(self, kind, reservation):
        need(kind == 'kit', 'Scaling controls reserve only their own Kit slots')
        while True:
            self.check_halt()
            for index in range(MAX_WORKERS):
                base = self.root/kind/str(index)
                base.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(base)+'.lock', os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                path = Path(str(base)+'.json')
                try:
                    old = B.read(path) if path.exists() else None
                    if old is not None and old.get('state') != 'released':
                        self.halt('Unreleased scaling reservation', slot=str(base), previous=old)
                        raise R.Quarantined('Scaling slot requires lifetime review: '+str(base))
                    state = dict(reservation, resource=kind, slot=index, state='reserved',
                        reserved_unix_s=time.time())
                    B.save(path, state)
                    return dict(fd=fd, path=path, state=state)
                except BaseException:
                    os.close(fd)
                    raise
            time.sleep(.05)


def gpu_identity(executable, uuid):
    """NVML-backed nvidia-smi query of the declared GPU only."""
    value = subprocess.run([str(executable), '--id='+uuid,
        '--query-gpu=index,uuid,pci.bus_id,name', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10.)
    rows = [line.split(',', 3) for line in value.stdout.strip().splitlines()]
    need(len(rows) == 1 and len(rows[0]) == 4, 'Exactly one declared GPU required')
    index, actual, pci, name = (x.strip() for x in rows[0])
    need(index == '0' and actual == uuid, 'Declared physical GPU0 UUID changed')
    return dict(index=0, uuid=actual, pci_bus_id=pci, name=name,
        query_scope='Only declared GPU UUID; GPU1 is never selected')


def archive_sources(root, sources):
    """Preserve exact unique source bytes; never copy inputs or binaries."""
    root = Path(root)
    B.verify(sources)
    rows = {}
    for name, source in sources.items():
        key = (source['path'], source['sha256'])
        if key not in rows:
            path = root/'frozen_sources'/(source['sha256']+'_'+Path(source['path']).name)
            path.parent.mkdir(parents=True, exist_ok=True)
            content = Path(source['path']).read_bytes()
            if path.exists():
                need(B.sha(path) == source['sha256'], 'A source archive name collided')
            else:
                with path.open('xb') as stream:
                    stream.write(content)
            need(B.sha(path) == source['sha256'], 'Archived source differs from its frozen SHA')
            rows[key] = dict(source=source, archive=B.binding(path), binding_names=[])
        rows[key]['binding_names'].append(name)
    B.verify(sources)
    path = root/'SOURCE_ARCHIVE.json'
    B.save(path, dict(schema=SCHEMA, scope='Unique sources only; inputs and nvidia-smi binary not copied',
        sources=list(rows.values())), exclusive=True)
    return B.binding(path)


def freeze(output_dir, *, reference_protocol=REFERENCE):
    reference_protocol = Path(reference_protocol).resolve()
    old = B.read(reference_protocol)
    need(old.get('diagnostic_only') is True and old.get('exclude_training') is True,
        'Reviewed diagnostic GPU protocol required')
    job = old['jobs'][0]
    need(job['name'] == 'single_g6' and job['gestures'] == [6]
        and len(job['prepared_dirs']) == 1, 'One fixed original g6 template required')
    prepared = Path(job['prepared_dirs'][0]).resolve()
    from author_gpu_vector_execution import load_jobs, source_bindings
    checked = load_jobs([prepared])[0]  # Existing read-only ready/plan/geometry checks.
    need(checked['command_steps'] > PREFIX_TICKS, 'The prefix must remain incomplete')
    sources = copy.deepcopy(old['sources'])
    sources.update({str(p): dict(path=str(p), sha256=h) for p, h in
        ((Path(p).resolve(), h) for p, h in source_bindings().items())})
    for name in ('author_scaling_control.py', 'author_scaling_execution.py',
                 'author_gpu_vector_profile.py'):
        sources[name] = B.binding(B.HERE/name)
    inputs = copy.deepcopy(old['inputs'])
    inputs['reference_protocol'] = B.binding(reference_protocol)
    for key, binding in checked['bindings'].items():
        inputs['g6_'+key] = binding
    smi = shutil.which('nvidia-smi')
    need(smi is not None, 'nvidia-smi required to bind and sample physical GPU0')
    inputs['nvidia_smi'] = B.binding(smi)
    identity = gpu_identity(smi, old['gpu_uuid'])
    out = Path(output_dir).resolve()
    groups = []
    for mode in ('gpu', 'cpu'):
        for count in COUNTS:
            name = f'{mode}_{count}'
            workers = []
            for i in range(1 if mode == 'gpu' else count):
                worker_id = f'worker_{i:02d}'
                workers.append(dict(worker_id=worker_id, prepared_dir=str(prepared),
                    lanes=count if mode == 'gpu' else 1,
                    output_dir=str(out/name/worker_id/'result'),
                    stage_dir=str(out/name/worker_id)))
            groups.append(dict(name=name, mode=mode, count=count, workers=workers,
                physical_ticks_per_worker=PREFIX_TICKS,
                integrated_environment_steps=PREFIX_TICKS*count))
    protocol = dict(schema=SCHEMA, task=B.TASK, diagnostic_only=True,
        exclude_training=True, formal_training_eligible=False, label=None, raw_label=None,
        prefix_ticks=PREFIX_TICKS, groups=groups, stage_timeout_s=900., retry_budget=0,
        cancel_sigint_grace_s=30., cancel_sigterm_grace_s=5., cancel_sigkill_grace_s=5.,
        resource_limits=dict(kit=MAX_WORKERS), global_gpu_lock=str(G.LOCK),
        global_registry=str(G.REGISTRY), gpu_identity=identity, gpu_uuid=old['gpu_uuid'],
        prepared_template=dict(path=str(prepared), ready=B.binding(prepared/'READY.json'),
            commands=checked['bindings']['commands']),
        total_worker_attempt_budget=16, total_integrated_environment_steps=2600,
        total_group_prefix_ticks=600, total_worker_physical_clock_steps=1600,
        groups_are_sequential=True,
        worker_source=B.binding(WORKER), inputs=inputs, sources=sources,
        inherited_env=old['inherited_env'], env=dict(old['env'],
            CUDA_VISIBLE_DEVICES=old['gpu_uuid'], NVIDIA_VISIBLE_DEVICES='0'),
        created_unix_s=time.time(), sampling_interval_s=2.,
        limitations=['Early settle prefixes cannot establish full grasp validity or collection ETA',
            'One known template and one attempt per size are not independent statistical replicates',
            'CPU physics versus GPU physics may differ; original command/logging/gates remain enabled',
            'Host GPU samples do not attribute a unique bottleneck or include another GPU',
            'No null prefix becomes a negative training label'])
    B.verify(inputs); B.verify(sources)
    out.mkdir(parents=True, exist_ok=False)
    protocol['source_archive'] = archive_sources(out, sources)
    B.verify(inputs); B.verify(sources)
    B.save(out/PROTOCOL, protocol, exclusive=True)
    verify(out)
    return protocol


def verify(root):
    root = Path(root).resolve()
    p = B.read(root/PROTOCOL)
    need(p.get('schema') == SCHEMA and p.get('exclude_training') is True
        and p.get('diagnostic_only') is True and p.get('prefix_ticks') == PREFIX_TICKS
        and p.get('retry_budget') == 0 and p.get('stage_timeout_s') == 900.
        and p.get('resource_limits') == {'kit': MAX_WORKERS}
        and p.get('total_worker_attempt_budget') == 16
        and p.get('total_integrated_environment_steps') == 2600,
        'Frozen 100-tick scaling diagnostic required')
    expected = [(m, n) for m in ('gpu', 'cpu') for n in COUNTS]
    need([(g['mode'], g['count']) for g in p['groups']] == expected,
        'Fixed ordered GPU 1/4/8 then CPU 1/4/8 required')
    for g in p['groups']:
        need(g['name'] == f'{g["mode"]}_{g["count"]}' and len(g['workers']) ==
            (1 if g['mode'] == 'gpu' else g['count']), 'Frozen worker count differs')
        for i, w in enumerate(g['workers']):
            base = root/g['name']/f'worker_{i:02d}'
            need(w['worker_id'] == f'worker_{i:02d}' and w['stage_dir'] == str(base)
                and w['output_dir'] == str(base/'result')
                and w['prepared_dir'] == p['prepared_template']['path']
                and w['lanes'] == (g['count'] if g['mode'] == 'gpu' else 1),
                'A worker escaped the frozen g6/group/output contract')
    need(p['env']['CUDA_VISIBLE_DEVICES'] == p['gpu_uuid']
        and p['env']['NVIDIA_VISIBLE_DEVICES'] == '0', 'Only physical GPU0 allowed')
    B.verify(p['inputs']); B.verify(p['sources'])
    B.verify({'source_archive': p['source_archive']})
    archive = B.read(p['source_archive']['path'])
    need(archive.get('schema') == SCHEMA, 'Source archive schema differs')
    archived = {(row['source']['path'], row['source']['sha256']) for row in archive['sources']}
    need(archived == {(s['path'], s['sha256']) for s in p['sources'].values()},
        'Every frozen unique source must have an exact archive')
    for row in archive['sources']:
        need(row['source']['sha256'] == row['archive']['sha256'], 'Source/archive SHA differ')
        B.verify({'source': row['source'], 'archive': row['archive']})
    B.verify({'worker': p['worker_source'], 'ready': p['prepared_template']['ready'],
        'commands': p['prepared_template']['commands']})
    return p


class ResourceSampler:
    """Only declared GPU0 and the controller's own worker process identities."""
    def __init__(self, root, protocol):
        self.path = Path(root)/'RESOURCE_SAMPLES.jsonl'
        self.protocol = protocol
        self.processes = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def add(self, identity):
        if identity is not None:
            with self.lock:
                self.processes[identity['pid']] = identity

    def sample(self):
        with self.lock:
            identities = list(self.processes.values())
        owned = []
        for identity in identities:
            if R.process_identity(identity['pid']) != identity:
                continue
            try:
                raw = Path('/proc', str(identity['pid']), 'stat').read_text()
                fields = raw[raw.rfind(')')+2:].split()
                owned.append(dict(process=identity, utime_ticks=int(fields[11]),
                    stime_ticks=int(fields[12]), rss_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE')))
            except (FileNotFoundError, ProcessLookupError):
                pass
        row = dict(unix_s=time.time(), monotonic_s=time.monotonic(),
            live_owned_worker_count=len(owned), owned_workers=owned,
            cpu_clock_ticks_per_second=os.sysconf('SC_CLK_TCK'))
        try:
            result = subprocess.run([self.protocol['inputs']['nvidia_smi']['path'],
                '--id='+self.protocol['gpu_uuid'],
                '--query-gpu=uuid,utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                check=True, timeout=5.)
            row['gpu0_nvml_csv'] = result.stdout.strip()
        except Exception as error:
            row['gpu0_sampling_error'] = repr(error)
        return row

    def _loop(self):
        with self.path.open('x') as stream:
            while True:
                stream.write(json.dumps(self.sample(), allow_nan=False)+'\n')
                stream.flush()
                if self.stop_event.wait(self.protocol['sampling_interval_s']):
                    break

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10.)
        need(not self.thread.is_alive(), 'Resource sampler failed to finish')


def signal_owned(process, identity, signum):
    """Never signal a PID/PGID that no longer matches our live child leader."""
    if process.poll() is not None:
        return False
    if identity is None or R.process_identity(process.pid) != identity or identity['pgid'] != process.pid:
        raise R.Quarantined('Owned child identity changed; automatic signaling forbidden')
    try:
        os.killpg(process.pid, signum)
        return True
    except ProcessLookupError:
        return False


def finish_owned(process, identity, protocol):
    events = []
    for signum, grace in ((signal.SIGINT, protocol['cancel_sigint_grace_s']),
            (signal.SIGTERM, protocol['cancel_sigterm_grace_s']),
            (signal.SIGKILL, protocol['cancel_sigkill_grace_s'])):
        if process.poll() is not None:
            break
        sent = signal_owned(process, identity, signum)
        events.append(dict(signal=signal.Signals(signum).name, sent=sent, unix_s=time.time()))
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    need(process.poll() is not None, 'Owned process did not exit after bounded cancellation')
    return events


def admit(receipt, worker, group, protocol):
    lanes = worker['lanes']
    need(receipt.get('status') == 'partial_scaling' and receipt.get('complete') is False
        and receipt.get('label') is None and receipt.get('raw_label') is None
        and receipt.get('exclude_training') is True and receipt.get('mode') == group['mode']
        and receipt.get('lanes') == lanes and receipt.get('prefix_ticks') == PREFIX_TICKS
        and receipt.get('physical_clock_steps') == PREFIX_TICKS
        and receipt.get('actual_integrated_environment_steps') == PREFIX_TICKS*lanes
        and receipt.get('recorded_command_environment_steps') == PREFIX_TICKS*lanes,
        'Unlabeled complete 100-tick accounting required')
    need(receipt.get('returned_physical_steps') == PREFIX_TICKS
        and receipt.get('actual_simulation_step_counter') == PREFIX_TICKS
        and receipt.get('observations_completed_steps_per_lane') == [PREFIX_TICKS]*lanes
        and receipt.get('physical_steps_unknown') is False
        and receipt.get('inflight_step_unconfirmed') is False
        and receipt.get('physical_steps_lower_bound') == PREFIX_TICKS
        and receipt.get('physical_steps_upper_bound') == PREFIX_TICKS
        and receipt.get('integrated_environment_steps_lower_bound') == PREFIX_TICKS*lanes
        and receipt.get('integrated_environment_steps_upper_bound') == PREFIX_TICKS*lanes,
        'Native attempted counter, returned steps and every observed lane must agree')
    template = receipt['prepared_template']
    need(template['path'] == protocol['prepared_template']['path']
        and template['ready'] == protocol['prepared_template']['ready']
        and template['commands'] == protocol['prepared_template']['commands'],
        'Unchanged original g6 preparation and commands required')
    B.verify({'ready': template['ready'], 'commands': template['commands'],
        'group': receipt['group_receipt'], 'profile': receipt['profile']})
    need(Path(receipt['group_receipt']['path']) == Path(worker['output_dir'])/'GPU_VECTOR_EXECUTION.json'
        and Path(receipt['profile']['path']) == Path(worker['output_dir'])/'GPU_VECTOR_PROFILE.json',
        'Native receipts must belong to this fresh worker output')
    B.verify(B.normalized_bindings(receipt['source_sha256']))
    receipt_sources = B.normalized_bindings(receipt['source_sha256'])
    frozen_sources = {(s['path'], s['sha256']) for s in protocol['sources'].values()}
    need(all((s['path'], s['sha256']) in frozen_sources for s in receipt_sources.values()),
        'A worker source was not frozen before dispatch')
    native = B.read(receipt['group_receipt']['path'])
    profile = B.read(receipt['profile']['path'])
    need(native.get('status') == 'partial_profile' and native.get('complete') is False
        and native.get('physical_clock_steps') == PREFIX_TICKS
        and native.get('actual_integrated_environment_steps') == PREFIX_TICKS*lanes
        and native.get('recorded_command_environment_steps') == PREFIX_TICKS*lanes
        and native.get('clock', {}).get('lane_steps') == [PREFIX_TICKS]*lanes
        and len(native.get('jobs', [])) == lanes
        and all(j.get('completion_tick') == PREFIX_TICKS and j.get('label') is None
            and j.get('raw_label') is None for j in native['jobs']),
        'Original group prefix count or label differs')
    need(profile.get('status') == 'partial_profile' and profile.get('actual_physics_steps') == PREFIX_TICKS
        and profile.get('label') is None and profile.get('raw_label') is None
        and profile.get('exclude_training') is True, 'Original profile is incomplete or labeled')
    need(all(native.get(key, {}).get('status') == 'live_api_snapshot_pass'
        for key in ('initial_gpu_probe', 'first_tick_gpu_probe')),
        'Both original initial and first-tick contact/state probes must pass')
    for key in ('setup_wall_s', 'execution_wall_s', 'execution_prefix_started_unix_s',
            'execution_prefix_finished_unix_s'):
        need(type(receipt.get(key)) in (int, float) and math.isfinite(receipt[key])
            and receipt[key] >= 0., 'Finite nonnegative timing required: '+key)
    need(receipt['execution_prefix_finished_unix_s'] > receipt['execution_prefix_started_unix_s'],
        'A measured positive prefix interval required')
    need(receipt.get('device') == ('cuda:0' if group['mode'] == 'gpu' else 'cpu'),
        'Explicit requested physics backend required')
    need(receipt.get('worker_pid') is not None, 'Actual worker PID required')
    return receipt


def run_stage(worker, group, protocol, pool, sampler, stop):
    directory = Path(worker['stage_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    output = Path(worker['output_dir'])
    command = [sys.executable, '-B', protocol['worker_source']['path'],
        '--prepared-dir', worker['prepared_dir'], '--output-dir', str(output),
        '--mode', group['mode'], '--lanes', str(worker['lanes']),
        '--prefix-ticks', str(PREFIX_TICKS), '--protocol', protocol['_protocol_path']]
    state = dict(status='starting', complete=False, command=command, worker_id=worker['worker_id'],
        started_unix_s=time.time(), process=None, retry_count=0, signals=[],
        physics_steps_known=None, physical_steps_unknown=True)
    B.save(directory/'STARTED.json', state, exclusive=True)
    began = time.monotonic()
    process = None
    try:
        B.verify(protocol['inputs']); B.verify(protocol['sources'])
        with pool.stage(('kit',), directory):
            with (directory/'process.log').open('xb') as stream:
                process = pool.popen(command, cwd=B.ROOT, env=B._environment(protocol),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', process=R.process_identity(process.pid))
                sampler.add(state['process'])
                B.save(directory/'STAGE.json', state)
                deadline = time.monotonic()+protocol['stage_timeout_s']
                while process.poll() is None:
                    if stop.is_set() or time.monotonic() >= deadline:
                        state['cancellation_reason'] = 'group_stop' if stop.is_set() else 'stage_timeout'
                        state['signals'] = finish_owned(process, state['process'], protocol)
                        break
                    try:
                        process.wait(timeout=.2)
                    except subprocess.TimeoutExpired:
                        continue
                state['returncode'] = process.returncode
                need(state.get('cancellation_reason') is None and process.returncode == 0,
                    'Worker failed, was canceled or exceeded frozen timeout')
                receipt = admit(B.read(output/WORKER_RECEIPT), worker, group, protocol)
                need(receipt['worker_pid'] == process.pid, 'Receipt PID differs from its owned stage')
                state.update(status='partial_scaling', complete=True,
                    receipt=B.binding(output/WORKER_RECEIPT),
                    physics_steps_known=receipt['physical_clock_steps'], physical_steps_unknown=False,
                    physical_cost={k:receipt[k] for k in COST_FIELDS})
            B.verify(protocol['inputs']); B.verify(protocol['sources'])
    except BaseException:
        stop.set()
        state.update(status='infrastructure_failure', complete=False, error=traceback.format_exc())
        if process is not None and process.poll() is None:
            try:
                state['signals'] += finish_owned(process, state['process'], protocol)
            except BaseException:
                state['cleanup_error'] = traceback.format_exc()
        if (output/WORKER_RECEIPT).is_file():
            state['partial_receipt'] = B.binding(output/WORKER_RECEIPT)
            try:
                partial = B.read(output/WORKER_RECEIPT)
                state['physics_steps_known'] = partial.get('physical_clock_steps')
                state['physical_steps_unknown'] = partial.get('physical_steps_unknown', True)
                state['physical_cost'] = {k:partial.get(k) for k in COST_FIELDS}
                state['partial_count_scope'] = ('Saved completed-step counter only; an interrupted '
                    'simulate/fetch or observation may have an unconfirmed in-flight step')
            except Exception:
                pass
    state.update(finished_unix_s=time.time(), wall_s=time.monotonic()-began)
    if process is not None:
        state['returncode'] = process.poll()
    B.save(directory/'STAGE.json', state)
    return dict(worker_id=worker['worker_id'], status=state['status'],
        stage=B.binding(directory/'STAGE.json'), receipt=state.get('receipt'),
        partial_receipt=state.get('partial_receipt'), process=state.get('process'),
        returncode=state.get('returncode'), physics_steps_known=state['physics_steps_known'],
        physical_steps_unknown=state['physical_steps_unknown'], physical_cost=state.get('physical_cost'))


def interval_stats(intervals):
    need(bool(intervals) and all(math.isfinite(a) and math.isfinite(b) and b > a
        for a, b in intervals), 'Positive finite intervals required')
    events = sorted([(a, 1) for a, _ in intervals]+[(b, -1) for _, b in intervals])
    live = peak = 0
    union = 0.
    previous = events[0][0]
    for point, delta in events:
        if live:
            union += point-previous
        live += delta
        peak = max(peak, live)
        previous = point
    return dict(span_s=max(b for _, b in intervals)-min(a for a, _ in intervals),
        union_s=union, peak_concurrency=peak,
        all_worker_overlap_s=max(0., min(b for _, b in intervals)-max(a for a, _ in intervals)),
        evidence_scope='Measured execution-prefix intervals; exact simultaneous Kit activity also has host samples')


def run_group(root, group, protocol, pool, sampler):
    began = time.monotonic()
    start_unix = time.time()
    stop = threading.Event()
    workers = []
    error = None
    with ThreadPoolExecutor(max_workers=len(group['workers'])) as executor:
        futures = [executor.submit(run_stage, w, group, protocol, pool, sampler, stop)
            for w in group['workers']]
        try:
            for future in as_completed(futures):
                row = future.result()
                workers.append(row)
                if row['status'] != 'partial_scaling':
                    stop.set()
        except BaseException:
            stop.set()
            error = traceback.format_exc()
            workers = [f.result() for f in futures if not f.cancelled()]
    workers.sort(key=lambda w: w['worker_id'])
    complete = error is None and len(workers) == len(group['workers']) and all(
        w['status'] == 'partial_scaling' for w in workers)
    result = dict(name=group['name'], mode=group['mode'], count=group['count'],
        status='partial_scaling_complete' if complete else 'stopped_after_failure',
        started_unix_s=start_unix, finished_unix_s=time.time(),
        wall_s=time.monotonic()-began, workers=workers, error=error,
        exclude_training=True, label=None, raw_label=None)
    if complete:
        receipts = [B.read(w['receipt']['path']) for w in workers]
        result['prefix_interval_stats'] = interval_stats([(r['execution_prefix_started_unix_s'],
            r['execution_prefix_finished_unix_s']) for r in receipts])
        result['setup_sum_s'] = sum(r['setup_wall_s'] for r in receipts)
        result['execution_wall_sum_s'] = sum(r['execution_wall_s'] for r in receipts)
        result['environment_steps'] = PREFIX_TICKS*group['count']
        result['prefix_environment_steps_per_s'] = result['environment_steps']/result['prefix_interval_stats']['span_s']
        result['outer_environment_steps_per_s'] = result['environment_steps']/result['wall_s']
    B.save(Path(root)/group['name']/'GROUP_RESULTS.json', result, exclusive=True)
    return result


def run(output_dir):
    root = Path(output_dir).resolve()
    protocol = verify(root)
    need(not (root/'RUN_STARTED.json').exists(), 'A started scaling control is never resumed or retried')
    need(gpu_identity(protocol['inputs']['nvidia_smi']['path'], protocol['gpu_uuid']) ==
        protocol['gpu_identity'], 'Frozen GPU0 identity changed')
    start = time.monotonic()
    rows = []
    error = None
    with R.lease(root/'DRIVER.lock'), R.lease(G.LOCK) as fd:
        with R.active_run(root, G.REGISTRY, B.binding(root/PROTOCOL)):
            need(not (root/'RUN_STARTED.json').exists(), 'Started scaling control cannot relaunch')
            B.save(root/'RUN_STARTED.json', dict(protocol=B.binding(root/PROTOCOL),
                process=R.process_identity(os.getpid()), unix_s=time.time()), exclusive=True)
            pool = ScalingPool(root/'resource_slots', fd, B.binding(root/PROTOCOL))
            sampler = ResourceSampler(root, protocol)
            runtime_protocol = dict(protocol, _protocol_path=str(root/PROTOCOL))
            sampler.start()
            try:
                for group in protocol['groups']:
                    B.verify(protocol['inputs']); B.verify(protocol['sources'])
                    row = run_group(root, group, runtime_protocol, pool, sampler)
                    rows.append(row)
                    if row['status'] != 'partial_scaling_complete':
                        break
                B.verify(protocol['inputs']); B.verify(protocol['sources'])
            except BaseException:
                error = traceback.format_exc()
            finally:
                sampler.close()
            result = dict(schema=SCHEMA, status='complete' if len(rows) == 6 and error is None
                and all(g['status'] == 'partial_scaling_complete' for g in rows) else 'stopped_after_failure',
                protocol=B.binding(root/PROTOCOL), groups=rows, error=error,
                wall_s=time.monotonic()-start, resource_samples=B.binding(sampler.path),
                exclude_training=True, formal_training_eligible=False, diagnostic_only=True,
                label=None, raw_label=None, new_training_labels=0, training_updates=0,
                retries=0, full_physical_validation=False,
                scope='100-tick execution-only prefixes with full original logging; no complete grasp or collection ETA')
            B.save(root/RESULTS, result, exclusive=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output_dir) if args.action == 'freeze' else run(args.output_dir)
    print(json.dumps(result, indent=2))
    if args.action == 'run' and result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
