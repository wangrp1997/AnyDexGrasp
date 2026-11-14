"""Cross-process slots for the opt-in wave runner; no simulator imports.

Lock files are never replaced. Reservations are durable before spawning. A slot
is released only after its entire stage process group has disappeared. Inherited
file descriptions are closed, never explicitly unlocked.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

import author_collection_batch as B

LIMITS = {'observation': 1, 'kit': 3, 'prepare': 3, 'audit': 2}
ORDER = tuple(LIMITS)


class Quarantined(RuntimeError):
    pass


def process_identity(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
        tail = raw[raw.rfind(')')+2:].split()
        return {'pid': int(pid), 'pgid': int(tail[2]), 'start_ticks': int(tail[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_alive(pgid):
    try:
        os.killpg(int(pgid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def lease(path):
    """Close-only flock lease, safe to inherit through workers and stages."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Quarantined('GPU0 or driver lease is still held: '+str(path)) from error
        yield fd
    finally:
        os.close(fd)  # LOCK_UN would also unlock a still-live inherited stage.


def check_inherited_lease(fd, path, expected):
    actual = os.fstat(fd); target = Path(path).stat()
    identity = [actual.st_dev, actual.st_ino]
    if identity != [target.st_dev, target.st_ino] or identity != expected:
        raise ValueError('Inherited parent lease does not refer to the frozen GPU lock')
    info = Path('/proc/self/fdinfo', str(fd)).read_text()
    if not any('FLOCK' in line and 'WRITE' in line for line in info.splitlines() if line.startswith('lock:')):
        raise ValueError('Inherited file description does not own an exclusive flock')
    probe = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise ValueError('GPU lease has no independently observable exclusive holder')
    finally:
        os.close(probe)


def assert_quiescent(root):
    """The registry also protects other run directories from orphan grandchildren.

    Frozen prepare grandchildren need not inherit FDs. Their still-running PGID
    or unreleased durable reservation therefore blocks a fresh global lease user.
    Unknown launch windows require explicit review, never automatic recovery.
    """
    root = Path(root)
    if any((root/'resource_slots/quarantine').glob('*.json')):
        raise Quarantined('Prior run has unresolved quarantine: '+str(root))
    for path in (root/'resource_slots').glob('*/*.json'):
        state = B.read(path)
        if state.get('state') != 'released' or (state.get('pgid') and group_alive(state['pgid'])):
            raise Quarantined('Prior run has an unreleased stage reservation: '+str(path))
    for marker in (root/'children').glob('*/CHILD_STARTED.json'):
        state_path = marker.parent/'CHILD_PROCESS.json'
        if not state_path.exists():
            raise Quarantined('Prior worker launch PID is unknown: '+str(marker))
        identity = B.read(state_path)['process']
        if process_identity(identity['pid']) == identity or group_alive(identity['pgid']):
            raise Quarantined('Prior worker group is still alive: '+str(state_path))


@contextmanager
def active_run(root, registry, protocol_binding):
    """Call only under the global lease; no new run may bypass its predecessor."""
    root = Path(root).resolve(); registry = Path(registry)
    if registry.exists():
        previous = B.read(registry)
        if previous['state'] != 'released':
            B.verify({'previous_protocol': previous['protocol']})
            assert_quiescent(previous['run_dir'])
    state = dict(run_dir=str(root), protocol=protocol_binding, state='active',
        owner=process_identity(os.getpid()), started_unix_s=time.time())
    B.save(registry, state)
    try:
        yield
    finally:
        try:
            assert_quiescent(root)
        except BaseException as error:
            state.update(state='active_review_required', lifetime_error=repr(error))
        else:
            state.update(state='released', finished_unix_s=time.time())
        B.save(registry, state)


class ResourcePool:
    def __init__(self, root, lease_fd, protocol_binding, *, fault=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.lease_fd = lease_fd
        self.protocol_binding = protocol_binding
        self.local = threading.local()
        self.fault = fault or (lambda point: None)

    def halt(self, reason, **details):
        value = dict(reason=reason, unix_s=time.time(), **details)
        # Distinct immutable incidents avoid concurrent writers replacing evidence.
        B.save(self.root/'quarantine'/f'{uuid.uuid4().hex}.json', value, exclusive=True)

    def check_halt(self):
        if any((self.root/'quarantine').glob('*.json')):
            raise Quarantined('Unresolved resource/process lifetime; see '+str(self.root/'quarantine'))

    def _acquire(self, kind, reservation):
        while True:
            self.check_halt()
            for index in range(LIMITS[kind]):
                base = self.root/kind/str(index); base.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(base)+'.lock', os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd); continue
                state_path = Path(str(base)+'.json')
                try:
                    old = B.read(state_path) if state_path.exists() else None
                    if old is not None and old.get('state') != 'released':
                        self.halt('Unreleased durable reservation cannot be reused', slot=str(base), previous=old)
                        raise Quarantined('Stale or unknown slot reservation: '+str(base))
                    state = dict(reservation, resource=kind, slot=index, state='reserved', reserved_unix_s=time.time())
                    B.save(state_path, state)
                    return dict(fd=fd, path=state_path, state=state)
                except BaseException:
                    os.close(fd); raise
            time.sleep(.05)

    @contextmanager
    def stage(self, kinds, stage_dir):
        if getattr(self.local, 'context', None) is not None:
            raise ValueError('Nested resource contexts are forbidden')
        start = time.monotonic(); slots = []
        context = dict(slots=slots, process=None, spawn_attempted=False, queue_wait_s=None)
        reservation = dict(reservation_id=uuid.uuid4().hex, stage_dir=str(Path(stage_dir).resolve()),
            worker=process_identity(os.getpid()), parent_protocol=self.protocol_binding)
        self.local.context = context
        try:
            for kind in ORDER:
                if kind in kinds:
                    slots.append(self._acquire(kind, reservation))
            context['queue_wait_s'] = time.monotonic()-start
            B.save(Path(stage_dir)/'RESOURCE_WAIT.json', dict(queue_wait_s=context['queue_wait_s'],
                resources=list(kinds), reservation_id=reservation['reservation_id']), exclusive=True)
            self.fault('reservation_before_popen')
            yield context
        finally:
            process = context['process']
            # Even a completed STAGE receipt is insufficient if a grandchild lives.
            pgid = process.pid if process is not None else None
            if pgid is not None:
                deadline = time.monotonic()+2.
                while group_alive(pgid) and time.monotonic() < deadline:
                    time.sleep(.02)
            safe = process is not None and not group_alive(pgid)
            # No spawn means an unknown interrupted launch window, not a retry.
            if not safe:
                self.halt('Stage launch/lifetime is unresolved', stage_dir=str(stage_dir),
                    pgid=pgid, spawn_attempted=context['spawn_attempted'])
            for slot in slots:
                state = dict(slot['state'], state='released' if safe else 'quarantined',
                    finished_unix_s=time.time(), pgid=pgid)
                B.save(slot['path'], state)
                os.close(slot['fd'])
            self.local.context = None
            if not safe:
                raise Quarantined('Stage group did not verifiably exit: '+str(stage_dir))

    def popen(self, *args, **kwargs):
        context = getattr(self.local, 'context', None)
        if context is None:
            raise ValueError('Stage Popen requires its own thread-local resource context')
        self.check_halt()
        if not kwargs.get('start_new_session'):
            raise ValueError('The frozen stage must own a new process group')
        own = [slot['fd'] for slot in context['slots']]
        kwargs['pass_fds'] = tuple(sorted(set(kwargs.get('pass_fds', ())) | {self.lease_fd, *own}))
        context['spawn_attempted'] = True
        process = subprocess.Popen(*args, **kwargs)
        context['process'] = process
        try:
            self.fault('popen_before_pid_persist')
            identity = process_identity(process.pid)
            for slot in context['slots']:
                slot['state'].update(state='running', process=identity, pgid=process.pid)
                B.save(slot['path'], slot['state'])
        except BaseException:
            # B._stage has not received process yet; retain durable quarantine.
            self.halt('Spawned before PID persistence failed', pgid=process.pid,
                stage_dir=context['slots'][0]['state']['stage_dir'] if own else None)
            raise
        return process


class SubprocessProxy:
    """Do not mutate the real subprocess module shared by Python threads."""
    def __init__(self, pool): self.pool = pool
    def Popen(self, *args, **kwargs): return self.pool.popen(*args, **kwargs)
    def __getattr__(self, name): return getattr(subprocess, name)


def classify(command, protocol):
    paths = {str(Path(str(arg)).resolve()) for arg in command if str(arg).endswith('.py')}
    sources = protocol['sources']
    if str(B.HERE/'author_lift_pipeline.py') in paths and 'prepare' in command: return ('prepare',)
    if sources['capture_entry']['path'] in paths: return ('observation', 'kit')
    if str(B.HERE/'run_collection_representation.py') in paths: return ('observation',)
    if sources['execution_entry']['path'] in paths: return ('kit',)
    if sources['audit_entry']['path'] in paths: return ('audit',)
    if sources['scene_entry']['path'] in paths: return ()
    raise ValueError('Unclassified external command cannot bypass global slots')


@contextmanager
def install_worker_adapters(pool, legacy_path):
    old_lock, old_stage, old_subprocess = B._lock, B._stage, B.subprocess

    @contextmanager
    def child_lock(path):
        if Path(path).resolve() == Path(legacy_path).resolve():
            yield  # Only this duplicate lease is bypassed; parent FD is inherited.
        else:
            with old_lock(path): yield

    def stage(directory, command, expected, protocol):
        if (Path(directory)/'STAGE.json').exists():
            return old_stage(directory, command, expected, protocol)
        with pool.stage(classify(command, protocol), directory):
            return old_stage(directory, command, expected, protocol)

    B._lock, B._stage, B.subprocess = child_lock, stage, SubprocessProxy(pool)
    try: yield
    finally: B._lock, B._stage, B.subprocess = old_lock, old_stage, old_subprocess
