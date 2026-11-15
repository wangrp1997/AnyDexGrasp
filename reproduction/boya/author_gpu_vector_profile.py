"""Opt-in bounded wall-time instrumentation of the frozen GPU vector entry.

Uses the native barrier's existing step_limit; full state/contact/JSONL evidence
and every original physical gate remain enabled. Prefixes never have labels.
No simulator imports or launches occur merely by importing this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import author_gpu_vector_execution as V

SCHEMA = 'anydex_boya_author_gpu_vector_profile_v1'
VECTOR_SHA256 = '263cf9fd7b146da66a31bd43c70d43bac01361c9904b627cc03bbd357d092e0d'
MAX_PREFIX_TICKS = 200


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Timings:
    """Nested inclusive and exclusive host-wall timings; no CUDA synchronize."""
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.phase = 'initialization'
        self.stack = []
        self.values = {}

    @contextmanager
    def measure(self, name):
        entry = [self.clock(), 0.]
        phase = self.phase
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock()-entry[0]
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            value = self.values.setdefault(phase, {}).setdefault(name,
                dict(calls=0, inclusive_wall_s=0., exclusive_wall_s=0.))
            value['calls'] += 1
            value['inclusive_wall_s'] += elapsed
            value['exclusive_wall_s'] += max(0., elapsed-entry[1])

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured


class Patches:
    """Only this single process is adapted; always restore original objects."""
    def __init__(self):
        self.undo = []

    def set(self, target, name, value):
        self.undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def wrap(self, target, name, timings, label):
        self.set(target, name, timings.wrap(label, getattr(target, name)))

    def restore(self):
        for target, name, value in reversed(self.undo):
            setattr(target, name, value)
        self.undo.clear()


class InterfaceTimer:
    """Forward native calls exactly once, without an added synchronization."""
    def __init__(self, interface, timings):
        self.interface, self.timings = interface, timings

    def __getattr__(self, name):
        value = getattr(self.interface, name)
        if name in ('simulate', 'fetch_results'):
            return self.timings.wrap('physx.'+name, value)
        return value


def prefix_clock(streams, step, *, on_step, prefix_ticks, timings):
    """Call the original native barrier with its existing bounded prefix API."""
    from canondex.grasp.arm_parallel import advance_together
    ready = [next(stream) for stream in streams]
    V._need(all(e['kind'] == 'ready' for e in ready), 'Initialized original lanes required')
    lengths = [int(e['steps']) for e in ready]
    V._need(type(prefix_ticks) is int and 1 <= prefix_ticks <= MAX_PREFIX_TICKS
            and all(prefix_ticks < n for n in lengths), 'Prefix must be positive, bounded and incomplete for every lane')
    def replay(event, stream):
        yield dict(event, steps=max(lengths))
        yield from stream
    def measured_step(sim):
        with timings.measure('barrier.step_callback'):
            return step(sim)
    def measured_after(tick, counts):
        with timings.measure('barrier.after_tick'):
            if on_step is not None:
                return on_step(tick, counts)
    timings.phase = 'execution_prefix'
    began = timings.clock()
    try:
        result = advance_together([replay(e, s) for e, s in zip(ready, streams)],
            measured_step, on_step=measured_after, step_limit=prefix_ticks)
    finally:
        elapsed = timings.clock()-began
        timings.phase = 'finalization'
    result.update(original_command_steps=lengths, scheduler_barrier_upper_bound=max(lengths),
        barrier_metadata_only=True, commands_added=0,
        completed_lane_tail_integrated_steps=[0]*len(streams),
        actual_integrated_environment_steps=result['physics_ticks']*len(streams),
        prefix_limited=True, requested_prefix_ticks=prefix_ticks,
        prefix_clock_including_generator_close_wall_s=elapsed,
        next_unintegrated_command_may_have_been_prepared=True)
    V._need(not result['all_streams_complete'], 'A prefix must not become a completed trial')
    return result


def install_live(patches, timings, storage, lift):
    """Called only after the unchanged AppLauncher has started."""
    import warp
    import omni.physx
    from isaaclab.sim import SimulationContext
    import torch
    import numpy as np
    from isaaclab_physx.assets import Articulation, RigidObject
    from canondex.scene import runtime, contact_readback, execute_motion, object_contacts
    patches.wrap(SimulationContext, 'step', timings, 'sim.step')
    getter = omni.physx.get_physx_simulation_interface
    patches.set(omni.physx, 'get_physx_simulation_interface',
        lambda *a, **kw: InterfaceTimer(getter(*a, **kw), timings))
    patches.wrap(warp.array, 'numpy', timings, 'warp.numpy_host_copy')
    patches.wrap(torch.Tensor, 'cpu', timings, 'torch.cpu_host_copy')
    patches.wrap(runtime, 'snapshot', timings, 'robot.snapshot')
    patches.wrap(contact_readback, 'contact_data_to_host', timings, 'contact.filtered6_readback')
    patches.wrap(execute_motion.RobotContactReports, 'read', timings, 'robot.contact_decode_and_read')
    patches.wrap(object_contacts.ObjectContactReports, 'read', timings, 'primary.contact_decode_and_read')
    patches.wrap(object_contacts, 'decode_object_contacts', timings, 'primary.contact_decode')
    patches.wrap(lift.common, 'decode_contacts', timings, 'all_body.contact_decode')
    patches.wrap(lift.common, 'object_robot_loads', timings, 'all_body.contact_loads')
    patches.wrap(lift, 'source_metrics', timings, 'all_body.source_mesh_metrics')
    patches.wrap(Articulation, 'update', timings, 'robot.update')
    patches.wrap(RigidObject, 'update', timings, 'object.update')
    patches.wrap(Articulation, 'write_data_to_sim', timings, 'robot.control_write')
    patches.wrap(RigidObject, 'write_data_to_sim', timings, 'object.force_write')
    patches.wrap(storage.ContactJournal, 'append', timings, 'journal.encode_gzip_append')
    patches.wrap(storage.ContactJournal, 'close', timings, 'journal.close')
    patches.wrap(json, 'dumps', timings, 'json.encode')
    patches.wrap(gzip.GzipFile, 'write', timings, 'gzip.write')
    patches.wrap(np, 'savez_compressed', timings, 'artifact.savez_compressed')


def profile_receipt(value, prefix_ticks):
    """Make the partial diagnostic explicit without modifying scientific rows."""
    clock = value.get('clock', {})
    if clock.get('prefix_limited'):
        V._need(value['physical_clock_steps'] == prefix_ticks
                and value['complete'] is False
                and all(j.get('raw_label') is None and j.get('label') is None for j in value['jobs']),
                'Incomplete profiling cannot contain a label or completed trial')
        value = dict(value, status='partial_profile', purpose='performance_diagnostic_prefix',
            training_eligible=False, prefix_limited=True, requested_prefix_ticks=prefix_ticks)
    return value


def run(prepared_dirs, output_dir, *, env_roots, origins_m, prefix_ticks=20, protocol_path=None):
    """Launch only when called explicitly by the parent's bounded dispatcher."""
    V._need(type(prefix_ticks) is int and 1 <= prefix_ticks <= MAX_PREFIX_TICKS,
            'Explicit prefix ticks must be 1..200')
    V._need(sha(V.__file__) == VECTOR_SHA256, 'Frozen GPU vector entry changed')
    out = Path(output_dir).resolve()
    V._need(not out.exists(), 'Fresh profiling output required')
    jobs = V.load_jobs(prepared_dirs)
    V._need(all(prefix_ticks < j['command_steps'] for j in jobs), 'Every lane must remain incomplete')
    storage, lift, _, _ = V._modules()
    timings, patches = Timings(), Patches()
    sources = V.source_bindings()
    sources[str(Path(__file__).resolve())] = sha(__file__)
    bindings = dict(sources)
    state = dict(schema=SCHEMA, status='starting', complete=False, label=None,
        raw_label=None, formal_training_eligible=False, exclude_training=True,
        purpose='performance_diagnostic_prefix', prefix_ticks=prefix_ticks,
        source_sha256=sources, actual_physics_steps=None,
        timing_semantics='Host wall time without added CUDA synchronization; inclusive buckets overlap; exclusive buckets are disjoint among instrumented scopes; GPU async work may be charged to the later blocking read',
        control_semantics='Frozen original commands and all original gates/full pointwise journals; native step_limit ends an incomplete prefix; no success or failure label',
        next_unintegrated_command_may_have_been_prepared=True)
    original_adapted = V.adapted_methods
    original_args = lift.args_for_plan
    original_save = storage.save
    started = time.perf_counter()
    finalized = False

    def final_profile(value):
        nonlocal finalized
        state.update(status=('partial_profile' if value.get('prefix_limited') else 'infrastructure_failure'),
            actual_physics_steps=value.get('physical_clock_steps', 0),
            actual_integrated_environment_steps=value.get('actual_integrated_environment_steps', 0),
            wall_before_close_s=time.perf_counter()-started, timing=timings.values,
            group_receipt=dict(path=str(out/'GPU_VECTOR_EXECUTION.json'),sha256=storage.sha(out/'GPU_VECTOR_EXECUTION.json')),
            error=value.get('error'), native_execution_profiles=[])
        for job in value.get('jobs', []):
            evaluation = Path((job.get('evaluation') or {}).get('path', out/'missing')).with_name('evaluation.json')
            if evaluation.is_file():
                native = json.loads(evaluation.read_text())
                state['native_execution_profiles'].append(dict(path=str(evaluation),
                    sha256=storage.sha(evaluation),profile=native.get('execution_profile')))
        V.verify_sources(bindings)
        original_save(out/'GPU_VECTOR_PROFILE.json', state)
        finalized = True

    def save(path, value):
        path = Path(path)
        if path == out/'GPU_VECTOR_EXECUTION.json':
            value = profile_receipt(value, prefix_ticks)
            with timings.measure('artifact.save_json'):
                result = original_save(path, value)
            final_profile(value)  # The real App.close may call os._exit.
            return result
        with timings.measure('artifact.save_json'):
            return original_save(path, value)

    def adapted(module):
        install_live(patches, timings, storage, lift)
        methods, edits = original_adapted(module)
        methods['after_step'] = timings.wrap('all_body.after_step', methods['after_step'])
        return methods, edits

    def args_for_plan(*a, **kw):
        args = original_args(*a, **kw)
        args.profile_execution = True  # Existing native timing-only option.
        return args

    patches.set(V, 'source_bindings', lambda: dict(bindings))
    patches.set(V, 'adapted_methods', adapted)
    patches.set(V, 'shared_clock', lambda streams, step, on_step=None:
        prefix_clock(streams, step, on_step=on_step, prefix_ticks=prefix_ticks, timings=timings))
    patches.set(lift, 'args_for_plan', args_for_plan)
    patches.set(storage, 'save', save)
    try:
        return V.run_group(prepared_dirs, out, env_roots=env_roots, origins_m=origins_m,
                           device='cuda:0', protocol_path=protocol_path)
    except BaseException:
        if not finalized:
            state.update(status='infrastructure_failure',error=traceback.format_exc(),
                wall_before_close_s=time.perf_counter()-started,timing=timings.values)
            if out.exists():
                original_save(out/'GPU_VECTOR_PROFILE.json', state)
        raise
    finally:
        patches.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--env-roots', nargs='+', required=True)
    parser.add_argument('--origins-json', required=True)
    parser.add_argument('--prefix-ticks', type=int, default=20)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    run(args.prepared_dirs,args.output_dir,env_roots=args.env_roots,
        origins_m=json.loads(args.origins_json),prefix_ticks=args.prefix_ticks,protocol_path=args.protocol)


if __name__ == '__main__':
    main()
