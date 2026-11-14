"""One bounded RRT HOME preparation, optionally followed by same-path retiming.

This opt-in dispatcher never retries search, changes its path, or executes a
simulator. A failed collision gate cannot be repaired by time parameterization.
The separately frozen RRT and retiming implementations remain unmodified.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import author_home_route as home
import author_joint_home_route as joint
import author_home_retime as retime
import author_transport as transport

CORE_PINS = {
    str(Path(home.__file__).resolve()): '1bf86feaccd7880128b54ace69b1289027d313e7f8a641bc44ef6aa00d0ff0d8',
    str(Path(joint.__file__).resolve()): 'cc390e0363696c5c65c368672a385c885f5ccb4ba450bdb1323d7766b2a0fcc6',
    str(Path(retime.__file__).resolve()): '0beb0b8aa0640d6ced6c9ed3a7f3ad500fd6f335ccac17eaa8699986507bf884',
}
RETIME_REASON = 'first_rrt_connection_failed_final_tick_or_derivative_gate'
POLICY = dict(schema='anydex_boya_author_joint_home_driver_policy_v1',
    max_rrt_preparations=1, max_retime_preparations=1,
    rrt=copy.deepcopy(joint.POLICY), retime=copy.deepcopy(retime.POLICY),
    retime_condition='Known first-connection rejection; SHA-bound full-scene geometry is clear; only finite trajectory derivative limits fail',
    forbidden='No search retry, alternative connection, path change, candidate change, IK, relaxed geometry or simulator',
    ik_calls=0, physics_steps=0, training_updates=0, labels=0)


def _merge(locks, other):
    for name, digest in other.items():
        name = str(Path(name).resolve())
        if name in locks and locks[name] != digest:
            raise ValueError('Conflicting frozen input SHA: '+name)
        locks[name] = digest


def _verify(locks):
    for name, digest in locks.items():
        if transport.sha(name) != digest:
            raise ValueError('Frozen driver dependency changed: '+name)


def _freeze_inputs(seed_plan_dir, scene_path, config_path, archive_sources):
    seed = transport.read(seed_plan_dir/'plan.json')
    scene = transport.validate_scene(scene_path, archive_sources)
    locks = {str(p): transport.sha(p) for p in (seed_plan_dir/'plan.json', scene_path, config_path)}
    for key, name in (('commands_sha256', 'commands.npz'), ('path_geometry_sha256', 'path_geometry.json'),
                      ('lift_arm_geometry_sha256', 'lift_arm_geometry.json')):
        _merge(locks, {str(seed_plan_dir/name): seed[key]})
    for mapping in (seed['source_sha256'], seed.get('extra_input_sha256', {}), scene.get('input_sha256', {})):
        _merge(locks, mapping)
    bindings = transport.scene_source_bindings(scene, archive_sources)
    for binding in bindings.values(): _merge(locks, {binding['path']: binding['sha256']})
    pool = Path(seed['arguments']['candidates']).resolve()
    generation = transport.read(pool/'generation.json')
    _merge(locks, {str(pool/'generation.json'): seed['generation_sha256'],
                   str(pool/'candidates.npz'): seed['candidates_sha256']})
    _merge(locks, generation['input_sha256'])
    _verify(locks)
    return locks, bindings


def _child_receipt(directory, returned):
    receipt = transport.read(directory/'HOME_ROUTE_PREPARATION.json')
    if transport.json_sha(receipt) != transport.json_sha(returned):
        raise ValueError('Returned and persisted child receipts differ')
    expected = {'ready': (True, True), 'null_geometry': (True, False),
                'infrastructure_failure': (False, False)}
    status = receipt.get('status')
    if status not in expected or (receipt.get('complete'), receipt.get('admitted')) != expected[status]:
        raise ValueError('Invalid child completion/admission contract')
    duration = receipt.get('wall_s')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise ValueError('Finite child preparation cost required')
    if status != 'infrastructure_failure':
        protocol_path = directory/'HOME_ROUTE_PROTOCOL.json'
        if transport.sha(protocol_path) != receipt['protocol_sha256']:
            raise ValueError('Child protocol binding changed')
        protocol = transport.read(protocol_path)
        _verify(protocol['input_sha256']); _verify(protocol['source_sha256'])
    return receipt


def retime_eligible(receipt, prepared_route_dir):
    """Validate the narrow timing-only branch; malformed evidence raises."""
    if (receipt['status'] != 'null_geometry' or receipt.get('reason') != RETIME_REASON
            or receipt.get('search', {}).get('connected') is not True):
        return False
    directory = Path(prepared_route_dir)
    for key, name in (('geometry_sha256', 'full_scene_geometry.json'),
                      ('connected_path_sha256', 'connected_path.npz'),
                      ('candidate_commands_sha256', 'candidate_commands.npz')):
        if transport.sha(directory/name) != receipt[key]:
            raise ValueError('First-connection evidence changed: '+name)
    geometry = transport.read(directory/'full_scene_geometry.json')
    if geometry.get('ok') is not True or geometry.get('violations') != []:
        return False
    derivative = receipt['derivative_audit']
    ratios = [derivative[k] for k in ('velocity_ratio', 'acceleration_ratio', 'jerk_ratio')]
    excess = derivative['joint_limit_excess_rad']
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in [*ratios, excess]):
        raise ValueError('Finite nonnegative timing failure evidence required')
    return (derivative.get('ok') is False and excess <= 1e-7
            and max(ratios) > 1.+1e-8)


def _ready_plan(receipt, directory, seed_sha256):
    native = directory/'native'
    if Path(receipt['plan_dir']).resolve() != native.resolve():
        raise ValueError('Child plan must remain inside its own output')
    if transport.sha(native/'plan.json') != receipt['plan_sha256']:
        raise ValueError('Final native plan binding changed')
    plan = transport.read(native/'plan.json')
    if (plan.get('admitted') is not True or plan.get('complete') is not True
            or plan.get('author_home_route', {}).get('original_seed_plan_sha256') != seed_sha256):
        raise ValueError('Ready plan must retain the original seed identity')
    for key, name in (('commands_sha256', 'commands.npz'), ('path_geometry_sha256', 'path_geometry.json')):
        if transport.sha(native/name) != plan[key]: raise ValueError('Final native artifact changed: '+name)
    if plan['path_geometry_sha256'] != receipt['geometry_sha256']:
        raise ValueError('Final native geometry differs from the admitted child report')
    geometry = transport.read(native/'path_geometry.json')
    if geometry.get('ok') is not True or geometry.get('violations') != []:
        raise ValueError('Ready native plan requires an explicitly clear geometry report')
    _verify(plan['source_sha256']); _verify(plan.get('extra_input_sha256', {}))
    return native


def prepare_home_route(seed_plan_dir, scene_path, output, *, config_path, archive_sources=None):
    """Return a final ready/null/infra receipt; never consume an existing run.

    ``plan_dir`` is populated only for an admitted final child plan. The top
    protocol/receipt bind the complete dispatch; child plans and receipts stay
    untouched and retain their own original provenance contracts.
    """
    seed_plan_dir, scene_path, output, config_path = (Path(p).resolve()
        for p in (seed_plan_dir, scene_path, output, config_path))
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    receipt = dict(schema='anydex_boya_author_joint_home_driver_preparation_v1',
        status='preparing', complete=False, admitted=False, plan_dir=None,
        operations=[], policy=copy.deepcopy(POLICY), ik_calls=0, physics_steps=0,
        grasp_attempts=0, labels=0, training_updates=0)
    receipt_path = output/'HOME_ROUTE_PREPARATION.json'
    transport.save(receipt_path, receipt)
    locks = {}; sources = {}; child_records = []

    def invoke(stage, callback):
        entry = dict(stage=stage, completed=False)
        receipt['operations'].append(entry)
        transport.save(receipt_path, receipt)  # Attempt is counted before entry.
        started = time.monotonic()
        try:
            result = callback()
            result = _child_receipt(output/stage, result)
            child_records.append((output/stage, result))
            entry.update(completed=True, status=result['status'], reported_wall_s=result['wall_s'],
                receipt_path=str(output/stage/'HOME_ROUTE_PREPARATION.json'),
                receipt_sha256=transport.sha(output/stage/'HOME_ROUTE_PREPARATION.json'))
            if stage == 'rrt':
                entry.update(search_executed='search' in result,
                    search_wall_s=result.get('search_wall_s', 0.), search=result.get('search'))
            return result
        finally:
            entry['call_wall_s'] = time.monotonic()-started
            transport.save(receipt_path, receipt)

    try:
        _verify(CORE_PINS)
        sources = {str(Path(__file__).resolve()): transport.sha(__file__),
                   str(Path(transport.__file__).resolve()): transport.sha(transport.__file__), **CORE_PINS}
        _merge(sources, {str(Path(p).resolve()): digest for p, digest in transport.PINS.items()})
        locks, bindings = _freeze_inputs(seed_plan_dir, scene_path, config_path, archive_sources)
        _verify(sources)
        protocol = dict(schema='anydex_boya_author_joint_home_driver_protocol_v1',
            policy=copy.deepcopy(POLICY), source_sha256=sources, input_sha256=locks,
            seed_plan_dir=str(seed_plan_dir), scene_path=str(scene_path), config_path=str(config_path),
            capture_source_bindings=bindings,
            archive_sources={str(Path(k).resolve()):str(Path(v).resolve()) for k,v in (archive_sources or {}).items()})
        protocol_path = output/'HOME_ROUTE_PROTOCOL.json'
        transport.save(protocol_path, protocol)  # Before any child preparation.
        shutil.copyfile(__file__, output/'AUTHOR_JOINT_HOME_DRIVER_SOURCE.py')
        receipt.update(protocol_path=str(protocol_path), protocol_sha256=transport.sha(protocol_path))
        transport.save(receipt_path, receipt)
        first = invoke('rrt', lambda: joint.prepare_home_route(seed_plan_dir, scene_path,
            output/'rrt', config_path=config_path, archive_sources=archive_sources))
        final, final_directory = first, output/'rrt'
        if retime_eligible(first, output/'rrt'):
            final = invoke('retime', lambda: retime.retime_home_route(output/'rrt', seed_plan_dir,
                scene_path, output/'retime', config_path=config_path, archive_sources=archive_sources))
            final_directory = output/'retime'
            if final.get('connected_path_sha256') and final['connected_path_sha256'] != first['connected_path_sha256']:
                raise ValueError('Retiming changed the frozen connected path')
            if final['status'] == 'ready' and transport.sha(output/'retime'/'connected_path.npz') != first['connected_path_sha256']:
                raise ValueError('Retiming path bytes changed')
        _verify(locks); _verify(sources)
        for directory, result in child_records: _child_receipt(directory, result)
        receipt.update(status=final['status'], complete=final['complete'], admitted=final['admitted'],
            final_stage=final_directory.name, reason=final.get('reason'), error=final.get('error'))
        if final['status'] == 'ready':
            native = _ready_plan(final, final_directory, locks[str(seed_plan_dir/'plan.json')])
            receipt.update(plan_dir=str(native), plan_sha256=transport.sha(native/'plan.json'))
    except BaseException as error:
        receipt.update(status='infrastructure_failure', complete=False, admitted=False,
            plan_dir=None, error=repr(error), traceback=traceback.format_exc())
    operations = receipt['operations']
    receipt['cost'] = dict(rrt_preparations=sum(e['stage']=='rrt' for e in operations),
        retime_preparations=sum(e['stage']=='retime' for e in operations),
        measured_child_calls_wall_s=sum(e['call_wall_s'] for e in operations),
        reported_child_wall_s=sum(e.get('reported_wall_s', 0.) for e in operations),
        reported_search_wall_s=sum(e.get('search_wall_s', 0.) for e in operations),
        actual_search_runs=(None if any(e['stage']=='rrt' and not e['completed'] for e in operations)
                            else sum(e.get('search_executed', False) for e in operations)),
        ik_calls=0, physics_steps=0)
    receipt['wall_s'] = time.monotonic()-began
    transport.save(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-plan-dir', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-dependency', action='append', default=[])
    args = parser.parse_args()
    result = prepare_home_route(args.seed_plan_dir, args.scene, args.output, config_path=args.config,
        archive_sources=dict(item.split('=', 1) for item in args.frozen_dependency))
    print(__import__('json').dumps(result, allow_nan=False))
    raise SystemExit(1 if result['status'] == 'infrastructure_failure' else 0)


if __name__ == '__main__': main()
