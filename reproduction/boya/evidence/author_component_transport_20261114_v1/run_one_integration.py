"""One pure continuation, with a transparent count of original IK calls."""
import functools
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'reproduction/boya'))
import author_pipeline as pipeline


def main():
    gesture = int(sys.argv[1])
    protocol = pipeline.transport.read(OUT/'COMPONENT_INTEGRATION_PROTOCOL.json')
    job = next(row for row in protocol['jobs'] if row['gesture_id'] == gesture)
    pipeline._verify(protocol['sources']); pipeline._verify(protocol['inputs'])
    pipeline.transport._backend_path()
    from canondex.grasp.admm import arm_only_ik
    original = arm_only_ik.arm_only_palm_ik
    counts = dict(schema='anydex_boya_author_continuation_planning_counts_v1', gesture_id=gesture,
        original_ik_source=pipeline._binding(Path(arm_only_ik.__file__)),
        original_ik_calls_started=0, original_ik_calls_returned=0, original_ik_calls_raised=0,
        original_ik_wall_s=0., max_nfev_values=[], physics_steps=0,
        instrumentation='Forward identical args/kwargs once to the original callable; preserve its result or exception')
    began = time.monotonic()

    @functools.wraps(original)
    def counted(*args, **kwargs):
        counts['original_ik_calls_started'] += 1
        value = kwargs.get('max_nfev')
        if value not in counts['max_nfev_values']: counts['max_nfev_values'].append(value)
        started = time.monotonic()
        try:
            result = original(*args, **kwargs)
            counts['original_ik_calls_returned'] += 1
            return result
        except BaseException:
            counts['original_ik_calls_raised'] += 1
            raise
        finally:
            counts['original_ik_wall_s'] += time.monotonic()-started

    arm_only_ik.arm_only_palm_ik = counted
    try:
        value = pipeline.prepare_from_frozen_candidate_plan(job['prepared_dir'], job['routed_plan_dir'],
            job['output'], release_spec_path=protocol['inputs']['fixed_release']['path'],
            archive_sources=pipeline.transport.read(protocol['inputs']['scene_archive_sources']['path']))
        print(json.dumps(value, indent=2))
    finally:
        arm_only_ik.arm_only_palm_ik = original
        counts['child_wall_s'] = time.monotonic()-began
        with (OUT/f'PLANNING_COUNTS_g{gesture}.json').open('x') as stream:
            json.dump(counts, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
