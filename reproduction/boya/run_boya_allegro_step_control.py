"""Opt-in published-Allegro position staircase on the frozen Boya diagnostic.

Only the close-phase motor targets change: 10 equal increments, 100 ms each.
The original Boya open and close endpoints, 1 s closure, 200 ms hold and every
desired wrist pose are retained. This is a public-code timing control, NOT an
Allegro hardware/torque model or a learned contact-feedback controller.

The author reads measured joints at closure onset; this precomputed control
uses the frozen opening endpoint. That difference is explicit and unchanged
from the baseline. Calling this CLI performs NEW physics in job.output_dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_boya_floating_diagnostic as baseline

BASE_COMMAND_SEQUENCE = baseline.command_sequence
CLOSE_STEPS = 250
INCREMENTS = 10
STEPS_PER_INCREMENT = 25
DT_S = .004


def command_sequence(inputs):
    sequence = BASE_COMMAND_SEQUENCE(inputs)
    sequence = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in sequence.items()}
    close = np.flatnonzero(sequence['phase'] == 'close')
    if (len(close) != CLOSE_STEPS or np.any(np.diff(close) != 1)
            or baseline.PHASE_LENGTHS != [('prehold', 800), ('approach', 500),
                                         ('close', 250), ('hold', 50), ('lift', 500)]):
        raise ValueError('Only the frozen 250-step, 4 ms diagnostic is supported')
    progress = np.repeat(np.arange(1, INCREMENTS + 1, dtype=float) / INCREMENTS,
                         STEPS_PER_INCREMENT)
    sequence['motor_target'][close] = np.asarray([
        inputs['closure'].command(float(p), float(p)) for p in progress])
    return sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job.get('closure_control') != 'published_allegro_ten_position_steps_v1':
        raise ValueError('Explicit opt-in closure_control required in a NEW job')
    if Path(job['output_dir']).exists():
        raise ValueError('New output directory required; baseline output is immutable')
    # Import AppLauncher only after parsing, like the existing standalone entry.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device='cpu', visualizer=None,
                           kit_args='--/log/file=/dev/null')
    code = 1
    try:
        baseline.command_sequence = command_sequence
        result = baseline.run(job)
        result['closure_control'] = dict(
            name=job['closure_control'], increments=INCREMENTS,
            seconds_per_increment=.1, total_close_seconds=1., hold_seconds=.2,
            initial_state='Frozen opening target; author Allegro reads measured opening joints',
            force_or_tactile_feedback=False,
            source='ur_toolbox/ur_toolbox/robot/Allegro/Allegro.py:61-93')
        baseline.save(Path(job['output_dir']) / 'evaluation.json', result)
        print(json.dumps(dict(object_id=result['object_id'],
                              task_success=result['task_success'],
                              steps=result['total_physics_steps'])), flush=True)
        code = 0
    finally:
        baseline.command_sequence = BASE_COMMAND_SEQUENCE
        launcher.app.close(exit_code=code)


if __name__ == '__main__':
    main()
