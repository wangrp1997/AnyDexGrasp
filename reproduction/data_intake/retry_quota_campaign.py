#!/usr/bin/env python3
"""Retry an explicit public artifact manifest at low frequency, without cron.

The first retry is delayed, every round uses a new receipt, and each file gets
one attempt per round. Only this process inherits the selected nros proxy.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--interval-seconds', type=int, default=21600)
    parser.add_argument('--rounds', type=int, default=4, choices=range(1, 5))
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 3))
    args = parser.parse_args()
    if args.interval_seconds < 21600:
        parser.error('Quota retries must be at least six hours apart')
    original = json.loads(args.manifest.read_text())
    if not original.get('artifacts'):
        parser.error('An explicit nonempty artifact manifest is required')
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lockfile = args.state.with_suffix(args.state.suffix + '.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.state.exists():
        parser.error('Use a new state path; retain prior retry receipts')
    stop = threading.Event()
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.set())
    state = dict(pid=os.getpid(), started_unix=time.time(),
                 source_manifest=str(args.manifest), interval_seconds=args.interval_seconds,
                 max_rounds=args.rounds, per_file_attempts_per_round=1,
                 initial_retry_delayed=True, rounds=[], status='waiting_for_quota_retry',
                 complete_files=[], remaining_files=[x['name'] for x in original['artifacts']],
                 next_retry_unix=time.time()+args.interval_seconds,
                 transfer_active=False, finished=False)
    folder = args.state.with_suffix('')
    folder.mkdir(parents=True, exist_ok=True)

    def save():
        state['updated_unix'] = time.time()
        temporary = args.state.with_name(args.state.name + '.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(args.state)

    remaining = original['artifacts']
    save()
    try:
        for round_number in range(1, args.rounds+1):
            while not stop.is_set() and time.time() < state['next_retry_unix']:
                stop.wait(min(60, max(0, state['next_retry_unix']-time.time())))
                save()
            if stop.is_set():
                state['status'] = 'stopped_by_signal'
                break
            round_manifest = folder / ('round_%02d_manifest.json' % round_number)
            round_state = folder / ('round_%02d_state.json' % round_number)
            round_manifest.write_text(json.dumps(dict(artifacts=remaining), indent=2)+'\n')
            state.update(status='retry_round_active', transfer_active=True,
                         active_round=round_number, next_retry_unix=None)
            save()
            process = subprocess.Popen([
                sys.executable, '-B', '-u', str(Path(__file__).with_name('download_data_campaign.py')),
                '--manifest', str(round_manifest), '--state', str(round_state),
                '--workers', str(args.workers), '--attempts', '1'])
            while process.poll() is None:
                if stop.wait(5):
                    process.terminate()
                    break
            code = process.wait()
            round_result = json.loads(round_state.read_text()) if round_state.exists() else {}
            finished = {x['id'] for x in round_result.get('artifacts', [])
                        if x.get('status') == 'complete'}
            state['complete_files'].extend(x['name'] for x in remaining if x['id'] in finished)
            remaining = [x for x in remaining if x['id'] not in finished]
            state['rounds'].append(dict(round=round_number, state=str(round_state),
                                        subprocess_exit_code=code, finished_unix=time.time()))
            state.update(remaining_files=[x['name'] for x in remaining], transfer_active=False)
            if not remaining:
                state['status'] = 'all_artifacts_downloaded_and_validated'
                break
            state.update(status='waiting_for_quota_retry',
                         next_retry_unix=time.time()+args.interval_seconds)
            save()
        else:
            state['status'] = 'retry_limit_reached_remaining_external_failures'
    finally:
        state.update(finished=True, transfer_active=False, next_retry_unix=None)
        save()


if __name__ == '__main__':
    main()
