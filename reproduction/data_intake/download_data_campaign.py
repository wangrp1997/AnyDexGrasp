#!/usr/bin/env python3
"""Durably download an explicit official artifact manifest with bounded workers.

Uses the current process proxy configuration only. Raw assets remain in public;
the JSON state is updated atomically and counts partial bytes separately.
"""
import argparse
import concurrent.futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

_spec = importlib.util.spec_from_file_location(
    'public_artifact_fetch', Path(__file__).with_name('fetch_public_file.py'))
_fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fetch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--attempts', type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    items = manifest['artifacts']
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lockfile = args.state.with_suffix(args.state.suffix + '.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = threading.Lock()
    status = [dict(**x, status='pending', attempts=0) for x in items]
    started = time.time()
    done = threading.Event()

    def save(final=False):
        with lock:
            for item in status:
                p = Path(item['destination'])
                partial = p.with_name(p.name + '.partial')
                item['published_bytes'] = p.stat().st_size if p.exists() else 0
                item['partial_bytes'] = partial.stat().st_size if partial.exists() else 0
            value = dict(pid=os.getpid(), started_unix=started,
                         updated_unix=time.time(), manifest=str(args.manifest),
                         finished=final,
                         expected_total_bytes=sum(x['bytes'] for x in items),
                         complete_files=sum(x['status']=='complete' for x in status),
                         published_bytes=sum(x['published_bytes'] for x in status),
                         partial_bytes=sum(x['partial_bytes'] for x in status),
                         artifacts=[dict(x) for x in status])
            temporary = args.state.with_name(args.state.name + '.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(args.state)

    def heartbeat():
        while not done.wait(5):
            save()

    def worker(index):
        item = status[index]
        url = 'https://drive.usercontent.google.com/download?' + urlencode(
            dict(id=item['id'], export='download', confirm='t'))
        for attempt in range(1, args.attempts + 1):
            with lock:
                item.update(status='downloading_or_content_validation', attempts=attempt)
            save()
            try:
                receipt = _fetch.fetch(url, item['destination'], item['bytes'])
                with lock:
                    item.update(status='complete', sha256=receipt['sha256'],
                                format=receipt.get('format'))
                print(json.dumps(dict(file=item['destination'], status='complete',
                                      bytes=item['bytes'], sha256=receipt['sha256'])), flush=True)
                save()
                return
            except Exception as e:
                with lock:
                    item.update(status='retry_pending' if attempt < args.attempts else 'error',
                                error_type=type(e).__name__, error=str(e)[:800])
                print(json.dumps(dict(file=item['destination'], attempt=attempt,
                                      status=item['status'], error_type=type(e).__name__)), flush=True)
                save()
                if attempt < args.attempts:
                    time.sleep(2)

    save()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, range(len(items))))
    finally:
        done.set()
        thread.join()
        save(final=True)


if __name__ == '__main__':
    main()
