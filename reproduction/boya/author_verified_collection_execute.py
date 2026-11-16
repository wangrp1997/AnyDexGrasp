"""Serial collection entry with the already validated lossless journal.

Same PhysX executor and numeric rules; importing/checking starts no simulator.
This entry adds no candidates, retries, labels, training or physics settings.
"""
import argparse
from pathlib import Path
import author_verified_execution as VERIFIED


def execute(prepared):
    with VERIFIED.install():
        return VERIFIED.E.execute(Path(prepared))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.check_only:
        with VERIFIED.install():
            assert VERIFIED.E.ContactJournal is VERIFIED.J.ContactJournal
        print('Verified journal installation OK; Kit/physics/labels/training=0')
    elif args.prepared_dir is not None:
        execute(args.prepared_dir)
    else:
        p.error('--prepared-dir required to execute')
