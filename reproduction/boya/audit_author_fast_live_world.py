"""Original full live-world audit with an independently compiled ray loop."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import time
import audit_author_live_world as A
import author_fast_geometry_checks as F

B=A.B


@contextmanager
def ray_loop():
    original=A.capture_auditor;timing=dict(calls=0,wall_s=0.,triangle_tests=0)
    def factory():
        fn=original()
        def measured(triangles,origin,direction):
            start=time.perf_counter()
            try:return F.first_depth(triangles,origin,direction)
            finally:
                timing['calls']+=1;timing['triangle_tests']+=len(triangles);timing['wall_s']+=time.perf_counter()-start
        fn.__globals__['first_depth']=measured;return fn
    A.capture_auditor=factory
    try:yield timing
    finally:A.capture_auditor=original


def capture(path,protocol):
    with ray_loop()as timing:result=A.capture(path,protocol)
    if timing['calls']!=result['independent_full_source_rays']:raise ValueError('Full independent ray coverage missing')
    result.update(full_source_ray_implementation=B.binding(F.__file__),ray_timing=timing,
        full_original_contact_and_geometry_checks_retained=True,fast_auditor=B.binding(__file__))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['capture','runtime'])
    p.add_argument('--capture-dir',type=Path);p.add_argument('--protocol',type=Path);p.add_argument('--prepared-dir',type=Path)
    p.add_argument('--split-manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='capture':F.precompile();result=capture(a.capture_dir,a.protocol)
    else:result=A.runtime(a.prepared_dir,a.split_manifest)
    B.save(a.output,result,exclusive=True);print({k:result.get(k)for k in ('status','wall_s','ray_timing')})
