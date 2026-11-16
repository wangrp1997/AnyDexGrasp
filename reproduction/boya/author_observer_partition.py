"""Direct timers around physics, observation, metrics and contact serialization.

The native generator's inclusive yield interval is retained in historical
reports. This separate measurement avoids attributing its caller to PhysX.
"""
from contextlib import contextmanager
import time
import author_fast_geometry_checks as F


@contextmanager
def installed(sim,lift,journal):
    before=dict(step=sim.step,after=lift.MultiObjectLiftRuntime.after_step,
        metrics=lift.source_metrics,append=journal.ContactJournal.append)
    result=dict(sim_step_calls=0,sim_step_s=0.,observer_calls=0,observer_s=0.,
        full_vertex_metric_calls=0,full_vertex_metric_s=0.,contact_json_append_calls=0,contact_json_append_s=0.,phases={})
    def step(*args,**kwargs):
        begin=time.perf_counter()
        try:return before['step'](*args,**kwargs)
        finally:result['sim_step_calls']+=1;result['sim_step_s']+=time.perf_counter()-begin
    def after(runtime,phase):
        begin=time.perf_counter()
        try:return before['after'](runtime,phase)
        finally:
            elapsed=time.perf_counter()-begin;result['observer_calls']+=1;result['observer_s']+=elapsed
            r=result['phases'].setdefault(str(phase),dict(steps=0,observer_s=0.));r['steps']+=1;r['observer_s']+=elapsed
    def metrics(*args,**kwargs):
        begin=time.perf_counter()
        try:return F.source_metrics(*args,**kwargs)
        finally:result['full_vertex_metric_calls']+=1;result['full_vertex_metric_s']+=time.perf_counter()-begin
    def append(instance,row):
        begin=time.perf_counter()
        try:return before['append'](instance,row)
        finally:result['contact_json_append_calls']+=1;result['contact_json_append_s']+=time.perf_counter()-begin
    sim.step=step;lift.MultiObjectLiftRuntime.after_step=after;lift.source_metrics=metrics;journal.ContactJournal.append=append
    try:yield result
    finally:
        sim.step=before['step'];lift.MultiObjectLiftRuntime.after_step=before['after']
        lift.source_metrics=before['metrics'];journal.ContactJournal.append=before['append']
