"""Opt-in local API evidence and read-only live PhysX tensor snapshots.

Importing/scanning never imports Isaac, Torch, Warp or launches Kit. Live probes
only read caller-owned, already initialized views: no stepping, resets, writes,
cooking-cache changes or fallback to a CPU simulation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re

import numpy as np

SITE = Path('/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages')
SCHEMA = 'anydex_boya_gpu_capability_probe_v1'


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _need(value, message):
    if not value:
        raise ValueError(message)


def _one(pattern):
    found = sorted(SITE.glob(pattern))
    _need(len(found) == 1, 'Expected exactly one installed source: '+pattern)
    return found[0]


def static_report():
    """Read this installation's primary sources; never claim a live GPU pass."""
    paths = dict(
        tensors=_one('isaacsim/extscache/omni.physics.tensors-*/omni/physics/tensors/api.py'),
        manager=SITE/'isaaclab_physx/physics/physx_manager.py',
        cfg=SITE/'isaaclab_physx/physics/physx_manager_cfg.py',
        sensor=SITE/'isaaclab_physx/sensors/contact_sensor/contact_sensor.py',
        convex=_one('isaacsim/extscache/omni.physx.asset_validator-*/omni/physxassetvalidator/scripts/cookingApproximationChecker.py'),
        mimic=SITE/'isaacsim/exts/isaacsim.asset.transformer.rules/docs/CHANGELOG.md',
    )
    sources = {name:binding(path) for name,path in paths.items()}
    anchors = [
        ('gpu_enabled_from_device', 'manager', '"physxScene:enableGPUDynamics", is_gpu', 'GPU dynamics is explicitly enabled for cuda device; check live readback too'),
        ('gpu_broadphase', 'manager', '"GPU" if is_gpu else "MBP"', 'GPU and CPU broadphase differ'),
        ('gpu_disables_ccd', 'manager', 'enable_ccd = cfg.enable_ccd and not is_gpu', 'This backend disables CCD on GPU'),
        ('initialization_disables_processing', 'manager', '"/physics/disableContactProcessing", True', 'Manager initialization disables contact processing'),
        ('gpu_suppresses_usd_readback', 'manager', '"/physics/suppressReadback", True', 'GPU initialization suppresses USD readback; read tensor states'),
        ('sensor_enables_processing', 'sensor', '"/physics/disableContactProcessing", False', 'Official sensor reenables processing'),
        ('gpu_frontend_excludes_numpy', 'tensors', 'The Numpy frontend cannot be used with GPU pipelines', 'Use Warp/Torch frontend for GPU pipeline'),
        ('physx_parameters_on_cpu', 'tensors', 'Returns -1 (CPU) for PhysX', 'CPU material/property tensors do not imply CPU dynamics'),
        ('raw_contacts', 'tensors', 'def get_raw_contact_data(self, dt)', 'All sensor contacts with other_actor_ids; no filter required'),
        ('raw_actor_mapping_on_cpu', 'tensors', "Must be a CPU tensor in the frontend's format", 'Resolve actor IDs after copying uint64 IDs to CPU'),
        ('filtered_contacts', 'tensors', 'def get_contact_data(self, dt)', 'Normal scalar force, point, normal, separation, counts and starts'),
        ('friction_contacts', 'tensors', 'def get_friction_data(self, dt)', 'Per-filter friction vector, point, counts and starts'),
        ('shared_pair_tables', 'sensor', 'would otherwise overwrite (it shares those buffers with get_contact_data)', 'Copy normal count/start before reading friction'),
        ('gpu_capacity_fixed', 'cfg', 'GPU simulation feature is unable to dynamically grow all the buffers', 'Freeze sufficient capacity and reject overflow; do not silently drop contacts'),
        ('convex_gpu_check', 'convex', 'convex_gpu_compatibility_is_valid(stage_id, prim_id)', 'Official GPU compatibility query exists; full checker resets/detaches simulation and is not called here'),
        ('mimic_runtime_schema', 'mimic', 'runtime consumes `NewtonMimicAPI` directly', 'Runtime consumes Newton mimic schema; this alone does not establish GPU trajectory correctness'),
    ]
    facts = []
    for name,key,needle,meaning in anchors:
        lines=paths[key].read_text().splitlines()
        hits=[i+1 for i,line in enumerate(lines) if needle in line]
        _need(hits, 'Installed source anchor missing: '+name)
        facts.append(dict(id=name, source=key, lines=hits, interpretation=meaning))
    tree=ast.parse(paths['tensors'].read_text())
    methods={c.name:{n.name:n.lineno for n in c.body if isinstance(n,ast.FunctionDef)}
             for c in tree.body if isinstance(c,ast.ClassDef)}
    versions={}
    for name in ('isaacsim','isaaclab','isaaclab_physx','warp-lang','torch'):
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: versions[name]=None
    return dict(schema=SCHEMA, status='static_sources_verified', versions=versions,
        sources=sources, facts=facts, tensor_api_methods=methods,
        live_gpu_validated=False, Kit_starts=0, physics_steps=0,
        limitations=['No live device/contact/force/mimic/cooked-hull check is inferred from source availability',
                     'CPU callback availability and completeness on GPU remain a live measurement',
                     'No fixed convex vertex limit is inferred without matching installed evidence'])


def to_host(value):
    """Snapshot immediately: native views may overwrite buffers on next call."""
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, 'numpy'):
        return value.numpy().copy()
    return np.asarray(value).copy()


def tensor_info(value):
    return dict(type=type(value).__module__+'.'+type(value).__name__,
                device=str(getattr(value,'device','cpu')), shape=list(value.shape),
                dtype=str(value.dtype))


def validate_contact_buffers(buffers, *, mode, capacity, sensor_count, filter_count=0):
    """Check complete referenced intervals, not uninitialized unused capacity."""
    _need(mode in ('filtered','raw','friction'), 'Unknown contact mode')
    _need(capacity > 0 and sensor_count > 0, 'Nonzero contact capacity/sensors required')
    size = 4 if mode == 'friction' else 7 if mode == 'raw' else 6
    _need(len(buffers)==size, 'Wrong native contact buffer count')
    a=[np.asarray(value) for value in buffers]
    pair_shape=(sensor_count,) if mode=='raw' else (sensor_count,filter_count)
    if mode!='raw': _need(filter_count>0, 'Filtered contacts/friction require filters')
    ci=2 if mode=='friction' else 4
    counts,starts=a[ci:ci+2]
    _need(counts.shape==starts.shape==pair_shape, 'Contact count/start shape differs')
    _need(counts.dtype.kind in 'iu' and starts.dtype.kind in 'iu', 'Integer contact count/start required')
    _need(np.all(counts>=0) and np.all(starts>=0), 'Negative contact interval')
    total=sum(map(int,counts.flat))
    _need(total < capacity, 'Contact capacity reached; truncation cannot be excluded')
    active=counts>0
    ends=starts[active].astype(object)+counts[active].astype(object)
    end=int(max(ends,default=0))
    _need(end<=capacity, 'Contact interval exceeds capacity')
    widths=(3,3) if mode=='friction' else (1,3,3,1)
    for value,width in zip(a,widths):
        _need(value.shape==(capacity,width), 'Native contact payload shape differs')
    if mode=='raw':
        _need(a[6].shape==(capacity,) and a[6].dtype==np.dtype('uint64'), 'Raw actor IDs must be uint64')
    indices=np.concatenate([np.arange(int(s),int(s)+int(c)) for s,c in zip(starts[active],counts[active])]) if total else np.empty(0,dtype=int)
    for value in a[:len(widths)]: _need(np.isfinite(value[indices]).all(), 'Nonfinite active contact data')
    if mode!='friction' and total:
        _need(np.min(a[0][indices])>=-1e-6, 'Negative normal force')
        normal_lengths=np.linalg.norm(a[2][indices],axis=1)
        loaded=a[0][indices,0]>1e-6
        _need(np.all(abs(normal_lengths[loaded]-1)<.02), 'Loaded contact normal is not unit length')
    return dict(mode=mode, active_contact_records=total, largest_referenced_endpoint=end,
                capacity=capacity, saturation_detected=False, sensor_count=sensor_count,
                filter_count=filter_count, active_indices=indices.tolist())


def capture_contact_view(view, dt, *, mode='filtered', require_cuda=True):
    """Read one already-initialized view; exception stays a failed probe."""
    _need(float(dt)>0, 'Positive physics dt required')
    names=dict(filtered='get_contact_data',raw='get_raw_contact_data',friction='get_friction_data')
    _need(mode in names, 'Unknown contact mode')
    values=getattr(view,names[mode])(dt=float(dt))
    info=[tensor_info(value) for value in values]
    if require_cuda:
        _need(all('cuda' in item['device'].lower() for item in info), 'Contact payload is not actually on CUDA')
    host=[to_host(value) for value in values]
    result=validate_contact_buffers(host,mode=mode,capacity=int(view.max_contact_data_count),
        sensor_count=int(view.sensor_count),filter_count=int(view.filter_count))
    result.pop('active_indices')
    result.update(tensors=info,sensor_paths=list(view.sensor_paths),
                  filter_paths=list(view.filter_paths),dt_s=float(dt),
                  physics_steps_added=0,contact_fields_dropped=False)
    return result,host


def probe_live(sim, output_dir, *, contact_views, rigid_views=None, articulation_views=None):
    """Record current GPU state/API evidence; never advance caller's clock.

    contact_views is name -> (initialized native view, 'filtered'|'raw'|'friction').
    Call after physics initialization and a caller-controlled tick with contacts.
    A valid empty contact buffer is recorded as empty, not proof of reporting.
    """
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=SCHEMA,status='running',read_only=True,physics_steps_added=0,
                source=binding(__file__),contacts={},rigid_bodies={},articulations={},
                errors=[],live_reporting_nonempty=False,production_admitted=False)
    arrays={}
    before=int(sim.get_physics_step_count())
    scene=sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
    attrs={key:scene.GetAttribute('physxScene:'+key).Get() for key in ('enableGPUDynamics','broadphaseType','enableCCD','solverType')}
    report.update(scene_attributes=attrs,simulation_view_device_ordinal=int(sim.physics_sim_view.device_ordinal),
        physics_clock_before=before,disable_contact_processing=sim.get_setting('/physics/disableContactProcessing'),
        suppress_usd_readback=sim.get_setting('/physics/suppressReadback'))
    if not attrs['enableGPUDynamics'] or attrs['broadphaseType']!='GPU' or report['simulation_view_device_ordinal']<0:
        report['errors'].append('Live GPU dynamics/broadphase/view ordinal not established')
    if report['disable_contact_processing'] is not False:
        report['errors'].append('Contact processing disabled or unavailable')
    for name,(view,mode) in contact_views.items():
        _need(re.fullmatch(r'[A-Za-z0-9_]+',name), 'Safe unique view name required')
        try:
            meta,values=capture_contact_view(view,sim.get_physics_dt(),mode=mode)
            report['contacts'][name]=dict(status='pass',**meta)
            report['live_reporting_nonempty'] |= meta['active_contact_records']>0
            arrays.update({f'contact_{name}_{i}':value for i,value in enumerate(values)})
        except Exception as error:
            report['contacts'][name]=dict(status='failed',error=repr(error))
            report['errors'].append('Contact view failed: '+name)
    for kind,views,getters in (
        ('rigid_bodies',rigid_views or {},('get_transforms','get_velocities','get_masses','get_material_properties')),
        ('articulations',articulation_views or {},('get_dof_positions','get_dof_velocities','get_dof_position_targets','get_dof_velocity_targets'))):
        for name,view in views.items():
            _need(re.fullmatch(r'[A-Za-z0-9_]+',name), 'Safe unique view name required')
            result={}
            for method in getters:
                try:
                    value=getattr(view,method)();result[method]=dict(status='pass',**tensor_info(value))
                    arrays[f'{kind}_{name}_{method}']=to_host(value)
                except Exception as error:
                    result[method]=dict(status='failed',error=repr(error));report['errors'].append(f'{kind}/{name}/{method} failed')
            report[kind][name]=result
    report['physics_clock_after']=int(sim.get_physics_step_count())
    if report['physics_clock_after']!=before:report['errors'].append('Physics clock advanced during read-only probe')
    np.savez(out/'TENSOR_SNAPSHOT.npz',**arrays)
    report['snapshot']=binding(out/'TENSOR_SNAPSHOT.npz')
    report['status']='failed' if report['errors'] else 'live_api_snapshot_pass'
    report['reporting_completeness_proven']=False
    report['limitations']=['Nonempty snapshot is not complete-run label/gate equivalence',
        'Forces applied persistently, mimic error, native cooked geometry, callback completeness and all phases require actual run evidence']
    (out/'PROBE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('scan',));p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    report=static_report();(args.output_dir/'STATIC_CAPABILITIES.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','facts','tensor_api_methods')},indent=2))


if __name__=='__main__':main()
