"""Serial planning resources and exact-state collision results for one worker.

The original planners, tolerances, IK, retiming and validators remain callable.
Only identical, content-bound geometry/state queries reuse their prior result.
No quantization, approximate geometry, simulation or historical labels.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack, redirect_stdout, redirect_stderr
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

_OWNER = threading.Lock()


def array_key(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError('Object arrays cannot identify planning geometry')
    return a.dtype.str, a.shape, hashlib.sha256(a.tobytes(order='C')).digest()


def mesh_key(mesh):
    return array_key(mesh.vertices), array_key(mesh.faces)


class PlanningSession:
    def __init__(self, *, max_states=100000, max_geometries=32):
        if max_states < 1 or max_geometries < 1:
            raise ValueError('Positive bounded planning cache capacities required')
        self.max_states, self.max_geometries = max_states, max_geometries
        self.files, self.models, self.meshes, self.references = {}, {}, {}, {}
        self.geometries, self.states = OrderedDict(), OrderedDict()
        self.stats = dict(model_builds=0, model_reuses=0, mesh_loads=0, mesh_reuses=0,
            reference_loads=0, reference_reuses=0, geometry_builds=0, geometry_reuses=0,
            audit_calls=0, logical_states=0, evaluated_states=0, reused_states=0,
            logical_distance_queries=0, evaluated_distance_queries=0,
            native_prepare_calls=0, state_evictions=0, geometry_evictions=0)

    def bind(self, files):
        rows = []
        for value in sorted({str(Path(p).resolve()) for p in files}):
            digest = hashlib.sha256(Path(value).read_bytes()).hexdigest()
            if value in self.files and self.files[value] != digest:
                raise ValueError('Static planning source changed; start a fresh session: '+value)
            self.files[value] = digest
            rows.append((value, digest))
        return tuple(rows)

    def verify(self):
        self.bind(list(self.files))

    def robot_key(self, path):
        path = Path(path).resolve()
        files = [path]
        for node in ET.parse(path).getroot().findall('.//collision/geometry/mesh'):
            files.append(path.parent / node.get('filename').removeprefix('file://'))
        return self.bind(files)

    def model(self, original, path):
        key = self.robot_key(path)
        if key not in self.models:
            self.models[key] = original(path)
            self.stats['model_builds'] += 1
        else:
            self.stats['model_reuses'] += 1
        # Each caller owns mutable model metadata. Static meshes live separately.
        result = copy.deepcopy(self.models[key])
        result._planning_resource_key = key
        return result

    def collision_mesh(self, original, model, name):
        key = (self.robot_key(model.urdf), name)
        if key not in self.meshes:
            self.meshes[key] = original(model, name)
            self.stats['mesh_loads'] += 1
        else:
            self.stats['mesh_reuses'] += 1
        return self.meshes[key].copy()

    def reference(self, original, root, pose_name='home'):
        root = Path(root).resolve()
        sources = [root/'ur_by_description/urdf/ur_by.urdf.xacro',
                   root/'ur_by_moveit_config/config/ur_with_dexhand.srdf']
        key = (self.bind(sources), pose_name)
        if key not in self.references:
            result = original(root, pose_name)
            # Extra configuration files used by e.g. dexonomy_home are also frozen.
            self.bind(row['path'] for row in (result.arm_home_source or {}).get('source_files', []))
            self.references[key] = result
            self.stats['reference_loads'] += 1
        else:
            self.stats['reference_reuses'] += 1
        return copy.deepcopy(self.references[key])

    def geometry(self, original, instance, model, ignored_pairs, object_parts, object_pose,
                 *, collision_links=None, vectorized_broadphase=False):
        parts = list(object_parts)
        links = tuple(model.collision_links if collision_links is None else collision_links)
        key = (self.robot_key(model.urdf), links,
               json.dumps(ignored_pairs, sort_keys=True, separators=(',', ':')),
               tuple(mesh_key(p) for p in parts), array_key(object_pose), bool(vectorized_broadphase))
        if key in self.geometries:
            self.stats['geometry_reuses'] += 1
            resource = self.geometries[key]
            self.geometries.move_to_end(key)
            instance.__dict__.update(resource)
            instance.model = model
        else:
            original(instance, model, ignored_pairs, parts, object_pose,
                     collision_links=collision_links, vectorized_broadphase=vectorized_broadphase)
            self.stats['geometry_builds'] += 1
            self.geometries[key] = dict(instance.__dict__)
            if len(self.geometries) > self.max_geometries:
                self.geometries.popitem(last=False)
                self.stats['geometry_evictions'] += 1
        instance._planning_geometry_key = key

    def audit(self, original, geometry, states, *, tolerance_m=1e-7):
        """Reconstruct the original full report, including logical query counts."""
        self.stats['audit_calls'] += 1
        # The unmodified empty audit supplies all geometry metadata and scope.
        result = original(geometry, [], tolerance_m=tolerance_m)
        violations, minimum, queries = [], float('inf'), 0
        for index, state in enumerate(states):
            names = tuple(sorted(state))
            key = (geometry._planning_geometry_key, np.float64(tolerance_m).tobytes(),
                   names, np.asarray([state[n] for n in names], dtype=np.float64).tobytes())
            self.stats['logical_states'] += 1
            if key in self.states:
                row = self.states[key]
                self.states.move_to_end(key)
                self.stats['reused_states'] += 1
            else:
                report = original(geometry, [state], tolerance_m=tolerance_m)
                row = (report['min_floor_m'], report['signed_distance_queries'],
                       copy.deepcopy(report['violations']))
                self.states[key] = row
                self.stats['evaluated_states'] += 1
                self.stats['evaluated_distance_queries'] += row[1]
                if len(self.states) > self.max_states:
                    self.states.popitem(last=False)
                    self.stats['state_evictions'] += 1
            minimum = min(minimum, row[0])
            queries += row[1]
            for violation in row[2]:
                item = copy.deepcopy(violation)
                item['sample'] = index
                violations.append(item)
        self.stats['logical_distance_queries'] += queries
        result.update(ok=not violations, samples=len(states), signed_distance_queries=queries,
                      min_floor_m=minimum, violations=violations)
        return result

    def receipt(self):
        return dict(stats=dict(self.stats), static_sources=dict(self.files),
            cached_states=len(self.states), cached_geometries=len(self.geometries),
            max_states=self.max_states, max_geometries=self.max_geometries,
            matching='Exact float64 state bytes, full mesh bytes, object transform, links, exclusions and tolerance',
            logical_reports_unchanged=True, physics_steps=0, labels=0)

    @contextmanager
    def installed(self, *, native_prepare=True):
        if not _OWNER.acquire(blocking=False):
            raise RuntimeError('Planning cache requires one serial owning worker')
        try:
            import author_transport as T
            T._backend_path()
            import canondex.planning.model as models
            import canondex.robots.reference as refs
            import canondex.grasp.template_arm_bridge as bridge
            import author_pipeline as pipeline
            model_type, mesh, reference = models.PlanningModel, models.PlanningModel.collision_mesh, refs.load_reference
            init, audit = bridge.ArmPathGeometry.__init__, bridge.ArmPathGeometry.audit
            with ExitStack() as stack:
                stack.enter_context(patch.object(models, 'PlanningModel', lambda path: self.model(model_type, path)))
                stack.enter_context(patch.object(model_type, 'collision_mesh', lambda model, name: self.collision_mesh(mesh, model, name)))
                stack.enter_context(patch.object(refs, 'load_reference', lambda root=refs.UR_REFERENCE, pose_name='home': self.reference(reference, root, pose_name)))
                stack.enter_context(patch.object(bridge.ArmPathGeometry, '__init__',
                    lambda instance, *args, **kwargs: self.geometry(init, instance, *args, **kwargs)))
                stack.enter_context(patch.object(bridge.ArmPathGeometry, 'audit',
                    lambda instance, states, **kwargs: self.audit(audit, instance, states, **kwargs)))
                if native_prepare:
                    stack.enter_context(patch.object(pipeline, 'subprocess', NativePrepareProxy(pipeline.subprocess, self)))
                yield self
                self.verify()
        finally:
            _OWNER.release()


@contextmanager
def deadline(seconds, command):
    """Retain the old native subprocess timeout inside the serial worker."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Native planning owner must run on its process main thread')
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    began = time.monotonic()
    def expired(*unused):
        raise subprocess.TimeoutExpired(command, seconds)
    use_previous = 0 < previous_timer[0] <= seconds
    signal.signal(signal.SIGALRM, previous if use_previous else expired)
    signal.setitimer(signal.ITIMER_REAL, previous_timer[0] if use_previous else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,
                max(1e-6, previous_timer[0]-(time.monotonic()-began)), previous_timer[1])


class NativePrepareProxy:
    """Only the original CPU --stage prepare call runs in this owning process."""
    def __init__(self, original, session):
        self.original, self.session = original, session

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, command, **kwargs):
        import author_transport as T
        entry = str(T.ROOT/'reproduction/boya/native_arm.py')
        if entry not in list(map(str, command)):
            return self.original.run(command, **kwargs)
        args = list(map(str, command))[list(map(str, command)).index(entry)+1:]
        if args[args.index('--stage')+1] != 'prepare' or '--device' not in args or args[args.index('--device')+1] != 'cpu':
            raise ValueError('Only original zero-physics CPU preparation may be resident')
        self.session.bind(Path(args[args.index('--'+n.replace('_', '-'))+1])
                          for n in ('robot_urdf', 'hand_urdf', 'hand_scene', 'package', 'physics_config', 'diagnostic_config'))
        import run_template_arm_positive as backend
        self.session.stats['native_prepare_calls'] += 1
        code = 0
        # Duplicate the binary stage log FD; leave its caller-owned FD open.
        with os.fdopen(os.dup(kwargs['stdout'].fileno()), 'w', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log), deadline(kwargs['timeout'], command):
                try:
                    backend.main(args)
                except (KeyboardInterrupt, subprocess.TimeoutExpired):
                    raise
                except SystemExit as error:
                    code = error.code if isinstance(error.code, int) else 1
                except Exception:
                    traceback.print_exc()
                    code = 1
        return subprocess.CompletedProcess(command, code)
