"""Frozen 100-tick GPU execution with default or disabled Fabric publication.

The existing scaling/vector/profile runners supply all physics and logging.
Only two publication settings change, after App creation and before scene setup.
No simulator is imported until the explicitly authorized worker runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import author_scaling_execution as S

P, V = S.P, S.V
SCHEMA = 'anydex_boya_author_fabric_execution_v1'
SCALING_SHA = '4037bebc5296977848cc2413d833d396eb2669df470b9c74a53863c91dbf0375'
KEYS = ('/physics/fabricUpdateTransformations', '/physics/fabricUpdateVelocities')
OTHER_KEYS = ('/physics/fabricUseGPUInterop', '/physics/suppressReadback',
              '/physics/disableContactProcessing')


def setting_values(settings):
    return {key: settings.get(key) for key in KEYS + OTHER_KEYS}


def configure(settings, mode):
    V._need(mode in ('baseline', 'both_disabled'), 'Explicit frozen Fabric arm required')
    before = setting_values(settings)
    if mode == 'both_disabled':
        for key in KEYS:
            settings.set_bool(key, False)
    after = setting_values(settings)
    V._need(all(before[k] == after[k] for k in OTHER_KEYS), 'Unrelated setting changed')
    if mode == 'both_disabled':
        V._need(all(after[k] is False for k in KEYS), 'Both Fabric publication settings must be disabled')
    else:
        V._need(before == after, 'Baseline must preserve all original settings')
    return dict(before=before, after=after)


def check_settings(values, mode, baseline=None):
    V._need(all(type(values[k]) is bool for k in KEYS), 'Loaded Fabric defaults must be explicit booleans')
    if mode == 'both_disabled':
        V._need(all(values[k] is False for k in KEYS), 'Fabric publication was reenabled')
    elif baseline is not None:
        V._need(all(values[k] == baseline[k] for k in KEYS), 'Baseline Fabric settings drifted')
    V._need(values['/physics/disableContactProcessing'] is False, 'Full contacts must remain enabled')


def convex_compatibility(sim):
    """Read only the official per-prim predicate, after the measured prefix.

    Never invoke CheckStage, release cooking caches, attach/detach or reset.
    Compatibility is evidence, not a geometry-changing admission workaround.
    """
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    enabled_before = manager.is_extension_enabled('omni.physx.asset_validator')
    if not enabled_before:
        manager.set_extension_enabled_immediate('omni.physx.asset_validator', True)
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    from omni.physxassetvalidator import get_physx_asset_validator_interface
    stage = sim.stage
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    V._need(stage_id != -1, 'Already attached stage required; do not insert or reattach')
    before = sim.get_physics_step_count()
    stage_before = hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest()
    query = get_physx_asset_validator_interface().convex_gpu_compatibility_is_valid
    rows = []
    started = time.monotonic()
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if str(approximation) not in ('convexHull', 'convexDecomposition'):
            continue
        path = str(prim.GetPath())
        valid = bool(query(stage_id, PhysicsSchemaTools.sdfPathToInt(prim.GetPath())))
        rows.append(dict(path=path, approximation=str(approximation), gpu_compatible=valid))
    after = sim.get_physics_step_count()
    stage_after = hashlib.sha256(stage.GetRootLayer().ExportToString().encode()).hexdigest()
    V._need(rows and before == after and stage_before == stage_after,
            'Compatibility query must preserve stage and physics clock')
    return dict(status='queried', checked_convex_prims=len(rows),
        gpu_compatible=sum(r['gpu_compatible'] for r in rows),
        gpu_incompatible=sum(not r['gpu_compatible'] for r in rows), rows=rows,
        physics_clock_before=before, physics_clock_after=after, physics_steps_added=0,
        root_layer_sha256_before=stage_before, root_layer_sha256_after=stage_after,
        wall_s=time.monotonic()-started, measured_prefix_includes_this_query=False,
        validator_extension_enabled_before_query=enabled_before,
        scope='Official per-prim compatibility predicate only; no actual CPU/GPU kernel attribution')


def arm_for_output(output_dir, protocol):
    matches = [(g, w) for g in protocol['groups'] for w in g['workers']
               if Path(w['output_dir']).resolve() == Path(output_dir).resolve()]
    V._need(len(matches) == 1, 'Output must identify exactly one frozen worker')
    group, worker = matches[0]
    V._need(group['name'] in ('baseline', 'both_disabled') and group['mode'] == 'gpu'
            and group['count'] == worker['lanes'] == 1, 'Single-GPU-lane Fabric pair only')
    return group['name'], worker


def run(prepared_dir, output_dir, *, protocol_path):
    V._need(P.sha(S.__file__) == SCALING_SHA, 'Reviewed scaling runner changed')
    protocol_path, out = Path(protocol_path).resolve(), Path(output_dir).resolve()
    protocol = json.loads(protocol_path.read_text())
    V._need(protocol['schema'] == 'anydex_boya_author_fabric_control_v1'
            and protocol['prefix_ticks'] == 100 and protocol['retry_budget'] == 0,
            'Frozen once-only 100-tick Fabric pair required')
    mode, worker = arm_for_output(out, protocol)
    V._need(Path(prepared_dir).resolve() == Path(worker['prepared_dir']).resolve(), 'Original template changed')
    V.verify_sources({v['path']: v['sha256'] for v in protocol['sources'].values()})
    storage = V._modules()[0]
    old_adapted, old_scene, old_sources, old_save = V.adapted_methods, S.scene_class, V.source_bindings, storage.save
    patches = P.Patches()
    live = {}
    state = dict(schema=SCHEMA, mode=mode, status='starting', label=None, raw_label=None,
        exclude_training=True, formal_training_eligible=False, full_lift_certified=False,
        prefix_ticks=100, protocol=dict(path=str(protocol_path), sha256=P.sha(protocol_path)),
        settings_after_ticks=[], only_changed_settings=list(KEYS) if mode == 'both_disabled' else [])
    bindings = old_sources()
    for path in (Path(__file__).resolve(), Path(S.__file__).resolve()):
        bindings[str(path)] = P.sha(path)

    def adapted(lift):
        import carb.settings
        settings = carb.settings.get_settings()
        state['after_app_before_scene'] = configure(settings, mode)
        live['settings'] = settings
        return old_adapted(lift)

    def scene_class(device, capture=None):
        base = old_scene(device, capture)
        class FabricScene(base):
            def prepare_guard(self):
                super().prepare_guard()
                live['scene'] = self
                from pxr import Usd, UsdGeom
                cameras = [str(p.GetPath()) for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath('/World/envs/env_0'))
                           if p.IsA(UsdGeom.Camera)]
                V._need(not cameras and not self.sim.cfg.visualizer_cfgs,
                        'Fabric-off diagnostic must have no task cameras or visualizers')
                values = setting_values(live['settings'])
                check_settings(values, mode)
                state['before_prefix'] = dict(settings=values, cameras=cameras,
                    use_fabric=self.sim.cfg.use_fabric, physics_clock=self.sim.get_physics_step_count(),
                    physics_device=str(self.sim.device), visualizer_cfg_count=len(self.sim.cfg.visualizer_cfgs or []))

            def check_cross(self):
                super().check_cross()
                values = setting_values(live['settings'])
                check_settings(values, mode, state['before_prefix']['settings'])
                state['settings_after_ticks'].append(dict(tick=self.sim.get_physics_step_count(), values=values))
        return FabricScene

    def save(path, value):
        result = old_save(path, value)
        if Path(path) == out/'SCALING_EXECUTION.json':
            state.update(status='partial_fabric' if value['status'] == 'partial_scaling' else 'infrastructure_failure',
                scaling_receipt=dict(path=str(path), sha256=P.sha(path)),
                source_sha256=value['source_sha256'], physical_cost={k:v for k,v in value.items()
                    if k in ('physical_clock_steps','returned_physical_steps','actual_simulation_step_counter',
                        'observations_completed_steps_per_lane','physical_steps_unknown','inflight_step_unconfirmed',
                        'physical_steps_lower_bound','physical_steps_upper_bound')})
            # After profile and data finalization, before fast App.close exits.
            scene = live.get('scene')
            if scene is not None:
                state['after_prefix'] = dict(settings=setting_values(live['settings']),
                    physics_clock=scene.sim.get_physics_step_count())
                try:
                    state['convex_compatibility'] = convex_compatibility(scene.sim)
                except Exception as error:
                    state['convex_compatibility'] = dict(status='query_failed', error=repr(error),
                        physics_clock_after=scene.sim.get_physics_step_count())
            old_save(out/'FABRIC_EXECUTION.json', state)
        return result

    patches.set(V, 'adapted_methods', adapted)
    patches.set(S, 'scene_class', scene_class)
    patches.set(V, 'source_bindings', lambda: dict(bindings))
    patches.set(storage, 'save', save)
    try:
        return S.run(prepared_dir, out, mode='gpu', lanes=1, prefix_ticks=100, protocol_path=protocol_path)
    finally:
        patches.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['gpu'], required=True)
    parser.add_argument('--lanes', type=int, choices=[1], required=True)
    parser.add_argument('--prefix-ticks', type=int, choices=[100], required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared_dir, args.output_dir, protocol_path=args.protocol)


if __name__ == '__main__':
    main()
