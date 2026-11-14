"""Prepare the six predeclared EGAD validation assets, without physics or scores.

Uses the previous external12's uniform V-HACD parameters and normalizes every
component's original convex envelope before packaging. No outcome-based retry.
"""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
import traceback
import prepare_egad_external_scenes as base
import prepare_egad_external_scenes_v2 as vh
from prepare_egad_external_scenes_v3 import normalize_component

ROOT = Path(__file__).resolve().parents[2]
IDS = tuple('egad_' + r + str(c) for r in 'ACE' for c in (2, 5))
INVENTORY = Path('/mnt/public/datasets/AnyDexGrasp/object_alignment_20261106_v1/egad_eval_inventory.json')
AUDIT = ROOT / 'outputs/boya_augmented_arm_audit_20261114_v1/SELECTION_AUDIT.json'
read = lambda p: json.loads(Path(p).read_text())
sha, save = base.sha, base.save


def worker(job_path):
    job = read(job_path)
    directory = Path(job_path).parent
    start = time.monotonic()
    result = dict(object_id=job['object_id'], complete=False, physics_steps=0,
                  network_forwards=0, training_steps=0)
    try:
        sys.path.insert(0, str(ROOT / 'outputs/native_deps/python_overlay'))
        import pybullet as p
        import trimesh
        def forbidden(*args, **kwargs):
            raise RuntimeError('Asset preparation must not create a physics world')
        p.connect = forbidden
        p.stepSimulation = forbidden
        if sha(job['source']['path']) != job['source']['sha256']:
            raise ValueError('Source mesh changed')
        _, visual = base.scaled_visual(job['source']['path'])
        source = directory / 'scaled_source.obj'
        source.write_text(trimesh.exchange.obj.export_obj(visual, include_normals=False,
                         include_color=False, include_texture=False, digits=16))
        raw = directory / 'vhacd_components.obj'
        t = time.monotonic()
        p.vhacd(str(source), str(raw), str(directory / 'vhacd.log'), **vh.PARAMETERS)
        native_s = time.monotonic() - t
        components, checks = [], []
        for v, f in vh.fallback.split_obj(raw):
            part, check = normalize_component(v, f)
            components.append(part)
            checks.append(check)
        measured = vh.approximation(visual, components)
        if not measured['AABB_pass']:
            raise ValueError('Same fixed approximation AABB gate failed')
        runtime = dict(method='V-HACD plus uniform source-vertex convex-envelope normalization',
                       parameters=vh.PARAMETERS, native_wall_s=native_s,
                       raw_components=str(raw), raw_components_sha256=sha(raw),
                       pybullet_native_sha256=sha(p.__file__))
        record = base.package(job, components, runtime)
        geo = record['geometry_approximation']
        geo.pop('coacd_parameters', None)
        geo.pop('coacd_runtime', None)
        geo.update(decomposition_method=runtime['method'], decomposition_parameters=vh.PARAMETERS,
                   decomposition_runtime=runtime, approximation_checks=measured,
                   component_normalization=checks)
        save(record['metadata_path'], geo)
        record.update(metadata_sha256=sha(record['metadata_path']),
                      source_split='official_EGAD49_augmented_arm6_20261114')
        save(Path(record['object_root']) / 'PROVENANCE.json', record)
        result.update(complete=True, status='ready', record=record)
    except Exception:
        result.update(status='asset_preparation_failed', error=traceback.format_exc())
    result['wall_s'] = time.monotonic() - start
    save(directory / 'RESULT.json', result)
    return 0 if result['complete'] else 2


def prepare(root, workers=3):
    root = root.resolve()
    root.mkdir(exist_ok=False)
    scene = root / 'scenes'
    scene.mkdir()
    audit = read(AUDIT)
    if [r['object_id'] for r in audit['objects']] != list(IDS):
        raise ValueError('Independent six-object selection differs')
    inventory = read(INVENTORY)
    base.validate_inventory(inventory)
    byid = {r['object_id']: r for r in inventory['objects']}
    exclusions = base.exclusion_receipt([*base.EXCLUSION_REFS,
                        ROOT / 'outputs/egad_external12_20261107_v1/objects.json',
                        ROOT / 'outputs/boya_fast6_20261111_v1/objects.json'])
    sources = {str(Path(p).resolve()): sha(p) for p in
               [__file__, base.__file__, vh.__file__,
                ROOT / 'reproduction/boya/prepare_egad_external_scenes_v3.py',
                vh.FALLBACK, vh.GEOMETRY]}
    inputs = {str(INVENTORY): sha(INVENTORY), str(AUDIT): sha(AUDIT)}
    for ref in exclusions['manifest_refs']:
        inputs[ref['path']] = ref['sha256']
    jobs = []
    for oid in IDS:
        row = byid[oid[5:]]
        inputs[row['path']] = row['sha256']
        directory = scene / oid
        directory.mkdir()
        job = dict(object_id=oid, source=row, object_root=str(root / 'assets' / oid),
                   inventory_sha256=sha(INVENTORY), exclusion=exclusions, run_dir=str(root))
        save(directory / 'JOB.json', job)
        jobs.append(directory / 'JOB.json')
    selection = dict(schema='anydex_boya_augmented_arm6_asset_selection_v1', approved_object_ids=list(IDS),
                     rule='EGAD rows A,C,E x columns2,5; no score/outcome/shape filtering',
                     source_inventory=str(INVENTORY), source_inventory_sha256=sha(INVENTORY),
                     source_native_units_to_m=.001, mass_kg=.1, friction=.6,
                     vhacd_parameters=vh.PARAMETERS, normalize_all_convex_envelopes=True,
                     timeout_s=120, exclusion_manifest_refs=exclusions['manifest_refs'],
                     source_sha256=sources, input_sha256=inputs,
                     ranking_and_outcomes_not_yet_observed=True, new_physics_steps=0)
    save(scene / 'SELECTION.json', selection)
    started = time.monotonic()
    def one(path):
        with (path.parent / 'worker.log').open('xb') as out:
            proc = subprocess.run([sys.executable, '-B', __file__, '--worker', str(path)],
                                  stdout=out, stderr=subprocess.STDOUT, timeout=120)
        value = read(path.parent / 'RESULT.json')
        if proc.returncode or not value['complete']:
            raise RuntimeError('Asset failure retained: ' + str(path))
        return value['record']
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, p): p for p in jobs}
        for future in as_completed(futures):
            row = future.result()
            results[row['object_id']] = row
            print('FAST6_ASSET', row['object_id'], flush=True)
    for path, digest in {**inputs, **sources}.items():
        if sha(path) != digest:
            raise ValueError('Frozen preparation input changed: ' + path)
    manifest = dict(schema='anydex_boya_external_scene_manifest_v1', objects=[results[i] for i in IDS],
                    approved_object_ids=list(IDS), active_object_count=6, object_count=6,
                    source_inventory=str(INVENTORY), source_inventory_sha256=sha(INVENTORY),
                    source_active_list_sha256=sha(INVENTORY), source_dataset_root=str(root / 'assets'),
                    scale=1., camera=base.CAMERA, exclusion_manifest_refs=exclusions['manifest_refs'],
                    source_native_units_to_m=.001, selection=str(scene / 'SELECTION.json'),
                    selection_sha256=sha(scene / 'SELECTION.json'), physics_steps=0,
                    split_boundary='Six new-to-Boya validation objects; no training writeback',
                    original_AnyDex_CAD_identity_verified=False, original_scene_pose_reproduced=False)
    save(scene / 'objects.json', manifest)
    save(scene / 'RESULTS.json', dict(complete=True, ready_objects=6, physics_steps=0,
         scene_manifest=str(scene / 'objects.json'), scene_manifest_sha256=sha(scene / 'objects.json'),
         wall_s=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.worker))
    if args.run_dir is None:
        parser.error('--run-dir required')
    prepare(args.run_dir)
