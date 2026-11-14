"""Read existing object assets and split manifests; never launch any work queue."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PUBLIC = Path('/mnt/public/datasets/AnyDexGrasp')


def main():
    start = time.monotonic()
    bindings, mismatches = {}, []

    def sha(path):
        path = Path(path).resolve()
        if str(path) not in bindings:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            bindings[str(path)] = digest.hexdigest()
        return bindings[str(path)]

    def read(path):
        sha(path)
        return json.loads(Path(path).read_text())

    def verify(path, expected):
        if sha(path) != expected:
            mismatches.append(str(path))

    inventory = read(PUBLIC/'object_alignment_20261106_v1/graspnet_s40_inventory.json')
    egad = read(PUBLIC/'object_alignment_20261106_v1/egad_eval_inventory.json')
    source_ids = set()
    for scene in inventory['scenes']:
        verify(scene['path'], scene['sha256'])
        ids = [int(i) for i in Path(scene['path']).read_text().split()]
        assert ids == scene['object_ids']
        source_ids.update(ids)
    assert source_ids == set(inventory['object_ids']) and len(source_ids) == 40
    for row in inventory['objects']:
        for file in row['files'].values():
            verify(file['path'], file['sha256'])
    asset_root = PUBLIC/'boya_s40_assets_20261106_v1'
    assets = []
    for directory in sorted(asset_root.glob('graspnet_*')):
        provenance = read(directory/'PROVENANCE.json')
        for file in provenance['convex_parts']:
            verify(file['path'], file['sha256'])
        for name, expected in provenance['source_input_sha256'].items():
            verify(name, expected)
        urdf = directory/'urdf/coacd.urdf'
        source_parts = [(urdf.parent/n.get('filename')).resolve()
                        for n in ET.parse(urdf).findall('.//collision/geometry/mesh')]
        assert len(source_parts) == len(provenance['convex_parts']) and all(p.is_file() for p in source_parts)
        assets.append(dict(object_id=directory.name, source_visual_mesh_sha256=provenance['source_visual_mesh_sha256'],
            converted_visual_mesh_sha256=sha(directory/'mesh/simplified.obj'), collision_urdf_sha256=sha(urdf),
            source_convex_parts=len(source_parts), AnyDex_CAD_verified=provenance['original_AnyDex_CAD_identity_verified']))
    data = ROOT/'outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset'
    manifest = read(data/'manifest.json')
    ranked = sorted(manifest['objects'], key=lambda r: hashlib.sha256(
        ('s40-internal-20261107:'+r['source_visual_mesh_sha256']).encode()).hexdigest())
    splits = {name: [r['object_id'] for r in rows]
              for name, rows in [('fit', ranked[:32]), ('internal_check', ranked[32:])]}
    for name, rows in [('train', ranked[:32]), ('internal_check', ranked[32:])]:
        assert set(manifest['mesh_groups'][name]) == {r['source_visual_mesh_sha256'] for r in rows}
    assert not set(splits['fit']) & set(splits['internal_check'])
    counts = []
    for g in [1,6,7,8]:
        path = data/f'gesture_{g}.npz'; sha(path)
        with np.load(path, allow_pickle=False) as z:
            bysplit = {k:int(np.count_nonzero(z['split'] == k)) for k in ['train','internal_check']}
            counts.append(dict(gesture_id=g, historical_labels=len(z['labels']), fit=bysplit['train'],
                internal_check=bysplit['internal_check'], target_new_compatible_fit=1000,
                conditional_gap_if_old_fit_later_accepted=1000-bysplit['train']))
    native = read(ROOT/'outputs/s40_collection_20261107_v1/native_preprocessing/SUMMARY.json')
    native_rows = []
    for row in native['results']:
        result = read(row['result_json'])
        for path, expected in result['artifact_sha256'].items():
            artifact = Path(path)
            if not artifact.is_absolute():
                artifact = Path(result['cache_directory'])/artifact
            verify(artifact, expected)
        native_rows.append(dict(object_id=row['object_id'], complete=result['complete'],
            native_20um=result['original_or_refined_within_20um'], source_partitions=result['source_partitions'],
            refined_partitions=result['refined_partitions'], repair_required=result['repair_required']))
    upstream = Path(inventory['source_root'])/'meshdata'
    upstream_rows = []
    for directory in sorted(p for p in upstream.iterdir() if p.is_dir()):
        urdf = directory/'coacd/coacd.urdf'
        paths = [(urdf.parent/n.get('filename')).resolve() for n in ET.parse(urdf).findall('.//collision/geometry/mesh')]
        upstream_rows.append(dict(directory=directory.name, visual=(directory/'nontextured.ply').is_file(),
            simplified_visual=(directory/'nontextured_simplified.ply').is_file(), collision_urdf=True,
            collision_part_files=len(paths), all_collision_files_present=all(p.is_file() for p in paths)))
    for row in egad['objects']:
        verify(row['path'], row['sha256'])
    converted = []
    for base in ['egad_external12_20261107_v1/assets_v3', 'boya_fast6_20261111_v1/assets',
                 'boya_augmented_arm_external6_20261114_v1/assets']:
        for directory in sorted((ROOT/'outputs'/base).glob('egad_*')):
            urdf = directory/'urdf/coacd.urdf'
            if urdf.exists():
                paths = [(urdf.parent/n.get('filename')).resolve() for n in ET.parse(urdf).findall('.//collision/geometry/mesh')]
                converted.append(dict(object_id=directory.name, root=str(directory),
                    urdf_sha256=sha(urdf), all_part_files_present=all(p.is_file() for p in paths), parts=len(paths)))
    for path in ['outputs/paper_code_audit/source/sections/methods.tex',
                 'outputs/paper_code_audit/source/sections/supp.tex', 'reproduction/boya/author_scene.py',
                 'reproduction/boya/author_lift_pipeline.py']:
        sha(ROOT/path)
    receipt = dict(schema='anydex_boya_formal_lift_source_inventory_v1', ok=not mismatches,
        S40=dict(source_scene_files=100, source_identity_union=40, source_ids=sorted(source_ids),
            converted_asset_objects=len(assets), assets=assets,
            native_preflight_objects=len(native_rows), native_preflight=native_rows,
            identity_boundary='GraspNet scene0-99 ID union from DGN2-derived local copies; not byte-compared to official GraspNet models.zip or AnyDex CAD'),
        upstream_public_derived_meshdata=dict(objects=len(upstream_rows), objects_detail=upstream_rows,
            role='88 local GraspNet-derived mesh/collision sets include S40; remaining48 are not automatically training objects or L144 identities',
            validation_scope='All files referenced by88 collision URDFs exist; no new cooking/dynamic validation for remaining48'),
        EGAD=dict(official_meshes=len(egad['objects']), source_mesh_SHA_verified=not mismatches,
            converted_collision_objects_in_three_frozen_external_campaigns=len({r['object_id'] for r in converted}),
            converted=converted, role='Existing external/development objects, excluded from new training by default; not L144 completion'),
        L144=dict(complete_machine_readable_ID_table_available=False, verified_object_to_CAD_correspondence_rows=0,
            local_evidence='Existing graspnet_s40_inventory flags missing L144/T30; paper methods names AnyGrasp144, supplement says CAD available upon request',
            limit='Local inventory conclusion only; model folders named obj140 are weights, not a144-object mesh identity manifest'),
        old_split=dict(policy='SHA256(s40-internal-20261107:source_visual_mesh_sha256), first32 fit,last8 internal',
            objects=splits, fit_objects=32, internal_objects=8, per_gesture=counts,
            historical_total=manifest['sample_count'], historical_fit=sum(r['fit'] for r in counts),
            old_fit_compatibility_with_new_clutter_lift_not_certified=True),
        scene_capabilities=dict(current_build_fixed_objects=['graspnet_000','graspnet_002','graspnet_005'],
            validator_capture_runtime_accept_explicit_object_list=True, random_clutter_sampler_implemented=False,
            formal_split_isolated_scene_scheduler_implemented=False, frozen_4000_fit_queue_created=False,
            rule='Every body visible/physically present in a scene belongs to the same fit/internal split; compatibility primary does not define scene split'),
        new_compatible_fit_labels=0, formal_fit_target_per_gesture=1000, formal_fit_target_total=4000,
        internal_labels_are_additional=True, per_slot_target_1000=False,
        input_source_files_SHA=bindings, SHA_mismatches=mismatches, audit_source_sha256=sha(__file__),
        new_physics_steps=0, new_IK_calls=0, new_downloads=0, new_training_updates=0,
        wall_s=time.monotonic()-start)
    with Path(__file__).with_name('OBJECT_INVENTORY.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(ok=receipt['ok'], S40=len(assets), upstream=len(upstream_rows),
        EGAD=len(egad['objects']), converted_EGAD=receipt['EGAD']['converted_collision_objects_in_three_frozen_external_campaigns'],
        source_files=len(bindings), wall_s=receipt['wall_s']),indent=2))


if __name__ == '__main__':
    main()
