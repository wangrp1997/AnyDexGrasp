"""Read-only replay of the last paired batch and independent source clearance checks."""
import argparse
import ast
from collections import Counter
import hashlib
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

import author_collection_batch as B


APPROACH_M = .04
CLEARANCE_M = 2e-5
SOURCE = B.ROOT / 'outputs/boya_mjx_formal_pairs_20261120_v1/collection'


def hull(vertices):
    """Retain the exact support vertices of the enclosed source geometry."""
    return vertices[ConvexHull(vertices).vertices]


def aabb_separation(first, second):
    return float(max(np.max(second.min(0) - first.max(0)), np.max(first.min(0) - second.max(0))))


def separation(first, second):
    gap = aabb_separation(first, second)
    if gap > CLEARANCE_M:
        return gap
    # An optimizer proposes an axis. The measured support gap decides separation.
    constraints = np.vstack((np.c_[first, -np.ones(len(first)), np.zeros(len(first))],
                             np.c_[-second, np.zeros(len(second)), np.ones(len(second))]))
    solution = linprog([0., 0., 0., 1., -1.], A_ub=constraints, b_ub=np.zeros(len(constraints)),
                       bounds=[(-1., 1.)]*3 + [(None, None)]*2, method='highs')
    if not solution.success:
        return gap
    axis = solution.x[:3]
    norm = np.linalg.norm(axis)
    if norm == 0:
        return gap
    axis /= norm
    return max(gap, float(np.min(second @ axis) - np.max(first @ axis)))


def native_objects(model_path):
    """Read baked object collision vertices from the saved native input model."""
    model = ET.parse(model_path).getroot()
    floor = model.find("worldbody/geom[@name='floor']")
    assert floor.attrib['type'] == 'plane'
    assert not set(floor.attrib) & {'pos', 'quat', 'axisangle', 'euler', 'xyaxes', 'zaxis'}
    meshes = {element.attrib['name']: element.attrib for element in model.findall('asset/mesh')}
    bodies = {}
    for body in model.findall('worldbody/body'):
        name = body.attrib['name']
        if not name.startswith('graspnet_'):
            continue
        shapes = []
        for geom in body.findall('geom'):
            if geom.attrib.get('contype') == '0':
                assert geom.attrib.get('conaffinity') == '0'
                continue
            assert geom.attrib['type'] == 'mesh'
            assert not set(geom.attrib) & {'pos', 'quat', 'axisangle', 'euler', 'xyaxes', 'zaxis'}
            mesh = meshes[geom.attrib['mesh']]
            assert set(mesh) == {'name', 'vertex', 'face'}
            shapes.append(hull(np.fromstring(mesh['vertex'], sep=' ').reshape(-1, 3)))
        assert shapes
        bodies[name] = shapes
    return bodies


def scene_geometry(scene, native, mesh_cache):
    import trimesh

    obstacles = []
    # The comparison has one ground plane and three active bodies, without a bin.
    assert scene['bin'] is None and len(scene['objects']) == 3
    for obj in scene['objects']:
        transform = np.asarray(obj['world_T_object'])
        source_meshes = []
        for key in ('source_mesh', 'source_visual_mesh'):
            path = obj[key]
            digest = obj[key+'_sha256']
            if digest not in mesh_cache:
                B.verify({key: dict(path=path, sha256=digest)})
                mesh = trimesh.load(path, force='mesh', process=False)
                mesh_cache[digest] = hull(np.asarray(mesh.vertices))
            source_meshes.append(mesh_cache[digest] * obj['scale'])
        for vertices in source_meshes + native[obj['object_id']]:
            obstacles.append(vertices @ transform[:3, :3].T + transform[:3, 3])
    return obstacles


def instrument(author, observations):
    original = author.definitions

    def definitions(name, names, scope):
        if (name, names) != ('utils/collision_detector.py', {'ModelFreeCollisionDetectorMultifinger'}):
            return original(name, names, scope)
        detector = next(node for node in author.source(name).body
                        if isinstance(node, ast.ClassDef) and node.name in names)
        method = next(node for node in detector.body if isinstance(node, ast.FunctionDef) and node.name == 'detect')
        assert isinstance(method.body[-1], ast.Return)
        method.body.insert(-1, ast.parse(
            'observe(collision_mask, meshes_pointclouds_multifinger)').body[0])
        tree = ast.fix_missing_locations(ast.Module(body=[detector], type_ignores=[]))

        def observe(mask, meshes):
            observations.append((mask.copy(), [np.asarray(mesh.points).copy() for mesh in meshes]))

        namespace = dict(scope, observe=observe)
        exec(compile(tree, str(B.ROOT / name), 'exec'), namespace)
        return namespace

    return patch.object(author, 'definitions', definitions)


def opening_geometry(states, slot, library, assets):
    from author_hand_clouds import full_hand_mesh

    key = library.report['slot_to_state'][slot['slot_id']]
    if key not in states:
        state = library.states[key]
        mesh, parts = full_hand_mesh(assets, slot['independent_q'])
        vertices = np.ascontiguousarray(np.asarray(mesh.vertices))
        triangles = np.ascontiguousarray(np.asarray(mesh.triangles))
        assert hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest() == state['mesh_array_sha256']
        assert parts == state['parts']
        # A linear plane's minimum is attained at a support vertex.
        states[key] = [hull(vertices[p['vertex_offset']:p['vertex_offset'] + p['vertices']]) for p in parts]
    return states[key]


def audit(output):
    import author_resident_prepare as resident

    protocol = B.read(SOURCE / 'PROTOCOL.json')
    final = B.read(SOURCE.parent / 'FINAL_REVIEW.json')
    B.verify(protocol['sources'])
    B.verify(protocol['inputs'])
    B.verify({'runtime_audit': final['original_audit']})
    model_receipt = B.read(SOURCE / 'MODEL_RECEIPT.json')
    B.verify({'model': model_receipt['model']})
    native = native_objects(model_receipt['model']['path'])
    resident.bind_backend({'backend': protocol['inputs']['backend']['path']})
    import author_contact_pipeline as contact
    import author_protocol as author
    from asset_adapter import BoyaAssets, CGR_R_REFERENCE, rigid
    from author_hand_clouds import HandCloudLibrary

    assets = BoyaAssets()
    reference = np.eye(4)
    reference[:3, :3] = CGR_R_REFERENCE
    openings, object_cache = {}, {}
    frames = []
    for spec in protocol['rows'][::2]:
        index = spec['pair_index']
        pair = SOURCE / 'pairs' / f'{index:03d}'
        start = B.read(pair / 'SHARED_START.json')
        ranking = B.read(pair / 'RANKING.json')
        attempt = pair / 'arm_plans/00'
        bundle = B.read(attempt / 'mapping/BUNDLE.json')
        prepared = B.read(attempt / 'pipeline/collection/PREPARED.json')
        refs = dict(pool=ranking['shared_pool'], state=start['live_state'], scene=start['measured_scene'],
                    observation=start['observation'], trace=prepared['trace'], calibration=bundle['calibration'],
                    surface=bundle['surface_manifest'], saved_predicate=B.binding(attempt/'COLLISION_BRANCHES.json'))
        B.verify(refs)
        calibration = B.read(refs['calibration']['path'])
        slots = {slot['slot_id']: slot for slot in calibration['slots']}
        library = HandCloudLibrary(refs['surface']['path'])
        world = rigid(B.read(refs['observation']['path'])['T_WC'])
        with np.load(refs['trace']['path'], allow_pickle=False) as arrays:
            rows = arrays['actual_rows17_camera']
            extra = arrays['extra_depth_m']
            points = arrays['source_scene_camera']
            masked = []
            with instrument(author, masked):
                actual = contact.contact_detect(rows, np.ones(len(rows), dtype=int), extra, points,
                     calibration=calibration, library=library, profile='dh3_collection')
            np.testing.assert_array_equal(actual['accepted_source_indices'], arrays['collision_accepted_top500_indices'])
            row_ids = arrays['source_indices']
        assert actual['branch_checks'] == B.read(refs['saved_predicate']['path'])['branch_checks']
        masks, sampled = masked[0] if masked else ([], [])
        plane_points = np.abs(points @ world[2, :3] + world[2, 3]) <= .001
        approach = contact.C.PROFILES['dh3_collection']['approach']
        threshold = contact.C.PROFILES['dh3_collection']['threshold']
        assert approach == APPROACH_M
        source_scene = B.read(refs['scene']['path'])
        obstacles = None
        measurements = []
        already_accepted = set(actual['accepted_source_indices'])
        for check, mask, original_samples in zip(actual['branch_checks'], masks, sampled, strict=True):
            number = check['source_local_index']
            row = rows[number]
            slot = slots[check['slot_id']]
            pose = np.eye(4)
            pose[:3, :3] = row[4:13].reshape(3, 3)
            pose[:3, 3] = row[13:16]
            before_depth = rigid(rigid(pose) @ reference @ np.linalg.inv(slot['palm_T_reference']))
            axis = pose[:3, 0].astype(float)
            axis /= np.linalg.norm(axis)
            endpoint = before_depth.copy()
            endpoint[:3, 3] += np.asarray(axis * (float(row[3]) + extra[number]), dtype=np.float32)
            expected_samples = library.world_points(slot, endpoint, voxel_mm=3)
            np.testing.assert_allclose(expected_samples, original_samples, atol=2e-9, rtol=0)
            target = world @ endpoint
            retract = -APPROACH_M * (world[:3, :3] @ axis)
            link_hulls = [vertices @ target[:3, :3].T + target[:3, 3]
                          for vertices in opening_geometry(openings, slot, library, assets)]
            bottom_target = min(float(v[:, 2].min()) for v in link_hulls)
            bottom_sweep = bottom_target + min(0., float(retract[2]))
            rejected = int(mask.sum()) > threshold
            assert rejected != check['collision_clear']
            status, clearance = 'admitted_by_original', None
            if rejected and bottom_sweep < -CLEARANCE_M:
                status = 'full_source_hand_intersects_ground'
            elif rejected and bottom_sweep > CLEARANCE_M:
                if obstacles is None:
                    obstacles = scene_geometry(source_scene, native, object_cache)
                status = 'ground_clear_object_overlap_unresolved'
                clearance = bottom_sweep
                for link in link_hulls:
                    sweep = np.r_[link, link + retract]
                    gaps = [separation(sweep, obj) for obj in obstacles]
                    clearance = min(clearance, *gaps)
                    if clearance <= CLEARANCE_M:
                        break
                if clearance > CLEARANCE_M:
                    status = 'separated_source_and_native_objects_and_ground'
            elif rejected:
                status = 'within_ground_clearance_tolerance'
            measurements.append(dict(raw_row=int(row_ids[number]), local_index=number, slot=slot['slot_id'],
                occupancy=int(mask.sum()), near_ground_occupancy=int(mask[plane_points].sum()),
                above_height_band_occupancy=int(mask[~plane_points].sum()), rejected=rejected,
                open_hand_bottom_at_target_m=bottom_target, open_hand_bottom_over_approach_m=bottom_sweep,
                status=status, proven_separation_lower_bound_m=clearance if status.startswith('separated_') else None,
                original_row_already_admitted=number in already_accepted))
        counts = Counter(row['status'] for row in measurements)
        rescued = {row['raw_row'] for row in measurements
                   if row['status'].startswith('separated_') and not row['original_row_already_admitted']}
        frame = dict(pair=index, split=spec['split'], original_pool=ranking['candidate_count'], counts=dict(counts),
                     independently_clear_rejected_rows=sorted(rescued), inputs=refs, measurements=measurements)
        frames.append(frame)
        print(index, 'pool', ranking['candidate_count'], dict(counts), 'cleared rejected rows', sorted(rescued), flush=True)
    assert len(frames) == len(final['frames']) == 16 and sum(row['original_pool'] == 0 for row in frames) == 9
    all_counts = Counter()
    empty_counts = Counter()
    for frame in frames:
        all_counts.update(frame['counts'])
        if frame['original_pool'] == 0:
            empty_counts.update(frame['counts'])
    report = dict(status='offline_clearance_audit_complete', source_audit=final['original_audit'],
        source_protocol=B.binding(SOURCE/'PROTOCOL.json'), source_model=model_receipt['model'],
        implementation=B.binding(__file__), source_hand_opening_states=len(openings),
        counts=dict(all_counts), empty_pool_branch_counts=dict(empty_counts),
        scenes_with_clear_rejected_raw_rows=sum(bool(frame['independently_clear_rejected_rows']) for frame in frames),
        empty_scenes_with_clear_rejected_raw_rows=sum(not frame['original_pool']
              and bool(frame['independently_clear_rejected_rows']) for frame in frames),
        approach_m=APPROACH_M, comparison_clearance_m=CLEARANCE_M, frames=frames,
        nominal_open_hand_only=True, added_physics_steps=0, new_CGR_forwards=0, training_updates=0)
    B.save(output, report, exclusive=True)
    print({key: value for key, value in report.items() if key!='frames'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output.resolve())
