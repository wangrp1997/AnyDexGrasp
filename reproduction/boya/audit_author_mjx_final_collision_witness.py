"""Find positive native-object overlap witnesses for unresolved empty-pool branches."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

import author_collection_batch as B
from audit_author_mjx_final_collision import APPROACH_M, CLEARANCE_M, native_objects, opening_geometry


def swept_vertex_witness(vertices, displacement, normals, offsets):
    # Intersect every half-space inequality on the same sweep parameter.
    slope = normals @ displacement
    intercept = vertices @ normals.T + offsets + CLEARANCE_M
    positive, negative = slope > 0, slope < 0
    stationary = ~(positive | negative)
    lower, upper = np.zeros(len(vertices)), np.ones(len(vertices))
    if np.any(negative):
        lower = np.maximum(lower, np.max(-intercept[:, negative] / slope[negative], axis=1))
    if np.any(positive):
        upper = np.minimum(upper, np.min(-intercept[:, positive] / slope[positive], axis=1))
    good = (lower <= upper) & np.all(intercept[:, stationary] <= 0, axis=1)
    for index in np.flatnonzero(good):
        alpha = float((lower[index] + upper[index]) * .5)
        point = vertices[index] + alpha * displacement
        inset = -float(np.max(normals @ point + offsets))
        if inset >= CLEARANCE_M - 1e-12:
            return dict(alpha_from_target_to_pregrasp=alpha,
                        hand_source_vertex_world=point.tolist(), inset_m=inset)
    return None


def world_shapes(scene, native, planes_cache):
    for obj in scene['objects']:
        name, transform = obj['object_id'], np.asarray(obj['world_T_object'])
        for component, vertices in enumerate(native[name]):
            key = name, component
            if key not in planes_cache:
                planes_cache[key] = ConvexHull(vertices).equations
            planes = planes_cache[key]
            world_vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            normals = planes[:, :3] @ transform[:3, :3].T
            offsets = planes[:, 3] - normals @ transform[:3, 3]
            yield name, component, world_vertices.min(0), world_vertices.max(0), normals, offsets


def first_witness(hulls, names, target, displacement, shapes):
    for link, vertices in zip(names, hulls, strict=True):
        vertices = vertices @ target[:3, :3].T + target[:3, 3]
        low = vertices.min(0) + np.minimum(displacement, 0)
        high = vertices.max(0) + np.maximum(displacement, 0)
        for name, component, min_v, max_v, normals, offsets in shapes:
            if np.any(low >= max_v) or np.any(high <= min_v):
                continue
            witness = swept_vertex_witness(vertices, displacement, normals, offsets)
            if witness:
                return dict(witness, body=name, native_component=component, hand_link=link)
    return None


def audit(root):
    import author_resident_prepare as resident

    report = B.read(root/'RESULTS.json')
    B.verify({k:report[k] for k in ('source_model','source_protocol','implementation')})
    protocol = B.read(report['source_protocol']['path'])
    resident.bind_backend({'backend':protocol['inputs']['backend']['path']})
    from asset_adapter import BoyaAssets, CGR_R_REFERENCE, rigid
    from author_hand_clouds import HandCloudLibrary

    assets = BoyaAssets()
    native = native_objects(report['source_model']['path'])
    reference = np.eye(4)
    reference[:3, :3] = CGR_R_REFERENCE
    opening, planes_cache, results = {}, {}, []
    for frame in report['frames']:
        candidates = [m for m in frame['measurements']
                      if m['status'] == 'ground_clear_object_overlap_unresolved']
        if frame['original_pool'] or not candidates:
            continue
        refs = frame['inputs']
        B.verify(refs)
        world = rigid(B.read(refs['observation']['path'])['T_WC'])
        calibration = B.read(refs['calibration']['path'])
        slots = {slot['slot_id']:slot for slot in calibration['slots']}
        library = HandCloudLibrary(refs['surface']['path'])
        shapes = list(world_shapes(B.read(refs['scene']['path']), native, planes_cache))
        with np.load(refs['trace']['path'], allow_pickle=False) as data:
            rows, extra = data['actual_rows17_camera'], data['extra_depth_m']
        for record in candidates:
            idx = record['local_index']
            slot = slots[record['slot']]
            row = rows[idx]
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = row[4:13].reshape(3, 3), row[13:16]
            palm = rigid(rigid(pose) @ reference @ np.linalg.inv(slot['palm_T_reference']))
            axis = pose[:3, 0].astype(float)
            axis /= np.linalg.norm(axis)
            palm[:3, 3] += np.asarray(axis * (float(row[3]) + extra[idx]), dtype=np.float32)
            target = world @ palm
            displacement = -APPROACH_M * (world[:3, :3] @ axis)
            state = library.states[library.report['slot_to_state'][record['slot']]]
            links = [part['link'] for part in state['parts']]
            hulls = opening_geometry(opening, slot, library, assets)
            witness = first_witness(hulls, links, target, displacement, shapes)
            results.append(dict(pair=frame['pair'], raw_row=record['raw_row'], slot=record['slot'], witness=witness))
    assert len(results) == report['empty_pool_branch_counts']['ground_clear_object_overlap_unresolved']
    outcome = dict(status='native_object_vertex_witness_check_complete', source_audit=B.binding(root/'RESULTS.json'),
                   implementation=B.binding(__file__), clearance_m=CLEARANCE_M,
                   tested=len(results), with_interior_witness=sum(bool(item['witness']) for item in results),
                   measurements=results, additional_physics_steps=0)
    B.save(root/'OBJECT_WITNESSES.json', outcome, exclusive=True)
    print({key:value for key,value in outcome.items() if key != 'measurements'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root.resolve())
