"""Compare original DH3 and Boya wrist geometry on the same saved CGR rows."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

import author_collection_batch as B


def dh3_geometry(keys):
    import open3d as o3d
    import pybullet as bullet
    from transforms3d.quaternions import quat2mat
    from reproduction.runtime.generate_dh3_collision_assets import original_inputs, select

    source = B.ROOT/'generate_mesh_and_pointcloud'
    table_path = source/'dh3_urdf/width_12D_angle_2D_angle.json'
    inputs = original_inputs(table_path)
    table = B.read(table_path)['pose1']
    hand_path = B.ROOT/'ur_toolbox/ur_toolbox/robot/DH3/DH3_grasp.py'
    hand_types = select(hand_path, {'grasp_types'}, {})['grasp_types']
    scope = dict(np=np, copy=copy, json=json, math=math, os=os, o3=o3d, quat2mat=quat2mat,
                 grasp_types=hand_types, tqdm=lambda values: values)
    scope.update(select(source/'angles2stl.py', {'DH3Angles2STLs'}, scope))
    geometry = select(source/'recover_dh3_hand_to_stl.py',
                      {'get_meshes', 'get_mesh', 'joint_mesh_mapping', 'four_meta_to_matrix'}, scope)
    vertices = {}

    def load(urdf):
        hand = bullet.loadURDF(urdf, useFixedBase=True)
        if bullet.getNumJoints(hand) != 12:
            raise ValueError('Original DH3 joint table requires 12 slots')
        return bullet, hand

    def capture(pose, slot, mesh, output):
        vertices[str(np.round(slot[12], 1))] = np.asarray(mesh.vertices).copy()

    def forbid_step(*args, **kwargs):
        raise RuntimeError('Static geometry audit cannot integrate physics')

    geometry.update(open_pybullet=load, save_stl_and_pointcloud=capture)
    old_step = bullet.stepSimulation
    client = bullet.connect(bullet.DIRECT)
    if client != 0:
        raise RuntimeError('This audit requires a dedicated first DIRECT client')
    try:
        bullet.stepSimulation = forbid_step
        angles = [table[key]['12d'] + [float(key)] for key in sorted(keys)]
        urdf = source/'dh3_urdf/dh3_urdf/urdf/dh3_urdf.urdf'
        geometry['get_meshes']([angles], str(urdf.parent.parent/'meshes')+'/', '', '', str(urdf), False, False)
    finally:
        bullet.stepSimulation = old_step
        bullet.disconnect(client)
    if set(vertices) != keys:
        raise ValueError('Original DH3 source geometry is missing a lookup width')
    return table, vertices, inputs


def width_key(width):
    return str(np.round(np.clip(width * 1.7, .01, .099)*100, 1))


def bottom(vertices, frame):
    return float((vertices @ frame[2, :3]).min() + frame[2, 3])


def audit(source, output, numbers):
    import author_resident_prepare as resident

    protocol = B.read(source/'PROTOCOL.json')
    B.verify(protocol['sources']); B.verify(protocol['inputs'])
    resident.bind_backend({'backend': protocol['inputs']['backend']['path']})
    from asset_adapter import BoyaAssets, CGR_R_REFERENCE, rigid
    from author_hand_clouds import full_hand_mesh

    inputs, draws, widths = {}, [], set()
    for number in numbers:
        pair = source/'pairs'/number
        start = B.read(pair/'SHARED_START.json')
        directory = source/'scenes'/f"{start['indices'][0]:02d}"/'contact'
        bundle = B.read(directory/'mapping/BUNDLE.json')
        prepared = B.read(directory/'pipeline/collection/PREPARED.json')
        bindings = dict(observation=start['observation'], calibration=bundle['calibration'],
                        surfaces=bundle['surface_manifest'], trace=prepared['trace'])
        B.verify(bindings)
        inputs[number] = bindings
        cal = B.read(bindings['calibration']['path'])
        branches = cal['contact_mapping']['branch_slots']
        slots = {slot['slot_id']:slot for slot in cal['slots']}
        with np.load(bindings['trace']['path'], allow_pickle=False) as arrays:
            rows = arrays['actual_rows17_camera'].copy(); extra = arrays['extra_depth_m'].copy()
            row_ids = arrays['source_indices'].copy()
        widths.update(width_key(rows[slots[key]['source_local_index'], 1]) for key in branches)
        observation = B.read(bindings['observation']['path'])
        draws.append((number, np.asarray(observation['T_WC']), rows, extra, row_ids, slots,
                      branches, B.read(bindings['surfaces']['path'])))

    table, original_vertices, original_inputs = dh3_geometry(widths)
    assets = BoyaAssets()
    local_vertices = {}
    reference = np.eye(4); reference[:3, :3] = CGR_R_REFERENCE
    reports = []
    for number, world_camera, rows, extra, row_ids, slots, branches, surface in draws:
        captured = []
        state_by_key = {r['key']:r for r in surface['states']}
        for branch in branches:
            slot = slots[branch]; i = slot['source_local_index']; row = rows[i]
            source_slot = slots[slot['source_surface_slot_id']]
            state = state_by_key[surface['slot_to_state'][source_slot['slot_id']]]
            if state['key'] not in local_vertices:
                mesh, _ = full_hand_mesh(assets, slot['independent_q'])
                v = np.ascontiguousarray(np.asarray(mesh.vertices)); t = np.ascontiguousarray(np.asarray(mesh.triangles))
                if hashlib.sha256(v.tobytes()+t.tobytes()).hexdigest() != state['mesh_array_sha256']:
                    raise ValueError('Boya full source triangles differ from the stored hand surface geometry')
                local_vertices[state['key']] = v.copy()
            mesh = local_vertices[state['key']]
            source_cgr = np.eye(4); source_cgr[:3, :3] = row[4:13].reshape(3, 3); source_cgr[:3, 3] = row[13:16]
            world_cgr = world_camera @ rigid(source_cgr)
            delta = world_cgr[:3, 0] / np.linalg.norm(world_cgr[:3, 0]) * (row[3]+extra[i])
            dh3_width = width_key(row[1]); entry = table[dh3_width]
            original_frame = np.eye(4); original_frame[:3, :3] = entry['rotation']; original_frame[:3, 3] = entry['translation']
            dh3_target = world_cgr @ np.linalg.inv(original_frame)
            open_ref = np.asarray(source_slot['palm_T_reference'])
            preclose_ref = np.asarray(slot['palm_T_reference'])
            boya_target = world_cgr @ reference @ np.linalg.inv(preclose_ref)
            boya_open = world_cgr @ reference @ np.linalg.inv(open_ref)
            # Compare source geometry at the grip reference and after the same source depth translation.
            dz = float(delta[2])
            dh3_min = bottom(original_vertices[dh3_width], dh3_target)
            boya_min = bottom(mesh, boya_target)
            open_min = bottom(mesh, boya_open)
            captured.append(dict(source_row=int(row_ids[i]), opponent=slot['opposing_group'],
                raw_cgr_height_m=float(world_cgr[2, 3]), depth_z_m=dz,
                dh3_before_depth_m=dh3_min, dh3_at_target_m=dh3_min+dz,
                boya_before_depth_m=boya_min, boya_at_target_m=boya_min+dz,
                boya_open_reference_before_depth_m=open_min, boya_open_reference_at_target_m=open_min+dz))
        reports.append(dict(pair=number, branches=captured))
        print(number, 'full source meshes', len(captured), flush=True)
    B.verify({k:v for bindings in inputs.values() for k,v in [(b['path'],b) for b in bindings.values()]})
    B.save(output, dict(status='complete', inputs=inputs, dh3_original_inputs=original_inputs,
        dh3_lookup=B.binding(B.ROOT/'generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json'),
        compared_dh3_widths=len(widths), source=B.binding(__file__), original_boya_meshes_verified=len(local_vertices),
        draws=reports, plane_z_m=0., full_mesh_endpoint_comparison=True, physics_steps=0, new_cgr_forwards=0), exclusive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('pairs', nargs='+')
    args = parser.parse_args()
    audit(args.source.resolve(), args.output.resolve(), args.pairs)
