"""Full Boya source surfaces for the author's sampled-hand voxel predicate.

Asset generation only, not grasp collection or a physical calibration. All
collision links, including inactive fingers, are retained without convexifying
their triangles. The calibration table remains a computed Boya substitution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from asset_adapter import BoyaAssets, rigid, sha


def full_hand_mesh(assets, q):
    import open3d as o3d

    frames, _ = assets.hand.frames(np.asarray(q, dtype=float))
    vertices, triangles, parts = [], [], []
    offset = 0
    for link, mesh in assets.meshes.items():
        frame = rigid(frames[link])
        points = np.asarray(mesh.vertices) @ frame[:3, :3].T + frame[:3, 3]
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.min() < 0 or faces.max() >= len(points)):
            raise ValueError('Invalid source triangle mesh: ' + link)
        parts.append(dict(link=link, vertices=len(points), triangles=len(faces),
                          vertex_offset=offset, palm_T_link=frame.tolist()))
        vertices.append(points)
        triangles.append(faces + offset)
        offset += len(points)
    if set(assets.meshes) != set(assets.hand.collision_links):
        raise ValueError('Source collision links must all be retained')
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(np.concatenate(vertices))
    result.triangles = o3d.utility.Vector3iVector(np.concatenate(triangles))
    return result, parts


def surface_clouds(mesh, *, seed):
    """Original 300000 uniform triangle-area samples, then 1..5mm downsample.

    Source generators estimate normals too; collision uses positions only.
    Open3D RNG is explicitly seeded for reproducible simulation assets.
    """
    import open3d as o3d

    o3d.utility.random.seed(int(seed))
    cloud = mesh.sample_points_uniformly(300000)
    return {mm: np.asarray(cloud.voxel_down_sample(.001*mm).points).copy()
            for mm in range(1, 6)}


def build(calibration, output, *, gestures=(1, 6, 7, 8), seed=20261114):
    calibration, output = Path(calibration).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh surface asset directory required')
    table = json.loads(calibration.read_text())
    locked = {str(calibration): sha(calibration), str(Path(__file__).resolve()): sha(__file__)}
    for path, entry in table['source_lock']['source_files'].items():
        if sha(path) != entry['sha256']:
            raise ValueError('Source hand asset changed: ' + path)
        locked[path] = entry['sha256']
    assets = BoyaAssets()
    current = assets.source_lock()
    for path, digest in current['readonly_fk_dependencies'].items():
        locked[path] = digest
    if current['source_files'] != table['source_lock']['source_files']:
        raise ValueError('Source assets differ from the calibration table')
    states = [s for s in table['states'] if s['gesture_id'] in gestures]
    if not states or set(s['gesture_id'] for s in states) != set(gestures):
        raise ValueError('Requested hand states are absent')
    if len({(s['gesture_id'], s['sample']) for s in states}) != len(states):
        raise ValueError('Repeated hand state identity')
    output.mkdir(parents=True)
    report = dict(schema='anydex_boya_author_hand_surfaces_v1', status='generating',
        calibration=dict(path=str(calibration), sha256=sha(calibration)),
        source_and_input_sha256=locked, source_lock=current, gestures=list(gestures),
        sampling=dict(points_per_state=300000, voxel_sizes_m=[.001,.002,.003,.004,.005],
                      seed_base=seed, method='Open3D triangle-area uniform surface sampling'),
        states=[], slot_to_state={}, physics_steps=0, CGR_forwards=0,
        calibration_semantics_validated=False, dynamic_hand_model_validated=False)
    started = time.monotonic()
    try:
        for ordinal, state in enumerate(states):
            key = f"g{state['gesture_id']:02d}_s{state['sample']:03d}"
            mesh, parts = full_hand_mesh(assets, state['independent_q'])
            clouds = surface_clouds(mesh, seed=seed+ordinal)
            path = output/(key+'.npz')
            np.savez_compressed(path, **{f'points_{mm}mm': points for mm, points in clouds.items()})
            vertex_bytes = np.ascontiguousarray(np.asarray(mesh.vertices)).tobytes()
            face_bytes = np.ascontiguousarray(np.asarray(mesh.triangles)).tobytes()
            record = dict(key=key, gesture_id=state['gesture_id'], sample=state['sample'],
                independent_q=state['independent_q'], parts=parts, sampled_points=300000,
                seed=seed+ordinal, path=str(path), sha256=sha(path),
                mesh_array_sha256=hashlib.sha256(vertex_bytes+face_bytes).hexdigest(),
                downsampled_counts={str(mm): len(p) for mm,p in clouds.items()})
            report['states'].append(record)
            for slot in table['slots']:
                if (slot['gesture_id'],slot['sample']) == (state['gesture_id'],state['sample']):
                    report['slot_to_state'][slot['slot_id']] = key
            print(json.dumps(dict(state=key, done=len(report['states']), total=len(states))), flush=True)
        if any(sha(path) != digest for path,digest in locked.items()):
            raise ValueError('Source/input changed during surface generation')
        report.update(status='complete', source_and_inputs_unchanged=True,
            total_uniform_samples=300000*len(states), states_count=len(states),
            all_source_collision_links_retained=True, source_meshes_convexified=False)
    except BaseException as exc:
        report.update(status='generation_failed', error=repr(exc))
        raise
    finally:
        report['wall_s'] = time.monotonic()-started
        (output/'MANIFEST.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


class HandCloudLibrary:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.report = json.loads(self.path.read_text())
        if (self.report['schema'] != 'anydex_boya_author_hand_surfaces_v1'
                or self.report['status'] != 'complete'):
            raise ValueError('Complete source surface library required')
        for path,digest in self.report['source_and_input_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Surface source changed: ' + path)
        self.states = {s['key']:s for s in self.report['states']}
        self.cache = {}

    def world_points(self, slot, world_T_palm, *, voxel_mm=3):
        if type(voxel_mm) is not int or voxel_mm not in range(1,6):
            raise ValueError('Author source cloud voxel must be 1..5mm')
        key = self.report['slot_to_state'][slot['slot_id']]
        state = self.states[key]
        if not np.array_equal(state['independent_q'], slot['independent_q']):
            raise ValueError('Requested slot and source surface state differ')
        cache_key = key, voxel_mm
        if cache_key not in self.cache:
            path = Path(state['path'])
            if sha(path) != state['sha256']:
                raise ValueError('Surface cloud bytes changed')
            with np.load(path,allow_pickle=False) as data:
                self.cache[cache_key] = data[f'points_{voxel_mm}mm'].copy()
        palm = rigid(world_T_palm)
        return self.cache[cache_key] @ palm[:3,:3].T + palm[:3,3]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a = p.parse_args()
    result = build(a.calibration,a.output_dir)
    print(json.dumps({k:result[k] for k in ('status','states_count','total_uniform_samples','wall_s')}))
