"""Pinned author collection, voxel collision and training primitives.

Numerical statements are extracted from the untouched upstream files. No robot
module is imported. These primitives do not accept a Boya calibration as being
validated, start a simulator, collect labels or promote an existing model.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'collect_allegro_grasp_data.py': 'd833b95ab50afcd9924ca6e97b21e6b7f65aa304f9a808599c0cdf48c98b7bbb',
    'collect_dh3_grasp_data.py': '1bcb655fad63a7c2b4f65e29b426b6aa614785d3cea514e0cd290fd05eb8bbcb',
    'utils/collision_detector.py': 'f613f98484e790eb3c3c2337023697fa01a96a5d4020091c8bb98cd60e724ba1',
    'train_multifinger.py': '8f12a094c495ce8bf80fe787e7a583203053aa0e9831288cb8c1d5a7f206fbaf',
    'dataset/multifinger_hand.py': '29e57cb152b65e2c4f84148a8a2e6e2a2cb9247b95387a52b87e3c18d8528e7e',
    'models/minkowski_graspnet.py': '8b92ae7f48be0926d992d209f97345466fd365ed9d63ad4d6a0a3dab0af8e74d',
    'models/loss.py': '07c0d5de79f969e565b944635f7ed9924a98f4eb6eff5aae4faa008a75473aa8',
    'models/minkowski_graspnet_single_point.py': '1463ee5804ec72bafbfc0f469041425092817ff6e6b797923db77120e1167b26',
    'utils/solvers.py': 'a9f2b274f438a1a29c1ae17c22a1f553dbe978354ba7143222a653e17a38f713',
}


def source(name):
    path = ROOT / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise ValueError('Author source changed: ' + name)
    return ast.parse(path.read_text())


def definitions(name, names, scope):
    nodes = [n for n in source(name).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Author definitions missing')
    scope = dict(scope, __name__='isolated_author_protocol')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/name), 'exec'), scope)
    return scope


def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets)


class LocalNumpy:
    """Keep the author's MT19937 draws without touching global NumPy RNG."""
    def __init__(self, rng):
        self.random = rng

    def __getattr__(self, name):
        return getattr(np, name)


def collection_candidates(rows17, features2261, *, reference, reference_type, seed):
    """Author top500/single flip/fixed type/random extra; before hand mapping.

    Input must already have passed the chosen sensor/workspace domain. Original
    hand-specific rig filters are not silently transplanted to Boya. reference_type
    is an author-hand ordinal, not a claim Boya has that author's geometry.
    """
    if reference not in ('allegro', 'dh3'):
        raise ValueError('Explicit allegro or dh3 source reference required')
    max_type = 10 if reference == 'allegro' else 4
    if type(reference_type) is not int or not 1 <= reference_type <= max_type:
        raise ValueError('Invalid source-hand type ordinal')
    rows, features = np.asarray(rows17), np.asarray(features2261)
    if (rows.ndim != 2 or rows.shape[1] != 17 or features.shape != (len(rows), 2261)
            or not np.isfinite(rows).all() or not np.isfinite(features).all()):
        raise ValueError('Finite paired original GraspGroup Nx17/Nx2261 required')
    import torch

    name = f'collect_{reference}_grasp_data.py'
    tree = source(name)
    bodies = [n.body for n in ast.walk(tree) if isinstance(n, ast.While)]
    body = next(b for b in bodies if any(assigns(n, 'source_index') for n in b))
    start = next(i for i, n in enumerate(body) if assigns(n, 'source_index'))
    stop = next(i for i in range(start, len(body)) if assigns(body[i], 'two_fingers_ggarray'))
    rng = np.random.RandomState(seed)
    scope = definitions(name, {'flip_ggarray'}, dict(np=LocalNumpy(rng), copy=copy))
    scope.update(ggarray=rows.copy(), grasp_features=torch.from_numpy(features.copy()),
                 cfgs=SimpleNamespace(grasp_type=reference_type), NUM_OF_ALLEGRO_DEPTH=4)
    exec(compile(ast.Module(body=body[start:stop], type_ignores=[]), str(ROOT/name), 'exec'), scope)
    prefix = 'Allegro' if reference == 'allegro' else 'DH3'
    indices = scope['source_index'][::-1][:500]
    return dict(rows=scope['ggarray'], features=scope['grasp_features'],
                source_indices=indices, if_flip=np.asarray(scope['if_flip'], dtype=bool),
                extra_depth_m=scope[prefix+'_depths'], reference_types=scope[prefix+'_types'],
                rng_state_after=rng.get_state(), source_sha256=PINS[name],
                boundary='Author pre-mapping policy only; no Boya hand/geometry/arm admission')


def voxel_collision(scene_points, hand_points_world, approach_axis, *,
                    hand_voxel_m, approach_distance_m=.08, collision_threshold=2):
    """Author recursive approach sweep/voxel inclusion; mapped hand supplied.

    Hand points must be the independently generated open-hand surface cloud,
    already transformed once into scene coordinates with raw+extra depth.
    This does not implement the width/LUT/mesh-cloud generation steps.
    """
    import open3d as o3d

    scene = np.asarray(scene_points, dtype=float)
    hand = np.asarray(hand_points_world, dtype=float)
    axis = np.asarray(approach_axis, dtype=float)
    if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in (scene, hand)):
        raise ValueError('Finite Nx3 source surface and scene clouds required')
    if not len(hand) or axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.):
        raise ValueError('Nonempty hand cloud and unit approach axis required')
    if not (np.isfinite(hand_voxel_m) and hand_voxel_m > 0 and
            np.isfinite(approach_distance_m) and 0 <= approach_distance_m <= .2):
        raise ValueError('Explicit finite positive voxel and bounded approach distance required')
    if type(collision_threshold) is not int or collision_threshold < 0:
        raise ValueError('Nonnegative integer scene-point threshold required')
    cls = next(n for n in source('utils/collision_detector.py').body
               if isinstance(n, ast.ClassDef) and n.name == 'ModelFreeCollisionDetectorMultifinger')
    detect = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'detect')
    loop = next(n for n in detect.body if isinstance(n, ast.For))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For))
    stop = next(i for i, n in enumerate(loop.body) if assigns(n, 'output')) + 1
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(hand)
    scope = dict(np=np, o3d=o3d, copy=copy, time=time, meshes_pointclouds=cloud,
                 grasp_direction=axis, ps=o3d.utility.Vector3dVector(scene),
                 VoxelGrid=float(hand_voxel_m), approach_dist=float(approach_distance_m))
    exec(compile(ast.Module(body=loop.body[start:stop], type_ignores=[]),
                 str(ROOT/'utils/collision_detector.py'), 'exec'), scope)
    mask = np.asarray(scope['output'], dtype=bool)
    return dict(admitted=int(mask.sum()) <= collision_threshold, included_scene_points=int(mask.sum()),
                inclusion_mask=mask, scene_points=len(scene), collision_threshold=collision_threshold,
                swept_points_before_downsample=len(scope['meshes_pointclouds'].points),
                source_sha256=PINS['utils/collision_detector.py'],
                physics_steps=0, hand_surface_generation_validated=False)


def select_training_records(order, *, sample_cap):
    """Author first100 selection or explicit larger-data extension, no balancing.

    Selection is before shuffle; callers shuffle the selected IDs using their
    fixed DataLoader RNG. Singleton handling belongs to original_epoch.
    """
    order = np.asarray(order)
    if order.ndim != 1 or len(set(map(str, order))) != len(order):
        raise ValueError('Unique ordered record IDs required')
    if type(sample_cap) is not int or sample_cap not in (100, 300, 1000):
        raise ValueError('Explicit author100 or extended300/1000 sample cap required')
    selected = order[:sample_cap].copy()
    return selected


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


def original_epoch(model, optimizer, criterion, batches, *, scheduler, gesture, epoch=0):
    """Execute the actual author train_one_epoch, including singleton skip.

    Optimizer/model/loss configuration is supplied by a future frozen training
    protocol. This entry does not select checkpoints or certify data semantics.
    """
    logs = []
    scope = definitions('train_multifinger.py', {'train_one_epoch'},
        dict(np=np, time=time, multifinger_net=model, optimizer=optimizer, criterion=criterion,
             lr_scheduler=scheduler,
             TRAIN_DATALOADER=batches, convert_data_to_gpu=lambda d:d,
             TRAIN_WRITER=NullWriter(), log_string=logs.append,
             NUM_MULTIFINGER_TYPE=1, GRIPPER_TYPE='Boya', MULTIFINGER_TYPE=gesture, EPOCH_CNT=epoch))
    scope['train_one_epoch']()
    return logs
