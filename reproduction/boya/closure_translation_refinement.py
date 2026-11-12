"""Finite, opt-in palm translation from visible points and frozen closing FK.

This is a geometric diagnostic, not a grasp-success predictor. Source pad
normals point out of the finger surface; points behind that surface receive a
soft penalty. Occluded object surfaces are not reconstructed. No object mesh,
object pose, recorded contacts, execution outcome or learned model is read.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


PROGRESS = np.linspace(0., 1., 11)
GRID_M = (-.01, -.005, 0., .005, .01)


def protocol():
    return dict(world_xyz_grid_m=list(GRID_M), candidate_count=125,
                closure_progress=PROGRESS.tolist(), roi_radius_m=.1,
                roi_minimum_world_z_m=.002, distance_clip_m=.03,
                rear_distance_squared_multiplier=3., translation_penalty=.1,
                objective='min_t max(d_thumb(t),d_original_opponent(t)) + 0.1*norm(delta)',
                pad_distance='min_x sqrt(norm(x-pad)^2 + 3*max(0,-normal_dot(x-pad))^2), clipped at 0.03m',
                approach_length_m=.1, approach_samples=21,
                open_hand_table_minimum_m=.0005, hull_interior_tolerance_m=1e-7,
                tie_break='objective, displacement norm, world XYZ lexicographic',
                order='zero displacement first, then world XYZ lexicographic',
                closure_table_policy='reported nominal FK only; no new hard closure gate',
                limited_observation='Visible points only; no hidden-surface or physical-support guarantee')


def _finite(value, shape, label):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{label}: expected finite shape {shape}')
    return a.copy()


def translation_grid():
    zero = (0., 0., 0.)
    return np.asarray([zero] + [x for x in itertools.product(GRID_M, repeat=3) if x != zero])


def _open_geometry(palm, approach, cloud, link_vertices):
    """Cache precisely the hull representation used by build_trial_pool.geometry."""
    equations = []
    low = np.inf
    for vertices in link_vertices:
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
            raise ValueError('Every original collision link needs finite 3D vertices')
        hull = ConvexHull(v)
        equations.append((v.min(0), v.max(0), hull.equations))
        world = v @ palm[:3, :3].T + palm[:3, 3]
        low = min(low, float(world[:, 2].min()), float(world[:, 2].min() - .1 * approach[2]))
    if not equations:
        raise ValueError('All original hand collision links must be supplied')
    direction = approach @ palm[:3, :3]

    def check(delta):
        minimum = float(low + delta[2])
        if minimum < .0005:
            return dict(ok=False, reason='table', minimum_height_m=minimum)
        local = (cloud - palm[:3, 3] - delta) @ palm[:3, :3]
        for distance in np.linspace(0., .1, 21):
            points = local + distance * direction
            for a, b, eq in equations:
                subset = points[np.all((points >= a) & (points <= b), axis=1)]
                if len(subset) and np.any(np.all(subset @ eq[:, :3].T + eq[:, 3] < -1e-7, axis=1)):
                    return dict(ok=False, reason='visible_scene_point_in_hand_hull',
                                minimum_height_m=minimum, approach_distance_m=float(distance))
        return dict(ok=True, minimum_height_m=minimum, approach_samples=21)
    return check


def simultaneous_pad_distances(points, normals, cloud, delta):
    """Return one clipped distance for each of two pads at each shared time."""
    difference = cloud[None, None, :, :] - points[:, :, None, :] - delta
    projection = np.sum(difference * normals[:, :, None, :], axis=-1)
    distance2 = np.sum(difference * difference, axis=-1) + 3. * np.maximum(0., -projection) ** 2
    return np.minimum(np.sqrt(distance2.min(axis=2)), .03)


def refine_translation(palm, approach, cloud_world, pad_points_world,
                       pad_normals_world, open_link_vertices_palm):
    """Pure array core. Input pad order is thumb then the original opponent.

    ``open_link_vertices_palm`` must include every collision link, including
    inactive fingers. Closure is sampled at the fixed eleven shared times.
    No state or input array is mutated.
    """
    palm = _finite(palm, (4, 4), 'palm')
    if (not np.allclose(palm[3], [0., 0., 0., 1.], atol=2e-6, rtol=0)
            or not np.allclose(palm[:3, :3].T @ palm[:3, :3], np.eye(3), atol=2e-6, rtol=0)
            or abs(np.linalg.det(palm[:3, :3]) - 1.) > 2e-6):
        raise ValueError('A proper rigid palm transform is required')
    approach = _finite(approach, (3,), 'approach')
    if abs(np.linalg.norm(approach) - 1.) > 1e-8:
        raise ValueError('A unit approach direction is required')
    cloud = np.asarray(cloud_world, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError('Finite Nx3 visible world points required')
    points = _finite(pad_points_world, (11, 2, 3), 'closing pad points')
    normals = _finite(pad_normals_world, (11, 2, 3), 'closing pad normals')
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(length < 1e-10):
        raise ValueError('Pad normals must be nonzero')
    normals /= length
    center = points[0].mean(axis=0)
    roi_mask = (cloud[:, 2] > .002) & (np.linalg.norm(cloud - center, axis=1) <= .1)
    roi = cloud[roi_mask]
    check = _open_geometry(palm, approach, cloud, open_link_vertices_palm)
    rows = []
    for index, delta in enumerate(translation_grid()):
        gate = check(delta)
        row = dict(index=index, delta_world_m=delta.tolist(), norm_m=float(np.linalg.norm(delta)),
                   geometry=gate, objective_m=None, matched_progress=None,
                   side_distance_by_progress_m=None)
        if len(roi):
            distances = simultaneous_pad_distances(points, normals, roi, delta)
            balanced = distances.max(axis=1)
            matched = int(np.argmin(balanced))
            row.update(objective_m=float(balanced[matched] + .1 * np.linalg.norm(delta)),
                       matched_progress=float(PROGRESS[matched]),
                       side_distance_by_progress_m=distances.tolist())
        rows.append(row)
    eligible = [row for row in rows if row['geometry']['ok'] and row['objective_m'] is not None]
    selected = min(eligible, key=lambda row: (row['objective_m'], row['norm_m'], row['delta_world_m'])) if eligible else rows[0]
    report = dict(schema='anydex_boya_closure_translation_refinement_v1', protocol=protocol(),
                  roi_center_world_m=center.tolist(), visible_point_count=len(cloud), roi_point_count=len(roi),
                  candidates=rows, selected_index=selected['index'],
                  selected_delta_world_m=selected['delta_world_m'],
                  selected_geometry_ok=selected['geometry']['ok'],
                  changed=bool(selected['norm_m'] > 0),
                  status='selected' if eligible else ('no_visible_roi' if not len(roi) else 'no_geometrically_admitted_translation'),
                  physics_steps=0, training_updates=0, model_forwards=0,
                  nominal_simultaneous_contact_is_not_required=True,
                  source_object_geometry_read=False, execution_outcomes_read=False)
    return np.asarray(selected['delta_world_m']), report


def prepare_refinement(job):
    """Read only whitelisted saved inputs; build FK and call the pure core.

    Accepts a job dictionary or its JSON path. Even when a dictionary contains
    object poses, meshes, past executions or labels, those fields are unused.
    """
    from asset_adapter import BoyaAssets, rigid
    from build_trial_pool import geometry

    inputs = {}

    def freeze(path):
        path = Path(path).resolve()
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    if not isinstance(job, Mapping):
        job = json.loads(freeze(job).read_text())
    selected_job = {key: str(job[key]) for key in ('candidate_dir', 'hand_urdf', 'hand_scene', 'package')}
    pool = Path(selected_job['candidate_dir'])
    generation = json.loads(freeze(pool / 'generation.json').read_text())
    if generation.get('complete') is not True or len(generation['records']) != 1:
        raise ValueError('Exactly one frozen complete candidate is required')
    saved_path = freeze(pool / 'candidates.npz')
    if inputs[str(saved_path)] != generation['candidates_sha256']:
        raise ValueError('Frozen candidates changed')
    with np.load(saved_path, allow_pickle=False) as saved:
        names = tuple(saved['independent_motor_names'].tolist())
        arrays = {key: saved[key].copy() for key in ('proposal_q', 'proposal_q_close', 'proposal_palm', 'proposal_approach_direction')}
    if any(len(value) != 1 for value in arrays.values()):
        raise ValueError('Translation cannot choose a different source candidate')
    opened, closed = arrays['proposal_q'][0], arrays['proposal_q_close'][0]
    palm = rigid(arrays['proposal_palm'][0])
    approach = arrays['proposal_approach_direction'][0]
    observation = json.loads(freeze(generation['source_observation']).read_text())
    camera = rigid(observation['T_WC'])
    with np.load(freeze(Path(generation['source_frame']) / 'representation_outputs.npz'), allow_pickle=False) as saved:
        cloud_camera = saved['sparse_input_XYZ_features'].copy()
    cloud = cloud_camera @ camera[:3, :3].T + camera[:3, 3]
    assets = BoyaAssets(urdf=selected_job['hand_urdf'], scene=selected_job['hand_scene'], package=selected_job['package'])
    from canondex.grasp.template_gripper_closure import TwoSidedClosure
    family = assets.families[generation['gesture_id']]
    if names != tuple(assets.hand.names) or not np.array_equal(closed, family.nominal):
        raise ValueError('Frozen original hand command mapping differs')
    row = generation['records'][0]
    assignment = row['jaw_assignment']
    if assignment['thumb'] != 'th' or assignment['opponent'] not in family.active_groups:
        raise ValueError('Original thumb/opponent assignment is required')
    if 'candidate_record' in job and job['candidate_record']['jaw_assignment'] != assignment:
        raise ValueError('Job and saved candidate have different pad assignments')
    contacts = [assets.reference_contacts[key] for key in ('th', assignment['opponent'])]
    normal_audit = []
    for contact in contacts:
        # The reference seed explicitly states outward-normal orientation.
        n = np.asarray(contact['normal_link'])
        seed = np.asarray(contact['seed']['outward_normal_link'])
        cosine = float(n @ seed / (np.linalg.norm(n) * np.linalg.norm(seed)))
        if cosine <= 0:
            raise ValueError('Annotated pad normal opposes its source outward direction')
        normal_audit.append(dict(group=contact['group'], outward_seed_cosine=cosine,
                                 annotation_status=contact.get('status')))
    closure = TwoSidedClosure.from_template(family, opened, closed)
    pad_points, pad_normals, closure_low = [], [], []
    for progress in PROGRESS:
        q = closure.command(progress, progress)
        points, normals = family.hand.evaluate(q, [c['collision_link'] for c in contacts],
                    [c['point_link_m'] for c in contacts], [c['normal_link'] for c in contacts], jacobian=False)[:2]
        pad_points.append(points @ palm[:3, :3].T + palm[:3, 3])
        pad_normals.append(normals @ palm[:3, :3].T)
        frames, _ = assets.hand.frames(q)
        minimum = np.inf
        for name, mesh in assets.meshes.items():
            frame = frames[name]
            vertices = mesh.vertices @ frame[:3, :3].T + frame[:3, 3]
            minimum = min(minimum, float((vertices @ palm[:3, :3].T + palm[:3, 3])[:, 2].min()))
        closure_low.append(minimum)
    frames, _ = assets.hand.frames(opened)
    vertices = [mesh.vertices @ frames[name][:3, :3].T + frames[name][:3, 3] for name, mesh in assets.meshes.items()]
    delta, report = refine_translation(palm, approach, cloud, np.asarray(pad_points), np.asarray(pad_normals), vertices)
    # Recheck baseline and selected translation using the existing source function.
    gate_rechecks = []
    for index in dict.fromkeys([0, report['selected_index']]):
        candidate = report['candidates'][index]
        shifted = palm.copy()
        shifted[:3, 3] += candidate['delta_world_m']
        original = geometry(assets, opened, shifted, cloud, approach)
        if original['ok'] != candidate['geometry']['ok'] or original.get('reason') != candidate['geometry'].get('reason'):
            raise ValueError('Cached geometry disagrees with original geometry gate')
        gate_rechecks.append(dict(index=index, original_geometry=original))
    lock = assets.source_lock()
    for path, meta in lock['source_files'].items():
        inputs[path] = meta['sha256']
    inputs.update(lock['readonly_fk_dependencies'])
    freeze(__file__)
    freeze(Path(__file__).with_name('asset_adapter.py'))
    freeze(Path(__file__).with_name('build_trial_pool.py'))
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Refinement input changed: ' + path)
    report.update(input_sha256=inputs, whitelisted_job=selected_job,
                  original_source_candidate_index=row['source_candidate_index'],
                  original_gesture_id=generation['gesture_id'], original_jaw_assignment=assignment,
                  normal_orientation_audit=normal_audit, original_gate_rechecks=gate_rechecks,
                  nominal_closure_minimum_world_z_m=closure_low,
                  shifted_nominal_closure_minimum_world_z_m=(np.asarray(closure_low) + delta[2]).tolist(),
                  original_palm=palm.tolist(), original_open_q=opened.tolist(), original_close_q=closed.tolist(),
                  pad_points_world_by_progress=np.asarray(pad_points).tolist(),
                  pad_normals_world_by_progress=np.asarray(pad_normals).tolist(),
                  input_scope='Hand assets, one frozen candidate, camera extrinsic and single visible RGBD-derived point cloud only')
    return delta, report


# Short name retained for the parent driver contract.
refine = prepare_refinement
