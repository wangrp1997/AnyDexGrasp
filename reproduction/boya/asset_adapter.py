"""Read-only Boya assets and explicit CGR-to-palm geometry, without a hand head.

CanonDex supplies source FK, template metadata and its structural exclusions.
Its sampling, experience ranking, control and success results are not imported.
Reference separations are NOT object-diameter or maximum hand-opening claims.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONDEX = Path('/home/wangrenpeng/CanonDex')
DEFAULT_URDF = Path('/home/wangrenpeng/tendon-spin/assets/hand/boya.urdf')
DEFAULT_SCENE = Path('/home/wangrenpeng/tendon-spin/assets/grasp/scene.xml')
PACKAGE_REL = Path('assets/gestures/boya/packages/research_v3_five_finger/package.json')
# Boya reference axes: +X thumb -> opponent, +Z approach.
# CGR axes: +X approach, +Y jaw closing, +Z cross/height axis.
CGR_R_REFERENCE = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rigid(value, *, tolerance=2e-6):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=tolerance, rtol=0)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=tolerance, rtol=0)
            or abs(np.linalg.det(a[:3, :3])-1.) > tolerance):
        raise ValueError('A finite proper rigid transform is required')
    return a.copy()


def cgr_reference_pose(candidate, world_T_camera, *, insertion_m=0.):
    """Map a saved 15-column parser row to a declared contact-reference frame.

The raw two-finger depth is the +X contact-plane offset from its base frame.
Additional hand insertion is explicit and defaults to ZERO. This is a Boya
geometric calibration convention, not the unavailable physical calibration.
No DH3 actuator, depth adjustment, width clamp or learned hand score is used.
"""
    a = np.asarray(candidate, dtype=float)
    if (a.shape != (15,) or not np.isfinite(a).all() or a[1] <= 0
            or a[2] < 0 or not np.isfinite(insertion_m)):
        raise ValueError('Finite [score,width,depth,R9,t3] parser row required')
    camera_T_cgr = np.eye(4)
    camera_T_cgr[:3, :3] = a[3:12].reshape(3, 3)
    camera_T_cgr[:3, 3] = a[12:15]
    # The original parser stores float32 rotations. Reject invalid input first,
    # then remove only round-off so the existing strict SE(3) arm API accepts it.
    rigid(camera_T_cgr)
    u, _, vh = np.linalg.svd(camera_T_cgr[:3, :3])
    camera_T_cgr[:3, :3] = u @ vh
    cgr_T_ref = np.eye(4)
    cgr_T_ref[:3, :3] = CGR_R_REFERENCE
    cgr_T_ref[0, 3] = a[2] + insertion_m
    return rigid(rigid(world_T_camera) @ rigid(camera_T_cgr) @ cgr_T_ref)


def map_slot(candidate, world_T_camera, slot, *, insertion_m=0.):
    world_T_ref = cgr_reference_pose(candidate, world_T_camera, insertion_m=insertion_m)
    palm_T_ref = rigid(slot['palm_T_reference'])
    return rigid(world_T_ref @ np.linalg.inv(palm_T_ref))


class BoyaAssets:
    def __init__(self, canondex=DEFAULT_CANONDEX, urdf=DEFAULT_URDF,
                 scene=DEFAULT_SCENE, package=None):
        self.canondex, self.urdf, self.scene = map(lambda x: Path(x).resolve(), (canondex, urdf, scene))
        self.package = Path(package).resolve() if package else self.canondex/PACKAGE_REL
        for p in (self.urdf, self.scene, self.package):
            if not p.is_file():
                raise FileNotFoundError(p)
        sys.path.insert(0, str(self.canondex))
        module = importlib.import_module('canondex.grasp.template_gripper_adapter')
        actual_root = Path(module.__file__).resolve().parents[2]
        if actual_root != self.canondex:
            raise ValueError('Loaded CanonDex module differs from the requested source')
        self.family_class = module.TemplateFamily
        self.reference_path = self.canondex/'assets/gestures/boya/templates/five_finger_wrap.json'
        refs = json.loads(self.reference_path.read_text())
        if refs['model_urdf_sha256'] != sha(self.urdf):
            raise ValueError('Distal pad references differ from the source hand')
        self.reference_contacts = {c['group']: c for c in refs['contacts'] if c['group'] in ('th','ff','mf','rf','lf')}
        self.package_data = json.loads(self.package.read_text())
        self.families, self.excluded = {}, []
        for entry in self.package_data['entries']:
            groups = {c['group'] for c in entry['template']['contacts']}
            if 'th' not in groups or not (groups & {'ff', 'mf', 'rf', 'lf'}):
                self.excluded.append(dict(gesture_id=entry['gesture_id'],
                    reason='No annotated thumb-versus-opponent reference; not inferred or fabricated'))
                continue
            f = self.family_class.load(self.package, self.urdf, self.scene, entry['gesture_id'])
            self.families[f.gesture_id] = f
        self.hand = next(iter(self.families.values())).hand
        # All links, including inactive fingers, stay in every geometry query.
        self.meshes = {n: self.hand.mesh(n) for n in self.hand.collision_links}
        clearance = importlib.import_module('canondex.grasp.collision_geometry')
        self.clearance = clearance.SourceHullClearance(self.urdf, self.scene)

    def source_lock(self):
        files = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size)
                 for p in (self.urdf, self.scene, self.package, self.reference_path)}
        for n in self.hand.collision_links:
            # mesh() has already populated source checksums and supports exact path lookup.
            link = self.hand.fk.robot.find(f"link[@name='{n}']/collision/geometry/mesh")
            p = (self.urdf.parent/link.get('filename')).resolve()
            files[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if not name.startswith('canondex') or not getattr(module, '__file__', None):
                continue
            p = Path(module.__file__).resolve()
            if p.is_file() and p.is_relative_to(self.canondex):
                imported[str(p)] = sha(p)
        revision = subprocess.check_output(['git', '-C', str(self.canondex), 'rev-parse', 'HEAD'], text=True).strip()
        return dict(reference_datum='source distal pad pairs; CGR depth is referenced to this plane', source_files=files, readonly_fk_dependencies=imported, canondex_revision=revision,
                    external_asset_root=str(self.urdf.parent), asset_copy=False,
                    source_files_modified=False, independent_motor_names=list(self.hand.names),
                    lower_rad=self.hand.lower.tolist(), upper_rad=self.hand.upper.tolist(),
                    mimics=self.hand.mimics, collision_links=list(self.hand.collision_links),
                    wrist_coordinates_separate=True, wrist_control_in_this_stage=False)

    def calibrate(self, samples=33):
        if type(samples) is not int or samples < 3:
            raise ValueError('At least three finite calibration samples are required')
        slots, states = [], []
        for gesture_id, family in self.families.items():
            for i, fraction in enumerate(np.linspace(0., 1., samples)):
                q = family.motor_state(float(fraction), float(fraction))
                report = self.clearance.check(family.hand.full_state(q, family.wrist), margin_m=0.)
                state = dict(gesture_id=gesture_id, sample=i, fraction=float(fraction),
                    independent_q=q.tolist(), close_q=family.nominal.tolist(), wrist=family.wrist,
                    self_collision_clear=report['ok'], violations=report['violations'])
                states.append(state)
                contacts = [self.reference_contacts[g] for g in family.active_groups]
                for c in contacts:
                    family.hand.mesh(c['collision_link'])
                    if family.hand.mesh_hashes[c['collision_link']] != c['surface']['mesh_sha256']:
                        raise ValueError('Distal pad source mesh changed')
                points,normals = family.hand.evaluate(q,[c['collision_link'] for c in contacts],
                    [c['point_link_m'] for c in contacts],[c['normal_link'] for c in contacts],jacobian=False)[:2]
                thumb = next(j for j,c in enumerate(contacts) if c['group']=='th')
                n = normals[thumb]/np.linalg.norm(normals[thumb])
                approach = np.array([0.,1.,0.]);approach -= (approach@n)*n
                if np.linalg.norm(approach)<1e-6:
                    approach=np.eye(3)[int(np.argmin(abs(n)))];approach-=(approach@n)*n
                approach/=np.linalg.norm(approach)
                for other,c in enumerate(contacts):
                    if other==thumb:continue
                    delta=points[other]-points[thumb];width=float(np.linalg.norm(delta));x=delta/width
                    z=approach-(approach@x)*x;z/=np.linalg.norm(z)
                    frame=np.eye(4);frame[:3,:3]=np.column_stack((x,np.cross(z,x),z))
                    frame[:3,3]=(points[other]+points[thumb])/2
                    slots.append(dict(slot_id=f'g{gesture_id}:distal_{i:03d}:{c["group"]}',**state,
                        reference_kind='source_distal_pads',opposing_group=c['group'],
                        reference_width_m=width,palm_T_reference=frame.tolist()))
        return dict(schema='anydex_boya_distal_pad_calibration_v2', slots=slots, states=states,
            excluded_templates=self.excluded, samples_per_gesture=samples,
            source_lock=self.source_lock(), hand_score_head=None, new_training_steps=0,
            controller_executed=False, physics_steps=0, hardware_connections=0,
            width_semantics='Annotated thumb-to-single-opponent pad separation, NOT maximum opening',
            approach_semantics='Source thumb tangent projected orthogonal to the closing reference',
            inactive_fingers_retained=True, source_structural_exclusions_retained=True,
            calibration_type='Computed mesh/FK reference table, not hardware measurement')
