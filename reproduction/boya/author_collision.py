"""Run the entire pinned author detector with Boya surface-cloud/LUT adapters.

Width expansion, strict width filter, depth translation, recursive sweep and
voxel inclusion all execute inside the original class. Boya's sparse computed
reference table is a declared hand-specific substitution, not a hardware LUT.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import author_protocol as author
from asset_adapter import CGR_R_REFERENCE, rigid, sha

DEPLOYMENT_SOURCE_SHA = '7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797'

PROFILES = {
    'dh3_collection': dict(min_width=.01, voxel=.003, approach=.04, threshold=2),
    'dh3_deployment': dict(min_width=.045, voxel=.003, approach=.05, threshold=0),
}


def lookup_slot(expanded_width,slots):
    if not slots: return None
    width = float(np.clip(expanded_width,min(s['reference_width_m'] for s in slots),
                          max(s['reference_width_m'] for s in slots)))
    allowed = [s for s in slots if abs(s['reference_width_m']-width)<=.002]
    if not allowed: return None
    return min(allowed,key=lambda s:(abs(s['reference_width_m']-width),s['slot_id']))


class TwoFingerGroup:
    """The GraspGroup fields used by the author's detector, preserving row IDs."""
    def __init__(self,rows): self.rows = np.asarray(rows).copy()
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return TwoFingerGroup(self.rows[index].reshape(-1,17))
    @property
    def widths(self): return self.rows[:,1]
    @widths.setter
    def widths(self,value): self.rows[:,1] = value
    @property
    def heights(self): return self.rows[:,2]
    @property
    def depths(self): return self.rows[:,3]
    @property
    def rotation_matrices(self): return self.rows[:,4:13].reshape(-1,3,3)
    @property
    def translations(self): return self.rows[:,13:16]
    @translations.setter
    def translations(self,value): self.rows[:,13:16] = value


class BoyaGroup:
    def __init__(self,ids,gestures,extra,slots):
        self.ids,self.gestures,self.extra = map(np.asarray,(ids,gestures,extra))
        self.slots = slots
        self.mapped = []

    def __len__(self): return len(self.ids)
    def __iter__(self): return iter(self.mapped)
    def __getitem__(self,index):
        return BoyaGroup(self.ids[index],self.gestures[index],self.extra[index],self.slots)

    def graspgroupTR_2_TR(self,two_finger_group,unused_path):
        self.mapped = []
        for i,row in enumerate(two_finger_group.rows):
            slot = lookup_slot(row[1],self.slots[int(self.gestures[i])])
            if slot is None: raise ValueError('Missing Boya LUT entry after width filtering')
            cgr = np.eye(4);cgr[:3,:3] = row[4:13].reshape(3,3);cgr[:3,3] = row[13:16]
            cgr_reference = np.eye(4);cgr_reference[:3,:3] = CGR_R_REFERENCE
            palm = rigid(rigid(cgr) @ cgr_reference @ np.linalg.inv(rigid(slot['palm_T_reference'])))
            self.mapped.append(SimpleNamespace(
                get_grasp_type_with_finger_name=lambda name=slot['slot_id']:name,
                width=slot['reference_width_m'],translation=palm[:3,3],
                rotation_matrix=palm[:3,:3],depth=float(row[3])+float(self.extra[i]),slot=slot,
                palm_before_depth=palm,source_index=int(self.ids[i])))


def detect(rows17,gesture_ids,extra_depth_m,scene_points,*,calibration,library,profile):
    import open3d as o3d

    if profile not in PROFILES: raise ValueError('Explicit collection/deployment profile required')
    author.source('collect_dh3_grasp_data.py')
    if sha(author.ROOT/'robot_dh3.py') != DEPLOYMENT_SOURCE_SHA:
        raise ValueError('Pinned deployment profile source changed')
    bound_calibration = library.report['calibration']
    calibration_path = Path(bound_calibration['path'])
    if (sha(calibration_path) != bound_calibration['sha256']
            or json.loads(calibration_path.read_text()) != calibration):
        raise ValueError('Collision reference table differs from the surface library binding')
    rows = np.asarray(rows17)
    scene = np.asarray(scene_points)
    n = len(rows)
    gestures,extra = np.asarray(gesture_ids),np.asarray(extra_depth_m)
    if (rows.shape != (n,17) or not np.issubdtype(rows.dtype,np.floating)
            or not np.isfinite(rows).all() or (rows[:,1]<=0).any()
            or (rows[:,3]<0).any() or gestures.shape != (n,) or extra.shape != (n,)
            or not np.isin(gestures,[1,6,7,8]).all()
            or not np.isfinite(extra).all() or not np.isclose(extra[:,None],[0.,.01,.02,.03]).any(axis=1).all()
            or scene.ndim != 2 or scene.shape[1] != 3 or not np.isfinite(scene).all()):
        raise ValueError('Finite source GraspGroup rows/scene, four types and discrete extra depths required')
    slots = {g:[s for s in calibration['slots'] if s['gesture_id']==g
                and s['opposing_group']=='ff' and s['self_collision_clear']] for g in (1,6,7,8)}
    mapping_mask = np.array([lookup_slot(float(row[1])*1.7,slots[int(g)]) is not None
                             for row,g in zip(rows,gestures)],dtype=bool)
    source_indices = np.flatnonzero(mapping_mask)
    two = TwoFingerGroup(rows[mapping_mask])
    hand = BoyaGroup(source_indices,gestures[mapping_mask],extra[mapping_mask],slots)
    cls = author.definitions('utils/collision_detector.py',{'ModelFreeCollisionDetectorMultifinger'},
        dict(np=np,o3d=o3d,copy=copy,time=author.time,math=math))['ModelFreeCollisionDetectorMultifinger']
    meshes = {}
    # Source detect expands width once and remaps the LUT after its width gate.
    # Clouds remain in palm coordinates; its original loader adds depth once.
    for row,g in zip(rows[mapping_mask],gestures[mapping_mask]):
        slot = lookup_slot(float(row[1])*1.7,slots[int(g)])
        key = slot['slot_id']+'_'+str(round(slot['reference_width_m']*100,1))
        if key not in meshes:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(library.world_points(slot,np.eye(4),voxel_mm=3))
            meshes[key] = cloud
    cfg = PROFILES[profile]
    detector = cls(scene,voxel_size=.001)
    hand,two,empty_mask,width_mask = detector.detect(hand,two,'',meshes,
        min_grasp_width=cfg['min_width'],VoxelGrid=cfg['voxel'],approach_dist=cfg['approach'],
        collision_thresh=cfg['threshold'],adjust_gripper_centers=False,DEBUG=False)
    accepted = hand.ids[np.asarray(empty_mask,dtype=bool)]
    return dict(accepted_source_indices=accepted, hand=hand,two_finger_group=two,
        collision_admitted=np.asarray(empty_mask,dtype=bool),source_width_mask=width_mask,
        mapping_available_mask=mapping_mask,profile=profile,parameters=cfg,
        mapping_substitution='Boya ff-reference; family clip/nearest2mm computed table',
        original_width_expansion_and_depth_and_voxel_functions=True,
        physics_steps=0,hand_reference_semantics_validated=False)
