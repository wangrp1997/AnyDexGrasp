"""Use the original capture stationarity gate, retaining measured velocities.

The v1 live adapter inadvertently added a .02rad/s instantaneous object cap.
Original settling uses full-window pose stationarity and .5rad/s twist sanity.
This adapter requires that original gate and the independent capture audit;
it never zeroes velocity or changes any original control/collision/lift gate.
"""
from pathlib import Path
import numpy as np
import author_live_world_v3 as ORIGINAL
import author_collection_batch as B
from trial_record import json_sha
from settle_s40_scenes import window_metrics,POLICY


def observed_entry(actual,observed,capture):
    if actual!=observed:raise ValueError('World changed after RGB-D capture')
    capture=Path(capture);audit=B.read(capture/'INDEPENDENT_CAPTURE_AUDIT.json');result=B.read(capture/'RESULTS.json');scene=B.read(capture/'observed_scene.json')
    if audit['status']!='pass'or result['status']!='complete' or scene['scene_id']!=observed['scene_id']:raise ValueError('Original certified settling required before live entry')
    if np.max(np.abs(actual['joint_velocity']))>.02:raise ValueError('Robot moving at observation')
    metrics={}
    with np.load(capture/'initialization_trace.npz',allow_pickle=False)as data:
        for i,row in enumerate(scene['objects']):
            key=row['object_id'];state=observed['objects'][key]
            np.testing.assert_array_equal(data['pose_xyzw'][-1,i],state['pose_xyzw']);np.testing.assert_array_equal(data['velocity'][-1,i],state['velocity'])
            value=window_metrics(data['pose_xyzw'][:,i],data['velocity'][:,i]);metrics[key]=value
            if not value['stable']:raise ValueError('Original settling gate failed')
    return dict(sim_step=actual['sim_step'],state_sha256=json_sha(actual),actor_pose_or_velocity_writes=0,
        capture_audit=B.binding(capture/'INDEPENDENT_CAPTURE_AUDIT.json'),original_stationarity_policy=POLICY,
        source_stationarity=B.binding(Path(__file__).with_name('settle_s40_scenes.py')),recomputed_stationarity=metrics,
        entry_semantics='Original full-window stable capture; exact measured poses and velocities, no extra instantaneous object speed cutoff')


class World(ORIGINAL.World):
    def validate_entry(self):
        self.entry_proof=observed_entry(self.state(),self.observation,self.observation_path.parent)
        if self.capture_metadata()!=self.observed_handles:raise ValueError('Native actors/camera changed after observation')
        self.entry_proof.update(world_builds=self.build_count,boundary_resets=self.boundary_resets,handles=self.observed_handles)
        return self.entry_proof
