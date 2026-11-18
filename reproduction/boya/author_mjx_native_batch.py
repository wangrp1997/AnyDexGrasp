"""Persistent arm/hand worlds using one native MJX-Warp control graph.

The interface and saved observations match the frozen batch. CPU kinematics
are updated when a lane is observed; unused CPU mirrors need no per-tick work.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np

from author_mjx_live_batch_v4 import Batch as FrozenBatch,clock_table,decode_contacts,photograph
from author_mjx_native_graph import NativeControlGraph


class Batch(FrozenBatch):
    def __init__(self,plan,names,initial,scenes,*,count,max_ticks):
        import mujoco
        import warp as wp
        from author_mujoco_persistent_world import World
        from author_mjx_compatibility import apply_gpu_ccd_tolerance
        began=time.monotonic();self.count=count;self.ticks=0;self.max_ticks=max_ticks;self.wp=wp
        self.expected_clock=clock_table(max_ticks);self.events=[];self.step_s=0.;self.observer_s=0.
        self.journal=None;self.cost_journal=None;self.packet=None
        self.base=World(plan,names,initial,scenes,max_steps=max_ticks,camera=True,source_options=True,rigid_joint_limits=True,object_dynamics=True)
        self.model=self.base.model;self.gpu_model=copy.copy(self.model);self.gpu_model.opt.noslip_iterations=0
        self.compatibility=apply_gpu_ccd_tolerance(self.gpu_model)
        self.capacity=512*min(count,8)+32*max(count-8,0)
        self.views=[];self._synced=[]
        for lane in range(count):
            w=copy.copy(self.base);w.data=mujoco.MjData(self.model);w.observed=w.data
            w.lane=lane;w.steps=0;w.phase='boundary';w.active=[];w.exchanges=0;w.events=[]
            self.views.append(w);self._synced.append(0)
        q=np.broadcast_to(self.base.data.qpos,(count,self.model.nq)).astype(np.float32).copy()
        for i,(_, (a,b))in enumerate(self.base.addresses.items()):q[:,a:a+7]=[10.+i,10.,5.,1.,0.,0.,0.]
        self.targets=np.broadcast_to(initial,(count,len(names))).copy();self.speeds=np.zeros_like(self.targets)
        ctrl=np.zeros((count,self.model.nu),np.float32);ctrl[:,self.base.position_ids]=self.targets
        values=dict(qpos=q,qvel=np.zeros((count,self.model.nv),np.float32),time=np.zeros(count,np.float32),ctrl=ctrl,
            qacc_warmstart=np.zeros((count,self.model.nv),np.float32),xfrc_applied=np.zeros((count,self.model.nbody,6),np.float32))
        self.core=NativeControlGraph(self.gpu_model,values,count=count,capacity=self.capacity,njmax=8192)
        self.gpu=self.core.live
        self.q_object_indices=np.array([a+j for a,_ in self.base.addresses.values()for j in range(7)],np.int32)
        self.v_object_indices=np.array([b+j for _,b in self.base.addresses.values()for j in range(6)],np.int32)
        self.object_body_ids=np.array(list(self.base.bodyids.values()),np.int32)
        with wp.ScopedStream(self.core.stream):
            self.q_object_device=wp.array(self.q_object_indices,dtype=wp.int32)
            self.v_object_device=wp.array(self.v_object_indices,dtype=wp.int32)
            self.body_object_device=wp.array(self.object_body_ids,dtype=wp.int32)
        self.contacts=[[]for _ in range(count)]
        self.sync_kinematics(q,values['qvel'],values['time'])
        self.setup_wall_s=time.monotonic()-began

    def set_cost_journal(self,path):
        self.cost_journal=Path(path).open('x',buffering=1)
        self._cost(False)

    def _cost(self,pending):
        if self.cost_journal is not None:
            self.cost_journal.write(json.dumps(dict(completed_native_environment_steps=self.ticks*8*self.count,
                pending_native_environment_steps_upper=8*self.count if pending else 0),separators=(',',':'))+'\n')

    def step(self,targets,speeds):
        if self.ticks>=self.max_ticks:raise RuntimeError('Frozen native-step budget exhausted')
        targets=np.asarray(targets);speeds=np.asarray(speeds)
        if targets.shape!=self.targets.shape or speeds.shape!=targets.shape:raise ValueError('Complete batch controls required')
        if not np.isfinite(targets).all()or not np.isfinite(speeds).all():raise ValueError('Nonfinite controls')
        ctrl=np.zeros((self.count,self.model.nu),np.float32)
        ctrl[:,self.base.position_ids]=targets;ctrl[:,self.base.velocity_ids]=speeds[:,self.base.arm_columns]
        self._cost(True);start=time.perf_counter();packet=self.core.advance(ctrl)
        self.step_s+=time.perf_counter()-start;self.ticks+=1;self.packet=packet;self._cost(False)
        np.testing.assert_array_equal(packet['time'],np.full(self.count,self.expected_clock[self.ticks],np.float32))
        np.testing.assert_array_equal(packet['ctrl'],ctrl)
        if not np.isfinite(packet['qpos']).all()or not np.isfinite(packet['qvel']).all():raise RuntimeError('Nonfinite native state')
        if np.any(packet['overflow'])or np.any(packet['nefc']>8192):raise RuntimeError('Native capacity overflow')
        start=time.perf_counter();self.contacts=decode_contacts(packet,self.model,self.base.mapping,self.count)
        self.observer_s+=time.perf_counter()-start;self.targets=targets.copy();self.speeds=speeds.copy()
        if self.journal is not None:
            self.journal.write(json.dumps(dict(tick=self.ticks,native_time_s=packet['time'].tolist(),canonical_time_s=self.ticks*.004,
                qpos=packet['qpos'].tolist(),qvel=packet['qvel'].tolist(),ctrl=packet['ctrl'].tolist(),
                native_xpos=packet['xpos'].tolist(),native_xmat=packet['xmat'].tolist(),contacts=self.contacts,
                nefc=packet['nefc'].tolist(),overflow=packet['overflow'].tolist()),separators=(',',':'),allow_nan=False)+'\n')

    def _sync_lane(self,lane,q,v,clock):
        import mujoco
        w=self.views[lane];w.data.qpos[:]=q;w.data.qvel[:]=v;w.data.time=float(clock);w.steps=self.ticks
        mujoco.mj_kinematics(self.model,w.data);mujoco.mj_comPos(self.model,w.data)
        mujoco.mj_comVel(self.model,w.data);mujoco.mj_camlight(self.model,w.data);self._synced[lane]=self.ticks

    def observe(self,lane,phase,index):
        if self._synced[lane]!=self.ticks:self._sync_lane(lane,self.packet['qpos'][lane],self.packet['qvel'][lane],self.packet['time'][lane])
        return super().observe(lane,phase,index)

    def snapshot(self):
        with self.wp.ScopedStream(self.core.stream):
            return {n:getattr(self.gpu,n).numpy().copy()for n in ('qpos','qvel','time','ctrl','qacc_warmstart')}

    def assert_snapshot(self,values):
        for n,v in self.snapshot().items():np.testing.assert_array_equal(v,values[n])

    def exchange(self,lane,scene):
        from scipy.spatial.transform import Rotation
        from author_mjx_native_exchange import coordinates,forces
        wp=self.wp;w=self.views[lane]
        if w.phase!='boundary':raise ValueError('Object exchange requires round boundary')
        before=self.snapshot();old_force=self.gpu.xfrc_applied.numpy().copy()
        active={o['object_id']:o for o in scene['objects']}
        if not set(active)<=set(self.base.bodyids):raise ValueError('Object outside source bank')
        positions=[];velocity=[];force=[]
        for i,(key,(a,b))in enumerate(self.base.addresses.items()):
            if key in active:
                row=active[key]
                if {n:row[n]for n in ('object_root','scale','source_mesh')}!=self.base.bank_identity[key]:raise ValueError('Bank geometry identity changed')
                T=np.asarray(row['world_T_object']);positions.extend(np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)])
                force.append([0.,0.,-.98,0.,0.,0.])
            else:positions.extend([10.+i,10.,5.,1.,0.,0.,0.]);force.append([0.]*6)
            velocity.extend([0.]*6)
        with wp.ScopedStream(self.core.stream):
            qp=wp.array(np.array(positions,np.float32),dtype=wp.float32)
            qv=wp.array(np.array(velocity,np.float32),dtype=wp.float32)
            ff=wp.array(np.array(force,np.float32),dtype=wp.spatial_vector)
            wp.launch(coordinates,dim=len(positions),inputs=[self.gpu.qpos,self.q_object_device,qp,lane])
            wp.launch(coordinates,dim=len(velocity),inputs=[self.gpu.qvel,self.v_object_device,qv,lane])
            wp.launch(forces,dim=len(force),inputs=[self.gpu.xfrc_applied,self.body_object_device,ff,lane])
            wp.synchronize_stream(self.core.stream)
        after=self.snapshot();other=np.arange(self.count)!=lane
        np.testing.assert_array_equal(after['qpos'][:,self.base.qindices],before['qpos'][:,self.base.qindices])
        np.testing.assert_array_equal(after['qvel'][:,self.base.vindices],before['qvel'][:,self.base.vindices])
        for key in ('qpos','qvel'):
            np.testing.assert_array_equal(after[key][other],before[key][other])
        for key in ('time','ctrl','qacc_warmstart'):np.testing.assert_array_equal(after[key],before[key])
        np.testing.assert_array_equal(self.gpu.xfrc_applied.numpy()[other],old_force[other])
        w.active=list(active);w.scene=copy.deepcopy(scene);w.exchanges+=1
        event=dict(lane=lane,scene_id=scene['scene_id'],tick=self.ticks,robot_and_clock_exact=True,other_lanes_exact=True,
            only_object_coordinate_buffers_written=True,graph_addresses_unchanged=True)
        self.events.append(event);w.events.append(event)
        self._sync_lane(lane,after['qpos'][lane],after['qvel'][lane],after['time'][lane])

    def receipt(self):
        result=super().receipt();result.update(GPU_graph_mode='native_single_control_graph',full_native_state_staging=False,
            same_installed_MJX_Warp_kernels=True,native_graph_captures=1,observer_copy_bytes_per_control=self.core.copy_bytes,
            graph_capture_integrations=0,CPU_kinematics_lazily_updated=True,line_buffered_native_cost_journal=True)
        return result

    def close(self):
        if self.cost_journal is not None:self.cost_journal.close()
        super().close()
