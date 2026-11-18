"""The same installed MJX-Warp kernels in one stable CUDA control graph.

Eight native integrator steps, a separate full-state observer, then one batched
pinned-host readback. Array addresses remain fixed across controls and resets.
"""
import dataclasses
import time
import numpy as np

from author_mjx_live_batch_v4 import CONTACT_FIELDS


def array_pairs(source,target,wp,path=''):
    result=[]
    if isinstance(source,wp.array):
        if source.shape!=target.shape or source.dtype!=target.dtype:raise ValueError('Native observer layout mismatch '+path)
        if source.size:
            if source.ptr==target.ptr:raise ValueError('Native observer aliases live state '+path)
            result.append((path,source,target))
    elif dataclasses.is_dataclass(source):
        for field in dataclasses.fields(source):
            result+=array_pairs(getattr(source,field.name),getattr(target,field.name),wp,path+'.'+field.name)
    elif isinstance(source,(tuple,list)):
        if len(source)!=len(target):raise ValueError('Native observer tuple mismatch')
        for index,(a,b)in enumerate(zip(source,target)):result+=array_pairs(a,b,wp,path+'.'+str(index))
    return result


class NativeControlGraph:
    def __init__(self,model,initial,*,count,capacity,njmax=8192):
        import warp as wp
        from warp._src.types import type_size_in_bytes
        from mujoco.mjx.third_party import mujoco_warp as mjw
        self.wp=wp;self.mjw=mjw;self.count=count;self.controls=0
        self.device=wp.get_device('cuda:0');self.stream=wp.Stream(self.device)
        with wp.ScopedStream(self.stream):
            self.model=mjw.put_model(model)
            self.live=mjw.make_data(model,nworld=count,naconmax=capacity,njmax=njmax)
            self.observed=mjw.make_data(model,nworld=count,naconmax=capacity,njmax=njmax)
            for name,value in initial.items():
                destination=getattr(self.live,name)
                if destination.size:destination.assign(np.asarray(value))
            self.copy_pairs=array_pairs(self.live,self.observed,wp)
            self.copy_bytes=sum(a.size*type_size_in_bytes(a.dtype)for _,a,_ in self.copy_pairs)
            self.host_ctrl=wp.empty_like(self.live.ctrl,device='cpu',pinned=True)
            self.host_ctrl_view=self.host_ctrl.numpy()
            self.before_warmstart=wp.empty_like(self.live.qacc_warmstart)
            arrays={n:getattr(self.observed,n)for n in ('qpos','qvel','time','ctrl','xpos','xmat')}
            arrays.update(nacon=self.observed.nacon,nefc=self.observed.nefc,overflow=self.observed.overflow,
                efc_force=self.observed.efc.force,live_warmstart=self.live.qacc_warmstart,
                observer_before_live_warmstart=self.before_warmstart)
            arrays.update({'contact_'+n:getattr(self.observed.contact,n)for n in CONTACT_FIELDS})
            self.output_arrays=arrays
            self.host={n:wp.empty_like(a,device='cpu',pinned=True)for n,a in arrays.items()}
            self.host_views={n:a.numpy()for n,a in self.host.items()}
            before={n:getattr(self.live,n).numpy().copy()for n in ('qpos','qvel','time','ctrl','qacc_warmstart')}
            began=time.perf_counter()
            with wp.ScopedCapture(stream=self.stream)as capture:
                for _ in range(8):mjw.step(self.model,self.live)
                for _,source,target in self.copy_pairs:wp.copy(target,source)
                wp.copy(self.before_warmstart,self.live.qacc_warmstart)
                mjw.forward(self.model,self.observed)
            self.graph=capture.graph;wp.synchronize_stream(self.stream)
            self.capture_wall_s=time.perf_counter()-began
            for name,value in before.items():np.testing.assert_array_equal(getattr(self.live,name).numpy(),value,err_msg='Graph capture integrated live state')
        self.capture_performed_no_integrations=True

    def advance(self,ctrl):
        wp=self.wp;started=time.perf_counter()
        np.copyto(self.host_ctrl_view,np.asarray(ctrl,dtype=np.float32))
        with wp.ScopedStream(self.stream):
            wp.copy(self.live.ctrl,self.host_ctrl)
            wp.capture_launch(self.graph,stream=self.stream)
            for name,source in self.output_arrays.items():
                if source.size:wp.copy(self.host[name],source)
            wp.synchronize_stream(self.stream)
        packet=dict(self.host_views)
        # Match MJX's scalar-array representation of matrix-valued Warp arrays.
        packet['xmat']=packet['xmat'].reshape(self.count,-1,9)
        np.testing.assert_array_equal(packet['live_warmstart'],packet['observer_before_live_warmstart'])
        self.controls+=1;self.last_wall_s=time.perf_counter()-started
        return packet
