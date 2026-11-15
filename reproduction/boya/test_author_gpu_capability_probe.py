"""CPU contract tests only; these fixtures never establish GPU capability."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import author_gpu_capability_probe as P


def buffers(mode='filtered',capacity=8):
    force=np.zeros((capacity,1),np.float32);force[:2]=.5
    points=np.zeros((capacity,3),np.float32)
    normals=np.zeros_like(points);normals[:2,2]=1
    separation=np.zeros((capacity,1),np.float32)
    counts=np.array([2],np.uint32) if mode=='raw' else np.array([[2]],np.uint32)
    starts=np.zeros_like(counts)
    if mode=='raw':return [force,points,normals,separation,counts,starts,np.arange(capacity,dtype=np.uint64)]
    if mode=='friction':return [points.copy(),points,counts,starts]
    return [force,points,normals,separation,counts,starts]


class FixtureTensor:
    """A CPU fixture with device text; never counted as actual CUDA evidence."""
    device='cuda:0'
    def __init__(self,value):self.value=value;self.shape=value.shape;self.dtype=value.dtype
    def numpy(self):return self.value


class FixtureView:
    sensor_count=1;filter_count=1;max_contact_data_count=8
    sensor_paths=['/env0/Object'];filter_paths=['/env0/Robot/pad']
    def __init__(self):self.data=buffers();self.calls=[]
    def get_contact_data(self,dt):
        self.calls.append(('normal',dt));return [FixtureTensor(v) for v in self.data]
    def get_friction_data(self,dt):
        self.data[4][0,0]=1
        return [FixtureTensor(v) for v in (self.data[1],self.data[1],self.data[4],self.data[5])]


class ProbeTests(unittest.TestCase):
    def test_empty_and_unreferenced_nan_are_not_synthetic_contacts(self):
        values=buffers();values[1][2:]=np.nan
        result=P.validate_contact_buffers(values,mode='filtered',capacity=8,sensor_count=1,filter_count=1)
        self.assertEqual(result['active_contact_records'],2)
        values[4][:]=0
        self.assertEqual(P.validate_contact_buffers(values,mode='filtered',capacity=8,sensor_count=1,filter_count=1)['active_contact_records'],0)

    def test_saturated_out_of_bounds_and_unsigned_wrap_are_rejected(self):
        for count,start in ((8,0),(2,7),(2,2**32-1)):
            with self.subTest(count=count,start=start):
                a=buffers();a[4][:]=count;a[5][:]=start
                with self.assertRaises(ValueError):P.validate_contact_buffers(a,mode='filtered',capacity=8,sensor_count=1,filter_count=1)

    def test_active_nan_nonunit_normal_and_malformed_actor_ids_reject(self):
        for mutation in ('nan','normal','ids'):
            with self.subTest(mutation=mutation):
                a=buffers('raw')
                if mutation=='nan':a[1][0,0]=np.nan
                elif mutation=='normal':a[2][0]=0
                else:a[6]=a[6].astype(np.int64)
                with self.assertRaises(ValueError):P.validate_contact_buffers(a,mode='raw',capacity=8,sensor_count=1)

    def test_readback_is_copied_before_friction_overwrites_pair_tables(self):
        view=FixtureView();normal,old=P.capture_contact_view(view,.004)
        friction,new=P.capture_contact_view(view,.004,mode='friction')
        self.assertEqual(normal['active_contact_records'],2)
        self.assertEqual(friction['active_contact_records'],1)
        self.assertEqual(int(old[4][0,0]),2)
        self.assertEqual(int(new[2][0,0]),1)

    def test_cpu_payload_cannot_be_reported_as_gpu(self):
        v=FixtureView();v.get_contact_data=lambda dt:buffers()
        with self.assertRaisesRegex(ValueError,'not actually on CUDA'):P.capture_contact_view(v,.004)
        self.assertEqual(P.capture_contact_view(v,.004,require_cuda=False)[0]['active_contact_records'],2)

    def test_failed_contact_method_records_failure_and_never_steps(self):
        attrs={'enableGPUDynamics':True,'broadphaseType':'GPU','enableCCD':False,'solverType':'TGS'}
        class Prim:
            def GetAttribute(self,key):return SimpleNamespace(Get=lambda:attrs[key.split(':')[-1]])
        sim=SimpleNamespace(get_physics_step_count=lambda:12,get_physics_dt=lambda:.004,
            cfg=SimpleNamespace(physics_prim_path='/physicsScene'),
            stage=SimpleNamespace(GetPrimAtPath=lambda _:Prim()),physics_sim_view=SimpleNamespace(device_ordinal=0),
            get_setting=lambda key:False if key.endswith('disableContactProcessing') else True)
        view=FixtureView()
        def failed(dt):raise RuntimeError('backend does not support requested mode')
        view.get_contact_data=failed
        with tempfile.TemporaryDirectory() as root:
            d=P.probe_live(sim,Path(root)/'probe',contact_views={'object':(view,'filtered')})
            self.assertEqual(d['status'],'failed');self.assertFalse(d['live_reporting_nonempty'])
            self.assertEqual(d['physics_clock_before'],d['physics_clock_after'])
            self.assertEqual(json.loads((Path(root)/'probe/PROBE.json').read_text())['contacts']['object']['status'],'failed')

    def test_installed_source_scan_is_static_and_exposes_known_constraints(self):
        d=P.static_report();facts={row['id'] for row in d['facts']}
        self.assertTrue({'gpu_disables_ccd','raw_contacts','shared_pair_tables','gpu_frontend_excludes_numpy'}<=facts)
        self.assertFalse(d['live_gpu_validated']);self.assertEqual(d['physics_steps'],0)
        self.assertIn('get_raw_contact_data',d['tensor_api_methods']['RigidContactView'])


if __name__=='__main__':unittest.main()
