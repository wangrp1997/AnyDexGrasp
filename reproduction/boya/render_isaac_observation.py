"""Render a fresh Isaac Sim tabletop RGB-D input for the first Boya smoke.

No controller, robot pose, mesh IDs, segmentation or hand outcome is a network
input. The isolated object view excludes the robot; this is a synthetic
observation protocol, explicitly distinct from the paper's real camera data.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,enable_cameras=True,device='cpu',kit_args='--/log/file=/dev/null')
    try:
        import numpy as np
        import torch
        from PIL import Image
        import scipy.io
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab.sensors.camera import Camera,CameraCfg
        import trimesh
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.004,device='cpu',physics=PhysxCfg()))
        sim_utils.GroundPlaneCfg(color=(.65,.68,.72)).func('/World/ground',sim_utils.GroundPlaneCfg(color=(.65,.68,.72)))
        sim_utils.DomeLightCfg(intensity=2500.,color=(1.,1.,1.)).func('/World/light',sim_utils.DomeLightCfg(intensity=2500.,color=(1.,1.,1.)))
        center=np.array([-.75,.10,.025])
        cube=sim_utils.CuboidCfg(size=(.05,.05,.05),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.85,.60,.25)))
        cube.func('/World/ObservedObject',cube,translation=tuple(center))
        camera=Camera(CameraCfg(prim_path='/World/observation_camera',width=1280,height=720,
            update_period=0,data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=48.,clipping_range=(.1,1.))))
        sim.reset()
        camera.set_world_poses(torch.tensor([[-.75,.10,.5]],dtype=torch.float32),
            torch.tensor([[0.,1.,0.,0.]],dtype=torch.float32),convention='ros')
        for _ in range(16):
            sim.render();camera.update(.004,force_recompute=True)
        def cpu(value):
            t=value.torch if hasattr(value,'torch') else value
            return t.detach().cpu().numpy()
        rgb=cpu(camera.data.output['rgb'])[0,:,:,:3]
        depth=cpu(camera.data.output['distance_to_image_plane'])[0].squeeze()
        K=cpu(camera.data.intrinsic_matrices)[0]
        if rgb.shape!=(720,1280,3) or depth.shape!=(720,1280):raise ValueError('Unexpected camera shape')
        if not .445<float(np.median(depth[345:375,625:655]))<.455:
            raise ValueError('Actual Isaac depth differs from the declared 50mm cube surface')
        depth_mm=np.where(np.isfinite(depth)&(depth>.1)&(depth<.72),np.rint(depth*1000),0).astype(np.uint16)
        Image.fromarray(rgb.astype(np.uint8)).save(a.output_dir/'color.png')
        Image.fromarray(depth_mm).save(a.output_dir/'depth.png')
        scipy.io.savemat(a.output_dir/'meta.mat',dict(intrinsic_matrix=K,factor_depth=np.array([[1000.]])))
        object_root=a.output_dir/'object';(object_root/'mesh').mkdir(parents=True);(object_root/'urdf').mkdir()
        mesh=trimesh.creation.box(extents=[.05]*3)
        mesh.export(object_root/'mesh/simplified.obj');mesh.export(object_root/'urdf/cube.obj')
        (object_root/'urdf/coacd.urdf').write_text('<robot name="boya_smoke_cube"><link name="object"><collision><geometry><mesh filename="cube.obj"/></geometry></collision></link></robot>\n')
        pose=np.eye(4);pose[:3,3]=center
        T=np.eye(4);T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=[-.75,.1,.5]
        report=dict(schema='anydex_boya_isaac_observation_v1',status='rendered',
            simulator='Isaac Sim / PhysX',renderer='Isaac native RTX camera',
            robot_visible=False,observation_protocol='Isolated object/ground view before arm approach',
            T_WC=T.tolist(),intrinsic_matrix=K.tolist(),depth_factor=1000.,
            actual_center_surface_depth_m=float(np.median(depth[345:375,625:655])),
            files={name:dict(path=str((a.output_dir/name).resolve()),sha256=sha(a.output_dir/name)) for name in ['color.png','depth.png','meta.mat']},
            object_root=str(object_root.resolve()),world_T_object=pose.tolist(),object_scale=1.,
            object_geometry=dict(type='box',size_m=[.05]*3),object_mass_kg=.1,
            control_steps=0,grasp_attempts=0,pybullet_imported=False,object_pose_as_network_input=False)
        (a.output_dir/'observation.json').write_text(json.dumps(report,indent=2)+'\n')
        print('BOYA_OBSERVATION_READY '+json.dumps(report),flush=True)
    finally:
        launcher.app.close()


if __name__=='__main__':main()
