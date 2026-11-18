"""One MuJoCo world/reset and a fixed interchangeable source-object bank.

An independent backend candidate. Object exchange never writes robot joints,
velocities or the simulation clock. It does not certify physical grasp labels.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import xml.etree.ElementTree as ET

import numpy as np
import author_mujoco_control as C
import author_fast_geometry_checks as F

B = C.B


class World:
    def __init__(self, plan, names, initial, scenes, *, max_steps=17500, camera=False, verify_camera_physics=False, source_options=False, rigid_joint_limits=False, object_dynamics=False):
        import mujoco
        from run_template_arm_positive import object_asset
        began=time.monotonic(); self.max_steps=max_steps; self.steps=0; self.exchanges=0
        self.robot_state_writes=0; self.reset_calls=0; self.phase='boundary'
        self.spec,_,self.planning,self.reference,self.parameters,self.arm_names,excludes,_=C.build(plan,names,scenes[0])
        self.names=list(names);self.sources={};self.assets={};bank={}
        for scene in scenes:
            for row in scene['objects']:
                key=row['object_id']
                identity={n:row[n] for n in ('object_root','scale','source_mesh')}
                if key in bank and bank[key][0]!=identity:raise ValueError('One bank identity has differing geometry')
                bank[key]=(identity,row)
        self.bank_identity={key:identity for key,(identity,_) in bank.items()}
        existing={b.name for b in self.spec.bodies}
        for i,(key,(_,row)) in enumerate(bank.items()):
            asset=object_asset(SimpleNamespace(object_root=Path(row['object_root'])),np.asarray(row['world_T_object']),row['scale'])
            self.assets[key]=asset
            if key in existing:continue
            body=self.spec.worldbody.add_body(name=key,pos=[10.+i,10.,5.])
            body.add_freejoint(name=key+'_free')
            volume=sum(float(p.volume) for p in asset.collision_parts)
            for j,part in enumerate(asset.collision_parts):
                name=key+'_part_'+str(j)
                self.spec.add_mesh(name=name,uservert=np.asarray(part.vertices).ravel(),userface=np.asarray(part.faces).ravel())
                body.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=name,
                    mass=.1*float(part.volume)/volume,friction=[.6,.02,.0001],condim=4)
        # Source hand equality settings were absent from the earlier URDF port.
        source=ET.parse(plan['arguments']['hand_scene']).getroot()
        equalities={e.name:e for e in self.spec.equalities}
        self.equalities=[]
        for row in source.findall('equality/joint'):
            target=equalities[row.get('name')]
            for field in ('solref','solimp'):
                if row.get(field):
                    values=getattr(target,field).copy();v=list(map(float,row.get(field).split()))
                    values[:len(v)]=v;setattr(target,field,values)
            self.equalities.append(dict(name=target.name,solref=list(target.solref),solimp=list(target.solimp)))
        # Exact public DexGraspBench hand contact/solver settings, explicit change.
        hand_bodies={e.get('name') for e in source.findall('.//body')}
        self.legal_hand={self.planning.fk.fixed_attachment(n,np.eye(4))[0] for n in hand_bodies if n in self.planning.fk.links}
        for geom in self.spec.geoms:
            if geom.parent.name in hand_bodies:
                geom.solimp[:3]=[.5,.99,.0001];geom.solref[:2]=[.005,1.]
        self.spec.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
        self.spec.option.noslip_iterations=2;self.spec.option.impratio=10
        self.source_options=None
        if source_options:
            from author_mujoco_source_options import apply
            self.source_options=apply(self.spec,plan['arguments']['hand_scene'])
        self.joint_limit_options=None
        if rigid_joint_limits:
            from author_mujoco_joint_limits import apply
            self.joint_limit_options=apply(self.spec)
        self.object_dynamics_options=None
        if object_dynamics:
            from author_mujoco_object_dynamics import apply
            self.object_dynamics_options=apply(self.spec,self.assets)
        self.substeps=int(round(.004/self.spec.option.timestep));self.native_steps=0
        self.max_native_steps=max_steps*self.substeps
        if abs(self.substeps*self.spec.option.timestep-.004)>1e-12:raise ValueError('Source integrator does not divide 4ms commands')
        physics_reference=self.spec.compile() if verify_camera_physics else None
        self.extra_static_compiles=int(verify_camera_physics);self.visual_physics_parameter_check=None
        if camera:
            from author_mujoco_camera import configure
            if any(s['camera']!=scenes[0]['camera'] for s in scenes):raise ValueError('One fixed camera calibration required')
            configure(self.spec,scenes[0]['camera'],self.assets)
        self.model=self.spec.compile();self.data=mujoco.MjData(self.model);self.observed=mujoco.MjData(self.model)
        if physics_reference is not None:
            fields=('body_mass','body_inertia','body_ipos','body_iquat','dof_damping','dof_armature',
                'jnt_range','jnt_solref','jnt_solimp','eq_solref','eq_solimp','actuator_gainprm','actuator_biasprm')
            for field in fields:np.testing.assert_array_equal(getattr(self.model,field),getattr(physics_reference,field),err_msg=field)
            self.visual_physics_parameter_check=dict(all_exact=True,fields=list(fields),
                reference='Same full-precision in-memory spec before visual-only additions; no rounded XML roundtrip')
        mujoco.mj_resetData(self.model,self.data);self.reset_calls+=1
        C.R.put_state(self.model,self.data,self.names,initial,self.planning.mimics);self.robot_state_writes+=1
        self.bodyids={key:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,key) for key in bank}
        self.addresses={}
        for key,body in self.bodyids.items():
            joint=int(self.model.body_jntadr[body]);self.addresses[key]=(int(self.model.jnt_qposadr[joint]),int(self.model.jnt_dofadr[joint]))
            if abs(float(self.model.body_mass[body])-.1)>1e-12:raise ValueError('Source bank mass changed')
        self.mapping=C.body_mapping(self.model,self.planning,list(bank))
        self.qindices=C.R.joint_addresses(self.model,list(self.planning.joints))
        self.vindices=self.model.jnt_dofadr[[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.planning.joints]]
        self.position_ids=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in self.names])
        self.velocity_ids=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,'velocity_'+n) for n in self.arm_names])
        self.arm_columns=[self.names.index(n) for n in self.arm_names]
        self.active=[];self.events=[]
        F.precompile();mujoco.mj_forward(self.model,self.data)
        self.camera=None
        if camera:
            from author_mujoco_camera import Camera
            self.camera=Camera(self,scenes[0]['camera'])
        self.setup_wall_s=time.monotonic()-began

    def exchange(self, scene):
        import mujoco
        from scipy.spatial.transform import Rotation
        if self.phase!='boundary':raise ValueError('Object exchange is permitted only between rounds')
        # Freeze all robot state and time before the external object exchange.
        before=self.data.qpos[self.qindices].copy(),self.data.qvel[self.vindices].copy(),float(self.data.time)
        rows={r['object_id']:r for r in scene['objects']}
        if len(rows)!=len(scene['objects']):raise ValueError('Duplicate object identity')
        if not set(rows)<=set(self.bodyids):raise ValueError('Object outside the preloaded bank')
        for key,row in rows.items():
            if {n:row[n] for n in ('object_root','scale','source_mesh')}!=self.bank_identity[key]:
                raise ValueError('Object geometry differs from the preloaded bank')
            pose=np.asarray(row['world_T_object'],dtype=float)
            if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1],rtol=0,atol=1e-12):
                raise ValueError('Finite rigid object transform required')
            if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),rtol=0,atol=1e-9) or abs(np.linalg.det(pose[:3,:3])-1)>1e-9:
                raise ValueError('Rigid object rotation required')
        for i,(key,body) in enumerate(self.bodyids.items()):
            q,v=self.addresses[key]
            if key in rows:
                pose=np.asarray(rows[key]['world_T_object'])
                self.data.qpos[q:q+3]=pose[:3,3]
                self.data.qpos[q+3:q+7]=Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)
            else:self.data.qpos[q:q+7]=[10.+i,10.,5.,1.,0.,0.,0.]
            self.data.qvel[v:v+6]=0.;self.data.xfrc_applied[body]=0.
            if key in rows:self.data.xfrc_applied[body,2]=-.98
        mujoco.mj_forward(self.model,self.data)
        np.testing.assert_array_equal(self.data.qpos[self.qindices],before[0])
        np.testing.assert_array_equal(self.data.qvel[self.vindices],before[1])
        if self.data.time!=before[2]:raise ValueError('Object exchange changed the world clock')
        self.active=list(rows);self.scene=copy.deepcopy(scene);self.exchanges+=1
        self.events.append(dict(kind='object_exchange',scene_id=scene['scene_id'],world_step=self.steps,
            model_id=id(self.model),data_id=id(self.data),robot_state_writes=0,clock_reset=False))

    def begin_round(self):
        if self.phase!='boundary' or not self.active:raise ValueError('An exchanged scene is required before a round')
        self.phase='active'

    def end_round(self):
        if self.phase!='active':raise ValueError('No active round')
        self.phase='boundary'

    def control(self, target, velocity):
        target=np.asarray(target);velocity=np.asarray(velocity)
        if target.shape!=(len(self.names),) or velocity.shape!=(len(self.names),) or not np.isfinite(target).all() or not np.isfinite(velocity).all():
            raise ValueError('Complete finite source controls required')
        self.data.ctrl[self.position_ids]=target
        self.data.ctrl[self.velocity_ids]=np.asarray(velocity)[self.arm_columns]

    def step(self,target,velocity):
        import mujoco
        from author_mujoco_clock import validate
        if self.steps>=self.max_steps:raise RuntimeError('Finite persistent-world step budget exhausted')
        if self.native_steps+self.substeps>self.max_native_steps:raise RuntimeError('Native integration budget exhausted')
        self.control(target,velocity);before_time=float(self.data.time)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data);self.native_steps+=1
        self.steps+=1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise RuntimeError('Nonfinite world state')
        if self.native_steps!=self.steps*self.substeps or abs(float(self.data.time)-before_time-.004)>1e-12:
            raise RuntimeError('Native clock/counter discontinuity')
        validate(float(self.data.time),self.steps,self.substeps)

    def receipt(self):
        return dict(world_builds=1,active_model_builds=1,initial_compiler_calls=3+self.extra_static_compiles,model_id=id(self.model),data_id=id(self.data),
            scene_reset_calls=self.reset_calls,robot_initial_state_writes=self.robot_state_writes,
            object_exchanges=self.exchanges,bank_size=len(self.bodyids),steps=self.steps,
            native_integration_steps=self.native_steps,native_steps_per_control=self.substeps,max_native_integrations=self.max_native_steps,
            source_options=self.source_options,
            joint_limit_options=self.joint_limit_options,
            object_dynamics_options=self.object_dynamics_options,
            native_time_s=float(self.data.time),events=self.events,setup_wall_s=self.setup_wall_s,
            source_mimic_parameters=self.equalities,Kit_starts=0,training_updates=0,
            camera_builds=int(self.camera is not None),camera_frames=0 if self.camera is None else self.camera.frames,
            visual_physics_parameter_check=self.visual_physics_parameter_check,
            contact_model_changed=True,production_ready=False,
            camera_and_CGR_integration_validated=False)
