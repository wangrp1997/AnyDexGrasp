"""One native robot/floor/camera and a preloaded object bank for all rounds.

Actor pose/velocity writes are permitted only before settling each round.
Capture and execution share the same native handles, solver state and clock.
"""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

import author_collection_batch as B
import author_collection_capture as CAPTURE
import author_collection_service as SERVICE
import author_collection_service_bound as BOUND
import author_contact_pipeline as M
import author_live_world_contract as K


def capture_function(world):
    module,scope=SERVICE.capture_template();fn=module.body[0]
    first=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Try))
    body=fn.body[first].body
    start=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='profile')
    end=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Tuple)and [x.id for x in n.targets[0].elts]==['poses','velocities','contact_rows'])
    names='profile, sim, bodies, roots, initial, camera, T, material_readback, native, static_paths, targets, capacity, contacts'
    body[start:end]=ast.parse(names+' = world.capture_handles(scene, output_dir)').body
    loop=next(n for n in body if isinstance(n,ast.For)and ast.unparse(n.target)=='step')
    loop.body.insert(0,ast.parse('world.capture_before_step()').body[0])
    step=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Expr)and ast.unparse(n)=='sim.step(render=False)')
    loop.body.insert(step+1,ast.parse('world.robot.update(DT)').body[0])
    observed=next(i for i,n in enumerate(body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and ast.unparse(n.value.func)=='observed.update')
    body.insert(observed+1,ast.parse("observed['persistent_world'] = world.capture_metadata()").body[0])
    class Text(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='Measured common-scene pose; independent execution cold recreation at this pose and zero twist':
                n.value='Measured same-world pose and velocity; no reset or reconstruction after RGB-D'
            return n
    module=Text().visit(module);scope.update(world=world,resident_launcher=world.acquire)
    exec(compile(ast.fix_missing_locations(module),__file__+':live_capture','exec'),scope)
    return scope['capture']


def live_ready_function(world):
    import author_lift as L
    tree=ast.parse(Path(L.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='MultiObjectLiftRuntime')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='ready'))
    expected="error > 2e-05 or float(abs(actual_velocity).max()) > 2e-05";count=0
    class Entry(ast.NodeTransformer):
        def visit_If(self,n):
            nonlocal count
            if ast.unparse(n.test)==expected:
                count+=1;n.test=ast.parse("error > 2e-5 or not np.array_equal(actual_velocity, np.asarray(world.observation['objects'][key]['velocity']))",mode='eval').body
            return self.generic_visit(n)
    fn=Entry().visit(fn)
    if count!=1:raise ValueError('Original cold-entry predicate changed')
    fn.body.insert(0,ast.parse('world.validate_entry()').body[0])
    fn.body.extend(ast.parse("self.setup_audit['entry_semantics'] = 'Same measured live world, velocities preserved; no pose reset after photograph'\nself.setup_audit['live_state_observation'] = B.binding(world.observation_path)\nself.setup_audit['world_continuity'] = world.entry_proof\ncommon.save(self.output_dir/'MULTI_OBJECT_SETUP.json', self.setup_audit)").body)
    scope=dict(vars(L),world=world,B=B);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__+':live_entry','exec'),scope)
    return scope['ready']


class DeferredApp:
    def __init__(self,world):self.world=world
    def close(self,*args,**kwargs):self.world.deferred_closes+=1


class World:
    def __init__(self,root,protocol):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.p=protocol
        self.order=K.RoundOrder(len(protocol['rows']));self.launcher=None;self.sim=None
        self.bank={};self.assets={};self.paths={};self.objects=None;self.camera=None
        self.active=[];self.scene=None;self.scene_path=None;self.observation=None;self.observation_path=None
        self.boundary_resets=0;self.deferred_closes=0;self.build_count=0;self.entry_proof=None;self.events=[]
        self.refs=BOUND.bind_references(protocol['inputs']);self.capture_impl=capture_function(self)

    def acquire(self,launcher_type,**kwargs):
        if self.launcher is not None:return SimpleNamespace(app=DeferredApp(self))
        self.launcher=launcher_type(**kwargs);self.bootstrap()
        return SimpleNamespace(app=DeferredApp(self))

    def bootstrap(self):
        import author_transport as T
        T._backend_path()
        import execute_template_arm_positive as native
        import torch
        from scipy.spatial.transform import Rotation
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera,CameraCfg
        from run_template_arm_positive import object_asset
        from canondex.grasp.physics import load_physics_config
        from canondex.scene import grasp_object
        from settle_s40_scenes import namespaced_spawn
        from render_s40_observations import look_at_ros
        bootstrap=self.root/'bootstrap';bootstrap.mkdir(exist_ok=False)
        ready=B.read(self.p['inputs']['geometry_template']['path']);args=M.LIFT._backend().args_for_plan(Path(ready['plan']['path']).parent,bootstrap)
        template=B.read(ready['scene']['path']);primary=template['target_object_id'];self.template_primary=primary
        if primary not in {r['object_id']for r in self.p['object_bank']}:raise ValueError('Geometry template primary outside bank')
        profile=load_physics_config(args.physics_config)
        class Built(Exception):pass
        def before_reset(sim,robot,objects):
            self.objects=objects;self.bank[primary]=objects['object'];self.paths[primary]='/World/Object'
            for i,row in enumerate(self.p['object_bank']):
                matrix=np.asarray(row['world_T_object']).copy();matrix[:3,3]=[10.+i,10.,5.]
                asset=object_asset(SimpleNamespace(object_root=Path(row['object_root'])),matrix,row['scale']);self.assets[row['object_id']]=asset
                if row['object_id']==primary:continue
                path='/World/AuthorScene/'+row['object_id'];self.paths[row['object_id']]=path
                self.bank[row['object_id']]=namespaced_spawn(grasp_object,path)(sim,asset,physics_profile=profile,
                    collision_refinement=B.read(row['collision_refinement'])if row.get('collision_refinement')else None)
            cam=self.p['camera'];self.camera_pose=look_at_ros(cam['eye_world_m'],cam['target_world_m'],cam['up_world'])
            self.camera=Camera(CameraCfg(prim_path='/World/observation_camera',width=cam['width'],height=cam['height'],
                update_period=0,data_types=['rgb','distance_to_image_plane'],update_latest_camera_pose=True,
                spawn=sim_utils.PinholeCameraCfg(focal_length=cam['focal_length'],horizontal_aperture=cam['horizontal_aperture'],clipping_range=tuple(cam['clipping_range_m']))))
        def factory(args,initial,asset,build_scene):
            self.sim,self.robot,self.urdf,self.objects=build_scene(before_reset)
            self.assets[primary]=asset;self.build_count+=1
            raise Built()
        gen=native.execution_steps(args,runtime_factory=factory)
        try:next(gen);raise ValueError('Bootstrap unexpectedly reached a command')
        except Built:pass
        finally:gen.close()
        self.robot_root=self.robot.data.root_link_pose_w.torch[0].cpu().numpy().copy()
        self.camera.set_world_poses(torch.tensor([self.camera_pose[:3,3].tolist()],dtype=torch.float32),
            torch.tensor([Rotation.from_matrix(self.camera_pose[:3,:3]).as_quat().tolist()],dtype=torch.float32),convention='ros')
        self.controller=self.make_controller();self.save('bootstrapped')

    def make_controller(self):
        from canondex.scene.runtime import snapshot
        world=self
        class Controller:
            def __init__(self):
                self.runtime=(world.sim,world.robot,world.urdf,world.objects);self.contacts=None;self.binding=None;self.native_audit_count=0
            def check(self,binding,**kwargs):
                B.verify(world.p['sources']);return False
            def reset(self,initial,object_pose):
                world.validate_entry()
                q=np.asarray([initial[n]for n in world.observation['joint_names']])
                if np.max(abs(q-world.observation['joint_pos']))>2e-5:raise ValueError('Plan initial state differs from measured live robot')
            def accept_placement(self,binding):self.binding=binding
            def receipt(self):return dict(asset_setup_count=world.build_count,state_reset_count=0,native_audit_count=self.native_audit_count,
                live_entries=[world.entry_proof],scope='Same world and measured state; no actor state reset after observation')
        return Controller()

    def state(self):
        from canondex.scene.runtime import snapshot
        from scipy.spatial.transform import Rotation
        raw=snapshot(self.robot);objects={}
        for key in self.active:
            body=self.bank[key];objects[key]=dict(pose_xyzw=body.data.root_link_pose_w.torch[0].cpu().numpy().tolist(),velocity=body.data.root_link_vel_w.torch[0].cpu().numpy().tolist())
        primary=self.scene['target_object_id'];pose=np.asarray(objects[primary]['pose_xyzw']);matrix=np.eye(4)
        matrix[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();matrix[:3,3]=pose[:3]
        return dict(schema=K.SCHEMA,scene_id=self.scene['scene_id'],sim_step=self.sim.get_physics_step_count(),robot_body_pose=raw['body_pose'].tolist(),
            joint_names=list(self.robot.joint_names),joint_pos=raw['joint_pos'].tolist(),joint_velocity=raw['joint_vel'].tolist(),objects=objects,
            world_T_object=matrix.tolist(),object_pose_xyzw=pose.tolist(),object_velocity=objects[primary]['velocity'])

    def place(self,scene):
        import torch,warp as wp
        from run_template_arm_positive import object_asset
        from canondex.scene.runtime import forward_scene
        from canondex.robots.reference import load_reference
        from canondex.planning.model import PlanningModel
        backend=B.read(self.p['inputs']['backend']['path']);reference=load_reference(Path(backend['reference_root']))
        model=PlanningModel(Path(backend['robot_urdf']));contract=M.BASE_CONTRACT(self.p['inputs']['calibration']['path'])
        slot=contract.slots['g1:distal_000:ff'];full=model.full_state(reference.arm_home|dict(zip(contract.names,slot['independent_q']))|slot['wrist'])
        tensor=lambda a:torch.tensor(a,dtype=torch.float32,device=self.robot.device)
        before=self.sim.get_physics_step_count();names=list(self.robot.joint_names);q=np.asarray([full[n]for n in names])
        for i,(key,body)in enumerate(self.bank.items()):
            body.reset();body.permanent_wrench_composer.reset()
            body.write_root_link_pose_to_sim_index(root_pose=tensor([[10.+i,10.,5.,0.,0.,0.,1.]]))
            body.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1,6))));body.write_data_to_sim()
        self.robot.reset();self.robot.permanent_wrench_composer.reset()
        self.robot.write_root_link_pose_to_sim_index(root_pose=tensor(self.robot_root[None]));self.robot.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1,6))))
        self.robot.write_joint_state_to_sim(tensor(q[None]),tensor(np.zeros((1,len(names)))))
        self.robot.actuators.target_command.set_position_index(value=tensor(q[None]));self.robot.actuators.target_command.set_velocity_index(value=tensor(np.zeros((1,len(names)))))
        self.robot.write_data_to_sim();self.active=[row['object_id']for row in scene['objects']];self.scene=scene
        for row in scene['objects']:
            key=row['object_id'];body=self.bank[key];asset=object_asset(SimpleNamespace(object_root=Path(row['object_root'])),np.asarray(row['world_T_object']),row['scale']);self.assets[key]=asset
            body.write_root_link_pose_to_sim_index(root_pose=tensor(asset.pose_xyzw[None]));body.write_root_link_velocity_to_sim_index(root_velocity=tensor(np.zeros((1,6))))
            body.permanent_wrench_composer.add_forces_and_torques_index(forces=tensor([[[0.,0.,-.98]]]),is_global=True);body.write_data_to_sim()
            values=body.root_view.get_material_properties().numpy().copy();values[...,:2]=.6;values[...,2]=0.
            ids=wp.array(np.arange(len(values),dtype=np.int32),dtype=wp.int32,device='cpu');body.root_view.set_material_properties(wp.array(values,dtype=wp.float32,device='cpu'),ids)
        forward_scene(self.sim);self.robot.update(.004)
        for body in self.bank.values():body.update(.004)
        if self.sim.get_physics_step_count()!=before:raise ValueError('Round boundary unexpectedly integrated physics')
        self.boundary_resets+=1
        self.events.append(dict(kind='between_round_placement',scene_id=scene['scene_id'],sim_step=before,active=self.active.copy(),actor_state_writes=True,world_rebuild=False))

    def capture_handles(self,scene,output):
        import isaaclab.sim as sim_utils
        from pxr import Usd,UsdPhysics
        from carb.settings import get_settings
        from canondex.grasp.physics import load_physics_config
        from template_gripper_native_audit import audit_scene
        self.place(scene);bodies=[self.bank[k]for k in self.active];roots=[self.paths[k]for k in self.active]
        (output/'native').mkdir();native=audit_scene(self.sim,output/'native',['all'],roots=tuple(roots),fast=True)
        if not native['within_20um']:raise ValueError('Original native20um gate failed')
        static=[str(p.GetPath())for p in Usd.PrimRange(self.sim.stage.GetPrimAtPath('/World/ground'))if p.HasAPI(UsdPhysics.CollisionAPI)]
        B.save(output/'CONTACT_SENSOR_CONFIG.json',dict(static_collision_paths=static,disable_contact_processing=get_settings().get_as_bool('/physics/disableContactProcessing')))
        from canondex.scene.execute_motion import RobotContactReports
        self.capture_robot_paths=RobotContactReports(self.sim,capacity=262144,compact=True).bodies
        targets=roots+static+self.capture_robot_paths;capacity=262144;contacts=self.sim.physics_sim_view.create_rigid_contact_view(roots,filter_patterns=[targets]*len(roots),max_contact_data_count=capacity)
        materials=[dict(mass_kg=float(b.root_view.get_masses().numpy().ravel()[0]),friction=float(b.root_view.get_material_properties().numpy()[...,0].min()))for b in bodies]
        initial=[self.assets[k].pose_xyzw for k in self.active]
        return load_physics_config(scene['physics_config']),self.sim,bodies,roots,initial,self.camera,self.camera_pose,materials,native,static,targets,capacity,contacts

    def capture_before_step(self):self.robot.write_data_to_sim()
    def capture_metadata(self):return dict(schema=K.SCHEMA,world_builds=self.build_count,active_paths=[self.paths[k]for k in self.active],bank_size=len(self.bank),sim_step=self.sim.get_physics_step_count(),
        robot_contact_paths=self.capture_robot_paths,robot_handle=id(self.robot),camera_handle=id(self.camera),sim_handle=id(self.sim),active_handles={k:id(self.bank[k])for k in self.active})

    def capture(self,index,scene_path,output):
        self.order.place(index);self.scene_path=Path(scene_path);began=time.monotonic()
        self.capture_impl(Path(scene_path),Path(output));self.order.photograph()
        # Update only reference metadata to measured placements. Native actors
        # and their measured velocities remain untouched after photography.
        from run_template_arm_positive import object_asset
        self.scene=B.read(Path(output)/'observed_scene.json')
        for row in self.scene['objects']:
            self.assets[row['object_id']]=object_asset(SimpleNamespace(object_root=Path(row['object_root'])),np.asarray(row['world_T_object']),row['scale'])
        self.observation=self.state();self.observation_path=Path(output)/'LIVE_STATE.json'
        self.observed_handles=self.capture_metadata()
        B.save(self.observation_path,self.observation,exclusive=True)
        self.events.append(dict(kind='photograph',state_sha256=K.json_sha(self.observation),sim_step=self.observation['sim_step'],world_builds=self.build_count,wall_s=time.monotonic()-began))
        self.save('observed');return dict(status='complete',capture=B.binding(Path(output)/'RESULTS.json'),live_state=B.binding(self.observation_path))

    def validate_entry(self):
        self.entry_proof=K.live_entry_errors(self.state(),self.observation)
        if self.capture_metadata()!=self.observed_handles:raise ValueError('Native actors/camera changed after observation')
        self.entry_proof.update(world_builds=self.build_count,boundary_resets=self.boundary_resets,handles=self.observed_handles)
        return self.entry_proof

    def execute(self,prepared):
        import author_verified_execution as V
        import execute_template_arm_positive as native
        import author_lift as L
        prepared=Path(prepared)
        if not(prepared/'READY.json').exists():
            self.order.finish();self.save('skipped_no_plan');return dict(status='not_executed',label=None,physics_steps=0)
        self.validate_entry();scene=B.read(B.read(prepared/'READY.json')['scene']['path'])
        if scene['scene_id']!=self.scene['scene_id']:raise ValueError('Prepared scene is not the observed live scene')
        self.scene=scene;primary=scene['target_object_id'];self.objects.update(object=self.bank[primary],object_path=self.paths[primary],robot_path='/World/Robot')
        self.controller.contacts=None;ready=live_ready_function(self)
        original_native=native.execution_steps;original_launch=V.E._launch_native_gpu0
        with M.installed(),V.install():
            original_factory=V.E.runtime_class
            def factory(module,bindings,journals):
                bindings.update({v['path']:v['sha256']for v in self.p['sources'].values()});base,ast_sha=original_factory(module,bindings,journals);world=self
                class Runtime(base):
                    def factory(inner,*args,**kwargs):raise ValueError('Live execution must never reconstruct its world')
                    def ready(inner):
                        inner.sim,inner.robot=world.sim,world.robot;inner.args=world.args;inner.dt=.004;inner.target_id=primary;inner.initial_restores={}
                        inner.bodies={k:world.bank[k]for k in world.active};inner.assets={k:world.assets[k]for k in world.active};inner.paths={k:world.paths[k]for k in world.active}
                        return ready(inner)
                return Runtime,ast_sha
            def execute_steps(args,*,resident_scene=None,runtime_factory=None):
                self.args=args;args.live_state_observation=self.observation_path;args.restore_object_initial_state=False
                return original_native(args,resident_scene=self.controller,runtime_factory=runtime_factory)
            V.E.runtime_class=factory;V.E._launch_native_gpu0=lambda common:SimpleNamespace(app=DeferredApp(self));native.execution_steps=execute_steps
            try:result=V.E.execute(prepared)
            finally:V.E.runtime_class=original_factory;V.E._launch_native_gpu0=original_launch;native.execution_steps=original_native
        self.order.finish();self.events.append(dict(kind='execution_finished',sim_step=self.sim.get_physics_step_count(),world_builds=self.build_count,label=result.get('label')))
        self.save('round_complete');return result

    def save(self,status):
        B.save(self.root/'WORLD_RECEIPT.json',dict(schema=K.SCHEMA,status=status,Kit_starts=int(self.launcher is not None),world_builds=self.build_count,
            boundary_resets=self.boundary_resets,stage_rebuilds=0,camera_builds=int(self.camera is not None),bank_size=len(self.bank),
            actor_handles={k:id(v)for k,v in self.bank.items()},robot_handle=id(self.robot)if self.sim else None,
            camera_handle=id(self.camera)if self.camera else None,sim_handle=id(self.sim)if self.sim else None,
            sim_steps=self.sim.get_physics_step_count()if self.sim else 0,events=self.events,phase=self.order.phase,deferred_closes=self.deferred_closes))

    def close(self,code=0):
        self.save('closed'if code==0 else'closed_after_failure')
        if self.launcher is not None:self.launcher.app.close(exit_code=code)
