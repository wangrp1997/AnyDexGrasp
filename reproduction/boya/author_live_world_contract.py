"""Pure-Python live-world state and command handover. Imports start no Kit."""
import copy
from pathlib import Path
import numpy as np
import author_collection_batch as B
import author_contact_pipeline as M
from trial_record import json_sha

SCHEMA='anydex_boya_live_world_collection_v1'


class RoundOrder:
    def __init__(self,count):self.count=count;self.index=0;self.phase='boundary'
    def place(self,index):
        if index!=self.index or self.phase!='boundary' or index>=self.count:raise ValueError('Placement only between complete rounds')
        self.phase='settling'
    def photograph(self):
        if self.phase!='settling':raise ValueError('Fresh settled observation required')
        self.phase='observed'
    def finish(self):
        if self.phase!='observed':raise ValueError('Execution requires this round observation')
        self.index+=1;self.phase='boundary'if self.index<self.count else'complete'


def live_entry_errors(actual,observed):
    if actual!=observed:raise ValueError('World changed since RGB-D capture')
    if np.max(np.abs(actual['joint_velocity']))>.02:raise ValueError('Robot was moving at observation')
    for state in actual['objects'].values():
        velocity=np.asarray(state['velocity'])
        if np.max(abs(velocity[:3]))>.002 or np.max(abs(velocity[3:]))>.02:raise ValueError('Observed object was not stationary')
    return dict(sim_step=actual['sim_step'],state_sha256=json_sha(actual),actor_pose_or_velocity_writes=0)


def bind_live_plan(source,observation,output):
    """Fresh plan changes measured initial coordinates only; all commands exact.

    The original 50 settle targets preopen the hand at HOME after photography.
    No pose/velocity write occurs to make the live world fit a prepared plan.
    """
    import shutil
    source=Path(source).resolve();output=Path(output).resolve();obs=B.read(observation)
    ready=B.read(source/'READY.json');protocol=B.read(source/'LIFT_PIPELINE_PROTOCOL.json')
    original=B.read(ready['plan']['path']);candidate=B.read(ready['candidate']['path'])
    old_binding=B.read(ready['command_binding']['path']);contract=M.contract_factory(protocol['inputs']['calibration']['path'])
    if obs.get('schema')!=SCHEMA or obs['scene_id']!=B.read(ready['scene']['path'])['scene_id']:raise ValueError('Live world observation does not match prepared scene')
    arrays=M.LIFT._arrays(Path(ready['commands']['path']).parent)
    for k in ('collection_candidate_sha256','collection_execution_spec_sha256'):arrays.pop(k,None)
    original_initial=arrays['initial_position'].copy();names=list(map(str,arrays['joint_names']))
    measured=dict(zip(obs['joint_names'],obs['joint_pos']))
    for i,name in enumerate(names):
        if name in contract.names or name in ('FAJ1','FAJ3'):arrays['initial_position'][i]=measured[name]
        elif abs(measured[name]-original_initial[i])>2e-5:raise ValueError('Live arm is not at the original planned HOME')
    output.mkdir(parents=True,exist_ok=False);native=output/'native';native.mkdir()
    for name in ('path_geometry.json','lift_arm_geometry.json'):
        shutil.copyfile(Path(ready['plan']['path']).parent/name,native/name)
    protocol=copy.deepcopy(protocol);protocol['inputs']['live_state_observation']=B.binding(observation)
    protocol['inputs']['source_preparation']=B.binding(source/'READY.json')
    protocol['sources'][Path(__file__).name]=B.binding(__file__)
    protocol['persistent_world']=dict(schema=SCHEMA,source_preparation=B.binding(source/'READY.json'),
        observation=B.binding(observation),no_state_reset_after_photograph=True)
    B.save(output/'LIFT_PIPELINE_PROTOCOL.json',protocol,exclusive=True)
    spec=copy.deepcopy(old_binding['execution_spec']);spec['persistent_world']=protocol['persistent_world']
    binding=M.COLLECT.write_bound_commands(candidate,arrays,native/'commands.npz',execution_spec=spec,action_contract=contract)
    plan=copy.deepcopy(original);plan['commands_sha256']=binding['commands_sha256']
    plan['author_collection_binding']['command_binding']=binding
    plan['live_state_observation']=str(Path(observation).resolve())
    plan.setdefault('extra_input_sha256',{}).update({str(Path(observation).resolve()):B.sha(observation),str(Path(__file__).resolve()):B.sha(__file__),
        ready['commands']['path']:ready['commands']['sha256']})
    plan['persistent_world']=dict(schema=SCHEMA,source_plan=B.binding(ready['plan']['path']),
        initial_positions_changed_only=True,original_initial_position=original_initial.tolist(),
        actual_initial_position=arrays['initial_position'].tolist(),observation=B.binding(observation))
    B.save(native/'plan.json',plan,exclusive=True);B.save(native/'COMMAND_BINDING.json',binding,exclusive=True)
    # Native executor locates the original exact arm refinement at plan parent.
    repair=Path(ready['plan']['path']).parent.parent/'arm_collision_refinements.json'
    shutil.copyfile(repair,output/'arm_collision_refinements.json')
    fresh=copy.deepcopy(ready);fresh.update(protocol=B.binding(output/'LIFT_PIPELINE_PROTOCOL.json'),
        plan=B.binding(native/'plan.json'),commands=B.binding(native/'commands.npz'),command_binding=B.binding(native/'COMMAND_BINDING.json'))
    B.save(output/'READY.json',fresh,exclusive=True)
    preparation=copy.deepcopy(B.read(source/'LIFT_PIPELINE_RESULTS.json'));preparation['ready_sha256']=B.sha(output/'READY.json')
    preparation['persistent_world']=protocol['persistent_world'];B.save(output/'LIFT_PIPELINE_RESULTS.json',preparation,exclusive=True)
    # Check every command/phase/palm/time array is exactly the source array.
    with np.load(ready['commands']['path'],allow_pickle=False)as old,np.load(fresh['commands']['path'],allow_pickle=False)as new:
        for key in old.files:
            if key in ('initial_position','collection_execution_spec_sha256'):continue
            if old[key].dtype!=new[key].dtype or old[key].shape!=new[key].shape or old[key].tobytes()!=new[key].tobytes():raise ValueError('Live handover changed original command: '+key)
    return dict(status='ready',prepared_dir=str(output),observation=B.binding(observation),command_arrays_exact=True,
        changed_initial_hand_only=True,physics_steps=0)
