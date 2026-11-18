"""Physical between-round return, with original numeric gates and full contacts."""
import json
from pathlib import Path
import time

import numpy as np
import author_mujoco_control as C


def return_to_start(world, previous, initial, output):
    import mujoco
    if world.phase!='boundary':raise ValueError('Return requires a finished round')
    output=Path(output)
    # External removal occurs at the round boundary. Robot state is untouched.
    world.exchange(dict(scene_id='external_clear_between_rounds',objects=[]))
    targets=np.r_[previous[::-1],np.linspace(previous[0],initial,101)[1:]]
    velocity=C.source_velocity_targets(targets,world.names,world.arm_names)
    desired=[world.planning.fk.transforms(world.planning.command_state(dict(zip(world.names,row))),
        world.reference.base_transform)['palm'] for row in targets]
    rows=[];stop=None;began=time.monotonic();step_s=contact_s=0.
    with output.open('x') as log:
        for i,(target,speed) in enumerate(zip(targets,velocity),1):
            t=time.perf_counter();world.step(target,speed);step_s+=time.perf_counter()-t
            t=time.perf_counter();contacts=C.contact_rows(world.model,world.data,world.mapping);contact_s+=time.perf_counter()-t
            row=dict(step=i,phase='physical_return',world_step=world.steps,time_s=i*.004,
                native_time_s=float(world.data.time),qpos=world.data.qpos.tolist(),qvel=world.data.qvel.tolist(),
                joint_position=world.data.qpos[world.qindices].tolist(),joint_velocity=world.data.qvel[world.vindices].tolist(),
                joint_target=target.tolist(),velocity_target=speed.tolist(),contacts=contacts,
                base_transform=C.pose_matrix(world.model,world.data,mujoco.mj_name2id(world.model,
                    mujoco.mjtObj.mjOBJ_BODY,world.planning.fk.root)).tolist())
            rows.append(row);log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            if max([0.]+[-c['separation_m'] for c in contacts])>.005:
                stop='original_5mm_deep_collision_stop';break
    loop_s=time.monotonic()-began
    # All bank bodies are parked: retain their contact identities for detecting
    # unexpected arm support rather than treating empty active objects specially.
    maxima,first=C.numeric_maxima(rows,world.planning,list(world.planning.joints),desired,
        world.reference,set(),list(world.bodyids))
    gates=C.source_gates(maxima,len(rows),len(targets))
    ids=C.R.joint_addresses(world.model,world.names)
    return dict(steps=len(rows),expected_steps=len(targets),stop=stop,gates=gates,maxima=maxima,first_failure=first,
        wall_s=time.monotonic()-began,loop_wall_s=loop_s,step_s=step_s,contact_read_s=contact_s,
        robot_state_writes=0,scene_reset_calls=0,clock_reset=False,trace=C.B.binding(output),
        final_motor_error_rad=float(np.max(np.abs(world.data.qpos[ids]-initial))),
        source_control='Reverse complete prior target sequence, then original 100-tick linear alignment; no state teleport',
        pass_=stop is None and all(gates.values()))
