"""Original native/RRT planning with its supported measured six-arm start flag.

No planner, geometry gate or grasp command is patched. The existing native
prepare call receives --initial-arm-state; its own input SHA, full geometry,
RRT and timing checks bind and validate the measured start.
"""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import author_collection_batch as B


def run(requests,output):
    import author_resident_prepare as P
    if not 1<=len(requests)<=8:raise ValueError('One to eight explicit planning requests required')
    P.bind_backend(requests[0])
    from author_compiled_kinematics import PlanningSession
    import author_planning_reuse as reuse
    from canondex.robots.reference import ARM_HOME_JOINTS
    session=PlanningSession();rows=[];began=time.monotonic();cfg=B.read(requests[0]['backend'])
    session.bind(Path(cfg[n])for n in ('robot_urdf','hand_urdf','hand_scene','package'))
    original=reuse.NativePrepareProxy.run
    with session.installed():
        for request in requests:
            live_path=Path(request['observation']).parent/'LIVE_STATE.json';live=B.read(live_path)
            if live['scene_id']!=B.read(request['scene'])['scene_id']:raise ValueError('Measured start belongs to another observation')
            measured=dict(zip(live['joint_names'],live['joint_pos']))
            arm={n:float(measured[n])for n in ARM_HOME_JOINTS}
            if not np.isfinite(list(arm.values())).all():raise ValueError('Finite measured six-axis start required')
            path=Path(request['output_dir']).parent/'MEASURED_ARM_START.json';B.save(path,arm,exclusive=True)
            session.bind([path,live_path]);calls=[]
            def native(proxy,command,**kwargs):
                values=list(map(str,command));entry=str(B.HERE/'native_arm.py')
                if entry in values and '--stage'in values and values[values.index('--stage')+1]=='prepare':
                    if '--initial-arm-state'in values:raise ValueError('No duplicate initial-arm override')
                    actual=values+['--initial-arm-state',str(path.resolve())]
                    calls.append(dict(original_command=values,actual_command=actual,measured_start=B.binding(path),live_observation=B.binding(live_path)))
                    B.save(Path(request['output_dir']).parent/'NATIVE_START_COMMANDS.json',dict(calls=calls))
                    return original(proxy,actual,**kwargs)
                return original(proxy,command,**kwargs)
            with patch.object(reuse.NativePrepareProxy,'run',native):row=P.prepare(request,session)
            row.update(measured_arm_start=B.binding(path),live_state=B.binding(live_path),original_native_start_flag=True,
                initial_state_override_calls=len(calls),source_geometry_gates_unchanged=True,physics_steps=0)
            B.save(Path(request['output_dir'])/'MEASURED_START_RECEIPT.json',row,exclusive=True)
            rows.append(row)
            if row['status']=='infrastructure_failure':raise RuntimeError('Original measured-start planning infrastructure failed')
    B.save(output,dict(status='complete',jobs=rows,wall_s=time.monotonic()-began,cache=session.receipt(),
        Kit_starts=0,physics_steps=0,labels=0,training_updates=0,implementation=B.binding(__file__)),exclusive=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(B.read(a.requests),a.output)
