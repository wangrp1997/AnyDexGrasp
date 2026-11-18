"""Original CGR network/parser and preprocessing, with honest MuJoCo provenance."""
from pathlib import Path
import time

import author_collection_batch as B


class Stream:
    def __init__(self,args):
        from run_pilot_representation import PersistentOfficialCGR
        began=time.monotonic();self.args=args;self.engine=PersistentOfficialCGR(args)
        self.load_s=time.monotonic()-began;self.rows=[]

    def forward(self,observation,output):
        from run_pilot_representation import read_frame,array_sha,REVISION
        began=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
        points,coords,binding,calibration,files,prep_s=read_frame(dict(observation_json=str(Path(observation).resolve())),self.args)
        raw=self.engine.forward(points,coords,output)
        result=dict(status='representation_forward_complete',scope='Fresh same-world MuJoCo calibrated RGB-D; original CGR identity network/parser',
            full_representation_forward=True,test_time_augmentation_branches=1,hand_head_loaded=False,
            source_revision=REVISION,scene_binding=binding,calibration=calibration,input_files=files,
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items()},
            input_points_sha256=array_sha(points),input_coords_sha256=array_sha(coords),
            checkpoint=self.engine.checkpoint,native_dependencies=self.engine.native,
            representation_overlay=self.engine.overlay,candidate_parser_overlay=self.engine.parser_receipt,
            output_files={n:B.binding(output/n) for n in ('raw_camera_candidates.npz','representation_outputs.npz')},
            physical_grasps_executed=0,new_training_steps=0,**raw)
        result['timing_seconds'].update(input_preparation=prep_s,total_frame_wall=time.monotonic()-began)
        result['persistent_model_reused']=bool(self.rows)
        if B.sha(observation)!=binding['observation_sha256']:raise ValueError('Observation changed during CGR')
        B.verify({k:dict(path=k,sha256=v['sha256']) for k,v in files.items()})
        B.save(output/'RESULTS.json',result,exclusive=True);self.rows.append(B.binding(output/'RESULTS.json'))
        return result

    def receipt(self):
        return dict(model_load_count=1,model_load_wall_s=self.load_s,network_forward_count=self.engine.calls,
            hand_head_load_count=0,training_updates=0,frames=self.rows,checkpoint=self.engine.checkpoint)
