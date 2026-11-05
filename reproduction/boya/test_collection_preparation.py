"""Zero-physics collection-contract tests using real CGR/calibration artifacts.

Native planning is replaced only in queue tests; no Isaac/model imports, grasp
execution or training occurs. Synthetic commands are never training records.
Run with the existing Boya venv and -B. Fixtures live only under /tmp.
"""
from pathlib import Path
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
HERE=ROOT/'reproduction/boya'
sys.path.insert(0,str(HERE))
import asset_adapter as adapter
import build_collection_pool as builder
import prepare_collection_queue as queue
import trial_record
REAL_POOL=ROOT/'outputs/boya_collection_backend_20261105_v1/pools/box_g6_d1'
CAL=ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json'

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,allow_nan=False)+'\n')

class CollectionPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not REAL_POOL.exists() or not CAL.exists():
            raise unittest.SkipTest('Existing real depth1 pool/calibration not present')
        cls.real_gen=read(REAL_POOL/'generation.json')
        cls.real_obs=read(cls.real_gen['source_observation'])
        cls.cal=read(CAL)
        cls.slots={s['slot_id']:s for s in cls.cal['slots']}
        with np.load(Path(cls.real_gen['source_frame'])/'raw_camera_candidates.npz',allow_pickle=False) as z:
            cls.raw=z['candidates'].copy();cls.features=z['candidate_features'].copy()
        with np.load(REAL_POOL/'candidates.npz',allow_pickle=False) as z:
            cls.pool_arrays={k:z[k].copy() for k in z.files}

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='anydex_collection_contract_')
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.pool=self.base/'pool';self.pool.mkdir()
        self.frame=self.base/'frame';self.frame.mkdir();self.out=self.base/'queue';self.out.mkdir()
        self.obs_path=self.base/'observation.json';save(self.obs_path,self.real_obs)
        self.cal_path=self.base/'calibration.json';shutil.copyfile(CAL,self.cal_path)
        # Genuine paired source rows, preserving the original 2261 values.
        source_rows=sorted({r['source_candidate_index'] for r in self.real_gen['records']})
        np.savez_compressed(self.frame/'raw_camera_candidates.npz',
            candidates=self.raw[source_rows],candidate_features=self.features[source_rows])
        self.source=self.frame/'raw_camera_candidates.npz'
        shutil.copyfile(REAL_POOL/'candidates.npz',self.pool/'candidates.npz')
        shutil.copyfile(REAL_POOL/'world_object_pose.json',self.pool/'world_object_pose.json')
        self.gen=copy.deepcopy(self.real_gen)
        self.gen.update(source_frame=str(self.frame),source_observation=str(self.obs_path))
        for r in self.gen['records']:r['source_candidate_index']=source_rows.index(r['source_candidate_index'])
        self.gen['input_sha256']={str(p):adapter.sha(p) for p in [self.obs_path,self.source,self.cal_path,self.pool/'world_object_pose.json']}
        save(self.pool/'generation.json',self.gen)
        cfg=read(ROOT/'reproduction/boya/backend.example.json')
        arm=self.base/'arm.json';save(arm,{'synthetic_arm_refinement_fixture':True})
        cfg['arm_collision_refinement']=str(arm)
        self.cfg=self.base/'backend.json';save(self.cfg,cfg)
        self.job=dict(trial_id='trial_001',object_id=self.real_obs['object_id'],
            candidate_dir=str(self.pool),observation_json=str(self.obs_path),
            source_npz=str(self.source),calibration=str(self.cal_path),gesture_id=6,
            extra_depth_index=1,split='train',max_arm_plans=2,candidate_indices=[0,1])
        self.calls=[];self.original_commands={}

    def fake_plan(self,cmd,**kwargs):
        self.calls.append(cmd);i=int(cmd[cmd.index('--index')+1])
        d=Path(cmd[cmd.index('--output-dir')+1]);d.mkdir()
        arrays=dict(initial_position=np.zeros(22,dtype=np.float64),joint_names=np.array(['q'+str(k) for k in range(22)]),
            time_s=np.array([.004,.008,.012]),joint_target=np.arange(66,dtype=np.float64).reshape(3,22)/1000,
            phase=np.array(['approach','close','lift']),desired_palm=np.repeat(np.eye(4)[None],3,axis=0))
        self.original_commands[i]={k:v.copy() for k,v in arrays.items()}
        np.savez_compressed(d/'commands.npz',**arrays)
        save(d/'plan.json',dict(complete=True,admitted=i==1,commands_sha256=adapter.sha(d/'commands.npz')))
        return SimpleNamespace(returncode=0)

    def assert_rejected_before_plan(self,job=None):
        with patch.object(queue.subprocess,'run',side_effect=AssertionError('Native planner must not be dispatched')):
            with self.assertRaises(ValueError):queue.prepare_one(job or self.job,self.out,self.cfg)

    def test_real_all_four_depths_follow_world_approach(self):
        camera=np.asarray(self.real_obs['T_WC'])
        for meta in self.real_gen['records']:
            row=self.raw[meta['source_candidate_index']].copy()
            if meta['if_flip']:row[3:12].reshape(3,3)[:,1:3]*=-1
            slot=self.slots[meta['source_slot']]
            # Independent polar projection of the saved float32 CGR rotation.
            u,_,vh=np.linalg.svd(row[3:12].reshape(3,3).astype(float));rotation=u@vh
            world_axis=camera[:3,:3]@rotation[:,0]
            zero=adapter.map_slot(row,camera,slot,insertion_m=0.)
            for depth in range(4):
                got=adapter.map_slot(row,camera,slot,insertion_m=.01*depth)
                np.testing.assert_allclose(got[:3,3]-zero[:3,3],world_axis*.01*depth,rtol=0,atol=3e-16)
                np.testing.assert_array_equal(got[:3,:3],zero[:3,:3])
                # The contact reference includes raw depth exactly once.
                contact=got@np.asarray(slot['palm_T_reference'])
                expected=camera[:3,:3]@(row[12:15]+rotation[:,0]*(float(row[2])+.01*depth))+camera[:3,3]
                np.testing.assert_allclose(contact[:3,3],expected,rtol=0,atol=3e-16)

    def test_saved_real_depth1_pose_matches_calibrated_mapping(self):
        for i,meta in enumerate(self.real_gen['records']):
            row=self.raw[meta['source_candidate_index']].copy()
            if meta['if_flip']:row[3:12].reshape(3,3)[:,1:3]*=-1
            expected=adapter.map_slot(row,self.real_obs['T_WC'],self.slots[meta['source_slot']],insertion_m=.01)
            np.testing.assert_array_equal(expected,self.pool_arrays['proposal_palm'][i])
            self.assertEqual(meta['additional_depth_index'],1)
            self.assertEqual(meta['additional_depth_m'],.01)

    def test_each_shift_rechecks_table_and_visible_cloud(self):
        verts=np.array([[x,y,z] for x in [-.005,.005] for y in [-.005,.005] for z in [-.005,.005]])
        eq=np.array([[1,0,0,-.005],[-1,0,0,-.005],[0,1,0,-.005],[0,-1,0,-.005],[0,0,1,-.005],[0,0,-1,-.005]])
        cache=SimpleNamespace(shapes=lambda assets,q:[(verts,verts.min(0),verts.max(0),eq)])
        palm=np.eye(4);palm[2,3]=.006;direction=np.array([0,0,-1.])
        self.assertTrue(builder.geometry(None,[],palm,np.empty((0,3)),direction,cache)['ok'])
        shifted=palm.copy();shifted[2,3]-=.01
        self.assertEqual(builder.geometry(None,[],shifted,np.empty((0,3)),direction,cache)['reason'],'table')
        hit=np.array([[0.,0.,.006]])
        self.assertEqual(builder.geometry(None,[],palm,hit,direction,cache)['reason'],'visible_scene_point_in_hand_hull')

    def test_ready_plan_keeps_original_arrays_and_correct_binding(self):
        with patch.object(queue.subprocess,'run',side_effect=self.fake_plan):result=queue.prepare_one(self.job,self.out,self.cfg)
        self.assertEqual(result['status'],'ready');self.assertIsNone(result['training_label'])
        self.assertEqual(result['actual_physics_steps'],0);self.assertEqual(len(self.calls),2)
        binding=read(result['binding_json']);self.assertEqual(binding['multifinger_pose_depth_type'],1)
        self.assertEqual(binding['multifinger_pose_finger_type'],6);self.assertEqual(binding['extra_depth_m'],.01)
        self.assertEqual(len(binding['canonical_CGR480']),480)
        with np.load(Path(result['plan_dir'])/'commands.npz',allow_pickle=False) as z:
            for k,v in self.original_commands[1].items():
                np.testing.assert_array_equal(z[k],v);self.assertEqual(z[k].dtype,v.dtype)
            self.assertEqual(str(z['trial_binding_sha256'].item()),binding['trial_binding_sha256'])
        self.assertTrue(result['inputs_unchanged'])
        with self.assertRaises(FileExistsError):queue.prepare_one(self.job,self.out,self.cfg)

    def test_stale_observation_is_rejected_not_rehashed(self):
        x=read(self.obs_path);x['new_unbound_field']='edited';save(self.obs_path,x)
        self.assert_rejected_before_plan()

    def test_other_observation_is_rejected(self):
        other=self.base/'other_obs.json';shutil.copyfile(self.obs_path,other)
        self.assert_rejected_before_plan({**self.job,'observation_json':str(other)})

    def test_other_source_features_are_rejected(self):
        other=self.base/'other_source.npz';shutil.copyfile(self.source,other)
        self.assert_rejected_before_plan({**self.job,'source_npz':str(other)})

    def test_other_calibration_contents_are_rejected(self):
        other=self.base/'other_calibration.json';x=read(self.cal_path);x['unbound_extra']='changed calibration';save(other,x)
        self.assert_rejected_before_plan({**self.job,'calibration':str(other)})

    def test_wrong_object_identity_is_rejected(self):
        self.assert_rejected_before_plan({**self.job,'object_id':'wrong_source_mesh'})

    def test_wrong_declared_extra_depth_is_rejected(self):
        self.assert_rejected_before_plan({**self.job,'extra_depth_index':2})

    def test_wrong_gesture_is_rejected(self):
        self.assert_rejected_before_plan({**self.job,'gesture_id':1})

    def test_malformed_trial_id_cannot_escape_queue(self):
        self.assert_rejected_before_plan({**self.job,'trial_id':'../escape_trial'})
        self.assertFalse((self.base/'escape_trial').exists())

    def test_no_candidate_is_null_but_wrong_observation_rejected(self):
        (self.pool/'generation.json').unlink()
        save(self.pool/'no_candidate.json',dict(gesture_id=6,additional_depth_index=1,status='no_hand_geometry_in_declared_budget',
             physics_steps=0,training_label=None,source_candidates_sha256=adapter.sha(self.source),
             source_observation_sha256=adapter.sha(self.obs_path),calibration_sha256=adapter.sha(self.cal_path)))
        with patch.object(queue.subprocess,'run',side_effect=AssertionError('No pool means no planner')):
            result=queue.prepare_one(self.job,self.out,self.cfg)
        self.assertIsNone(result['training_label']);self.assertEqual(result['actual_physics_steps'],0)
        x=read(self.obs_path);x['edited']=True;save(self.obs_path,x)
        self.assert_rejected_before_plan({**self.job,'trial_id':'trial_002'})

    def test_duplicate_trial_id_is_rejected_before_queue_creation(self):
        jobs=self.base/'jobs.json';save(jobs,dict(backend_config=str(self.cfg),jobs=[self.job,self.job]))
        output=self.base/'new_queue'
        with patch.object(sys,'argv',['prepare_collection_queue.py','--jobs-json',str(jobs),'--output-dir',str(output)]):
            with self.assertRaises(ValueError):queue.main()
        self.assertFalse(output.exists())

    def test_selected_record_depth_cannot_disagree_with_pool(self):
        self.gen['records'][1]['additional_depth_index']=2
        save(self.pool/'generation.json',self.gen)
        self.assert_rejected_before_plan()

    def test_input_change_during_native_preparation_is_rejected(self):
        def mutate(cmd,**kwargs):
            result=self.fake_plan(cmd,**kwargs)
            x=read(self.obs_path);x['changed_during_preparation']=True;save(self.obs_path,x)
            return result
        with patch.object(queue.subprocess,'run',side_effect=mutate):
            with self.assertRaises(ValueError):queue.prepare_one(self.job,self.out,self.cfg)
        self.assertNotEqual(read(self.out/'trial_001/preparation.json')['status'],'ready')

    def test_existing_queue_directory_is_not_overwritten_or_dispatched(self):
        jobs=self.base/'jobs.json';save(jobs,dict(backend_config=str(self.cfg),jobs=[self.job]))
        marker=self.out/'prior_receipt';marker.write_text('retain')
        with patch.object(sys,'argv',['prepare_collection_queue.py','--jobs-json',str(jobs),'--output-dir',str(self.out)]):
            with patch.object(queue.subprocess,'run',side_effect=AssertionError('Existing queue must not dispatch')):
                with self.assertRaises(FileExistsError):queue.main()
        self.assertEqual(marker.read_text(),'retain')

    def test_distinct_trial_ids_may_prepare_repeat_without_copying_labels(self):
        with patch.object(queue.subprocess,'run',side_effect=self.fake_plan):
            a=queue.prepare_one(self.job,self.out,self.cfg)
            b=queue.prepare_one({**self.job,'trial_id':'explicit_replicate'},self.out,self.cfg)
        self.assertEqual(a['trial_binding_sha256'],b['trial_binding_sha256'])
        self.assertNotEqual(a['plan_dir'],b['plan_dir'])
        for row in [a,b]:
            self.assertEqual(row['actual_physics_steps'],0);self.assertIsNone(row['training_label'])
        self.assertFalse(any(self.out.rglob('training_record.json')))

    def test_explicit_ik_rejection_receipt_continues_to_next_candidate(self):
        def reject_first(cmd,**kwargs):
            i=int(cmd[cmd.index('--index')+1])
            if i!=0:return self.fake_plan(cmd,**kwargs)
            self.calls.append(cmd);d=Path(cmd[cmd.index('--output-dir')+1]);d.mkdir()
            save(d/'admission_failure.json',dict(complete=True,kind='arm_plan_rejected',physical_executed=False))
            return SimpleNamespace(returncode=1)
        with patch.object(queue.subprocess,'run',side_effect=reject_first):
            result=queue.prepare_one(self.job,self.out,self.cfg)
        self.assertEqual(result['status'],'ready');self.assertEqual(len(result['plans']),2)
        self.assertIn('admission_failure_sha256',result['plans'][0])
        self.assertIsNone(result['training_label']);self.assertEqual(result['actual_physics_steps'],0)

    def test_missing_ik_rejection_receipt_is_infrastructure_error(self):
        with patch.object(queue.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(RuntimeError):queue.prepare_one(self.job,self.out,self.cfg)
        self.assertIsNone(read(self.out/'trial_001/preparation.json')['training_label'])

    def test_invalid_ik_rejection_receipts_do_not_hide_infrastructure(self):
        invalid=[dict(complete=False,kind='arm_plan_rejected',physical_executed=False),
                 dict(complete=True,kind='unexpected_error',physical_executed=False),
                 dict(complete=True,kind='arm_plan_rejected',physical_executed=True)]
        for i,receipt in enumerate(invalid):
            with self.subTest(receipt=receipt):
                def reject(cmd,**kwargs):
                    d=Path(cmd[cmd.index('--output-dir')+1]);d.mkdir();save(d/'admission_failure.json',receipt)
                    return SimpleNamespace(returncode=1)
                with patch.object(queue.subprocess,'run',side_effect=reject):
                    with self.assertRaises(RuntimeError):
                        queue.prepare_one({**self.job,'trial_id':'bad_receipt_'+str(i)},self.out,self.cfg)

    def test_zero_step_infra_null_and_completed_failure_are_distinct(self):
        ev=dict(total_physics_steps=3,complete=True,completed_phases=['close','lift'],label_eligible=True,task_success=False)
        self.assertEqual(trial_record.execution_label(ev)[0],0)
        for change in [dict(total_physics_steps=0),dict(infrastructure_error='setup'),dict(complete=False),dict(label_eligible=False),dict(completed_phases=['close'])]:
            self.assertIsNone(trial_record.execution_label({**ev,**change})[0])

if __name__=='__main__':unittest.main(verbosity=2)
