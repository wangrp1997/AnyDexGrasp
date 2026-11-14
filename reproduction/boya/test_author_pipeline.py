"""Numerical integration checks; synthetic optimization and source assets only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import author_protocol as author
import author_training as training
import author_collision as collision
from author_hand_clouds import HandCloudLibrary, full_hand_mesh
from asset_adapter import BoyaAssets, sha


def synthetic(fit=301,check=7):
    n = fit+check
    rng = np.random.RandomState(421)
    return dict(features=rng.normal(size=(n,480)).astype(np.float32),
        labels=(np.arange(n)%2).astype(np.int64),target_index=(np.arange(n)%20).astype(np.int64),
        trial_id=np.array([f'synthetic_{i}' for i in range(n)]),
        mesh_sha256=np.array(['fit_source']*fit+['check_source']*check),
        split=np.array(['train']*fit+['internal_check']*check))


class AuthorPipelineTests(unittest.TestCase):
    def test_scaled_data_enters_actual_author_training_and_safe_best_checkpoints(self):
        torch.set_num_threads(2)
        data = synthetic()
        cls,inference,loss,poly = training.components()
        def positive_initialization(**kwargs):
            model = cls(**kwargs)
            # Deliberately exercise all save thresholds in this synthetic
            # integration check. Formal factory initialization stays unchanged.
            with torch.no_grad(): model.bn7.bias.fill_(1.2)
            return model
        with tempfile.TemporaryDirectory(prefix='anydex_author_training_') as tmp:
            for cap,updates in ((100,1),(300,3)):
                with self.subTest(cap=cap),patch.object(training,'components',
                        return_value=(positive_initialization,inference,loss,poly)):
                    result = training.fit_head(data,Path(tmp)/str(cap),gesture=1,
                        sample_cap=cap,seed=108,development_epochs=1)
                    self.assertEqual(result['status'],'development_complete')
                    self.assertEqual(result['fit_records'],cap)
                    self.assertEqual(result['optimizer_updates'],updates)
                    self.assertEqual(set(result['checkpoints']),{'0.5','0.7','0.9'})
                    self.assertTrue(all(result['strict_published_inference_reload'].values()))
                    self.assertEqual(result['fit_trial_ids'],list(data['trial_id'][:cap]))
                    self.assertFalse(result['task_semantics_certified'])

    def test_original_checkpoint_rule_and_no_fallback_are_preserved(self):
        saves = []
        with tempfile.TemporaryDirectory(prefix='anydex_author_selection_') as tmp:
            scope,*_ = training.source_scope(None,None,None,None,None,None,1,tmp,
                lambda model,path:saves.append(path.name),'cpu')
            best = [0.]*9
            rows = [([.4,.02,.03],0),([.4,.06,.1],3),([.395,.07,.11],6),
                    ([.39,.01,.02],6),([.6,.06,.12],9)]
            for epoch,(triplet,count) in enumerate(rows):
                best = scope['save_model'](epoch,np.tile(triplet,3),best)
                self.assertEqual(len(saves),count)
        # A zero-positive validation set is a data/model result, not a reason
        # to change the author selection rule or silently use final weights.
        data = synthetic(fit=2,check=2);data['labels'].fill(0)
        with tempfile.TemporaryDirectory(prefix='anydex_author_no_selection_') as tmp:
            result = training.fit_head(data,Path(tmp)/'run',gesture=1,sample_cap=100,
                seed=123,development_epochs=1,write_checkpoints=False)
        self.assertEqual(result['selected_checkpoint_count'],0)
        self.assertFalse(result['has_deployable_selected_state'])

    def test_external_rows_object_leakage_and_duplicate_trials_rejected(self):
        for mutation in ('external','object','duplicate'):
            with self.subTest(mutation=mutation):
                data = synthetic()
                if mutation=='external':
                    data['split'] = data['split'].astype('U30');data['split'][-1]='external_validation'
                elif mutation=='object': data['mesh_sha256'][-1]=data['mesh_sha256'][0]
                else: data['trial_id'][-1]=data['trial_id'][0]
                with self.assertRaises(ValueError): training.validate_arrays(data)

    def test_full_boya_surface_loader_and_original_detector(self):
        root = author.ROOT/'outputs/boya_full_alignment_20261114_v2/hand_clouds'
        self.assertTrue((root/'MANIFEST.json').is_file(),'Generate the frozen surface library first')
        library = HandCloudLibrary(root/'MANIFEST.json')
        cal = json.loads(Path(library.report['calibration']['path']).read_text())
        self.assertEqual(library.report['states_count'],132)
        self.assertEqual(library.report['total_uniform_samples'],39600000)
        assets = BoyaAssets()
        slot = next(s for s in cal['slots'] if s['gesture_id']==1 and s['sample']==16 and s['opposing_group']=='ff')
        mesh,parts = full_hand_mesh(assets,slot['independent_q'])
        self.assertEqual({p['link'] for p in parts},set(assets.hand.collision_links))
        self.assertEqual(len(mesh.triangles),sum(len(m.faces) for m in assets.meshes.values()))
        self.assertEqual(len(mesh.vertices),sum(len(m.vertices) for m in assets.meshes.values()))
        rows = np.zeros((4,17),dtype=np.float64)
        rows[:,0]=[.9,.8,.7,.6];rows[:,1]=.06;rows[:,2]=.03
        rows[:,3]=[.01,.02,.03,.04];rows[:,4:13]=np.eye(3).reshape(9)
        rows[:,13:16]=[.2,.1,.5];rows[:,16]=-1
        extra = np.array([0.,.01,.02,.03])
        altered = copy.deepcopy(cal)
        altered['slots'][0]['palm_T_reference'][0][3] += .001
        with self.assertRaises(ValueError):
            collision.detect(rows,[1,6,7,8],extra,np.array([[3.,3.,3.]]),
                calibration=altered,library=library,profile='dh3_collection')
        for profile in collision.PROFILES:
            with self.subTest(profile=profile):
                result = collision.detect(rows,[1,6,7,8],extra,np.array([[3.,3.,3.]]),
                    calibration=cal,library=library,profile=profile)
                np.testing.assert_array_equal(result['accepted_source_indices'],np.arange(4))
                # Independently compute the endpoint cloud; original loader
                # normalizes direction and casts depth translation tofloat32.
                loaded = author.definitions('utils/collision_detector.py',
                    {'ModelFreeCollisionDetectorMultifinger'},
                    dict(np=np,copy=copy,math=__import__('math'),time=author.time,
                         o3d=__import__('open3d')))['ModelFreeCollisionDetectorMultifinger']
                mapped = result['hand'].mapped[0]
                key = mapped.slot['slot_id']+'_'+str(round(mapped.width*100,1))
                import open3d as o3d
                cloud = o3d.geometry.PointCloud()
                local = library.world_points(mapped.slot,np.eye(4))
                cloud.points = o3d.utility.Vector3dVector(local)
                detector = loaded(np.empty((0,3)))
                actual = detector.load_meshes_pcls({key:cloud},
                    collision.TwoFingerGroup(rows[:1]),[mapped])[0]
                palm = mapped.palm_before_depth
                expected = local@palm[:3,:3].T+palm[:3,3]
                expected += np.array([mapped.depth,0.,0.],dtype=np.float32)
                np.testing.assert_allclose(np.asarray(actual.points),expected,atol=1e-12,rtol=0)
                if profile=='dh3_deployment':
                    scene = np.asarray(actual.points)[100:101].copy()
                    rejected = collision.detect(rows[:1],[1],extra[:1],scene,
                        calibration=cal,library=library,profile=profile)
                    self.assertEqual(len(rejected['accepted_source_indices']),0)
        self.assertEqual(sha(root/'MANIFEST.json'),sha(library.path))


if __name__=='__main__': unittest.main()
