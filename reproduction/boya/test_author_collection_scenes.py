"""Deterministic placement/split/provenance checks; zero Kit, IK or labels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import author_collection_scenes as scenes


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def box(path,extent=.035,tag=''):
    points=np.array([[x,y,z] for x in [-extent,extent] for y in [-extent,extent] for z in [-extent,extent]])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('# '+tag+'\n'+''.join('v '+' '.join(map(str,p))+'\n' for p in points)
        +'f 1 2 3\nf 2 4 3\nf 5 7 6\nf 6 7 8\nf 1 5 2\nf 2 5 6\nf 3 4 7\nf 4 8 7\nf 1 3 5\nf 3 7 5\nf 2 6 4\nf 4 6 8\n')
    return path


def fixture(root):
    rows,natives,groups=[],[],dict(fit=[],internal_check=[])
    physics=root/'physics.json';physics.write_text('{}')
    for index,number in enumerate(sorted(scenes.S40)):
        identity=f'graspnet_{number:03d}';directory=root/identity
        visual=box(directory/'mesh/simplified.obj',tag=identity+' converted')
        source=box(directory/'source.obj',tag=identity+' original')
        collision=box(directory/'urdf/meshes/part.obj',tag=identity+' collision')
        urdf=directory/'urdf/coacd.urdf'
        urdf.write_text('<robot name="fixture"><link name="object"><collision><geometry><mesh filename="meshes/part.obj"/></geometry></collision></link></robot>')
        T=np.eye(4)
        rows.append(dict(object_id=identity,graspnet_object_id=number,scale=1.,split='train',
            source_visual_mesh=str(source),source_visual_mesh_sha256=scenes.sha(source),
            object_root=str(directory),mesh_path=str(visual),mesh_sha256=scenes.sha(visual),
            settling_ready=True,world_T_object=T.tolist()))
        result=write(directory/'native_result.json',dict(complete=True,original_or_refined_within_20um=True,
            binding=dict(inputs=dict(object_id=identity,scale=1.,physics_profile_sha256=hashlib.sha256(b'{}').hexdigest(),
                source_files={'mesh/simplified.obj':scenes.sha(visual),
                'urdf/coacd.urdf':scenes.sha(urdf),'urdf/meshes/part.obj':scenes.sha(collision)}))))
        natives.append(dict(object_id=identity,result_json=str(result),artifacts=[str(result)],repair_path=None,repair_sha256=None))
        groups['fit' if index<32 else 'internal_check'].append(dict(object_id=identity,mesh_sha256=scenes.sha(source)))
    source_manifest=write(root/'objects.json',dict(schema='anydex_boya_s40_scene_manifest_v1',objects=rows,camera={}))
    native_summary=write(root/'native.json',dict(complete=True,results=natives))
    backend=write(root/'backend.json',dict(physics_config=str(physics)))
    split=write(root/'split.json',dict(internal_mesh_groups=groups))
    return source_manifest,native_summary,backend,split


class SceneQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.paths=fixture(self.root)
        self.sources=mock.patch.object(scenes,'_execution_sources',return_value={str(self.paths[2]):scenes.sha(self.paths[2])})
        self.sources.start()
    def tearDown(self):
        self.sources.stop();self.tmp.cleanup()
    def catalog(self):
        return scenes._catalog(*self.paths)

    def test_real_geometry_contained_separated_source_bottom_and_repeatable(self):
        catalog=self.catalog();a=scenes.sample_scene(catalog,'fit',0,2026111401)
        self.assertEqual(a,scenes.sample_scene(catalog,'fit',0,2026111401))
        self.assertEqual(a['status'],'ready')
        self.assertEqual(len(set(a['source_object_ids'])),3)
        for row in a['objects']:
            bounds=np.asarray(row['proxy_world_AABB_m']);domain=np.asarray(scenes.POLICY['xy_AABB_world_m'])
            self.assertTrue(np.all(bounds[0,:2]>=domain[0]))
            self.assertTrue(np.all(bounds[1,:2]<=domain[1]))
            self.assertAlmostEqual(row['source_world_AABB_m'][0][2],.005,places=13)
            self.assertLessEqual(row['attempts'],256)
        for i in range(3):
            for j in range(i):
                self.assertTrue(scenes.gap_clear(a['objects'][i]['proxy_world_AABB_m'],a['objects'][j]['proxy_world_AABB_m'],.003))
        self.assertNotEqual(a['objects'],scenes.sample_scene(catalog,'fit',1,2026111401)['objects'])

    def test_split_isolation_and_index_independence(self):
        catalog=self.catalog()
        fit=set(catalog['split_object_ids']['fit']);internal=set(catalog['split_object_ids']['internal'])
        self.assertTrue(fit.isdisjoint(internal))
        for split,pool in [('fit',fit),('internal',internal)]:
            for index in range(5):
                row=scenes.sample_scene(catalog,split,index,17)
                self.assertTrue(set(row['source_object_ids'])<=pool)
        first=scenes.sample_scene(catalog,'fit',0,17)
        scenes.sample_scene(catalog,'internal',45,17)
        self.assertEqual(first,scenes.sample_scene(catalog,'fit',0,17))

    def test_unplaceable_scene_is_terminal_without_object_or_yaw_replacement(self):
        catalog=self.catalog()
        cache={name:(np.array([[-1.,-1.,-1.],[1.,1.,1.]]),)*2 for name in catalog['objects']}
        failed=scenes.sample_scene(catalog,'fit',0,2026111401,geometry_cache=cache)
        clear=scenes.sample_scene(catalog,'fit',0,2026111401)
        self.assertEqual(failed['status'],'placement_null')
        self.assertEqual(failed['source_object_ids'],clear['source_object_ids'])
        self.assertEqual([r['yaw_world_rad'] for r in failed['objects']],[r['yaw_world_rad'] for r in clear['objects']])
        self.assertEqual([r['attempts'] for r in failed['objects']],[0,0,0])
        self.assertEqual(failed['objects'][1]['placement_status'],'not_attempted_after_scene_null')
        self.assertFalse(failed['placement_null_is_training_label'])

    def test_native_asset_drift_and_split_leak_rejected(self):
        split=scenes.read(self.paths[3]);bad=copy.deepcopy(split)
        bad['internal_mesh_groups']['internal_check'][0]=bad['internal_mesh_groups']['fit'][0]
        write(self.paths[3],bad)
        with self.assertRaisesRegex(ValueError,'leaking'):self.catalog()
        write(self.paths[3],split)
        row=scenes.read(self.paths[0])['objects'][0]
        part=Path(row['object_root'])/'urdf/meshes/part.obj';part.write_text(part.read_text()+'# changed\n')
        with self.assertRaisesRegex(ValueError,'native preflight'):self.catalog()

    def test_queue_prefix_build_and_identity_geometry_are_distinct(self):
        out=self.root/'queue'
        queue=scenes.generate_queue(*self.paths,out,seed=2026111401,fit_scene_count=2,internal_scene_count=1)
        self.assertEqual([s['split'] for s in queue['scenes']],['fit','fit','internal'])
        self.assertEqual(queue['maximum_gesture_scene_attempts'],12)
        self.assertFalse(queue['formal_collection_started'])
        selected=queue['scenes'][0]
        spec_path=selected['scene_spec']['path']
        scene=scenes.build_scene(spec_path,self.root/'scene')
        self.assertEqual(scene['dataset_split'],'fit')
        self.assertIsNone(scene['bin'])
        self.assertEqual(len(scene['objects']),3)
        self.assertEqual(scene['all_source_mesh_sha256'],[r['source_visual_mesh_sha256'] for r in scene['objects']])
        for row in scene['objects']:
            self.assertEqual(row['mesh_group_sha256'],row['source_visual_mesh_sha256'])
            self.assertNotEqual(row['source_mesh_sha256'],row['source_visual_mesh_sha256'])
        spec=scenes.read(spec_path);spec['objects'][0]['world_T_object'][0][3]+=.001;write(spec_path,spec)
        with self.assertRaisesRegex(ValueError,'placement'):scenes.build_scene(spec_path,self.root/'tampered')

    def test_gap_guard_keeps_exact_three_mm_and_rejects_smaller_gap(self):
        a=np.array([[0.,0.,0.],[.1,.1,.1]])
        b=np.array([[.103,0.,0.],[.203,.1,.1]])
        self.assertTrue(scenes.gap_clear(a,b,.003))
        b[:,0]-=.000001
        self.assertFalse(scenes.gap_clear(a,b,.003))


if __name__=='__main__':unittest.main()
