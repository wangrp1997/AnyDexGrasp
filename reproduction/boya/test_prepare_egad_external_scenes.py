"""External-selection, unit conversion and backend-layout admission contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
import trimesh

SOURCE = Path(__file__).with_name('prepare_egad_external_scenes.py')
spec = importlib.util.spec_from_file_location('egad_external_builder', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExternalAssetChecks(unittest.TestCase):
    def test_fixed_selection_is_grid_not_geometry_or_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=[]
            for letter in 'ABCDEFG':
                for col in range(7):
                    oid=f'{letter}{col}';p=Path(tmp)/(oid+'.obj');p.write_text(oid)
                    rows.append(dict(object_id=oid,path=str(p),sha256=m.sha(p)))
            inv=dict(schema='anydex_official_egad_eval_asset_inventory_v1',object_count=49,
                     object_ids=[r['object_id']for r in rows],objects=rows)
            selected=m.validate_inventory(inv)
            self.assertEqual([r['object_id']for r in selected],list(m.FIXED_IDS))
            self.assertEqual(len(selected),12)
            reordered=dict(inv,objects=list(reversed(rows)))
            with self.assertRaises(ValueError):m.validate_inventory(reordered)
            Path(rows[-1]['path']).write_text('changed')
            with self.assertRaises(ValueError):m.validate_inventory(inv)

    def test_uniform_meter_conversion_preserves_native_offset_and_triangles(self):
        source=trimesh.creation.box(extents=[100.,80.,60.]);source.vertices += [23.,47.,59.]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.obj';p.write_text(trimesh.exchange.obj.export_obj(source,digits=16))
            original,scaled=m.scaled_visual(p)
            np.testing.assert_array_equal(scaled.faces,original.faces)
            np.testing.assert_array_equal(scaled.vertices,np.asarray(original.vertices)*.001)
            self.assertGreater(scaled.vertices[:,1].min(),0.)
            self.assertNotEqual(float(scaled.vertices[:,0].mean()),0.)

    def make_job(self,base):
        source=trimesh.creation.box(extents=[80.,60.,40.]);source.vertices += [40.,30.,20.]
        p=base/'native.obj';p.write_text(trimesh.exchange.obj.export_obj(source,digits=16))
        return dict(object_id='egad_A0',source=dict(object_id='A0',path=str(p),sha256=m.sha(p)),
            object_root=str(base/'assets'/'egad_A0'),inventory_sha256='f'*64,run_dir=str(base/'run'),
            exclusion=dict(source_hashes=[],meter_geometry_hashes=[]))

    def test_backend_layout_identity_scale_external_identity_and_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.make_job(Path(tmp));_,visual=m.scaled_visual(job['source']['path'])
            result=m.package(job,[(visual.vertices,visual.faces)],dict(test_geometry_only=True))
            self.assertEqual(result['split'],'external_validation')
            self.assertEqual(result['source_mesh_sha256'],job['source']['sha256'])
            self.assertEqual(result['scale'],1.)
            root=Path(job['object_root'])
            self.assertTrue((root/'mesh/simplified.obj').is_file())
            self.assertTrue((root/'urdf/coacd.urdf').is_file())
            import xml.etree.ElementTree as ET
            xml=ET.parse(root/'urdf/coacd.urdf').getroot()
            collisions=xml.findall('link/collision')
            self.assertEqual(len(collisions),1)
            self.assertEqual(collisions[0].find('geometry/mesh').get('scale'),'1 1 1')
            self.assertEqual(collisions[0].find('origin').get('xyz'),'0 0 0')
            T=np.asarray(result['world_T_object'])
            np.testing.assert_array_equal(T[:3,:3],np.eye(3))
            world=visual.vertices+T[:3,3]
            np.testing.assert_allclose((world[:,:2].min(0)+world[:,:2].max(0))/2,[-.75,.1])
            self.assertAlmostEqual(world[:,2].min(),0.)
            self.assertEqual(result['observation_json'],str(Path(job['run_dir'])/'observations/egad_A0/observation.json'))
            self.assertFalse(result['grasp_outcome_consulted'])

    def test_meter_geometry_overlap_rejected_without_shape_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.make_job(Path(tmp));_,visual=m.scaled_visual(job['source']['path'])
            job['exclusion']['meter_geometry_hashes']=[m.meter_geometry_sha(visual)]
            with self.assertRaisesRegex(ValueError,'overlaps'):
                m.package(job,[(visual.vertices,visual.faces)],{})

    def test_invalid_convex_component_is_retained_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=self.make_job(Path(tmp))
            with self.assertRaisesRegex(ValueError,'Invalid CoACD'):
                m.package(job,[(np.zeros((3,3)),np.array([[0,1,2]]))],{})
            self.assertTrue(Path(job['object_root']).exists())

    def test_failed_worker_receipt_has_no_grasp_training_or_network_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);job=self.make_job(base);jobfile=base/'JOB.json';m.save(jobfile,job)
            with mock.patch('coacd.run_coacd',side_effect=RuntimeError('preparation_only_failure')):
                self.assertEqual(m.worker(jobfile),2)
            result=json.loads((base/'RESULT.json').read_text())
            self.assertEqual(result['status'],'asset_preparation_failed')
            self.assertFalse(result['complete'])
            self.assertEqual((result['physics_steps'],result['training_steps'],result['network_forwards']),(0,0,0))
            self.assertEqual(result['object_id'],'egad_A0')


if __name__=='__main__':
    unittest.main(verbosity=2)
