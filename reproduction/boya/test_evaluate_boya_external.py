import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,'/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
import evaluate_boya_external as m
from evaluate_s40_hand_domain import geometry_worker

class ExternalEvaluationContractTests(unittest.TestCase):
    def manifests(self):
        objects=[dict(object_id=f'fresh_{i}',observation_json='/not/read',representation_dir='/not/read',mesh_sha256=f'{i+1:064x}',split='external_validation')for i in range(3)]
        manifest=dict(schema=m.MANIFEST_SCHEMA,object_count=3,approved_object_ids=[r['object_id']for r in objects],objects=objects,exclusion_manifest_refs=[dict(path='/not/read',sha256='a'*64)])
        protocol=dict(schema=m.PROTOCOL_SCHEMA,objects=[{'object_id':r['object_id']}for r in objects],object_count=3,arm_budget_per_object=4,physical_attempts_per_object=1,outcome_dependent_retry=False,benchmark_only=True,training_writeback=False,head_recipe=m.VARIANT)
        return manifest,protocol
    def test_generic_external_identity_denominator_not_internal_eight(self):
        manifest,protocol=self.manifests();objects,ids=m.validate_external_manifests(manifest,protocol)
        self.assertEqual(ids,['fresh_0','fresh_1','fresh_2']);self.assertEqual(len(objects),3)
        manifest['approved_ids']=list(reversed(ids))
        with self.assertRaisesRegex(ValueError,'Conflicting'):m.validate_external_manifests(manifest,protocol)
    def test_protocol_cannot_expand_budget_retry_or_drop_failed_objects(self):
        for field,value in [('arm_budget_per_object',8),('physical_attempts_per_object',2),('outcome_dependent_retry',True),('training_writeback',True),('benchmark_only',False),('object_count',2)]:
            manifest,protocol=self.manifests();protocol[field]=value
            with self.assertRaises(ValueError):m.validate_external_manifests(manifest,protocol)
        manifest,protocol=self.manifests();manifest['objects'][0]['training_label']=0
        with self.assertRaisesRegex(ValueError,'precede grasp outcomes'):m.validate_external_manifests(manifest,protocol)
    def test_mesh_identity_exclusion_covers_source_and_prepared_without_all_shas(self):
        doc={'objects':[{'object_id':'used','mesh_sha256':'a'*64,'source_visual_mesh_sha256':'b'*64,'source_input_sha256':{'/image.png':'c'*64}}]}
        self.assertEqual(m.mesh_shas(doc),{'a'*64,'b'*64});self.assertEqual(m.object_ids(doc),{'used'})
        with self.assertRaisesRegex(ValueError,'Invalid mesh'):m.mesh_shas({'mesh_sha256':'not-a-sha'})
    def test_real_control_report_and_checkpoint_metadata_frozen(self):
        report=json.loads((m.ROOT/'outputs/s40_preactivation_control_20261107_v1/TRAINING_RESULTS.json').read_text());allowed,internal=m.validate_training(report)
        self.assertEqual((len(allowed),len(internal)),(32,8));self.assertFalse(allowed&internal)
        checkpoint=m.torch.load(report['internal_heads']['6']['path'],map_location='cpu',weights_only=True)
        fit=m.validate_checkpoint(checkpoint,6,report['dataset_manifest']['sha256'],allowed,internal,{'f'*64})
        with self.assertRaisesRegex(ValueError,'disjoint'):m.validate_checkpoint(checkpoint,6,report['dataset_manifest']['sha256'],allowed,internal,fit)
        changed=copy.deepcopy(checkpoint);changed['training_variant']='relu_original'
        with self.assertRaises(ValueError):m.validate_checkpoint(changed,6,report['dataset_manifest']['sha256'],allowed,internal,set())
    def test_observation_identity_and_mesh_binding(self):
        manifest,_=self.manifests();row=manifest['objects'][0];reader=m.Reader()
        with self.assertRaisesRegex(ValueError,'another external'):m.validate_scene_binding(row,{'object_id':'different'},reader)
        with self.assertRaisesRegex(ValueError,'mesh differs'):m.validate_scene_binding(row,{'object_id':row['object_id'],'object_geometry':{'source_mesh_sha256':'e'*64}},reader)
    def test_reuses_original_full_geometry_worker_and_existing_output_rejected(self):
        self.assertIs(m.geometry_worker,geometry_worker)
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaisesRegex(ValueError,'Fresh external'):m.evaluate('/missing','/missing','/missing',tmp,'/missing','/missing')

if __name__=='__main__':unittest.main()
