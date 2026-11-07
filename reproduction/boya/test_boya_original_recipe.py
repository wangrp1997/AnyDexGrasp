import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import evaluate_boya_original_recipe as m

class OriginalRecipeTests(unittest.TestCase):
    def test_real_code100_report_and_checkpoint_accept_modified_loss_reject(self):
        training=json.loads(m.TRAINING_REPORT.read_text());allowed,internal=m.validate_training(training)
        checkpoint=torch.load(training['internal_heads']['1']['path'],map_location='cpu',weights_only=True)
        self.assertTrue(m.validate_checkpoint(checkpoint,1,m.DATASET_SHA,allowed,internal,set()))
        modified=copy.deepcopy(checkpoint);modified['training_variant']='preactivation_selected_mse'
        with self.assertRaisesRegex(ValueError,'Modified-loss'):m.validate_checkpoint(modified,1,m.DATASET_SHA,allowed,internal,set())
        training['schema']='anydex_s40_preactivation_control_v1'
        with self.assertRaisesRegex(ValueError,'code100'):m.validate_training(training)

    def fixture(self):
        rows=[dict(object_id=x,observation_json='/obs',representation_dir='/cached',mesh_sha256='a'*64,split=m.ROLE)for x in m.APPROVED]
        manifest=dict(schema=m.MANIFEST_SCHEMA,objects=rows,approved_object_ids=list(m.APPROVED),exclusion_manifest_refs=[dict(path='/training',sha256='b'*64)])
        protocol=dict(schema=m.PROTOCOL_SCHEMA,objects=list(m.APPROVED),object_count=4,head_recipe=m.RECIPE,evaluation_role=m.ROLE,known_object_diagnostic=True,arm_budget_per_object=4,physical_attempts_per_object=1,outcome_dependent_retry=False,benchmark_only=True,training_writeback=False)
        return manifest,protocol

    def test_known_four_manifest_rejects_unseen_and_label_steering(self):
        manifest,protocol=self.fixture();self.assertEqual(m.validate_manifests(manifest,protocol)[1],list(m.APPROVED))
        for key,value in [('split','external_validation'),('success',True)]:
            changed=copy.deepcopy(manifest);changed['objects'][0][key]=value
            with self.assertRaises(ValueError):m.validate_manifests(changed,protocol)
        protocol['head_recipe']='preactivation_selected_mse'
        with self.assertRaises(ValueError):m.validate_manifests(manifest,protocol)

    def test_pipeline_swap_is_scoped_and_every_saved_artifact_marks_recipe(self):
        original=m.pipeline.VARIANT;original_reader=m.pipeline.Reader
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'artifact.json'
            def fake(*args):
                self.assertEqual(m.pipeline.VARIANT,m.RECIPE)
                reader=m.pipeline.Reader();self.assertIn(str(Path(m.__file__).resolve()),reader.files)
                m.pipeline.save(path,dict(scope='unseen',evaluation_role='external_validation',novelty_audit={}))
                return dict(scope='unseen',evaluation_role='external_validation')
            with patch.object(m.pipeline,'evaluate',side_effect=fake):
                result=m.evaluate('objects','protocol',m.TRAINING_REPORT,'out','cal','policy')
            saved=json.loads(path.read_text());self.assertEqual(saved['head_recipe'],m.RECIPE)
            self.assertFalse(saved['novelty_audit']['independent_unseen_validation']);self.assertEqual(result['evaluation_role'],m.ROLE)
        self.assertEqual(m.pipeline.VARIANT,original);self.assertIs(m.pipeline.Reader,original_reader)
        with patch.object(m.pipeline,'evaluate',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):m.evaluate('x','x',m.TRAINING_REPORT,'x','x','x')
        self.assertEqual(m.pipeline.VARIANT,original);self.assertIs(m.pipeline.Reader,original_reader)

if __name__=='__main__':unittest.main()
