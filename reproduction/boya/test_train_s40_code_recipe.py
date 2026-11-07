import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec=importlib.util.spec_from_file_location('code_recipe','/tmp/train_s40_code_recipe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RecipeBindingTests(unittest.TestCase):
    def test_published_ast_defaults_and_epoch_scheduler(self):
        r=m.recipe_from_ast(m.ROOT/'train_multifinger.py')
        self.assertEqual(r['defaults'],{'--max_epoch':100,'--batch_size':128,'--learning_rate':.0005,'--weight_decay':.0005})
        self.assertEqual(r['scheduler_step_unit'],'epoch')
        self.assertEqual((r['original_output_count'],r['boya_output_count']),(240,20))
        self.assertEqual(len(m.frozen_sources()),8)

    def test_ast_rejects_default_or_scheduler_drift(self):
        text=(m.ROOT/'train_multifinger.py').read_text()
        for altered in [text.replace('default=100,','default=20,'),text.replace('power=0.99','power=0.9'),text.replace('lr=BASE_LEARNING_RATE','lr=0.001')]:
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'changed.py';p.write_text(altered)
                with self.assertRaises(ValueError):m.recipe_from_ast(p)

    def test_existing_output_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'New output'):
                m.train('/missing',Path(tmp))

    def test_dataset_source_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'original435'):
                m.validate_fixed_dataset(p)

    def test_fixed_fit_only_loop_and_plan_before_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new';source=Path(tmp)/'source';source.write_text('source')
            data={g:{'split':np.array(['train','internal_check','train'])} for g in m.original.GESTURES}
            manifest={'mesh_groups':{'train':['a']*32,'internal_check':['b']*8}}
            calls=[]
            def fake_head(g,stage,a,fit,check,output,device,config,components,provenance,benchmark):
                self.assertTrue((out/'PLAN.json').is_file())
                plan=json.loads((out/'PLAN.json').read_text())
                self.assertEqual(plan['selected_epoch'],100)
                self.assertEqual(config,m.CONFIG)
                self.assertEqual(stage,'internal_check')
                self.assertEqual(fit.tolist(),[0,2]);self.assertEqual(check.tolist(),[1])
                self.assertFalse(benchmark);calls.append(g)
                return dict(epoch=100,optimizer_steps=100,fit_metric={'count':2,'mse':.2},
                    internal_check_metric={'count':1,'mse':.3},path='fake',sha256='fake')
            sources={str(source):m.original.sha(source)}
            with patch.object(m,'frozen_sources',return_value=sources),\
                 patch.object(m,'validate_fixed_dataset',return_value=(manifest,data,sources,{})),\
                 patch.object(m.original,'official_components',return_value=()),\
                 patch.object(m.original,'train_head',side_effect=fake_head),\
                 patch.object(m.torch.cuda,'is_available',return_value=True),\
                 patch.object(m.torch.cuda,'get_device_name',return_value='test'),\
                 patch.object(m.torch.cuda,'get_device_capability',return_value=(12,0)):
                r=m.train('fake',out)
            self.assertEqual(calls,[1,6,7,8]);self.assertEqual(r['heads'],{})
            self.assertEqual(r['new_optimizer_steps'],400)
            self.assertAlmostEqual(r['internal_weighted_metric']['mse'],.3)
            self.assertTrue(r['inputs_and_sources_unchanged']);self.assertTrue(r['plan_unchanged'])

if __name__=='__main__':unittest.main()
