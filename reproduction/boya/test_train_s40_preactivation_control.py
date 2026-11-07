import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(1,'/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
import train_s40_preactivation_control as m

class PreactivationControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.cls,cls.loss_cls,_=m.original.official_components()
    def model(self):return self.cls(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5)
    def test_negative_preactivation_positive_label_has_gradient(self):
        model=self.model().train();model.bn7.weight.data.zero_();model.bn7.bias.data.fill_(-2)
        x=torch.randn(4,480);y=torch.ones(4,dtype=torch.long);target=torch.zeros(4,dtype=torch.long);criterion=self.loss_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5,train_type=1)
        p=model(x)[0];loss=m.original.selected_loss(criterion,p,y,target,1);loss.backward();self.assertEqual(float(model.bn7.bias.grad[0]),0.)
        model.zero_grad(set_to_none=True)
        with m.PreactivationCapture(model)as capture:
            source_output=model(x)[0];pre=capture.take();self.assertTrue(torch.all(pre<0));self.assertTrue(torch.all(source_output==0))
            loss=m.original.selected_loss(criterion,pre,y,target,1);loss.backward();self.assertLess(float(model.bn7.bias.grad[0]),-1.)
        self.assertEqual(len(model.bn7._forward_hooks),0)
    def test_capture_preserves_source_forward_and_strict_state(self):
        torch.manual_seed(42);model=self.model().eval();x=torch.randn(4,480)
        with torch.no_grad():
            original=model(x)[0].clone()
            with m.PreactivationCapture(model)as capture:
                observed=model(x)[0];pre=capture.take();self.assertTrue(torch.equal(original,observed));self.assertTrue(torch.equal(torch.relu(pre),observed))
            reloaded=self.model();reloaded.load_state_dict(model.state_dict(),strict=True);reloaded.eval();self.assertTrue(torch.equal(original,reloaded(x)[0]))
    def test_hook_consumption_and_exception_removal(self):
        model=self.model().eval()
        with self.assertRaisesRegex(RuntimeError,'not consumed'):
            with m.PreactivationCapture(model)as capture:
                model(torch.randn(2,480));model(torch.randn(2,480))
        self.assertEqual(len(model.bn7._forward_hooks),0)
        with m.PreactivationCapture(model)as capture:
            with self.assertRaisesRegex(RuntimeError,'No captured'):capture.take()
            model(torch.randn(2,480));self.assertEqual(tuple(capture.take().shape),(2,1,20))
        self.assertEqual(len(model.bn7._forward_hooks),0)
    def test_plan_before_training_and_frozen_split_indices(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'new';source=Path(tmp)/'source';source.write_text('source')
            data={g:{'split':np.array(['train','internal_check','train'])}for g in m.original.GESTURES};manifest={'mesh_groups':{'train':['fit']*32,'internal_check':['heldout']*8}};sources={str(source):m.original.sha(source)};calls=[]
            def head(g,a,fit,check,output,device,config,components,provenance):
                self.assertTrue((out/'PLAN.json').is_file());self.assertEqual(fit.tolist(),[0,2]);self.assertEqual(check.tolist(),[1]);self.assertEqual(config,m.base.CONFIG);calls.append(g)
                return dict(epoch=100,optimizer_steps=100,fit_metric={'count':2,'mse':.2},internal_check_metric={'count':1,'mse':.3},path='fake',sha256='fake')
            with patch.object(m,'frozen_sources',return_value=sources),patch.object(m.base,'validate_fixed_dataset',return_value=(manifest,data,sources,{})),patch.object(m.original,'official_components',return_value=()),patch.object(m,'train_head',side_effect=head):r=m.train('fake',out,'cpu')
            self.assertEqual(calls,[1,6,7,8]);self.assertEqual(r['new_optimizer_steps'],400);self.assertTrue(r['inputs_and_sources_unchanged']);self.assertFalse(r['all40_refit']);self.assertEqual(r['training_variant'],'preactivation_selected_mse')
    def test_frozen_dataset_remains_364_fit_71_internal(self):
        manifest=json.loads(Path(m.BASE_REPORT).read_text())['dataset_manifest'];path=Path(manifest['path']);self.assertEqual(m.original.sha(path),m.base.DATASET_SHA256)
        dataset=json.loads(path.read_text());counts={}
        for g,e in dataset['gesture_files'].items():
            with np.load(e['path'],allow_pickle=False)as z:
                counts[g]={s:int(np.sum(z['split']==s))for s in ['train','internal_check']}
                fit=set(z['mesh_sha256'][z['split']=='train']);held=set(z['mesh_sha256'][z['split']=='internal_check']);self.assertFalse(fit&held)
        self.assertEqual(counts,{'1':{'train':83,'internal_check':15},'6':{'train':98,'internal_check':22},'7':{'train':98,'internal_check':19},'8':{'train':85,'internal_check':15}})
if __name__=='__main__':unittest.main()
