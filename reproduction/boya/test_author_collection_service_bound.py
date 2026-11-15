"""Native source defaults must bind before capture; zero Kit/physics tests."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import author_collection_service_bound as S

B=S.B


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
        self.urdf=self.root/'hand.urdf';self.xml=self.root/'scene.xml'
        self.urdf.write_text('<robot/>');self.xml.write_text('<mujoco/>')
        self.backend=self.root/'backend.json'
        B.save(self.backend,dict(hand_urdf=str(self.urdf),hand_scene=str(self.xml),reference_root=str(self.root)))
        self.inputs=dict(backend=B.binding(self.backend))
        self.calls=[]
        def bind(args):self.calls.append(args)
        self.native=patch.dict(sys.modules,{'run_template_arm_positive':SimpleNamespace(bind_hand=bind)})
        self.native.start();self.addCleanup(self.native.stop)
        self.paths=patch.object(S.T,'_backend_path');self.paths.start();self.addCleanup(self.paths.stop)

    def test_existing_assets_are_bound_before_runtime_constructor(self):
        events=[]
        def constructor(instance,*args):events.append('runtime');self.assertEqual(len(self.calls),1)
        with patch.object(S.ORIGINAL.CollectionRuntime,'__init__',constructor):
            S.BoundRuntime(self.root/'runtime',{},self.inputs)
        value=B.read(self.root/'runtime/NATIVE_REFERENCE_BINDING.json')
        self.assertEqual(value['hand_scene'],B.binding(self.xml));self.assertEqual(value['physics_steps'],0)
        self.assertEqual(events,['runtime'])

    def test_stale_imported_default_is_rejected_before_Kit(self):
        with patch.dict(sys.modules,{'canondex.robots.hand_actuation':SimpleNamespace(HAND_SCENE=Path('/missing/old.xml'))}):
            with self.assertRaisesRegex(ValueError,'Stale hand defaults'):S.bind_references(self.inputs)

    def test_modified_backend_is_rejected_before_binding(self):
        self.backend.write_text('{}')
        with self.assertRaises(ValueError):S.bind_references(self.inputs)
        self.assertEqual(self.calls,[])

    def test_plan_with_another_reference_cannot_execute(self):
        runtime=object.__new__(S.BoundRuntime);runtime.references=S.bind_references(self.inputs);runtime.poisoned=False
        directory=self.root/'prepared';B.save(directory/'plan.json',dict(arguments=dict(hand_urdf=str(self.urdf),
            hand_scene='/missing/old.xml',reference_root=str(self.root))))
        B.save(directory/'READY.json',dict(plan=B.binding(directory/'plan.json')))
        with patch.object(S.ORIGINAL.CollectionRuntime,'run_job',side_effect=AssertionError('Native execution dispatched')):
            with self.assertRaisesRegex(ValueError,'Prepared native reference differs'):
                runtime.run_job(dict(role='execute',prepared_dir=str(directory)))
        self.assertTrue(runtime.poisoned)


if __name__=='__main__':unittest.main()
