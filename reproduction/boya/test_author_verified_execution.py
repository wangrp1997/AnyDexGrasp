"""Storage-only adapter restores original hooks and retains cost on failure."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import author_verified_execution as V


class ExecutionAdapterTests(unittest.TestCase):
    def test_install_restores_original_and_returns_exact_finish_contract(self):
        original=V.E.ContactJournal
        class Runtime:
            def finish(self,*args,**kwargs):self.calls+=1;return self.original_result
        with tempfile.TemporaryDirectory()as temp,patch.object(V.E,'runtime_class',return_value=(Runtime,'unchanged_ast'))as factory:
            sources={};journals=[]
            with V.install():
                cls,ast=V.E.runtime_class(None,sources,journals)
                self.assertEqual(ast,'unchanged_ast');self.assertIs(V.E.ContactJournal,V.J.ContactJournal)
                r=cls();r.output_dir=Path(temp);r.steps=2;r.rows=[1,2];r.contact_rows=SimpleNamespace(records=2)
                r.calls=0;r.original_result={'label':1}
                self.assertIs(r.finish(),r.original_result);self.assertEqual(r.calls,1)
                self.assertEqual(set(sources),set(V.source_bindings()))
            self.assertIs(V.E.ContactJournal,original);factory.assert_called_once()
            cost=V.E.json.loads((Path(temp)/'CONTACT_STORAGE_COST.json').read_text())
            self.assertEqual(cost['original_runtime_steps'],2);self.assertEqual(cost['additional_physics_steps'],0)

    def test_original_finish_failure_keeps_reported_tick_cost_without_label(self):
        class Runtime:
            def finish(self,*args,**kwargs):raise ValueError('storage seal failed')
        with tempfile.TemporaryDirectory()as temp,patch.object(V.E,'runtime_class',return_value=(Runtime,'unchanged_ast')):
            with V.install():
                cls,_=V.E.runtime_class(None,{},[]);r=cls();r.output_dir=Path(temp);r.steps=3024
                r.rows=[None]*3024;r.contact_rows=SimpleNamespace(records=3024)
                with self.assertRaises(ValueError):r.finish()
            cost=V.E.json.loads((Path(temp)/'CONTACT_STORAGE_COST.json').read_text())
            self.assertEqual(cost['original_runtime_steps'],3024)
            self.assertEqual(cost['status'],'original_finish_failed');self.assertFalse(cost['labels_certified'])


if __name__=='__main__':unittest.main()
