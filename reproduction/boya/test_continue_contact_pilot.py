"""No replay of a started or mismatched stage during interruption recovery."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import continue_contact_pilot as K


class ContinuationTests(unittest.TestCase):
    def test_identical_saved_manifest_can_be_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'REPRESENTATION_MANIFEST.json';value={'objects':[{'object_id':'fixed'}]}
            K.B.save(path,value,exclusive=True);before=path.read_bytes()
            with K.preserve_completed_prefix():K.B.save(path,value,exclusive=True)
            self.assertEqual(before,path.read_bytes())

    def test_changed_manifest_and_other_existing_artifacts_still_reject(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'REPRESENTATION_MANIFEST.json';K.B.save(p,{'objects':[]},exclusive=True)
            q=Path(d)/'COLLECTION_RECORD.json';K.B.save(q,{},exclusive=True)
            with K.preserve_completed_prefix():
                with self.assertRaises(ValueError):K.B.save(p,{'objects':['replacement']},exclusive=True)
                with self.assertRaises(FileExistsError):K.B.save(q,{},exclusive=True)

    def test_any_durable_started_CGR_blocks_review_before_resource_release(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);stage=root/'scenes/01/stages/representation';stage.mkdir(parents=True)
            K.B.save(root/'RUN_STARTED.json',{'process':{'pid':123,'pgid':123,'start_ticks':1}})
            K.B.save(root/'resource_slots/observation/0.json',dict(state='reserved',stage_dir=str(stage)))
            K.B.save(stage/'STAGE.json',{'status':'running'})
            with patch.object(K.P,'verify',return_value={}):
                with self.assertRaises(ValueError):K.review(root)
            self.assertEqual(K.B.read(root/'resource_slots/observation/0.json')['state'],'reserved')
            self.assertFalse((root/'INTERRUPTION_REVIEW.json').exists())


if __name__=='__main__':unittest.main()
