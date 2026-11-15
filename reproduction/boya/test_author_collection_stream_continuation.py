"""The additional group must not dispatch/restart the saved cold control."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import author_collection_stream_continuation as S


class ContinuationTests(unittest.TestCase):
    def test_freeze_has_one_extra_Kit_and_only_new_outputs(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)/'control'
            with patch.object(S.B.subprocess,'Popen',side_effect=AssertionError('Freeze dispatched')):p=S.freeze(root)
            self.assertEqual(p['additional_Kit_starts'],1);self.assertLess(p['max_combined_environment_steps'],35000)
            cp=S.B.read(p['protocol']['path']);self.assertEqual(cp['sources']['service_entry'],S.B.binding(S.SERVICE.__file__))
            self.assertEqual(len(cp['worker_jobs']),4);self.assertFalse((root/'RUN_STARTED.json').exists())
            p['additional_Kit_starts']=2;S.B.save(root/'CONTROL.json',p)
            with self.assertRaises(ValueError):S.verify(root)

    def test_saved_comparison_roots_are_read_only_path_routes(self):
        root=Path('/tmp/example');view=S.SavedRoots(root)
        self.assertEqual(view/'cold/scenes/00',S.SOURCE/'cold/scenes/00')
        self.assertEqual(view/'resident/runtime/WORKER_RECEIPT.json',root/'resident/runtime/WORKER_RECEIPT.json')
        with self.assertRaises(ValueError):view/'cold/../resident'


if __name__=='__main__':unittest.main()
