"""Exact-byte storage compatibility and real corruption rejection; no Kit/physics."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_collection_execute as OLD
import author_contact_compressor as W
import author_verified_contact_journal as J


def rows():
    return [dict(step=i,zero=0.0,negative_zero=-0.0,force=[1e-30,-3.141592653589793,0.],
        contacts=[dict(bodies=['body','floor'],force_n=0.,position_world_m=[.1,.2,.3]),
                  dict(bodies=['body','floor'],force_n=0.,position_world_m=[.1,.2,.3])],
        text='原始完整接触',values=list(range(i)))for i in range(1,6)]


class VerifiedJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.addCleanup(self.temp.cleanup)

    def journal(self,name='new.gz'):
        return J.ContactJournal(self.root/name,spool_root=self.root/'raw')

    def test_complete_bytes_match_original_and_worker_is_isolated(self):
        old=OLD.ContactJournal(self.root/'old.gz');new=self.journal()
        for row in rows():old.append(row);new.append(row)
        old.close();pointer=new.pointer()
        with gzip.open(old.path,'rb')as a,gzip.open(new.path,'rb')as b:self.assertEqual(a.read(),b.read())
        seal=J.verify_pointer(pointer);self.assertTrue(seal['isolated']);self.assertTrue(seal['no_site'])
        self.assertEqual(seal['loaded_native_framework_modules'],[])
        self.assertNotEqual(seal['worker_pid'],os.getpid());self.assertTrue(new.raw_path.exists())
        self.assertEqual(pointer,new.pointer())
        self.assertEqual(json.loads(new.state.read_text())['compressor_attempts'],1)

    def test_original_complete_pointer_reader_accepts_extra_seal_without_omission(self):
        import audit_author_collection_runtime as A
        journal=self.journal()
        for row in rows():journal.append(row)
        pointer=self.root/'pointer.json';pointer.write_text(json.dumps(journal.pointer()))
        self.assertEqual(list(A.iter_contact_rows(pointer)),rows())

    def test_raw_valid_json_mutation_fails_before_publication_and_never_retries(self):
        journal=self.journal();journal.append(dict(step=1,value=.125));journal.raw.flush()
        with journal.raw_path.open('r+b')as stream:
            content=stream.read().replace(b'0.125',b'0.126');stream.seek(0);stream.write(content)
        with self.assertRaises(J.JournalIntegrityError):journal.close()
        seal=json.loads(journal.receipt.read_text());self.assertEqual(seal['status'],'failed')
        self.assertFalse(journal.path.exists());self.assertTrue(journal.raw_path.exists())
        with patch.object(J.subprocess,'run',side_effect=AssertionError('Retried compressor')):
            with self.assertRaises(J.JournalIntegrityError):journal.pointer()

    def test_source_change_rejects_publication_with_raw_retained(self):
        source=self.root/'source.py';source.write_text('before')
        journal=J.ContactJournal(self.root/'new.gz',spool_root=self.root/'raw',source_bindings={str(source):W.sha(source)})
        journal.append(dict(step=1));source.write_text('after')
        with self.assertRaises(J.JournalIntegrityError):journal.close()
        self.assertTrue(journal.raw_path.exists());self.assertFalse(journal.path.exists())

    def test_published_gzip_change_is_rejected_by_pointer_verifier(self):
        journal=self.journal();journal.append(dict(step=1));pointer=journal.pointer()
        with journal.path.open('r+b')as stream:stream.seek(12);stream.write(b'corrupted')
        with self.assertRaises(J.JournalIntegrityError):J.verify_pointer(pointer)

    def test_seal_change_is_rejected(self):
        journal=self.journal();journal.append(dict(step=1));pointer=journal.pointer()
        journal.receipt.write_text('{}')
        with self.assertRaises(J.JournalIntegrityError):J.verify_pointer(pointer)

    def test_duplicate_writer_or_existing_output_never_overwrites(self):
        journal=self.journal();journal.append(dict(step=1))
        with self.assertRaises(FileExistsError):self.journal()
        journal.close();before=W.sha(journal.path)
        with self.assertRaises(FileExistsError):self.journal()
        self.assertEqual(W.sha(journal.path),before)

    def test_nonfinite_or_out_of_order_input_is_rejected(self):
        journal=self.journal()
        with self.assertRaises(ValueError):journal.append(dict(step=2))
        with self.assertRaises(ValueError):journal.append(dict(step=1,value=float('nan')))
        journal.append(dict(step=1));self.assertEqual(journal.pointer()['record_count'],1)

    def test_empty_prefix_is_explicit_zero_records_not_a_label(self):
        pointer=self.journal().pointer();self.assertEqual(pointer['record_count'],0)
        self.assertEqual(pointer['logical']['sha256'],hashlib.sha256(b'').hexdigest())
        self.assertNotIn('label',pointer)

    def test_real_deflate_corruption_during_publication_never_gets_passing_seal(self):
        raw=self.root/'source.jsonl';data=b'{"step":1,"value":"abcdef"}\n';raw.write_bytes(data)
        target=self.root/'target.gz';receipt=self.root/'seal.json';replace=W.os.replace
        def corrupt(source,dest):
            replace(source,dest)
            if Path(dest)==target:
                with target.open('r+b')as stream:stream.seek(12);stream.write(b'corrupt-deflate')
        with patch.object(W.os,'replace',side_effect=corrupt):
            result=W.compress(raw,target,receipt,dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),records=1))
        self.assertEqual(result['status'],'failed');self.assertTrue(raw.exists());self.assertTrue(target.exists())
        self.assertNotIn('logical_readback',result)


if __name__=='__main__':unittest.main()
