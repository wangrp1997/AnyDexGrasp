from pathlib import Path
import tempfile
import unittest

from author_lossless_journal import Journal,records,MAGIC,HEADER


class Unencodable:
    def __reduce__(self):raise OSError('encode failed')


class LosslessJournalTests(unittest.TestCase):
    def test_order_fields_and_float_bits_survive_backpressure(self):
        rows=[dict(tick=i,values=[-0.,5e-324,1.7976931348623157e308],
            native_fields=dict(pos=[.1,.2,.3],flex=None,empty=[]),flag=True,index=2**63+3)for i in range(100)]
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'trace'
            with Journal(path,capacity=1)as writer:
                for row in rows:writer.append(row)
            restored=list(records(path));self.assertEqual(rows,restored)
            self.assertEqual(restored[0]['values'][0].hex(),(-0.).hex())
            self.assertEqual(writer.records,100)

    def test_missing_footer_is_rejected_even_at_record_boundary(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'trace'
            with Journal(path,asynchronous=False)as writer:writer.append({'x':1})
            raw=path.read_bytes();path.write_bytes(raw[:-20])
            with self.assertRaises(ValueError):list(records(path))

    def test_corrupt_payload_is_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'trace'
            with Journal(path)as writer:writer.append({'x':[1,2,3]})
            raw=bytearray(path.read_bytes());raw[len(MAGIC)+HEADER.size+2]^=1;path.write_bytes(raw)
            with self.assertRaises(ValueError):list(records(path))

    def test_background_encoding_failure_reaches_close(self):
        with tempfile.TemporaryDirectory()as d:
            writer=Journal(Path(d)/'trace',capacity=1);writer.append({'bad':Unencodable()})
            with self.assertRaises(RuntimeError):writer.close()
            self.assertFalse(writer.worker.is_alive())


if __name__=='__main__':unittest.main()
