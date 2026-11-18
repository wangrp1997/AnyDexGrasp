"""Bounded, lossless binary records for locally produced simulation traces.

After append, the producer must leave the record and nested values unchanged.
The worker owns serialization and file IO. Close drains every queued record;
encoding/IO errors propagate to the producer. The footer detects lost tails.
"""
import os
from pathlib import Path
import pickle
import queue
import struct
import threading
import zlib

MAGIC=b'AnyDexTrace\x00\x01'
HEADER=struct.Struct('<QI')
COUNT=struct.Struct('<Q')
STOP=object()


class Journal:
    def __init__(self,path,*,asynchronous=True,capacity=4):
        self.path=Path(path);self.stream=self.path.open('xb',buffering=1<<20)
        self.stream.write(MAGIC);self.records=0;self.error=None;self.closed=False
        self.asynchronous=asynchronous;self.queue=queue.Queue(maxsize=capacity)
        self.worker=None
        if asynchronous:
            self.worker=threading.Thread(target=self._consume,name='trace-writer',daemon=False)
            self.worker.start()

    def _write(self,record):
        payload=pickle.dumps(record,protocol=5)
        self.stream.write(HEADER.pack(len(payload),zlib.crc32(payload)))
        self.stream.write(payload);self.records+=1

    def _consume(self):
        try:
            while True:
                record=self.queue.get()
                if record is STOP:return
                self._write(record)
        except BaseException as error:self.error=error

    def _send(self,record):
        while True:
            if self.error is not None:raise RuntimeError('Trace writer failed')from self.error
            try:self.queue.put(record,timeout=.05);return
            except queue.Full:pass

    def append(self,record):
        if self.closed:raise ValueError('Closed trace journal')
        if self.asynchronous:self._send(record)
        else:self._write(record)

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.asynchronous:
                try:self._send(STOP)
                finally:self.worker.join()
                if self.error is not None:raise RuntimeError('Trace writer failed')from self.error
            self.stream.write(HEADER.pack(0,0));self.stream.write(COUNT.pack(self.records))
            self.stream.flush();os.fsync(self.stream.fileno())
        finally:self.stream.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def records(path):
    """Read a complete, locally generated trace; reject damage or lost tails."""
    with Path(path).open('rb')as stream:
        if stream.read(len(MAGIC))!=MAGIC:raise ValueError('Unknown trace format')
        count=0
        while True:
            header=stream.read(HEADER.size)
            if len(header)!=HEADER.size:raise ValueError('Missing trace footer or truncated header')
            size,crc=HEADER.unpack(header)
            if size==0:
                footer=stream.read(COUNT.size)
                if crc or len(footer)!=COUNT.size or COUNT.unpack(footer)[0]!=count or stream.read(1):
                    raise ValueError('Invalid trace footer')
                return
            if size>1<<30:raise ValueError('Invalid trace record size')
            payload=stream.read(size)
            if len(payload)!=size or zlib.crc32(payload)!=crc:raise ValueError('Truncated or damaged trace record')
            yield pickle.loads(payload);count+=1
