"""Opt-in original capture with complete, streamed gzip contact storage.

The pinned capture's physics, observation and admission calculations stay
unchanged. A small AST overlay replaces contact storage and selects GPU 0.
Importing this module never starts Kit or a simulator.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
import gzip
import hashlib
import json
from pathlib import Path

import author_scene as original

ORIGINAL_SHA = 'c5a7cb962bb3d38961d31cc6a70a4a2c748a905d41b9bcc06e6df345387e7fe5'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


class ContactJournal:
    """One full JSON frame per line; bounded support/count accumulators only."""
    def __init__(self, output_dir, roots, *, floor_paths=('/World/ground/collisionPlane',), compression_level=1):
        self.path = Path(output_dir)/'initialization_contacts.jsonl.gz'
        self.roots = tuple(roots)
        self.floor_paths = frozenset(floor_paths)
        if not self.floor_paths: raise ValueError('Explicit original floor collision paths required')
        if len(set(self.roots)) != len(self.roots): raise ValueError('Unique bodies required')
        self.counts = dict.fromkeys(self.roots, 0)
        self.support = {root:deque(maxlen=250) for root in self.roots}
        self.frames = self.contacts = 0
        self.raw = self.path.open('xb')
        self.stream = gzip.GzipFile(filename='', fileobj=self.raw, mode='wb',
                                    compresslevel=compression_level, mtime=0)
        self.complete = self.closed = False
        self.logical = hashlib.sha256()

    def append(self, frame):
        if self.closed or frame['step'] != self.frames+1:
            raise ValueError('Once-only consecutive contact frames required')
        supported = set()
        for row in frame['contacts']:
            body, other = row['bodies']
            if body not in self.counts: raise ValueError('Unknown original contact body')
            self.counts[body] += 1
            if other in self.floor_paths and row['normal_force_n'] > .001:
                supported.add(body)
        encoded = (json.dumps(frame, separators=(',', ':'), allow_nan=False)+'\n').encode()
        self.stream.write(encoded); self.logical.update(encoded)
        for root in self.roots: self.support[root].append(root in supported)
        self.frames += 1; self.contacts += len(frame['contacts'])

    def evidence(self):
        return {root:dict(final_window_supported_ticks=sum(self.support[root]),
                          contact_rows=self.counts[root]) for root in self.roots}

    def finish(self):
        if self.frames != original.STEPS:
            raise ValueError('Complete original 750 initialization contact frames required')
        self.complete = True
        self.close()

    def close(self):
        if self.closed: return
        self.stream.close(); self.raw.close(); self.closed = True
        original.save(self.path.parent/'CONTACT_JOURNAL.json', dict(
            schema='anydex_boya_initialization_contact_journal_v1',
            status='complete' if self.complete else 'incomplete',
            path=str(self.path.resolve()), sha256=digest(self.path),
            logical_jsonl_sha256=self.logical.hexdigest(), frames=self.frames,
            contacts=self.contacts, original_object_paths=list(self.roots),
            floor_collision_paths=sorted(self.floor_paths),
            evidence=self.evidence(), compression='gzip level1, full JSONL frames',
            sampling=False, contact_fields_dropped=False,
            original_capture_sha256=ORIGINAL_SHA, overlay_source_sha256=digest(__file__)))


def overlay_receipt():
    return dict(schema='anydex_boya_streamed_capture_overlay_v1',
        original_source=dict(path=str(Path(original.__file__).resolve()), sha256=ORIGINAL_SHA),
        overlay_source=dict(path=str(Path(__file__).resolve()),sha256=digest(__file__)),
        changes=['stream full contact frames through gzip level1',
                 'same original contact counters and last250 support predicate',
                 'close journal on exceptions', 'explicit renderer GPU0, multiGPU disabled'],
        physics_or_admission_changes=False, initialization_steps=750, dt_s=.004,
        renderer_gpu=0, physics_device='cpu')


def capture_function():
    """Compile exactly matched storage replacements; reject any source drift."""
    if digest(original.__file__) != ORIGINAL_SHA: raise ValueError('Pinned original capture changed')
    tree = ast.parse(Path(original.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='capture')
    replacements = {
        'poses, velocities, contact_rows = [], [], []':
            "poses, velocities, contact_rows = [], [], ContactJournal(output_dir, roots, floor_paths=[p for p in static_paths if p.startswith('/World/ground')])",
        "save(output_dir/'initialization_contacts.json', contact_rows)": 'contact_rows.finish()',
        'final_contacts = contact_rows[-250:]': 'final_contacts = None',
    }
    maps = {ast.dump(ast.parse(k).body[0], include_attributes=False):ast.parse(v).body[0]
            for k,v in replacements.items()}
    seen = dict.fromkeys(maps,0)
    extras = dict(evidence=0, launcher=0, protocol=0, finally_close=0)
    class Overlay(ast.NodeTransformer):
        def visit_Assign(self,node):
            key=ast.dump(node,include_attributes=False)
            if key in maps:
                seen[key]+=1;return maps[key]
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='contact_evidence':
                extras['evidence']+=1
                return ast.parse('contact_evidence = contact_rows.evidence()').body[0]
            return self.generic_visit(node)
        def visit_Expr(self,node):
            key=ast.dump(node,include_attributes=False)
            if key in maps:
                seen[key]+=1;return maps[key]
            key_mkdir=ast.dump(ast.parse('output_dir.mkdir(parents=True, exist_ok=False)').body[0],include_attributes=False)
            if key==key_mkdir:
                extras['protocol']+=1
                return [node,ast.parse("save(output_dir/'CAPTURE_OVERLAY.json', overlay_receipt())").body[0]]
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='AppLauncher':
                kw=next(k for k in node.keywords if k.arg=='kit_args')
                if kw.value.value!='--/log/file=/dev/null':raise ValueError('Original launcher arguments changed')
                kw.value=ast.Constant('--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false')
                extras['launcher']+=1
            return self.generic_visit(node)
        def visit_Try(self,node):
            node=self.generic_visit(node)
            if node.finalbody:
                extras['finally_close']+=1
                node.finalbody.insert(0,ast.parse("if 'contact_rows' in locals() and isinstance(contact_rows, ContactJournal):\n    contact_rows.close()").body[0])
            return node
    function=Overlay().visit(function)
    if set(seen.values())!={1} or set(extras.values())!={1}:
        raise ValueError('Pinned storage-only overlay match differs: '+repr((seen,extras)))
    scope=dict(vars(original),ContactJournal=ContactJournal,overlay_receipt=overlay_receipt)
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module, str(Path(__file__).resolve())+':storage_overlay', 'exec'),scope)
    return scope['capture']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    capture_function()(args.scene.resolve(),args.output_dir.resolve())


if __name__=='__main__': main()
