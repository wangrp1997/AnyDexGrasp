"""Opt-in export reusing an already completed independent numeric audit.

Every certificate input is freshly SHA256 checked, in full, before reuse and
again before publication. No mtime cache, contact decoding, audit fallback,
simulator, collection or training is dispatched. The unchanged exporters own
all admission, array construction, quotas and whole-scene split isolation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import threading
import time
import traceback

import author_collection_batch as B
import author_collection_export as E
import author_collection_wave_export as X
import author_collection_waves as W

SCHEMA = 'anydex_boya_author_certified_collection_export_v1'
PINS = {
    'author_collection_batch.py': '153527cbbfa38a3533bf5759a06331ab1c1896da95f8a575d8b18f7e38240959',
    'author_collection_export.py': 'a7fb4159bd9d649d290753d5a92891540407728ea59d88a25001106d485f9fe1',
    'author_collection_wave_export.py': '3252143090a204ccd5fa450fc613b301701d4a4be1d6924ba584081f2b6a3e1f',
    'author_collection_waves.py': 'afc7d78face25ce8102458c9a351c4a2cd6f0db45103c2c3dd13478d6496561c',
    'audit_author_collection_runtime.py': 'be0ae018d3b4ab569764602a1516db3f2767b6fb6effe41e54706fd576bb03b5',
}
CHECKS = {
    'all_scene_original_visual_identity_and_converted_geometry_split',
    'source_candidate_ready_actual_commands_record_and_execution_bindings',
    'all_contacts_lossless_decode_and_all_body_force_recomputation',
    'all_source_vertex_bottoms_and_final_lift_labels',
    'actual_28_targets_from_24_source_coordinates_and_mimics',
}
_active = False


def _pins():
    result = {}
    for name, digest in PINS.items():
        path = B.HERE/name
        if B.sha(path) != digest:
            raise ValueError('Certified export frozen source changed: '+str(path))
        result[name] = dict(path=str(path), sha256=digest)
    return result


class _FullHashes:
    """Invocation-local full-content deduplication, followed by full rehash."""
    def __init__(self):
        self.expected = {}; self.bytes_read = 0; self.files_read = 0; self.wall_s = 0.

    def _read(self, path, digest):
        began = time.monotonic(); h = hashlib.sha256(); size = 0
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block); size += len(block)
        self.bytes_read += size; self.files_read += 1; self.wall_s += time.monotonic()-began
        if h.hexdigest() != digest:
            raise ValueError('Certified audit input SHA changed: '+path)

    def add(self, path, digest):
        path = str(Path(path).resolve())
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Complete lowercase SHA256 required')
        if path in self.expected:
            if self.expected[path] != digest:
                raise ValueError('Conflicting certified input SHA: '+path)
        else:
            self._read(path, digest); self.expected[path] = digest
        return path

    def verify_again(self):
        for path, digest in self.expected.items():
            self._read(path, digest)


def _contexts(source, mode):
    if mode == 'batch':
        protocol = B.read(source/'BATCH_PROTOCOL.json')
        if protocol.get('schema') != B.SCHEMA or protocol.get('task') != B.TASK:
            raise ValueError('Explicit frozen lift batch required')
        B.verify(protocol['inputs']); B.verify(protocol['sources'])
        rows = B.ledger(source, protocol)
        entries = [(row, row, protocol, protocol['queue']['scenes'][row['queue_index']],
                    source/'ledger'/f'{row["queue_index"]:07d}_g{row["gesture_id"]}.json')
                   for row in rows if row.get('label') is not None]
    elif mode == 'wave':
        protocol, plan, index = W.verify_protocol(source)
        rows = W.ledger(source, protocol)
        entries = [(row, raw, cp, spec, Path(row['raw_ledger']['path']))
                   for row, raw, cp, spec in X._ordered_rows(rows, protocol, plan, index)]
    else:
        raise ValueError('mode must be batch or wave')
    contexts = {}
    for parent, raw, cp, spec, raw_path in entries:
        directory = Path(raw['trial']['record']['path']).resolve().parent
        if directory in contexts:
            raise ValueError('Duplicate certificate trial directory')
        parent_path = source/'ledger'/f'{parent["queue_index"]:07d}_g{parent["gesture_id"]}.json'
        contexts[directory] = dict(parent=parent, raw=raw, protocol=cp, spec=spec,
            ledger=B.binding(raw_path), parent_ledger=B.binding(parent_path))
    return contexts


def _flag(command, name):
    if command.count(name) != 1:
        raise ValueError('Independent audit stage requires one '+name)
    return command[command.index(name)+1]


class CertificateReuse:
    def __init__(self, contexts):
        self.contexts = contexts; self.hashes = _FullHashes(); self.receipts = {}

    def __call__(self, prepared_dir, protocol):
        began = time.monotonic(); before = (self.hashes.files_read, self.hashes.bytes_read)
        directory = Path(prepared_dir).resolve()
        context = self.contexts.get(directory)
        if context is None or context['protocol'] != protocol:
            raise ValueError('Certificate is outside this exact frozen trial/protocol')
        raw, spec, trial = context['raw'], context['spec'], context['raw']['trial']
        trial_id = raw['trial_id']
        if trial_id in self.receipts or raw.get('status') != 'labelled':
            raise ValueError('Exactly one original labelled audit certificate per trial required')
        for binding in (context['ledger'], context['parent_ledger'], spec['scene_spec']):
            self.hashes.add(binding['path'], binding['sha256'])
        self.hashes.add(Path(context['ledger']['path']).parent.parent/'BATCH_PROTOCOL.json', raw['protocol_sha256'])
        artifact = raw.get('artifact_bindings', {})
        cert_binding = artifact.get('runtime_audit'); stage_binding = artifact.get('audit_stage')
        if (not cert_binding or not stage_binding
                or Path(cert_binding['path']).resolve() != directory/'RUNTIME_AUDIT.json'):
            raise ValueError('Original ledger-bound runtime audit certificate is missing or belongs to another trial')
        self.hashes.add(cert_binding['path'], cert_binding['sha256'])
        self.hashes.add(stage_binding['path'], stage_binding['sha256'])
        certificate = B.read(cert_binding['path']); stage = B.read(stage_binding['path'])
        split = protocol['queue'].get('split_manifest')
        audit_source = protocol['sources']['audit_entry']
        source_path = B.HERE/'audit_author_collection_runtime.py'
        if (not split or audit_source != dict(path=str(source_path), sha256=PINS[source_path.name])
                or certificate.get('audit_source') != audit_source):
            raise ValueError('Certificate must bind the current exact frozen independent numeric audit source')
        self.hashes.add(audit_source['path'], audit_source['sha256'])
        self.hashes.add(split['path'], split['sha256'])
        command = stage.get('command', [])
        if (stage.get('status') != 'complete' or stage.get('complete') is not True
                or stage.get('returncode') != 0
                or stage.get('outputs', {}).get(cert_binding['path']) != cert_binding
                or command.count(audit_source['path']) != 1
                or Path(_flag(command, '--prepared-dir')).resolve() != directory
                or Path(_flag(command, '--split-manifest')).resolve() != Path(split['path']).resolve()
                or Path(_flag(command, '--output')).resolve() != directory/'RUNTIME_AUDIT.json'):
            raise ValueError('Original completed audit process/output binding does not certify this trial')
        if (certificate.get('schema') != 'anydex_boya_author_collection_runtime_audit_v1'
                or Path(certificate.get('prepared_dir', '')).resolve() != directory
                or not CHECKS.issubset(set(certificate.get('checks', [])))
                or certificate.get('scene_id') != spec['scene_id']
                or certificate.get('candidate_sha256') != trial['candidate_sha256']
                or certificate.get('commands_sha256') != trial['commands']['sha256']
                or certificate.get('runtime', {}).get('physics_steps') != trial['actual_physics_steps']
                or type(certificate.get('runtime', {}).get('physics_steps')) is not int
                or trial['actual_physics_steps'] <= 0):
            raise ValueError('Certificate is not a completed full numeric audit of this actual trial')
        B.certify_audit(certificate, trial, spec)
        inputs = certificate.get('audit_inputs_sha256')
        if not isinstance(inputs, dict) or not inputs:
            raise ValueError('Full original audit input map is required')
        required = {str(directory/name) for name in ('LIFT_PIPELINE_PROTOCOL.json', 'LIFT_PIPELINE_RESULTS.json',
            'READY.json', 'COLLECTION_RECORD.json', 'EXECUTION_STARTED.json', 'LIFT_PIPELINE_EXECUTION.json')}
        required.update(str(Path(trial[key]['path']).resolve()) for key in
                        ('record', 'candidate', 'evaluation', 'commands', 'command_binding', 'scene'))
        required.update((str(source_path), str(Path(split['path']).resolve())))
        execution = directory/'execution'
        required.update(str(execution/name) for name in ('MULTI_OBJECT_TRACE.json', 'MULTI_OBJECT_CONTACTS.json'))
        scene = B.read(trial['scene']['path'])
        if (scene.get('scene_id') != spec['scene_id'] or scene.get('dataset_split', spec['split']) != spec['split']
                or len(scene.get('objects', [])) != 3):
            raise ValueError('Certificate scene/spec identity or split differs')
        for obj in scene['objects']:
            for name in ('source_mesh', 'source_visual_mesh'):
                if not obj.get(name):
                    raise ValueError('Original visual and converted source meshes must both be explicit')
                required.add(str(Path(obj[name]).resolve()))
        pointer = B.read(execution/'MULTI_OBJECT_CONTACTS.json')
        if pointer.get('schema') == 'anydex_boya_contact_journal_pointer_v1':
            required.add(str(Path(pointer['data']['path']).resolve()))
        if not required.issubset(inputs):
            raise ValueError('Original numeric audit input map is missing required trial/mesh/contact bindings')
        for path, digest in inputs.items():
            if not Path(path).is_absolute() or str(Path(path).resolve()) != path:
                raise ValueError('Canonical absolute input paths required')
            self.hashes.add(path, digest)
        for key in ('record', 'candidate', 'evaluation', 'commands', 'command_binding', 'scene'):
            binding = trial[key]
            if inputs.get(str(Path(binding['path']).resolve())) != binding['sha256']:
                raise ValueError('Audit inputs and original trial binding differ: '+key)
        receipt = dict(original_certificate=copy.deepcopy(cert_binding), original_audit_stage=copy.deepcopy(stage_binding),
            original_ledger=context['ledger'], parent_ledger=context['parent_ledger'],
            prepared_dir=str(directory), original_audit_source=audit_source,
            original_numeric_audit_wall_s=certificate.get('wall_s'),
            logical_input_files=len(inputs), numeric_replays=0,
            verification_wall_s=time.monotonic()-began,
            newly_hashed_files=self.hashes.files_read-before[0],
            newly_hashed_bytes=self.hashes.bytes_read-before[1])
        self.receipts[trial_id] = receipt
        return certificate

    def finalize(self):
        if len(self.receipts) != len(self.contexts):
            raise ValueError('Not all admitted trials reused exactly one valid certificate')
        self.hashes.verify_again()


@contextmanager
def _reuse_context(reuse):
    global _active
    if _active or threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError('Certified export supports only one main-thread export in a dedicated CLI process')
    _active = True; original = E._audit_trial
    try:
        E._audit_trial = reuse
        yield
    finally:
        E._audit_trial = original; _active = False


def _publish_paths(value, work, output):
    if isinstance(value, dict):
        return {key: (str(output/Path(item).relative_to(work))
                      if key == 'path' and isinstance(item, str) and Path(item).is_relative_to(work)
                      else _publish_paths(item, work, output)) for key, item in value.items()}
    if isinstance(value, list):
        return [_publish_paths(item, work, output) for item in value]
    return value


def export(source_dir, output_dir, *, mode='batch'):
    began = time.monotonic(); sources = _pins()
    entry_source = B.binding(__file__)
    source = Path(source_dir).resolve(); output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Fresh certified export directory required')
    work = output.parent/f'.{output.name}.certified-work-{os.getpid()}'
    if work.exists():
        raise FileExistsError('Preserved prior certified export staging directory exists')
    reuse = CertificateReuse(_contexts(source, mode))
    try:
        with _reuse_context(reuse):
            manifest = E.export(source, work) if mode == 'batch' else X.export(source, work)
            reuse.finalize()
        _pins()
        B.verify({'certified_entry': entry_source})
        original_schema = manifest['schema']
        manifest = _publish_paths(manifest, work, output)
        manifest.update(schema=SCHEMA, wrapped_export_schema=original_schema, export_mode=mode,
            independent_audit_mode='reuse_original_pass_certificate_after_full_input_sha_revalidation',
            certificate_reuse=reuse.receipts,
            verification_cost=dict(wall_s=time.monotonic()-began, hash_wall_s=reuse.hashes.wall_s,
                full_file_reads=reuse.hashes.files_read, full_bytes_read=reuse.hashes.bytes_read,
                unique_verified_files=len(reuse.hashes.expected), verification_passes=2,
                deduplication='Only invocation-local exact (canonical path, expected SHA); complete content rehashed before publication',
                mtime_cache_used=False, sampled_hashes_used=False, numeric_replays=0),
            certified_export_source=entry_source, wrapped_sources=sources,
            new_simulator_launches=0, new_physics_steps=0, new_training_updates=0)
        B.save(work/'MANIFEST.json', manifest)
        work.rename(output)
        return manifest
    except BaseException as error:
        if work.exists():
            B.save(work/'CERTIFIED_EXPORT_FAILURE.json', dict(schema=SCHEMA,status='failed',
                error=repr(error), traceback=traceback.format_exc(), numeric_replays=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--batch-dir', type=Path); source.add_argument('--run-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.batch_dir or args.run_dir, args.output_dir, mode='batch' if args.batch_dir else 'wave')
    print(dict(schema=result['schema'], output=str(args.output_dir.resolve()),
               reused_certificates=len(result['certificate_reuse']), verification_cost=result['verification_cost']))


if __name__ == '__main__':
    main()
