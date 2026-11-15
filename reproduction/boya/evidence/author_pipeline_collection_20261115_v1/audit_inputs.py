"""Read-only reuse of reviewed input/admission auditors with one filename adapter.

The old auditor files remain immutable. Only their protocol filename literals are
changed in a private AST; numerical and admission checks run unchanged. Existing
first execution audits are verified, never rerun or decoded a second time.
"""
import argparse
import ast
from pathlib import Path
import time

import author_pipeline_collection as P

B=P.B
HERE=Path(__file__).resolve().parent
OLD_NAME='RESIDENT_COLLECTION_PROTOCOL.json'


def load(path,name,adapt=False):
    tree=ast.parse(path.read_text());changes=[]
    if adapt:
        class Edit(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value==OLD_NAME:
                    changes.append(node.lineno)
                    return ast.copy_location(ast.Constant(P.PROTOCOL),node)
                return node
        tree=Edit().visit(tree)
        assert changes
    ns=dict(__name__=name,__file__=str(path))
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
    return ns,dict(source=B.binding(path),protocol_literal_lines=changes)


def audit(root):
    began=time.monotonic();protocol=P.verify(root)
    assert B.read(root/'PIPELINE_COLLECTION_RESULTS.json')['pipeline_validity']=='pass'
    assert protocol['total_scene_budget']==2 and protocol['total_gesture_attempt_budget']==8
    reader,reader_source=load(B.HERE/'audit_author_resident_collection.py','_pipeline_compare',True)
    ledger,ledger_source=load(B.HERE/'evidence/author_resident_collection_inputs_20261115_v1/audit_ledger.py',
        '_pipeline_admission',True)
    semantics,semantic_source=load(B.HERE/'evidence/author_resident_collection_inputs_20261115_v1/compare_semantics.py',
        '_pipeline_semantics')
    comparison=reader['compare_run'](root)
    assert len(comparison['scenes'])==2
    semantic_rows=[];reference=Path(protocol['source_batch_protocol']['path']).parent
    for scene in comparison['scenes']:
        assert all(v['exact'] for v in scene['capture'].values())
        assert all(v['exact'] for v in scene['representation'].values())
        name=f'{scene["source_queue_index"]:07d}_{scene["scene_id"]}'
        for g,record in scene['gestures'].items():
            assert record['status']==record['reference_status']=='ready'
            current=root/'scenes'/name/f'g{g}/pool/selected_author_candidate.json'
            old=reference/'scenes'/name/f'g{g}/pool/selected_author_candidate.json'
            x,y=B.read(current),B.read(old)
            checks={k:x[k]==y[k] for k in semantics['FIELDS']}
            assert all(checks.values()),checks
            differences={k:[n for n,v in content['arrays'].items() if not v['exact']]
                for k,content in record.items() if isinstance(content,dict) and 'arrays' in content}
            assert len(differences)==3
            assert all(set(v)<=set(semantics['METADATA']) for v in differences.values()),differences
            semantic_rows.append(dict(scene_id=scene['scene_id'],gesture_id=int(g),
                semantic_fields_exact=len(checks),nonexact_metadata=differences,
                current_candidate=B.binding(current),reference_candidate=B.binding(old)))
    assert len(semantic_rows)==8
    comparison.update(status='pass',protocol_filename_adapter=reader_source)
    B.save(HERE/'INPUT_COMPARISON.json',comparison,exclusive=True)
    B.save(HERE/'PLAN_SEMANTICS.json',dict(status='pass',rows=semantic_rows,source=semantic_source,
        metadata_scope='Fresh source paths and bindings preserved; only candidate/spec hash metadata may differ'),exclusive=True)
    admission=ledger['audit'](root)
    admission.update(protocol_filename_adapter=ledger_source,verifier=B.binding(__file__))
    B.save(HERE/'ADMISSION_AUDIT.json',admission,exclusive=True)
    sessions=[]
    for path in sorted(root.glob('scenes/*/g*/resident_session/EXECUTE_MANY_RESULTS.json')):
        result=B.read(path);receipt=B.read(path.with_name('SESSION_RECEIPT.json'))
        assert result['status']=='complete' and receipt['status']=='closed'
        assert len(result['ordered_prepared_dirs'])==len(result['jobs'])==len(receipt['jobs'])==1
        assert receipt['Kit_starts']==receipt['asset_setup_count']==1
        assert receipt['state_reset_count']==receipt['scene_rebuild_count']==0
        assert result['execution_steps_unknown'] is False
        sessions.append(dict(result=B.binding(path),receipt=B.binding(path.with_name('SESSION_RECEIPT.json'))))
    assert len(sessions)==8
    value=dict(status='pass',protocol=B.binding(root/P.PROTOCOL),sessions=sessions,
        comparison=B.binding(HERE/'INPUT_COMPARISON.json'),semantics=B.binding(HERE/'PLAN_SEMANTICS.json'),
        admission=B.binding(HERE/'ADMISSION_AUDIT.json'),wall_s=time.monotonic()-began,
        new_Kit_launches=0,new_physics_steps=0,new_model_forwards=0,verifier=B.binding(__file__))
    B.save(HERE/'INPUTS_VERIFIED.json',value,exclusive=True)
    print({'status':'pass','sessions':len(sessions),'semantic_fields':len(semantics['FIELDS'])})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    audit(parser.parse_args().run_dir)
