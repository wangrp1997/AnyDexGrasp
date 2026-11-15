"""Only the original fifteen unstarted scenes, with verified contact storage.

Privately compile the source-pinned production adapter. Draws, original first
candidate, commands, physical rules and numerical auditor are unchanged.
"""
import ast
from contextlib import contextmanager
from pathlib import Path

import author_production_collection as ORIGINAL
import author_verified_execution as EXECUTION
import author_verified_contact_journal as JOURNAL

PIN='99f73043c4013437d39f20835b3d0dd4938fa03e03ef70117c541b5f1df1bc4a'
if ORIGINAL.B.sha(ORIGINAL.__file__)!=PIN:raise ValueError('Pinned production source changed')
tree=ast.parse(Path(ORIGINAL.__file__).read_text());changes={}
mapping={
    "[r['split'] for r in queue['scenes']] != ['fit', 'internal'] * 16":
        "[r['split'] for r in queue['scenes']] != ['internal','fit']*7+['internal']",
    "list(range(w * WINDOW, (w + 1) * WINDOW))":"list(range(w*WINDOW,min((w+1)*WINDOW,15)))",
    "p['total_scene_budget'] != WINDOW":"p['total_scene_budget'] != min(WINDOW,15-p['window_index']*WINDOW)",
    "p['total_gesture_attempt_budget'] != 32":"p['total_gesture_attempt_budget'] != 4*min(WINDOW,15-p['window_index']*WINDOW)",
    'range(4)':'range(2)',"len(index['windows']) != 4":"len(index['windows']) != 2",
    '0 <= w < 4':'0 <= w < 2',
}
patterns={ast.dump(ast.parse(a,mode='eval').body,include_attributes=False):(a,b)for a,b in mapping.items()}


class Adapt(ast.NodeTransformer):
    def visit(self,node):
        if isinstance(node,ast.expr):
            hit=patterns.get(ast.dump(node,include_attributes=False))
            if hit:
                a,b=hit;changes[a]=changes.get(a,0)+1
                return ast.copy_location(ast.parse(b,mode='eval').body,node)
        return super().visit(node)

    def visit_Constant(self,node):
        replacement={32:15,128:60}.get(node.value)if type(node.value)is int else None
        strings={'anydex_boya_author_production_pilot_v1':'anydex_boya_author_production_tail_v1',
                 'run_production_collection.py':'run_production_tail.py',
                 'observe_production_collection.py':'observe_production_tail.py'}
        if isinstance(node.value,str):replacement=strings.get(node.value)
        if replacement is not None:
            key=repr(node.value);changes[key]=changes.get(key,0)+1
            return ast.copy_location(ast.Constant(replacement),node)
        return node


tree=Adapt().visit(tree)
expected={a:(2 if a.startswith('list(')else 1)for a in mapping}
expected.update({'32':7,'128':5,repr('anydex_boya_author_production_pilot_v1'):1,
                 repr('run_production_collection.py'):1,repr('observe_production_collection.py'):1})
if changes!=expected:raise ValueError('Reviewed remaining-budget AST sites changed: '+repr(changes))
ns=dict(__name__='_anydex_production_remaining15',__file__=str(Path(__file__).resolve()))
exec(compile(ast.fix_missing_locations(tree),ORIGINAL.__file__,'exec'),ns)
globals().update({k:v for k,v in ns.items()if not k.startswith('__')})
base_freeze,base_verify,base_verify_window,base_terminal=ns['freeze'],ns['verify'],ns['verify_window'],ns['terminal']
SPOOL='/dev/shm/anydex_production_tail_20261115_v1'
AST_ADAPTATIONS=dict(original=B.binding(ORIGINAL.__file__),edits=changes,remaining_scenes=15,
                     remaining_slots=60,sampler_redraws=0,failed_scene_retries=0,scientific_changes=False)


def freeze(source_batch,output):
    path=Path(source_batch)/'BATCH_PROTOCOL.json';base=B.read(path)
    base['env']=dict(base['env'],ANYDEX_CONTACT_SPOOL_ROOT=SPOOL)
    base['sources'].update(original_production_adapter=B.binding(ORIGINAL.__file__),
        original_resident_execution=B.binding(B.HERE/'author_resident_execution.py'))
    for name in ('author_verified_execution.py','author_verified_contact_journal.py','author_contact_compressor.py',
                 'freeze_production_tail.py'):
        base['sources'][name]=B.binding(B.HERE/name)
    base['sources']['author_resident_execution.py']=B.binding(EXECUTION.__file__)
    B.save(path,base)
    return base_freeze(source_batch,output)


def verify_window(root):
    p=base_verify_window(root)
    if (p['env'].get('ANYDEX_CONTACT_SPOOL_ROOT')!=SPOOL
            or p['sources']['author_resident_execution.py']!=B.binding(EXECUTION.__file__)):
        raise ValueError('Verified resident entry and tmpfs spool required')
    return p


def verify(root):
    p=base_verify(root)
    if p['env'].get('ANYDEX_CONTACT_SPOOL_ROOT')!=SPOOL:raise ValueError('Frozen raw spool changed')
    return p


def verified_contact_artifacts(trial):
    prepared=Path(trial['record']['path']).parent;directory=prepared/'execution'
    pointer=B.read(directory/'MULTI_OBJECT_CONTACTS.json');JOURNAL.verify_pointer(pointer)
    cost=B.read(directory/'CONTACT_STORAGE_COST.json');steps=trial['actual_physics_steps']
    if (cost['status']!='original_finish_returned'or cost['original_runtime_steps']!=steps
            or cost['observed_trace_rows']!=steps or cost['contact_records']!=steps
            or pointer['logical']['records']!=steps):
        raise ValueError('Original returned-step/full-log counters changed')
    return dict(contact_integrity_seal=pointer['integrity_seal'],
                contact_storage_cost=B.binding(directory/'CONTACT_STORAGE_COST.json'),
                contact_storage_state=B.binding(directory/'MULTI_OBJECT_CONTACTS.jsonl.gz.STATE.json'))


def terminal(root,protocol,i,row,g,status,*,artifacts,trial=None,**kwargs):
    artifacts=dict(artifacts)
    if trial is not None:artifacts.update(verified_contact_artifacts(trial))
    return base_terminal(root,protocol,i,row,g,status,artifacts=artifacts,trial=trial,**kwargs)


@contextmanager
def semantic_guarded_stages(control):
    with ORIGINAL.guarded_stages(control):
        previous=B._stage
        def stage(directory,command,expected,protocol):
            result=previous(directory,command,expected,protocol)
            if result['status']=='complete'and 'audit_author_collection_runtime.py'in[Path(str(x)).name for x in command]:
                audit=B.read(expected[0])
                if(audit.get('status')!='pass'or audit.get('formal_training_eligible')is not True
                   or type(audit.get('label'))is not int or audit['label']not in(0,1)):
                    control.stop('Original numeric audit certificate failed',certificate=B.binding(expected[0]))
            return result
        B._stage=stage
        try:yield
        finally:B._stage=previous


ns.update(freeze=freeze,verify=verify,verify_window=verify_window,terminal=terminal,
          guarded_stages=semantic_guarded_stages)
if __name__=='__main__':ns['main']()
