"""Future production control: stop dispatch on a failed audit certificate.

The original auditor may exit0 after saving status=failed. Preserve all frozen
sources and original numeric checks; inspect that certificate before allowing
another non-audit stage. No collection is started by importing this adapter.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
import author_production_collection as ORIGINAL

if ORIGINAL.B.sha(ORIGINAL.__file__)!='99f73043c4013437d39f20835b3d0dd4938fa03e03ef70117c541b5f1df1bc4a':
    raise ValueError('Pinned production source changed')
tree=ast.parse(Path(ORIGINAL.__file__).read_text());changes={}
mapping={'anydex_boya_author_production_pilot_v1':'anydex_boya_author_production_pilot_v2',
    'run_production_collection.py':'run_production_control_v2.py',
    'observe_production_collection.py':'observe_production_control_v2.py'}
class Adapt(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str)and node.value in mapping:
            changes[node.value]=changes.get(node.value,0)+1
            return ast.copy_location(ast.Constant(mapping[node.value]),node)
        return node
tree=Adapt().visit(tree)
if changes!={k:1 for k in mapping}:raise ValueError('Reviewed v2 scheduling literals changed')
ns=dict(__name__='_anydex_production_control_v2',__file__=str(Path(__file__).resolve()))
exec(compile(ast.fix_missing_locations(tree),ORIGINAL.__file__,'exec'),ns)
globals().update({k:v for k,v in ns.items()if not k.startswith('__')})
base_freeze=ns['freeze']


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


def freeze(source_batch,output):
    path=Path(source_batch)/'BATCH_PROTOCOL.json';base=B.read(path)
    base['sources']['frozen_production_v1']=B.binding(ORIGINAL.__file__);B.save(path,base)
    return base_freeze(source_batch,output)


ns['guarded_stages']=semantic_guarded_stages;ns['freeze']=freeze
if __name__=='__main__':ns['main']()
