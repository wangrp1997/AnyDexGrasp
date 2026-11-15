"""Continue only the31 unstarted scenes of the failed finite pilot.

The original32-scene protocol, zero-tick import failure, and runtime sources stay
immutable. Compile its pinned production adapter privately, changing only the
remaining budget/order and the final window size. Never retry the failed scene.
"""
import ast
from pathlib import Path
import author_production_collection as ORIGINAL

ORIGINAL_SHA='99f73043c4013437d39f20835b3d0dd4938fa03e03ef70117c541b5f1df1bc4a'
if ORIGINAL.B.sha(ORIGINAL.__file__)!=ORIGINAL_SHA:raise ValueError('Pinned production source changed')
tree=ast.parse(Path(ORIGINAL.__file__).read_text());changes={}
mapping={
    "[r['split'] for r in queue['scenes']] != ['fit', 'internal'] * 16":
        "[r['split'] for r in queue['scenes']] != ['internal','fit']*15+['internal']",
    "list(range(w * WINDOW, (w + 1) * WINDOW))":"list(range(w*WINDOW,min((w+1)*WINDOW,31)))",
    "p['total_scene_budget'] != WINDOW":"p['total_scene_budget'] != min(WINDOW,31-p['window_index']*WINDOW)",
    "p['total_gesture_attempt_budget'] != 32":"p['total_gesture_attempt_budget'] != 4*min(WINDOW,31-p['window_index']*WINDOW)",
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
        replacement={32:31,128:124}.get(node.value)if type(node.value)is int else None
        strings={
            'anydex_boya_author_production_pilot_v1':'anydex_boya_author_production_continuation_v1',
            'run_production_collection.py':'run_production_continuation.py',
            'observe_production_collection.py':'observe_production_continuation.py',
        }
        if isinstance(node.value,str):replacement=strings.get(node.value)
        if replacement is not None:
            key=repr(node.value);changes[key]=changes.get(key,0)+1
            return ast.copy_location(ast.Constant(replacement),node)
        return node


tree=Adapt().visit(tree)
expected={a:(2 if a.startswith('list(')else 1)for a in mapping}
expected.update({'32':7,'128':5,repr('anydex_boya_author_production_pilot_v1'):1,
    repr('run_production_collection.py'):1,repr('observe_production_collection.py'):1})
if changes!=expected:raise ValueError('Reviewed remaining-budget AST edit sites changed: '+repr(changes))
ns=dict(__name__='_anydex_production_remaining31',__file__=str(Path(__file__).resolve()))
exec(compile(ast.fix_missing_locations(tree),ORIGINAL.__file__,'exec'),ns)
globals().update({k:v for k,v in ns.items()if not k.startswith('__')})
AST_ADAPTATIONS=dict(original=ORIGINAL.B.binding(ORIGINAL.__file__),edits=changes,
    scientific_changes=False,failed_scene_retries=0,remaining_scenes=31,remaining_slots=124)
