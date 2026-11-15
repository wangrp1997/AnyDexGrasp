"""Run the independent pilot auditor with the31-scene remaining-budget schema."""
import ast
from pathlib import Path

import author_production_continuation as D

source=D.B.HERE/'audit_production_pilot.py'
if D.B.sha(source)!='132f9491b91551484d5f73ff02395708d69b99419c3f9f197ac726ba243a566c':
    raise ValueError('Reviewed independent auditor changed')
tree=ast.parse(source.read_text());edits={32:0,128:0}
class Adapt(ast.NodeTransformer):
    def visit_Import(self,node):
        for alias in node.names:
            if alias.name=='author_production_collection':alias.name='author_production_continuation'
        return node
    def visit_Constant(self,node):
        if type(node.value)is int and node.value in edits:
            edits[node.value]+=1
            return ast.copy_location(ast.Constant({32:31,128:124}[node.value]),node)
        return node
tree=Adapt().visit(tree)
if edits!={32:1,128:2}:raise ValueError('Independent remaining-budget auditor edits changed')
namespace=dict(__name__=__name__,__file__=str(Path(__file__).resolve()))
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)
