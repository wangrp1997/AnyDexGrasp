"""Opt-in CPU PhysX/Fabric publication, retaining all physical settings.

The pinned CanonDex function remains read-only. Only its Fabric selection and
the existing initial-reset compatibility guard change in this scoped overlay.
"""
import ast
from contextlib import contextmanager
from pathlib import Path

RUNTIME = Path('/home/wangrenpeng/CanonDex/canondex/scene/runtime.py')


def scene_factory(original):
    tree = ast.parse(RUNTIME.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_scene')
    changes = {'fabric': 0, 'reset_guard': 0}

    class Overlay(ast.NodeTransformer):
        def visit_keyword(self, node):
            if node.arg == 'use_fabric':
                if ast.unparse(node.value) != "device != 'cpu'":
                    raise ValueError('Unreviewed native Fabric selection')
                node.value = ast.Constant(True)
                changes['fabric'] += 1
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == "device != 'cpu'":
                if [ast.unparse(n) for n in node.body] != ['install_first_reset_fix()']:
                    raise ValueError('Unreviewed native reset guard')
                node.test = ast.Constant(True)
                changes['reset_guard'] += 1
            return self.generic_visit(node)

    function = Overlay().visit(function)
    if changes != {'fabric': 1, 'reset_guard': 1}:
        raise ValueError('Expected precisely two publication changes')
    scope = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 __file__ + ':cpu_fabric', 'exec'), scope)
    return scope['create_scene']


@contextmanager
def installed(native):
    original = native.create_scene
    native.create_scene = scene_factory(original)
    try:
        yield
    finally:
        native.create_scene = original
