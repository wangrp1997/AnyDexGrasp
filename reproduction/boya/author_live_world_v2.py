"""Bind fixed camera metadata omitted by the first live capture adapter.

The first failed source/750-step capture remains immutable. The camera is
fixed across the entire frozen world, so its declared metadata is shared.
"""
import builtins
import dis
import types
import author_live_world as ORIGINAL


def unresolved_globals(function):
    def names(code):
        result={i.argval for i in dis.get_instructions(code)if i.opname=='LOAD_GLOBAL'}
        for value in code.co_consts:
            if isinstance(value,types.CodeType):result|=names(value)
        return result
    return names(function.__code__)-set(function.__globals__)-set(vars(builtins))


def capture_function(world):
    function=ORIGINAL.capture_function(world);function.__globals__['cam']=world.p['camera']
    missing=unresolved_globals(function)
    if missing:raise ValueError('Live capture has unbound globals: '+str(sorted(missing)))
    return function


class World(ORIGINAL.World):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.capture_impl=capture_function(self)
