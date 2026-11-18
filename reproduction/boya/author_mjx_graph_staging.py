"""Retain all declared mutable native state through MJX staged CUDA graphs.

MJX 3.11's generated stage_out lists omit native contacts, counters and EFC
buffers. They are already declared in-out FFI arrays. This adapter includes
those existing arrays in the staging copies without changing any kernel,
array schema, alias declaration, control or physics option.
"""


def install():
    from mujoco.mjx.warp import ffi
    original=ffi.jax_callable_variadic_tuple
    if getattr(original,'_anydex_full_state_staging',False):return
    def full_state(*args,**kwargs):
        if int(kwargs.get('graph_mode',2))in(3,4):
            mutable=set(kwargs.get('in_out_argnames')or())
            kwargs['stage_in_argnames']=set(kwargs.get('stage_in_argnames')or())|mutable
            kwargs['stage_out_argnames']=set(kwargs.get('stage_out_argnames')or())|mutable
        return original(*args,**kwargs)
    full_state._anydex_full_state_staging=True
    ffi.jax_callable_variadic_tuple=full_state
