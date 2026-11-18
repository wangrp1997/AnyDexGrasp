"""Read the hand's source solver/contact settings; no outcome-based constants."""
from pathlib import Path
import numpy as np
import author_collection_batch as B


def apply(spec,path):
    import mujoco
    source=mujoco.MjSpec.from_file(str(path))
    fields=('timestep','integrator','solver','iterations','ls_iterations','ls_tolerance',
        'tolerance','ccd_tolerance','cone','impratio','noslip_iterations','noslip_tolerance')
    before={};after={}
    for field in fields:
        before[field]=float(getattr(spec.option,field));setattr(spec.option,field,getattr(source.option,field))
        after[field]=float(getattr(spec.option,field))
    ratio=.004/spec.option.timestep
    if abs(ratio-round(ratio))>1e-12 or not 1<=round(ratio)<=8:raise ValueError('Source timestep must divide original 4ms control into at most eight native steps')
    # Preserve the declared masses, friction and actuator settings. Transfer
    # only contact solver fields from the same named source hand body/floor.
    by_body={}
    for geom in source.geoms:
        if geom.parent.name in by_body:raise ValueError('Ambiguous source-body contact settings')
        by_body[geom.parent.name]=geom
    changed=[]
    for geom in spec.geoms:
        key='world' if geom.name=='floor' else geom.parent.name
        original=by_body.get(key)
        if original is None:continue
        for field in ('solref','solimp','solmix','margin','gap'):
            value=getattr(original,field);setattr(geom,field,value.copy() if isinstance(value,np.ndarray) else value)
        changed.append(dict(geom=geom.name,body=geom.parent.name,source_geom=original.name,
            solref=list(geom.solref),solimp=list(geom.solimp)))
    return dict(source=B.binding(Path(path)),option_before=before,option_after=after,contact_geoms=changed,
        native_steps_per_4ms_control=int(round(ratio)),source_gravity_and_floor_pose_not_copied=True,
        declared_friction_mass_and_drives_preserved=True)
