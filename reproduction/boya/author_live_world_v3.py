"""Include the standard primary actor in the full capture geometry audit."""
import author_live_world_v2 as ORIGINAL


class World(ORIGINAL.World):
    def capture_handles(self,scene,output):
        # The source auditor treats '/World/Object' separately even when 'all'
        # is requested. Include it explicitly; retain every original gate.
        import template_gripper_native_audit as audit
        original=audit.audit_scene
        def all_active(sim,directory,selectors,**kwargs):
            return original(sim,directory,list(dict.fromkeys([*selectors,'object'])),**kwargs)
        audit.audit_scene=all_active
        try:return super().capture_handles(scene,output)
        finally:audit.audit_scene=original
