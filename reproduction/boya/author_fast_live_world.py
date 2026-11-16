"""Same CPU world with full-vertex metric acceleration and direct profiling."""
from pathlib import Path
import time
from author_live_world_v4 import World as OriginalWorld
import author_collection_batch as B
import author_fast_geometry_checks as F
import author_observer_partition as P


class World(OriginalWorld):
    def bootstrap(self):
        import omni.physx
        from author_cpu_fabric_control import startup_steps
        events=[]
        with startup_steps(omni.physx,events):super().bootstrap()
        B.save(self.root/'NATIVE_STARTUP_STEPS.json',dict(integrations=events,
            returned_steps=sum(e['returned']for e in events),scope='Additional native warmup outside SimulationContext tick count'))

    def execute(self,prepared):
        import author_lift as lift
        import author_verified_contact_journal as journal
        F.precompile();began=time.monotonic()
        with P.installed(self.sim,lift,journal)as timing:
            try:return super().execute(prepared)
            finally:
                B.save(Path(prepared)/'DIRECT_EXECUTION_PARTITION.json',dict(status='recorded',timing=timing,
                    wall_s=time.monotonic()-began,physics_changed=False,full_source_vertices_retained=True,
                    implementation=B.binding(F.__file__),profiler=B.binding(P.__file__),
                    scope='sim_step is direct wall time; observer includes metrics and JSON append, nested fields must not be added'))
