# Frozen rigid-transmission test

Historical stage: later fixed-grid results and cumulative costs are in [TIME_RESOLUTION_RESULTS.md](TIME_RESOLUTION_RESULTS.md).

The100Nm structural-constraint variant preserves the original cube top1 and
actually lifts90.752mm, but **still fails** the unchanged .01rad coupling and
1.5rad/s velocity criteria. Empty regression passes and is array-identical to
prior empty v5. No hardware validation or original paper benchmark is claimed.

The original geometric relation is q=A*u+b. The constant nine-row matrix C
satisfies rank(C)=9 and C*A=0. Ideal structural reaction power is
lambda^T*C*qdot=0. Such internal multipliers are different from the two active
motor torques. The old .8Nm gear ceiling effectively permits numerical slip;
it is not a measured DH3 clutch. The new model interprets these rows as ideal
structural relations with a high finite numerical ceiling, not extra motors.
Active motor caps remain .6/.3Nm and speeds1/.5rad/s, wrist cap30 per row,
inner20, geometry, collision matrix, scene, controls and residual/velocity gates
all remain unchanged.

Before either new run, MODEL.json and the entire new controller source were
frozen. Original URDF tree-origin lengths plus every STL vertex and COM bound
the geometry diameter by.4655799m. The declared wrist norm scales are each
sqrt(3)*30; the engineering moment expression

```
sqrt(3)*30 + D*(sqrt(3)*30 + (m_hand+.1)*9.81) + .6+.3
```

is79.5589Nm.100Nm was declared before trials as a numerical structure rating.
Only the geometry diameter is a triangle-inequality bound. This load scale
is **not a rigorous multiplier bound**: impacts, force indeterminacy and
contact conditioning are not bounded by it. It is not hardware identification.
Bullet's original gear source lines107-108 define J=[1,gearRatio]; line146
passes symmetric constraint impulse limits. The complete C/A/b and source
hashes are in the frozen model evidence.

The new empty run has max residual.0015704rad, speed1.0741rad/s and structural
reaction.353575Nm. Its exact trace SHA matches old empty v5. The new cube run
has max residual.162744rad, speed2.7112rad/s and max reaction.858026Nm, leaving
99.141974Nm cap margin. The largest error is on finger2 tip at step1578 /
6.575s, during lift, with row reaction-.231881Nm and12.780N finger-table
normal load. The object contact at this instant is1.794N. Cap saturation does
not explain this remaining failure. The last hold settles below.00548rad and
.00436rad/s; this does not erase the transient.

The cube's lowest point ends87.039mm above table and each final120 steps has
positive-normal hand-object contacts (720 rows total). Hand-object/table
minimum signed distances are-.262/-.717mm. Motor, gear and wrist force bounds
are obeyed, wrist error<.834mm; zero dynamic reset/attachment, gravity retained.
205 real640×480 frames are saved, sampled every12 steps for20fps playback.
The sampled sum(abs(lambda*delta residual)) is.094963J, signed sum-.028514J.
This is a sampled correction-work **proxy**, not exact energy or any energy
upper bound. Object mgh is.08903J: the comparable scale warrants investigation,
but neither proves nor excludes artificial lift contribution because the wrist
also does physical work.

Exactly1980 empty+2460 cube=4440 new steps /18.5s simulated; reported clocks
sum12.6116s including actual cube rendering but exclude process startup.
All earlier runs remain: cumulative16,740 physical steps /69.75s before any
later timestep study. There was no new network inference, new object, grasp
search, force/friction change, scene modification or post-result cap selection.

Tracked artifacts:

- `dh3_rigid_transmission_smoke.py`: independent new source; old controller unchanged.
- `evidence/dh3_rigid_transmission_model_20261103_v1.json`: exact frozen model,
  SHA571f31eef67c719a5fb21f9adef9ed939374de314b4f8d04c5dba4f92025f4fd.
- `evidence/dh3_rigid_transmission_20261103_v1.json`: results, costs and transient.

Full outputs are `outputs/dh3_rigid_{empty,cube}_physics_20261103_v1/` and
`outputs/dh3_rigid_transmission_model_20261103_v1/`; source SHA is
bad3f61d58b2861b109080eca0c72bb252f9f3e8c3c529bc956aa3cafe5a2c6d.
These failed structural-validation results remain frozen if a separate
numerical-resolution study follows.
