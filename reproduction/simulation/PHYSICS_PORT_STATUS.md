# DH3 physical-port status, 2026-11-03

Historical stage: later fixed-grid results and cumulative costs are in [TIME_RESOLUTION_RESULTS.md](TIME_RESOLUTION_RESULTS.md).

The original-network top1 **actually lifted and held the fixed50mm cube in
both physical trials** (+91.52/+91.73mm), but **neither cube trial passes the
predeclared coupling/velocity acceptance**. Empty-hand mechanism v5 passes.
Increasing non-contact inner solver iterations20→100 did not resolve the
loaded transient. All physical work is paused; no additional pose, force,
friction, object or acceptance search follows from this report.

This is an independent simulation port, not the author's supplied simulator
or real-robot benchmark. No CanonDex or shared Python/Torch/CUDA files changed.

## Preserved source and declared port

The original revision is `c9c4a43df33e40860417c7e2dd02f5122d3b2da2`.
The unchanged source formula is extracted from
`generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py` by
`reproduction/runtime/dh3_author_kinematics.py`; all355 released lookup rows
have already matched it. Source geometric endpoint position0 is evaluable,
but no released measured position0 pose or calibrated dynamic model exists.

For original registers position p and rotation r, let

```
f = 1.333 * (1 - p/95)
a = -1.08 + 1.51 * r/95
```

The helperless adapter has all three main flexions=f; each outlet joint is
`.01+(1+.05/1.333)*f`; each tip joint is `(1+.1/1.333)*f`.
Finger1 base=a, finger3 base=-a, finger2 base remains fixed. The actual code
composes the original affine formula and adapter manifest rather than
substituting a separately tuned hand pose.

Only finger1 flexion and base are active motors. The other nine coordinates
are passive motors with force0, coupled by finite gear constraints:
`q_dependent - factor*q_master = offset`. Nonzero offsets are preserved.
Every actual gear ratio/offset and constraint info is saved in CONTROLLER.
Original force70/speed100 are uncalibrated registers; they are **not**
translated to70N/100rad/s or claimed as reproduced firmware behavior.

The explicit SI assumptions are flexion torque≤.6Nm/command speed≤1rad/s,
rotation torque≤.3Nm/command speed≤.5rad/s, gear reaction≤.8Nm and ERP.2.
A finite world-fixed wrist constraint has maxForce30 per linear/angular row
(N/Nm) and ERP.2; it tracks a free hand base, not a kinematically reset body.
Observed passive velocities can exceed command speed under contact; they
are recorded and are part of the failed cube acceptance. Gear constraints
are an ideal finite transmission approximation, not an identified DH3
mechanism or adaptive-contact firmware. ERP correction can add energy;
empty traces report its absolute reaction×residual-change work proxy.

Gravity is -9.81m/s², dt1/240s, one substep,100 solver iterations. The tested
non-contact inner count is recorded separately (1,20 or100). Hand friction.5,
restitution0 and zero joint damping are declared defaults. The fixed scene
keeps cube mass.1kg, size50mm, analytic inertia and friction1 from its manifest.

## Geometry, collision and frames

`outputs/dh3_geometry_adapter_20261103_v2/` preserves all13 original entity
meshes/masses/inertias. Load uses `URDF_USE_INERTIA_FROM_FILE`. It removes the
zero-mass moving helper that PyBullet kept motionless in an actual isolated
probe. Only the third-tip source Z discrepancy3.673205e-6rad is approximated;
all original triangle vertices at355 published+96 closure configurations
have maximum error **.122001µm**, below the stated1µm geometry tolerance.
The exact static adapter and its failed dynamics remain in v1 outputs.

Dynamic STL collision imports use Bullet convex hulls. Concave-mesh physical
contact equivalence has not been established; no unmeasured hull-error bound
is asserted. Initial convex geometry has palm/outlet overlap1.303mm, still
.303mm at zero margin, and same-finger tip/outlet overlap10.44mm. The active
adapter therefore declares exactly these additional mechanism collision
exclusions for each finger: palm↔outlet, main flexion link↔outlet and
tip↔outlet. Direct parent collisions are excluded by the normal URDF flag.
These are explicit internal mechanism assumptions, not hardware validation.
Different fingers, other nonadjacent pairs, hand-object and hand-table
collisions remain enabled; no whole-finger exclusion or object attachment
is used. Original STL sizes and13 entity inertials are unchanged. Bullet's
usual mesh margin remains in force; no margin was reduced for the trials.

The cube candidate is the original learned pipeline's selected top1,
pre-collision index3, type4, width63mm, depth39.999999mm, registers[95,100].
Its RGB-D comes from the actual frozen cube scene v2, not a photograph of a
different scene. The existing independent frame audit binds rendered depth,
11 branches,111 original collision admissions and original top10/top1 arrays.

`T_WC` is read from `scene_manifest['camera']['T_WC']`. The selected DH3 hand
datum is transformed into world as a **base_Link frame**. Final insertion is
`depth - position/100*5mm` along world-transformed original two-finger +X.
The prepose subtracts70mm along that axis. The measured base local inertial
transform explicitly converts every target to the COM returned by Bullet;
this composition uses double precision rather than float32 multiplyTransforms.
A further100mm world-Z lift is the declared script, replacing the unavailable
UR arm/firmware execution. It is not claimed as original UR path planning.

Every run initializes q and base pose only before integration. The two reset
APIs are then replaced by guards that raise if called. Wrist targets change
only a finite physical constraint. No state reset, object attachment,
zero-gravity grasp or post-contact teleport occurred in either cube run.
Each cube trial is a separate frozen-initial-scene experiment, not a claim of
continuous现场重抓 across trials.

## Actual results and total cost

| Run | Actual steps | Outcome |
| --- | ---: | --- |
| Empty v1/v2 |0+0| Initial COM precision assertion; both failures retained |
| Empty v3 |1980| Exact static helper; mass0 coordinate stuck and CAD collision conflicts; failed |
| Massless isolation |240+240| No-gravity/fixed-root/no-selfcollision diagnostic only: mass0 stays fixed; tiny positive helper mass moves |
| Gear isolation |960| Gravity on, selfcollision off, tiny helper; diagnostic only, not acceptance |
| Empty v4 |1980| Helperless/internal adjacency corrected; inner1 fails under actual inter-finger contact |
| Empty v5 |1980| inner20: **passes**; gear max.001570rad, wrist max.9998mm, peak speed1.0741rad/s |
| Cube v1 |2460| inner20: **lifted**, +91.517mm, lowest87.927mm; coupling/velocity failed |
| Cube v2 |2460| Only inner20→100 changed: **lifted**, +91.733mm, lowest88.517mm; coupling/velocity still failed |

Total **12,300 actual physical steps /51.25s simulated time**, including all
invalid/isolated diagnostics and both cube trials. Five integrated main-run
clocks sum32.622s, including synchronous TinyRenderer in both cube trials;
that is not total campaign wall time. Python startup and the two isolated
probe wall clocks were not instrumented. Native builds, network inference
and prior zero-step geometry/RGB-D validation have separate receipts and do
not add physical steps to this count. No failed costs were discarded.

Empty v5 fingers meet before the unmeasured geometric endpoint. A measured
self-contact stop is accepted and its remaining target error.1921rad reported;
it is not called endpoint tracking. Gear residual remains small at the stop.

Cube v1's worst coupling event is step1574 /6.5583s during lift, on the
finger2 flexion gear: error.115207rad, reaction at its .8Nm limit. At that
step hand-table normal forces sum14.928N but hand-object forces sum.372N.
At6.6292s finger2 tip reaches5.784rad/s. The last lift-hold phase settles to
max gear.002233rad and max speed.005724rad/s. Table contact competition and
the finite transmission model are implicated; this is not all attributable
to solver numerical error. V1 deepest hand-object/table contact distances
are -.331/-.491mm. Both motor caps and the gear cap are obeyed; wrist error
stays below.834mm.

The single numerical pair at inner100 has worst gear.114253rad and speed
6.763rad/s, again at the same lift transition, with14.949N hand-table versus
.354N hand-object normal load. Last lift-hold gear max is.013476rad. It
therefore **does not establish convergence or repair**. Its hand-object/table
minimum distances are -.403/-.679mm. No acceptance threshold was loosened.

Both cube trials have positive hand-object contacts throughout the final120
steps (.5s),480 positive-normal contact rows, and each has205 real640×480
frames sampled every12 actual physics steps. Encode at20fps for normal
speed (10.25s); media processing requires zero extra physics steps. V1 video:
`outputs/dh3_cube_physics_20261103_v1/grasp_normal_speed.mp4`.

## Evidence, source freezing and replay

Independent trace/candidate/frame audit (zero new physics):
`evidence/dh3_cube_physics_independent_audit_20261103_v1.json`. It independently
recomputes gears from actual q and confirms continuous final positive contacts.

Lightweight all-run results, source receipts, costs and transient fields:
`evidence/dh3_physics_smoke_20261103.json`. Full per-step traces, original
CONTROLLER/RESULTS, contacts and frames remain in the correspondingly named
`outputs/dh3_*physics_20261103_v*/` directories; these assets are not for Git.

Each actual cube and empty v4/v5 source has been preserved as
`controller_source_frozen.py` and matched to the pre-existing receipt SHA.
Cube v1/empty v5 source SHA is
`8d0a0f0180c223f626813fa84d3adae63509807c93136796258b2e5c23f8fb0e`;
cube v2 is
`54536fe4b955af99925e93ee22e60f0359822d61f15d15d4915d9a78b1329070`.
The earlier versions were recovered by reversing subsequent source edits
and **matched byte-exactly to those existing SHAs**; the receipts were not
rewritten. Early empty v3 did not record a source SHA at launch: its numeric
CONTROLLER/trace remain, but exact source identity is a stated limitation.
The two isolated diagnostic scripts were also frozen. The current controller
now automatically freezes its exact source before its first physical step.

`trace.npz` has per-step q/v, motor torque, gear residual/reaction, COM and
velocity, wrist reaction/tracking error, commanded registers and joint targets;
cube traces additionally save object pose/velocity/AABB. `contacts.npy` columns
are step,bodyA,bodyB,linkA,linkB,signed distance(m),normal force(N),two lateral
forces(N). CONTROLLER includes actual constraint info and body IDs, allowing
an independent check that only hand-internal and hand-world constraints exist.
The cube scene manifest/sourceSHA are bound before the first step.

For an explicitly authorized future run, use the existing Python/overlay and
a **new output directory**, for example:

```bash
PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$PY" -B reproduction/simulation/dh3_physics_smoke.py --mode empty \
  --position 95 --rotation 100 --non-contact-inner-iterations 20 \
  --output-dir outputs/NEW_EMPTY_RUN
```

The current physical work remains paused.
Do not rerun either frozen directory or interpret these primitive trials as
a reproduced real-robot success rate or as a validated new hand controller.
