# Original DH3 simulation readiness

更新：本文保留此前原模型/源码审计；后续独立DH3单方块物理烟测已通过，当前结果见[仿真说明](../simulation/README.md)。

The original DH3 is ready for the already completed geometry/inference
reproduction. Its released assets are **not yet a validated physical DH3
simulation**. This audit loaded the unchanged URDF twice in private PyBullet
DIRECT, read its metadata and performed zero-time joint resets: **zero
physics steps, zero grasps, no hardware connection and no CanonDex changes**.
No new packages or environments were installed for this audit.

## What the author actually controls

`ur_toolbox/ur_toolbox/robot/DH3/DH3.py:169-213` sends two device registers:
`position` and `rotation`. They are not 12 independently actuated joint
commands. The selected original prediction `[94,67]` opens with
`SetTargetPosition(94)`, then `SetTargetRotation(67)`. The subsequent original
`close_gripper()` sends only `SetTargetPosition(0)` and retains rotation67.
The exact original method bodies were replayed against a recording-only
interface without executing the socket-opening constructor.

The constructor sets force70 and speed100. The source gives **no conversion
to newtons, joint torques or radians/second**, no calibrated actuator model,
and no software contact-feedback closure loop. Firmware behavior cannot be
recovered from those two numbers alone. The original DH3 close call does not
use a different 12D target/torque per type; type selects its preshape/rotation
and calibrated grasp frame. This differs from the author's Allegro branch,
which explicitly passes per-type close-pose/torque arrays.

`robot_dh3.py:732-757` opens for0.9s, while `close_gripper` waits its default
0.2s. The original robot entry sets `a=v=.01`, then calls the arm routine
with acceleration `.02` and velocity `.03` in that API's units. These arm
parameters are separate from device speed100. In `ur.py:375-377`, DH3 also
subtracts a preshape-dependent `position/100*5mm` from predicted depth.
For this selected pose: stored depth30mm, backoff4.7mm, net target insertion
25.3mm. Those geometric conventions can be reproduced without inventing a
force controller.

## What the lookup provides, and what it cannot provide

The original `recover_dh3_hand_to_stl.py:96-135` converts two register values
to12 geometry joint angles using fixed affine ranges. **All355 published
lookup slots match that formula exactly.** Those ranges were used by the
original mesh/FK generator, which resets joints directly rather than
integrating physical actuation.

| Original pose | Published position range | Rotation |
| --- | --- | --- |
| pose1 | 7-95 | 0 |
| pose2 | 5-95 | 67 |
| pose3 | 9-95 | 100 |
| pose4 | 2-95 | 100 |

No released slot has position0. The original XLSX was also read directly
with Python's standard ZIP/XML reader: its four sheets have headers
`width,position,rotation` and **no position0 record in columnB**. The
hardware closing endpoint therefore cannot be reconstructed exactly merely
by looking up a smaller width. Evaluating the author affine formula at0 is
possible, but supplies an unvalidated geometric endpoint, not measured
closed-hand dynamics or contact response.

## URDF facts from the actual zero-step load

The SolidWorks export has13 links,11 revolute joints and1 fixed joint,
with9 `mimic` tags. All13 link masses are positive; total mass is
0.448484kg. All inertia tensors are positive definite and satisfy the
principal-moment triangle condition. These are useful mathematical checks,
**not hardware identification or validation**.

Eight revolute joints have lower=upper=effort=velocity=0. The three other
revolutes have effort100 and velocity100; the primary flexion joint's upper
limit is1.16rad. The selected published open pose violates eight declared
joint ranges. The nominal formula endpoint for close(position0) violates
nine ranges: primary flexion becomes1.333rad, above1.16rad. No limits were
changed or relaxed in this audit.

The published geometry also conflicts with the URDF mimic equations. At
this open pose, `finger1_tip_joint` is+0.010526rad while its mimic expects
-0.014032rad; `finger1_outlet_joint` is+0.001053rad while its mimic also
expects-0.014032rad. Therefore blindly enabling these mimic equations would
change the author's reconstructed geometry. In the actual default PyBullet
load, there are zero user constraints; zero-step resetting the primary
flexion to0.5rad leaves its mimic joint states at0. This checks the absence
of automatic kinematic propagation for that operation; it does not test
any integrated dynamics.

All13 links load collision meshes. The URDF supplies no contact material
or joint damping/friction tags and no transmission. Observed PyBullet
lateral friction0.5, restitution0 and joint damping/friction0 are defaults,
not author-validated pad properties. Loading with `URDF_USE_INERTIA_FROM_FILE`
produces different inertial diagonals from default loading; the audit saves
both. A later physics port must choose and document this explicitly.

## Smallest justified next steps

1. Keep original files immutable. In a separately versioned simulation
   adapter, reproduce the released preshape, hand datum, depth/backoff and
   approach direction. Confirm loaded link poses/collision shapes agree
   with the original mesh reconstruction before any lifting test.
2. Establish one explicit physical actuation model: resolve the conflicting
   joint limits/mimic equations and the0-position endpoint, then document
   coupling, effort, speed and contact parameters. Author geometry can
   guide the motion, but guessed force/PID values must be labeled a physics
   port. Do not call independently commanding all12 joints the original
   hardware controller.
3. After that adapter is validated, perform one primitive-object tabletop
   close/lift with a scripted wrist, reporting contacts, object motion and
   integration settings. Only then extend to the selected real-scene pose
   or compare object-level success. Such a test would validate the port;
   it would not by itself reproduce the paper's real-robot success rate.

If hardware-consistent DH3 dynamics cannot be obtained, porting the original
AnyDex representation/learned-selection method to a separately validated
Boya hand is a clearer engineering experiment than claiming arbitrary DH3
closure is the author's physical strategy. It remains a new-hand transfer
requiring its own hand-geometry mapping and execution validation; present
DH3 learned labels cannot silently be treated as Boya labels. Neither route
has been implemented or tested by this audit.

Run this bounded audit again with the existing isolated overlay:

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/audit_dh3_simulation_readiness.py
```

Full local receipt:
`outputs/dh3_simulation_readiness_20261103_v1/RESULTS.json`.
Lightweight tracked source/control/XML/PyBullet evidence:
`evidence/dh3_simulation_hand_readiness_20261103.json`.

## Original kinematic interface now implemented

`dh3_author_kinematics.py` is a pure NumPy/stdlib interface that extracts the
original unchanged `rate()` and `pos_scope` AST from the SHA-pinned author
source. `joint_angles(position,rotation)` supports finite position[0,95] and
rotation[0,100], and returns the original12 geometric values in radians.
The original formula denominator remains0..95 even for the author's released
rotation100 values. **All355 published12D rows match exactly**, with zero
maximum difference.

`closure_path(94,67,samples=96)` now actually saves96 geometric samples down
to the original close register `[0,67]`, with constant rotation. The NPZ has
normalized progress and register/joint values; it has no time, torque, speed,
contact response or physical-success claim. The formula's fifth value
(zero-based slot4) stays1, preserving the original array; the original URDF
joint at this index is fixed and is not made movable.

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
"$ANYDEX_PY" -B reproduction/runtime/dh3_author_kinematics.py
```

The saved trajectory is
`outputs/dh3_author_kinematics_20261103_v1/selected_top1_to_position0_geometric_path.npz`.
The lightweight source/355-row/trajectory receipt is
`evidence/dh3_author_kinematics_20261103.json`.

A further original geometry boundary must be resolved before adjusting URDF
limits: `get_meshes` at `recover_dh3_hand_to_stl.py:213-220` swaps **only link
positions**, not rotations, for pairs2/3,6/7,10/11 when assembling STLs. The
original raw URDF link ordering is also tip then outlet, while the author's
mesh map assigns outlet then tip. Thus direct URDF FK/collision geometry
cannot be assumed to match the assembled geometry used by the original
inference collision gate. A later separately versioned port needs explicit
per-link geometry comparison; merely increasing joint limits does not make
it validated. No URDF edit, geometry-port implementation or dynamic test was
performed in this stage.

## Later geometric adaptation completed

The next authorized stage resolved the original position-only link swap with
an explicit static tree adapter, documented in
`../simulation/DH3_GEOMETRY_ADAPTER.md`. All355 preshapes plus96 formula closure
points now match every original entity STL vertex (max3.2514e-8m); original
13 entity inertias/STLs remain unchanged, with one zero-mass coordinate helper.
This solves geometry alignment without per-frame mesh resets. Original URDF
observations above remain unchanged, and this later geometry check does not
by itself validate dynamic coupling, hardware force calibration or lifting.

The later exact adapter's zero-mass helper was found dynamically locked in
PyBullet by separate controller diagnostics. The single helperless v2 keeps
all13 original entity inertias/STLs and bounds the third-tip source-Z
approximation:451-case/all-vertex maximum0.122001µm, below1µm. Its explicit
joint map and geometry evidence are documented in
`../simulation/DH3_GEOMETRY_ADAPTER.md`; no helper mass, mesh scaling or
control/scene change is introduced by that alternative.
