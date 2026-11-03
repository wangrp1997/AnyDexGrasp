# Exact original DH3 mesh geometry adapter

The independent static URDF adapter aligns all355 released preshapes and96
points on the selected `[94,67]→[0,67]` formula closure path. Every triangle
vertex of all13 original entity STLs is checked, without sampling. Maximum
actual PyBullet/source-frame vertex error is **3.2514e-8m**; independent
float64 transform algebra error is **2.22e-16**. Original unadapted URDF
geometry gives an81.489mm negative-control maximum difference. Verification
uses only zero-time FK joint resets: **zero physics steps**.

The author source/STLs and all13 original entity mass/inertial elements stay
unchanged. One zero-mass frame with no visual/collision geometry is added to
the derived URDF. This is a geometry adapter; dynamic constraints/actuator
limits and the numerical validity of that massless frame still require their
own integrated validation.

## Why swapping STL filenames alone is insufficient

Original `recover_dh3_hand_to_stl.py:get_meshes` obtains raw URDF link
transforms, then swaps **only translations** for pairs2/3,6/7,10/11. Its mesh
map also assigns the outlet STL to the raw tip index and tip STL to the raw
outlet index. Thus the desired outlet mesh has its original outlet position
and original tip rotation; desired tip mesh has its original tip position and
original outlet rotation. A static visual origin cannot correct these
variable relative rotations for all closure positions.

For one finger, write `f,t,o` for original flex, tip and outlet geometry
coordinates, `R_B` for base rotation, and `A,B` for the original main-flex and
outlet fixed Z rotations. Original transforms include:

```
R_main = R_B A Ry(-f)
R_tip  = R_B A Ry(-(f+t))
R_out  = R_B B Ry(o)
p_tip  = p_main + R_main d_tip
p_out  = p_base + R_B d_out
```

The adapter expresses the mixed transforms as an ordinary fixed URDF tree:

* Outlet mesh remains on its original outlet entity, attached at `d_out` to
  the finger base. Its joint origin rotation is `A`, axis `-Y`, coordinate
  `f+t`. Its position and rotation are exactly the desired original mesh.
* For fingers1/2, `A=B`. The original tip entity stays at `d_tip` under the
  main link, with axis `+Y` and coordinate `f+o`, cancelling parent flex and
  leaving `R_B B Ry(o)`.
* Finger3 has `A_z=1.5707963267949` and `B_z=1.5708`. The source difference
  `delta=3.6732051e-6rad` is preserved. A zero-mass counterflex frame at
  `d_tip` rotates `+Y*f`; the final tip joint applies fixed `Rz(delta)` and
  `+Y*o`. This yields `R_B A Ry(-f) Ry(f) Rz(delta) Ry(o) = R_B B Ry(o)`.

No mesh is moved or reset per dynamic frame. The driver receives only joint
coordinates. The original fifth formula slot (index4) remains ignored by the
fixed `finger2_base_joint`; it is not changed into a movable joint.

## Files and driver interface

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$ANYDEX_PY" -B \
  reproduction/simulation/dh3_geometry_adapter.py
```

Local derived outputs:

* `outputs/dh3_geometry_adapter_20261103_v1/dh3_author_geometry.urdf`
* `ADAPTER.json`: explicit joint-name→original12D linear map, source/STL SHA,
  helper frame and all declared differences.
* `VALIDATION.json`: all-case/all-link maximum errors and negative control.
* `case_vertex_errors.json`: all451 case errors, retained under ignored outputs.

The module API `adapter_targets(author12,manifest)` returns targets keyed by
movable joint name. The exact map is stored in `movable_joint_map`, allowing
control code to derive its own finite coupling constraints. No force, gains,
velocity, timestep or controller is selected by this module. `--generate-only`
creates the derived files without claiming verification.

The adapter removes incompatible original mimic tags and changes position
limits to envelopes of the original formula over position[0,95] and
rotation[0,100]. Existing effort/velocity values, including zeros, are
preserved; they are not asserted to be usable motor calibration. A separate
control/physics adapter must explicitly document any subsequent settings.

The base link's unchanged local inertial origin is
`[-0.000222201146896917, 2.99284922239407e-6, 0.0365598116000577]`m with
identity quaternion in the source URDF. Bullet may diagonalize the full
inertia tensor into a nonidentity principal-inertia orientation; read the
actual position and quaternion from `getDynamicsInfo` rather than assuming
the source identity orientation. `getBasePositionAndOrientation` reports
that center-of-mass/principal-inertia pose; an explicit base reset must
use `T_world_COM = T_world_baseLink * T_baseLink_COM`. The author's hand datum
must not accidentally be used as the center of mass.

Tracked lightweight receipts are in `evidence/dh3_geometry_*_20261103.json`.
The original source and all entity inertia/mesh bytes remain separate from
this generated adapter. This verification claims geometry correspondence,
not stable dynamics, measured closed pose, grasp/lift success or equivalence
to the real DH3 actuator firmware.

## Helperless v2 for integrated physics

Separate dynamic diagnostics found that the exact version's zero-mass
counterflex coordinate remains locked in PyBullet. The exact v1 remains a
geometry reference and is not relabeled dynamically usable. The single v2
alternative removes that helper and explicitly omits only the third tip's
source fixed-Z difference3.6732051e-6rad. It uses `f+o` for the third tip,
matching the first two fingers. No source mesh is scaled, no original entity
mass/inertia changes, and no tiny helper mass is introduced.

All451 cases and every triangle vertex of the13 original entity meshes were
actually checked again: maximum error **1.2200099e-7m =0.122001µm**, below
the declared1µm bound. The maximum occurs on the third fingertip; other links
retain the earlier float32-scale errors. This is a transparent bounded
geometry approximation, not an exact algebra claim or successful grasp.
Original negative-control geometry remains81.489mm apart at maximum.

```bash
ANYDEX_PY=/home/wangrenpeng/boya-inhand-wm/.venv/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$ANYDEX_PY" -B \
  reproduction/simulation/dh3_geometry_adapter.py --helperless \
  --output-dir outputs/dh3_geometry_adapter_20261103_v2
```

V2 `dh3_author_geometry.urdf`, `ADAPTER.json` and `VALIDATION.json` live in that
new directory. It has13 entity links, zero auxiliary links and no
`finger3_author_counterflex_joint`. The explicit third-tip mapping is now
`finger3_tip_joint = author12[9] + author12[11]`; read each version's own
manifest rather than reusing the v1 joint list. Both versions retain the
same original mass/STL provenance and zero-step validation protocol.
V2 verification took3.747s. Lightweight receipts are
`evidence/dh3_geometry_{adapter,alignment}_helperless_20261103.json`.
The geometry adapter does not choose self-collision exclusions or change
control/scene code; those remain independently audited physics decisions.
