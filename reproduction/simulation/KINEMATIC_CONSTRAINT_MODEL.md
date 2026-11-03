# What the two-register formula does and does not specify

The released DH3 geometry defines an affine preshape/closure curve. After
removing the fixed coordinate and applying the independently checked mesh
frame adapter, write its eleven movable coordinates as

```
q = A u + b,                 u = [position_register, rotation_register]
C q = d,                    rank(C) = 9,     C A = 0
```

Every row of `C` is the explicitly recorded relation
`q_dependent - factor*q_master = offset`. The two independent coordinates
are the finger1 flexion and finger1 base rotation. This algebra describes a
two-dimensional configuration subspace; it is not a hardware transmission
calibration. Measured contact compliance/differentials are not provided by
the affine geometry table.

A rigid idealization has dynamics of the form

```
M(q) qdd + h = B tau_motor + J_contact^T f_contact + C^T lambda
C qdot = 0
P_internal_constraint = lambda^T C qdot = 0
```

Here `lambda` is structural reaction, not an additional position motor.
It can balance externally applied wrist/contact loads even when the two
motor torques are small. A chosen cap on `lambda` therefore has a different
meaning from the two active motor caps. If that cap saturates and `Cq=d`
slips, the simulator is implementing a slipping transmission, not the
original formula as a rigid configuration relation. The previous 0.8Nm
cap is a disclosed simulation assumption, not a published DH3 specification.

A high finite solver cap can approximate the rigid relation while retaining
finite motor/wrist forces. Its admissibility must be checked from the actual
constraint residual, speeds, force readback, cap margin and contact dynamics;
raising a cap alone does not prove physical validity. External wrench times
a geometric lever gives an engineering scale only. Contact configurations,
mechanical advantage and constraint conditioning prevent treating that scale
as a general theorem bounding every reaction. No cap used here is identified
from real DH3 hardware.

Bullet's original `btMultiBodyGearConstraint.cpp` constructs a joint-coordinate
Jacobian with coefficients `1` and `gearRatio`, an ERP position correction,
and a symmetric constraint-impulse bound. It does not create a separate
active finger motor. The unchanged source file is available in the pinned
local PyBullet3.2.7 sdist under
`outputs/native_deps/build_src/pybullet-3.2.7/src/BulletDynamics/Featherstone/`.

With finite numerical error, the recorded sampled quantity
`sum_t |lambda_t * (residual_t - residual_(t-1))|` measures a constraint
correction-work proxy. It is not a strict bound on continuous-time energy
injection or proof of passivity. Full discrete energy accounting would need
additional solver/contact information. The receipts retain legacy field
names containing `upper_proxy`; interpret them with this limitation.

A validated idealized reference model can support an algorithm smoke test
in its declared simulated physics. It cannot establish identified firmware,
real-world success rate, or equivalence to the paper's UR5/D415 experiments.
