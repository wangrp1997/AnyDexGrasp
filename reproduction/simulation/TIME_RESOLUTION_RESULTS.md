# DH3 time-resolution results

**The1920Hz and3840Hz runs both lift/hold the same cube and pass the original
coupling, speed, motor, gear and wrist criteria.** Their final lift differs
only.073854mm. This is one declared primitive simulation-port validation,
not hardware identification or the author's real-robot benchmark.

All rows use the same original learned top1,100Nm structural row ceiling,
inner20, .6/.3Nm active motors, wrist30 per row, physical-time control,
geometry, internal collision matrix, object and camera. Plans/source SHAs
were frozen before each fixed group. No new inference or pose search occurred.

| Hz | Max gear error(rad) | Peak speed(rad/s) | Final lift(mm) | Absolute sampled correction-work proxy(J) | Original gates |
| ---: | ---: | ---: | ---: | ---: | --- |
|240|.162744|2.71120|90.7524|.0949629|Fail|
|480|.0504754|2.60468|89.8060|.0834456|Fail|
|960|.0112864|1.88535|88.8802|.0164004|Fail|
|1920|.00171178|1.37707|88.5180|.0115727|Pass|
|3840|.000453969|1.26194|88.5919|.00767370|Pass|

The two fine grids retain motor bounds.6/.3Nm; observed position motor peak
is.6Nm, rotation peak.1427/.1699Nm. Structural reaction peaks.9486/1.0827Nm,
far below the100Nm numerical ceiling, and wrist COM errors.1042/.0521mm.
Hand-object/table minimum signed distances are -27.37/-26.01µm at1920 and
-27.25/-14.19µm at3840. Every step of the final.5s has positive-normal object
contacts. Both runs have zero dynamic reset/attachment and retain gravity.

The960Hz empty regression passes (gear4.061e-5rad, speed1.0747rad/s). Its
accumulated absolute correction proxy is not monotonic against240Hz, so a
universal energy convergence law is not claimed. For the cube, adjacent-grid
peak-error orders are1.69,2.16,2.72,1.91. These are empirical, consistent with
significant discretization error, not proof of a general quadratic order for
nonsmooth contacts. The final absolute proxy decreases by34% on the last
pair; it is endpoint-sampled sum(abs(lambda*delta residual)), **not exact
energy or an energy upper bound**. Passing gates does not remove this limit.

The source observation manifest retains dt1/240 and its original signature.
Each run explicitly overrides integration dt after zero-step observation/state
binding; geometry, object, camera, candidate and source images stay unchanged.
Every actual substep is recorded with numSubSteps=1. There are205 actual frames
per cube,20fps, sampled every24/48/96/192 steps respectively. No hidden coarse
output discards substep contacts or constraint reactions.

All history remains: **98,460 actual integration steps /119s physical time at
mixed dt**, including low-cap failures, isolated diagnostics, rigid coarse-grid
failures and all refinement runs. The final pair adds59,040 steps /20.5s;
reported clocks23.494/34.217s include actual rendering but exclude process
startup and overlap because the runs were concurrent. Do not sum them as
campaign elapsed wall time. Refinement stopped at3840; no further run follows.

Full per-step outputs:
`outputs/dh3_dt{1920,3840}_cube_physics_20261103_v1/`.
Light complete curve/cost/source/contact/frame evidence:
`evidence/dh3_time_resolution_results_20261103_v1.json`.
Frozen plans:
`evidence/dh3_time_resolution_plan_20261103_v1.json` and
`evidence/dh3_time_refinement_plan_20261103_v1.json`.
Independent source versions are `dh3_time_resolution_smoke.py` and
`dh3_time_refinement_smoke.py`; earlier source/reports and failed receipts
remain intact. No claim about arbitrary objects, firmware accuracy or paper
success rate is inferred from this single frozen cube and scripted wrist.
