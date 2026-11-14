"""Freeze a Boya world-palm release pose without candidate/object input.

The public robot uses a fixed throw SE3 selected by global_cam. Its numeric TCP
poses belong to that installation. This tool declares a separate simulator
adaptation: original arm-home palm rotation plus fixed world XYZ, not an
assertion that the author's TCP frame or numeric pose has been reproduced.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import author_transport as transport

SCHEMA = 'anydex_boya_author_fixed_release_v1'
FIXED_WORLD_XYZ_M = (-.4, -.25, .35)
UR_SOURCE = transport.ROOT/'ur_toolbox/ur_toolbox/robot/ur.py'


def source_fixed_throw_policy():
    """Inspect literal fixed TCP poses and the actual throw-before-open call."""
    if transport.sha(UR_SOURCE) != transport.PINS[UR_SOURCE]:
        raise ValueError('Pinned public fixed-throw source changed')
    tree = ast.parse(UR_SOURCE.read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    pose = functions['throw_pose']
    if [arg.arg for arg in pose.args.args] != ['self']:
        raise ValueError('Public fixed throw must take no grasp/object arguments')
    branch = next(node for node in pose.body if isinstance(node, ast.If))
    if not (isinstance(branch.test, ast.Attribute) and branch.test.attr == 'global_cam'
            and isinstance(branch.test.value, ast.Name) and branch.test.value.id == 'self'):
        raise ValueError('Public throw pose must depend only on the camera installation')
    fixed = {}
    for name, nodes in (('global_camera', branch.body), ('wrist_camera', branch.orelse)):
        returns = [node for node in nodes if isinstance(node, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            raise ValueError('One literal fixed throw pose required per camera branch')
        value = np.asarray(ast.literal_eval(returns[0].value.args[0]), dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError('Six finite literal XYZ/rotation-vector throw coordinates required')
        fixed[name] = value.tolist()
    def calls(name, method):
        return [node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == method
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self']
    movements = calls('throw', 'movel')
    if (len(movements) != 1 or not isinstance(movements[0].args[0], ast.Call)
            or not isinstance(movements[0].args[0].func, ast.Attribute)
            or movements[0].args[0].func.attr != 'throw_pose'):
        raise ValueError('Public throw must command its fixed throw_pose')
    throws, opens = calls('grasp_and_throw', 'throw'), calls('grasp_and_throw', 'open_gripper')
    if len(throws) != 1 or not any(node.lineno > throws[0].lineno for node in opens):
        raise ValueError('Actual grasp_and_throw must call fixed throw before opening')
    return dict(source=dict(path=str(UR_SOURCE), sha256=transport.sha(UR_SOURCE)),
        throw_pose_line=pose.lineno, throw_movel_line=movements[0].lineno,
        actual_throw_call_line=throws[0].lineno,
        subsequent_open_line=min(node.lineno for node in opens if node.lineno > throws[0].lineno),
        literal_tcp_xyz_rotvec_poses=fixed, coordinate_units='metres/radians; original robot TCP/base convention',
        selection='Fixed camera-installation branch only; no object or grasp position input',
        numeric_tcp_poses_copied_into_boya=False)


def _reference_home_state(model, reference):
    """Complete FK coordinates; zero fingers do not define an executed hand pose."""
    targets = dict.fromkeys(model.active_names, 0.)
    targets.update(reference.arm_home)
    targets.update(FAJ1=0., FAJ3=0.)
    return model.full_state(targets)


def _home_fk(config):
    transport._backend_path()
    from run_template_arm_positive import bind_hand
    bind_hand(SimpleNamespace(hand_urdf=Path(config['hand_urdf']), hand_scene=Path(config['hand_scene'])))
    from canondex.planning.model import PlanningModel
    from canondex.robots.reference import load_reference, home_source_receipt

    model = PlanningModel(Path(config['robot_urdf']))
    reference = load_reference(Path(config['reference_root']))
    home = _reference_home_state(model, reference)
    palm = model.fk.transforms(home, reference.base_transform)['palm']
    source_files = [Path(config[key]).resolve() for key in ('robot_urdf', 'hand_urdf', 'hand_scene')]
    source_files += [Path(config['reference_root'])/'ur_by_description/urdf/ur_by.urdf.xacro',
                     Path(config['reference_root'])/'ur_by_moveit_config/config/ur_with_dexhand.srdf',
                     transport.CANON/'canondex/planning/model.py',
                     transport.CANON/'canondex/robots/reference.py',
                     transport.CANON/'canondex/robots/kinematics.py']
    return transport.transform(palm), dict(home_full_state=home,
        coordinate_semantics=dict(independent_coordinates=len(model.active_names),
            expanded_coordinates=len(home), independent_names=list(model.active_names),
            zero_hand_coordinates='Two wrists and sixteen finger motors; four distal mimics are derived',
            purpose='Complete FK input only; finger zero is not an execution/open-hand command',
            palm_rotation_depends_on_finger_joints=False),
        reference_home=home_source_receipt(reference), base_transform=reference.base_transform.tolist(),
        source_sha256={str(path.resolve()): transport.sha(path) for path in source_files})


def build_release_spec(scene_path, backend_config, output):
    """Generate exactly one declared pose; read no candidate or body transform."""
    scene_path, backend_config, output = (Path(path).resolve() for path in (scene_path, backend_config, output))
    scene = transport.read(scene_path)
    if scene.get('schema') != 'anydex_boya_author_scene_v1':
        raise ValueError('Explicit captured author scene required')
    # Only scene/bin identity is read here; body transforms have no role in
    # computing either rotation or translation of the fixed release target.
    transport.transform(scene['bin']['world_T_bin'])
    config = transport.read(backend_config)
    author = source_fixed_throw_policy()
    home, fk = _home_fk(config)
    release = np.eye(4)
    release[:3, :3] = home[:3, :3]
    release[:3, 3] = FIXED_WORLD_XYZ_M
    spec = dict(schema=SCHEMA, scene=dict(path=str(scene_path), sha256=transport.sha(scene_path)),
        bin_sha256=transport.json_sha(scene['bin']), world_T_release_palm=release.tolist(),
        frame='world', link='palm', pose_rule='fixed_world_se3',
        depends_on_candidate_pose=False, depends_on_object_pose=False,
        source_sha256={str(UR_SOURCE): transport.sha(UR_SOURCE),
                       str(Path(__file__).resolve()): transport.sha(__file__),
                       str(backend_config): transport.sha(backend_config), **fk['source_sha256']},
        source_policy=author,
        adaptation=dict(method='fixed_home_fk_rotation_and_declared_world_xyz',
            fixed_world_xyz_m=list(FIXED_WORLD_XYZ_M), home_palm_world_se3=home.tolist(),
            home_full_state=fk['home_full_state'], reference_home=fk['reference_home'],
            fk_coordinate_semantics=fk['coordinate_semantics'],
            base_transform=fk['base_transform'],
            declaration='Boya simulator frame adaptation, not the original robot TCP numeric pose; freeze before candidates',
            physical_or_reachability_acceptance=False), physics_steps=0, candidate_reads=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(spec, stream, indent=2, allow_nan=False); stream.write('\n')
    transport.load_release_spec(scene_path, output, scene=scene)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True, help='Actual observed_scene.json, unchanged')
    parser.add_argument('--backend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh explicit sidecar JSON')
    args = parser.parse_args()
    result = build_release_spec(args.scene, args.backend, args.output)
    print(json.dumps(dict(schema=result['schema'], output=str(args.output.resolve()),
        sha256=transport.sha(args.output), world_T_release_palm=result['world_T_release_palm'],
        physics_steps=0), indent=2))


if __name__ == '__main__':
    main()
