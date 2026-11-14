"""Read saved failed commands; no planning, state changes, or admission changes."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, sys, time
import numpy as np
import trimesh, coal

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
OUT = ROOT/'outputs/boya_author_pipeline_home_release_20261114_v1'
sys.path[:0] = [str(ROOT/'reproduction/boya'), '/home/wangrenpeng/CanonDex']
import author_transport as transport
from canondex.planning.model import PlanningModel

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')
def bvh(mesh):
    value = coal.BVHModelOBBRSS()
    value.beginModel(len(mesh.faces), len(mesh.vertices))
    vertices, faces = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
    for p in np.asarray(mesh.vertices): vertices.append(p)
    for f in np.asarray(mesh.faces): faces.append(coal.Triangle(*map(int, f)))
    value.addSubModel(vertices, faces)
    value.endModel()
    return value
def T(frame): return coal.Transform3s(frame[:3, :3], frame[:3, 3])
req = coal.CollisionRequest()
req.num_max_contacts = 1
req.enable_contact = False
dreq = coal.DistanceRequest(True, 0., 1e-10)
dreq.enable_signed_distance = False
dreq.gjk_tolerance = 1e-10
def exact_pair(a, ta, b, tb):
    result = coal.CollisionResult()
    coal.collide(a, T(ta), b, T(tb), req, result)
    hit = bool(result.isCollision())
    distance = 0.
    if not hit:
        d = coal.DistanceResult()
        coal.distance(a, T(ta), b, T(tb), dreq, d)
        distance = max(0., float(d.min_distance))
    return dict(source_triangle_surfaces_intersect=hit, source_triangle_unsigned_distance_m=distance)

start = time.monotonic()
protocol = read(OUT/'COMPONENT_INTEGRATION_PROTOCOL.json')
scene = read(protocol['inputs']['observed_scene']['path'])
owners = {}
for i in range(27):
    owners[i] = ('graspnet_000' if i < 2 else 'graspnet_002' if i < 19 else
                 'graspnet_005' if i < 22 else 'bin:'+transport.bin_boxes(scene['bin'])[i-22]['name'])
model = None
mesh_cache, bvh_cache = {}, {}
boxes = transport.bin_boxes(scene['bin'])
for i, box in enumerate(boxes, 22):
    mesh = trimesh.creation.box(extents=box['extents_m'])
    mesh.apply_transform(box['world_T_box'])
    bvh_cache[owners[i]] = bvh(mesh)
records, exact = [], []
for g in (1, 6, 7):
    d = OUT/f'integration_g{g}/transport_raw'
    plan, geo, opened = read(d/'plan.json'), read(d/'lift_arm_geometry.json'), read(d/'path_geometry.json')
    with np.load(d/'commands.npz', allow_pickle=False) as z: a = {k:z[k].copy() for k in z.files}
    names = list(map(str, a['joint_names']))
    phases = a['phase'].astype(str)
    post = np.flatnonzero(~np.isin(phases, ['settle', 'home_to_pregrasp', 'approach']))
    v = geo['violations']
    def item(row):
        n = int(post[row['sample']])
        return dict(row, command_sample=n, time_s=float(a['time_s'][n]), phase=str(phases[n]),
                    owner=owners[row['other']] if row['kind']=='object' else row.get('other', row['kind']))
    groups = defaultdict(list)
    for row in v:
        it = item(row)
        groups[(it['phase'], it['kind'], it['link'], str(it['owner']))].append(it)
    group_records = []
    for (phase, kind, link, owner), rows in sorted(groups.items()):
        group_records.append(dict(phase=phase, kind=kind, link=link, owner=owner, count=len(rows),
            first=min(rows,key=lambda x:x['sample']), last=max(rows,key=lambda x:x['sample']),
            deepest=min(rows,key=lambda x:x['distance_m'])))
    endpoint = [item(x) for x in v if x['sample']==len(post)-1]
    counts = read(OUT/f'PLANNING_COUNTS_g{g}.json')
    records.append(dict(gesture_id=g, commands=len(phases), phases=dict(Counter(phases)),
        post_samples=len(post), count=len(v), counts_by_kind=dict(Counter(x['kind'] for x in v)),
        counts_by_object_owner=dict(Counter(owners[x['other']] for x in v if x['kind']=='object')),
        counts_by_phase=dict(Counter(str(phases[post[x['sample']]]) for x in v)),
        first=item(min(v,key=lambda x:x['sample'])), deepest=item(min(v,key=lambda x:x['distance_m'])),
        final_command_violations=endpoint, grouped=group_records, open_ok=opened['ok'],
        seed_full_scene_parts=opened['seed_target_geometry']['object_collision_parts'],
        seed_full_scene_samples=opened['seed_target_geometry']['samples'],
        extra_open_parts=opened['multi_object_open']['object_collision_parts'],
        extra_open_samples=opened['multi_object_open']['samples'],
        original_ik_counts=counts, transport_ik_audits=len(plan['transport_ik_waypoints']),
        inputs={str(p.resolve()):sha(p) for p in [d/'plan.json', d/'commands.npz', d/'lift_arm_geometry.json', d/'path_geometry.json']}))
    if model is None: model = PlanningModel(Path(plan['arguments']['robot_urdf']))
    selected = [('deepest_self', min((x for x in v if x['kind']=='self'),key=lambda x:x['distance_m']))]
    selected += [('final_bin', x) for x in v if x['sample']==len(post)-1 and x['kind']=='object']
    for tag, row in selected:
        it = item(row)
        frames = model.fk.transforms(model.full_state(dict(zip(names,map(float,a['joint_target'][it['command_sample']])))))
        link = row['link']
        other = row['other'] if row['kind']=='self' else owners[row['other']]
        for name in [link]+([other] if row['kind']=='self' else []):
            if name not in bvh_cache:
                mesh_cache[name] = model.collision_mesh(name)
                bvh_cache[name] = bvh(mesh_cache[name])
        tb = frames[other] if row['kind']=='self' else np.eye(4)
        exact.append(dict(gesture_id=g, representative=tag, **it,
            **exact_pair(bvh_cache[link],frames[link],bvh_cache[other],tb),
            source_link_triangles=len(mesh_cache[link].faces),
            source_other_triangles=len(mesh_cache[other].faces) if row['kind']=='self' else 12,
            source_link_watertight=bool(mesh_cache[link].is_watertight),
            source_other_watertight=bool(mesh_cache[other].is_watertight) if row['kind']=='self' else True))

save(OUT/'POST_GEOMETRY_BREAKDOWN.json',dict(schema='anydex_boya_author_saved_post_geometry_breakdown_v1',
    scope='Saved violations only; post sample indexes mapped to full command indexes without replanning.',
    owner_indexes=owners, results=records, physics_steps=0, new_planning_calls=0))
save(OUT/'SOURCE_TRIANGLE_DIAGNOSTIC.json',dict(schema='anydex_boya_author_fixed_post_source_triangles_v1',
    scope='Each gesture deepest self state and final bin-wall states only. Exact URDF collision triangle surfaces and declared bin boxes, not whole-link convex hulls.',
    results=exact, physics_steps=0, new_planning_calls=0, wall_s=time.monotonic()-start,
    limitations='Surface intersection is established only at selected states. A negative surface result with separation does not alone rule out volume containment. Proxy penetration depths are retained from unchanged planning audit, not exact triangle penetration depths.',
    bindings={str(Path(__file__).resolve()):sha(__file__),str(Path(plan['arguments']['robot_urdf'])):sha(plan['arguments']['robot_urdf']),
              **{v['path']:v['sha256'] for v in protocol['sources'].values()}}))
print(json.dumps(dict(results=exact, total_violations=[x['count'] for x in records], wall_s=time.monotonic()-start),indent=2))
