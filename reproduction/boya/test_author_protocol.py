"""Author policy tests; no physical steps, real-data training or checkpoints."""
import copy
import unittest

import numpy as np

import author_protocol as port


class AuthorProtocolTests(unittest.TestCase):
    def test_collection_policy_pairing_flip_and_local_rng(self):
        n = 603
        rows = np.zeros((n, 17), np.float32)
        rows[:, 0] = np.arange(n) / n
        rows[:, 1:4] = [.05, .02, .01]
        rotations = np.repeat(np.eye(3)[None], n, axis=0)
        rotations[::2] = np.diag([1., -1., -1.])
        rows[:, 4:13] = rotations.reshape(n, 9)
        features = np.repeat(np.arange(n, dtype=np.float32)[:, None], 2261, axis=1)
        state = np.random.get_state()
        for ref in ('allegro', 'dh3'):
            with self.subTest(reference=ref):
                out = port.collection_candidates(rows, features, reference=ref, reference_type=1, seed=741)
                ids = np.arange(n-1, n-501, -1)
                np.testing.assert_array_equal(out['source_indices'], ids)
                np.testing.assert_array_equal(out['features'][:, :2261], features[ids])
                np.testing.assert_array_equal(out['features'][:, -1], out['if_flip'])
                self.assertEqual(len(out['rows']), 500)
                sign = out['rows'][:, 4:13].reshape(-1, 3, 3)[:, 1, 1]
                self.assertTrue(np.all(sign >= 0) if ref == 'allegro' else np.all(sign <= 0))
                rng = np.random.RandomState(741)
                rng.randint(1, 2, 500)
                np.testing.assert_array_equal(out['extra_depth_m'], rng.randint(0, 4, 500)*.01)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        np.testing.assert_array_equal(rows[:, 4:13], rotations.reshape(n, 9))

    def test_voxel_loop_matches_whole_original_detector(self):
        import open3d as o3d
        import math
        original = port.definitions('utils/collision_detector.py',
            {'ModelFreeCollisionDetectorMultifinger'}, dict(np=np, copy=copy, o3d=o3d, time=port.time, math=math))
        cls = original['ModelFreeCollisionDetectorMultifinger']
        rng = np.random.RandomState(173)
        hand = rng.uniform(-.02, .02, (1200, 3))

        class Group:
            widths = np.array([.05])
            heights = np.array([.02])
            depths = np.array([.01])
            translations = np.zeros((1, 3))
            rotation_matrices = np.eye(3)[None]

            def __len__(self): return 1
            def __iter__(self): return iter([self])
            def __getitem__(self, item): return self
            def graspgroupTR_2_TR(self, *args): pass

        for distance, included in ((d, n) for d in (0., .04, .08, .1) for n in (0, 2, 3, 30)):
            with self.subTest(approach_distance=distance, included=included):
                scene = np.vstack((hand[:included], [[1., 1., 1.], [-1., -1., -1.]]))
                detector = cls(scene)
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(hand)
                detector.load_meshes_pcls = lambda *args: [copy.deepcopy(cloud)]
                whole = detector.detect(Group(), Group(), '', {}, min_grasp_width=.01,
                    VoxelGrid=.003, approach_dist=distance, collision_thresh=2, DEBUG=False)[2]
                part = port.voxel_collision(scene, hand, [1., 0., 0.], hand_voxel_m=.003,
                    approach_distance_m=distance, collision_threshold=2)
                self.assertEqual(part['admitted'], bool(whole[0]))
                self.assertEqual(part['included_scene_points'], included)
                self.assertEqual(part['admitted'], included <= 2)
                self.assertFalse(part['hand_surface_generation_validated'])

    def test_original_epoch_skips_singleton_in_large_dataset(self):
        import torch
        import train_boya_decision as components
        torch.set_num_threads(2)
        cls = port.definitions('models/minkowski_graspnet.py',
            {'MinkowskiGraspNetMultifingerType1'},
            dict(torch=torch, nn=torch.nn, F=torch.nn.functional))['MinkowskiGraspNetMultifingerType1']
        torch.manual_seed(231)
        model = cls(num_multifinger_type=1, num_multifinger_depth=4,
                    num_two_finger_angle=1, num_two_finger_depth=5)
        criterion = components.official_components()[1](num_multifinger_type=1,
            num_multifinger_depth=4, num_two_finger_angle=1, num_two_finger_depth=5, train_type=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)

        def batch(n):
            target = torch.arange(n) % 20
            return dict(grasp_preds_features=torch.randn(n, 480),
                result=(torch.arange(n) % 2)[:, None],
                two_fingers_pose_angle_type=torch.zeros(n, 1, dtype=torch.long),
                two_fingers_pose_depth_type=(target//4)[:, None],
                multifinger_pose_depth_type=(target%4)[:, None],
                multifinger_pose_finger_type=torch.ones(n, 1, dtype=torch.long),
                if_flip=torch.zeros(n, 1, dtype=torch.long))

        scheduler = components.official_components()[2](optimizer, max_iter=100, power=.99, last_step=-1)
        port.original_epoch(model, optimizer, criterion, [batch(128), batch(1)], scheduler=scheduler, gesture=1)
        self.assertTrue(all(int(model.state_dict()[f'bn{i}.num_batches_tracked']) == 1 for i in range(1, 8)))
        self.assertTrue(all(int(state['step']) == 1 for state in optimizer.state.values()))
        inference = components.official_components()[0]()
        inference.load_state_dict(model.state_dict(), strict=True)
        model.eval();inference.eval();data = batch(11)
        with torch.no_grad():
            expected = model(data)[0]
            actual = inference(data['grasp_preds_features'])[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_nested_data_sizes_are_not_silently_capped_at100(self):
        ids = np.array([f't{i}' for i in range(1100)])
        for count in (100, 300, 1000):
            selected = port.select_training_records(ids, sample_cap=count)
            np.testing.assert_array_equal(selected, ids[:count])
        with self.assertRaises(ValueError): port.select_training_records(['a', 'a'], sample_cap=100)
        with self.assertRaises(ValueError): port.select_training_records(ids, sample_cap=0)


if __name__ == '__main__':
    unittest.main()
