"""Wave scheduling and real POSIX lifecycle tests; no Kit/CGR/robot planning."""
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import author_collection_batch as B
import author_collection_waves as W
import author_collection_wave_resources as R
from test_author_collection_batch import fixture, FakeStages, fake_trial


def wave_fixture(root, splits=('fit','fit','fit','fit','fit','fit','internal')):
    _, original = fixture(root, splits)
    q = root/'QUEUE.json'; dummy = root/'dummy.json'
    out = root/'waves'
    W.freeze(q, out, calibration=dummy, cloud_manifest=dummy, backend=dummy,
        checkpoint=dummy, scene_entry=Path(__file__), capture_entry=Path(__file__),
        execution_entry=Path(__file__), audit_entry=Path(__file__),
        fit_per_gesture=1, internal_per_gesture=1)
    return out


@contextmanager
def no_lock(path): yield


class WaveTests(unittest.TestCase):
    def fake_spawn(self, root, index, launched):
        def spawn(command, **kw):
            i = int(command[command.index('--child-index')+1]); child = index['children'][i]
            directory = Path(child['protocol']['path']).parent.parent; launched.append(i)
            with patch.object(B,'_stage',side_effect=FakeStages()), patch.object(B,'_lock',no_lock):
                B.run(directory/'batch')
            B.save(directory/'WORKER_RESULT.json',W._child_result(directory,child,'complete'),exclusive=True)
            class Done:
                pid=900000+i; returncode=0
                def wait(self): return 0
            return Done()
        return spawn

    def test_freeze_has_acyclic_bindings_original_seed_and_no_legacy_parent(self):
        with tempfile.TemporaryDirectory() as td, patch.object(W.subprocess,'Popen') as spawn:
            root = wave_fixture(Path(td)); spawn.assert_not_called()
            protocol, plan, index = W.verify_protocol(root)
            self.assertEqual(protocol['resource_limits'],dict(observation=1,kit=3,prepare=3,audit=2))
            self.assertFalse((root/'BATCH_PROTOCOL.json').exists())
            for i, child in enumerate(index['children']):
                q=B.read(child['queue']['path'])
                self.assertEqual(q['seed'],123);self.assertEqual(q['parent_queue_index'],i)
                self.assertEqual(q['scenes'],[plan['queue']['scenes'][i]])
                self.assertEqual(q['inputs']['wave_plan'],protocol['plan'])
            with self.assertRaises(FileNotFoundError):B.run(root)

    def test_quota_admits_first_wave_prefix_keeps_excess_cost_and_skips_later_wave(self):
        with tempfile.TemporaryDirectory() as td:
            root=wave_fixture(Path(td)); protocol,plan,index=W.verify_protocol(root)
            launched=[]
            def spawn(command,**kw):
                i=int(command[command.index('--child-index')+1]);child=index['children'][i]
                directory=Path(child['protocol']['path']).parent.parent
                launched.append(i)
                with patch.object(B,'_stage',side_effect=FakeStages()),patch.object(B,'_lock',no_lock):
                    B.run(directory/'batch')
                B.save(directory/'WORKER_RESULT.json',W._child_result(directory,child,'complete'),exclusive=True)
                class Done:
                    pid=900000+i;returncode=0
                    def wait(self):return 0
                return Done()
            with patch.object(W,'LEGACY_LOCK',Path(td)/'gpu.lock'),patch.object(W.subprocess,'Popen',side_effect=spawn),patch.object(B,'validate_trial',side_effect=fake_trial):
                result=W.run(root)
                W.run(root)
            rows=W.ledger(root)
            self.assertEqual(launched,[0,1,2,6]);self.assertTrue(result['quota_met'])
            self.assertEqual(result['known_final_execution_physics_steps'],16*23)
            self.assertEqual(result['quota_excess_labels'],8)
            self.assertEqual([r['queue_index'] for r in rows if r['label']is not None],[0]*4+[6]*4)
            self.assertTrue(all(r['status']=='quota_reached' and not r['execution_steps_unknown'] for r in rows if r['queue_index']in(3,4,5)))
            self.assertTrue(all(r['raw_label']==0 and r['actual_physics_steps']==23 for r in rows if r['status']=='quota_excess_not_admitted'))

    def test_unknown_started_worker_is_never_relaunched(self):
        with tempfile.TemporaryDirectory() as td:
            root=wave_fixture(Path(td),('fit',));_,_,index=W.verify_protocol(root)
            directory=Path(index['children'][0]['protocol']['path']).parent.parent
            B.save(directory/'CHILD_STARTED.json',dict(state='unknown'))
            with patch.object(W,'LEGACY_LOCK',Path(td)/'gpu.lock'),patch.object(W.subprocess,'Popen') as spawn:
                with self.assertRaises(R.Quarantined):W.run(root)
                spawn.assert_not_called()

    def test_actual_live_worker_is_never_restarted_even_with_result(self):
        with tempfile.TemporaryDirectory() as td:
            root=wave_fixture(Path(td),('fit',));_,_,index=W.verify_protocol(root);child=index['children'][0]
            directory=Path(child['protocol']['path']).parent.parent
            B.save(directory/'CHILD_PROCESS.json',dict(process=R.process_identity(os.getpid())))
            B.save(directory/'WORKER_RESULT.json',W._child_result(directory,child,'complete'))
            with self.assertRaises(R.Quarantined):W._reconcile(root,child)

    def test_selected_worker_verification_does_not_hash_other_child_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=wave_fixture(Path(td),('fit','internal'));calls=[];original=B.verify
            def observe(bindings):
                calls.extend(e['path']for e in bindings.values());return original(bindings)
            with patch.object(B,'verify',side_effect=observe):W.verify_protocol(root,child_index=0)
            self.assertEqual(calls.count(str(Path(td)/'dummy.json')),1)

    def test_parent_partial_admission_resumes_without_restarting_children(self):
        with tempfile.TemporaryDirectory() as td:
            root=wave_fixture(Path(td),('fit','fit','fit')); _,_,index=W.verify_protocol(root)
            launched=[]; original=B.save; interrupted=[False]
            def save(path,value,**kw):
                original(path,value,**kw)
                if Path(path)==root/'ledger/0000000_g1.json' and not interrupted[0]:
                    interrupted[0]=True; raise RuntimeError('parent stopped after durable g1 admission')
            with patch.object(W,'LEGACY_LOCK',Path(td)/'gpu.lock'), patch.object(W.subprocess,'Popen',side_effect=self.fake_spawn(root,index,launched)), patch.object(B,'validate_trial',side_effect=fake_trial):
                with patch.object(B,'save',side_effect=save):
                    with self.assertRaises(RuntimeError):W.run(root)
                result=W.run(root)
            self.assertEqual(launched,[0,1,2]); rows=W.ledger(root)
            self.assertEqual([r['gesture_id']for r in rows[:4]],list(B.GESTURES))
            self.assertEqual(sum(r['label']is not None for r in rows),4)
            self.assertEqual(result['known_final_execution_physics_steps'],12*23)

    def test_dead_partial_worker_keeps_one_label_and_missing_outcomes_null(self):
        with tempfile.TemporaryDirectory() as td:
            root=wave_fixture(Path(td),('fit',)); protocol,_,index=W.verify_protocol(root);child=index['children'][0]
            directory=Path(child['protocol']['path']).parent.parent
            with patch.object(B,'_stage',side_effect=FakeStages()),patch.object(B,'validate_trial',side_effect=fake_trial),patch.object(B,'_lock',no_lock):B.run(directory/'batch')
            for g in (6,7,8):(directory/f'batch/ledger/0000000_g{g}.json').unlink()
            (directory/'batch/LEDGER.json').unlink()
            B.save(directory/'CHILD_STARTED.json',dict(state='started_no_retry'))
            B.save(directory/'CHILD_PROCESS.json',dict(process=dict(pid=900001,pgid=900001,start_ticks=1)))
            with patch.object(W,'LEGACY_LOCK',Path(td)/'gpu.lock'),patch.object(W.subprocess,'Popen')as spawn,patch.object(B,'validate_trial',side_effect=fake_trial):
                result=W.run(root);spawn.assert_not_called()
            rows=W.ledger(root,protocol)
            self.assertEqual([r['label']for r in rows],[0,None,None,None])
            self.assertEqual(result['known_final_execution_physics_steps'],4*23)
            self.assertEqual(result['execution_starts_with_unknown_steps'],0)


class ResourceTests(unittest.TestCase):
    def pool(self,root,fd,**kw):return R.ResourcePool(root/'slots',fd,dict(path='protocol',sha256='a'*64),**kw)

    def test_unlocked_same_inode_is_rejected_and_real_inherited_lease_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lease';path.touch();fd=os.open(path,os.O_RDWR)
            try:
                st=os.fstat(fd)
                with self.assertRaises(ValueError):R.check_inherited_lease(fd,path,[st.st_dev,st.st_ino])
            finally:os.close(fd)
            with R.lease(path) as fd:
                st=os.fstat(fd);R.check_inherited_lease(fd,path,[st.st_dev,st.st_ino])

    def test_close_only_lease_remains_locked_by_actual_child(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lease'
            with R.lease(path) as fd:
                child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.3)'],pass_fds=(fd,),start_new_session=True)
            try:
                with self.assertRaises(R.Quarantined):
                    with R.lease(path):pass
            finally:child.wait(timeout=5)
            with R.lease(path):pass

    def test_no_resource_build_with_live_child_group_quarantines(self):
        with tempfile.TemporaryDirectory() as td,R.lease(Path(td)/'lease') as fd:
            pool=self.pool(Path(td),fd);child=None
            try:
                with self.assertRaises(R.Quarantined):
                    with pool.stage((),Path(td)/'build'):
                        child=pool.popen([sys.executable,'-c','import time; time.sleep(8)'],start_new_session=True)
                self.assertTrue(any((Path(td)/'slots/quarantine').glob('*.json')))
            finally:
                if child:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

    def test_normal_stage_releases_only_after_actual_process_group_exit(self):
        with tempfile.TemporaryDirectory() as td,R.lease(Path(td)/'lease') as fd:
            pool=self.pool(Path(td),fd)
            with pool.stage(('prepare',),Path(td)/'prepare'):
                child=pool.popen([sys.executable,'-c','pass'],start_new_session=True);child.wait(timeout=5)
            state=B.read(Path(td)/'slots/prepare/0.json')
            self.assertEqual(state['state'],'released')

    def test_reservation_before_popen_failure_is_durable_and_not_reusable(self):
        with tempfile.TemporaryDirectory() as td,R.lease(Path(td)/'lease') as fd:
            def fail(point):
                if point=='reservation_before_popen':raise RuntimeError('fault window1')
            pool=self.pool(Path(td),fd,fault=fail)
            with self.assertRaises(R.Quarantined):
                with pool.stage(('prepare',),Path(td)/'prepare'):pass
            self.assertEqual(B.read(Path(td)/'slots/prepare/0.json')['state'],'quarantined')
            with self.assertRaises(R.Quarantined):
                with pool.stage(('prepare',),Path(td)/'again'):pass

    def test_spawn_before_pid_persist_failure_retains_child_lock_and_quarantine(self):
        with tempfile.TemporaryDirectory() as td,R.lease(Path(td)/'lease') as fd:
            def fail(point):
                if point=='popen_before_pid_persist':raise RuntimeError('fault window2')
            pool=self.pool(Path(td),fd,fault=fail);child=None
            try:
                with self.assertRaises(R.Quarantined):
                    with pool.stage(('kit',),Path(td)/'execute') as ctx:
                        try:pool.popen([sys.executable,'-c','import time; time.sleep(8)'],start_new_session=True)
                        finally:child=ctx['process']
                slot=os.open(Path(td)/'slots/kit/0.lock',os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
                finally:os.close(slot)
                self.assertEqual(B.read(Path(td)/'slots/kit/0.json')['state'],'quarantined')
            finally:
                if child:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

    def test_two_prepare_threads_inherit_only_their_own_slot_fds(self):
        with tempfile.TemporaryDirectory() as td,R.lease(Path(td)/'lease') as fd:
            pool=self.pool(Path(td),fd);barrier=threading.Barrier(2);seen={};errors=[]
            class Finished:
                pid=900001
            def spawn(*args,**kw):
                name=threading.current_thread().name;seen[name]=set(kw['pass_fds']);barrier.wait(timeout=5);return Finished()
            def work():
                try:
                    name=threading.current_thread().name
                    with pool.stage(('prepare',),Path(td)/name):pool.popen(['synthetic'],start_new_session=True)
                except BaseException as e:errors.append(e)
            with patch.object(R.subprocess,'Popen',side_effect=spawn),patch.object(R,'group_alive',return_value=False):
                ts=[threading.Thread(target=work,name=f't{i}')for i in range(2)]
                for t in ts:t.start()
                for t in ts:t.join(timeout=8)
            self.assertFalse(errors,errors);self.assertEqual(len(seen),2)
            self.assertEqual(seen['t0']&seen['t1'],{fd})
            self.assertEqual([len(v)for v in seen.values()],[2,2])

    def test_observation_only_capture_cgr_and_execution_does_not_take_observation(self):
        protocol=dict(sources={k:dict(path=str(B.HERE/(k+'.py')))for k in('capture_entry','execution_entry','audit_entry','scene_entry')})
        self.assertEqual(R.classify(['x',str(B.HERE/'capture_entry.py')],protocol),('observation','kit'))
        self.assertEqual(R.classify(['x',str(B.HERE/'execution_entry.py')],protocol),('kit',))
        self.assertEqual(R.classify(['x',str(B.HERE/'run_collection_representation.py')],protocol),('observation',))
        protocol['sources']['execution_entry']['path']=str(B.HERE/'author_lift_pipeline.py')
        self.assertEqual(R.classify(['x',str(B.HERE/'author_lift_pipeline.py'),'prepare'],protocol),('prepare',))

    def test_new_run_cannot_bypass_prior_orphan_without_inherited_fds(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); old=root/'old';new=root/'new';old.mkdir();new.mkdir()
            protocol=root/'protocol.json';B.save(protocol,{})
            child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(8)'],start_new_session=True)
            try:
                # This child has no lease FD: only the durable global registry protects it.
                B.save(old/'resource_slots/prepare/0.json',dict(state='running',pgid=child.pid))
                B.save(root/'active.json',dict(state='active',run_dir=str(old),protocol=B.binding(protocol)))
                with R.lease(root/'lease'):
                    with self.assertRaises(R.Quarantined):
                        with R.active_run(new,root/'active.json',B.binding(protocol)):pass
            finally:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

    def test_resource_slots_enforce_each_global_limit_under_thread_contention(self):
        with tempfile.TemporaryDirectory() as td,R.lease(Path(td)/'lease') as fd:
            pool=self.pool(Path(td),fd);mutex=threading.Lock();counts={k:0 for k in R.LIMITS};maximum=dict(counts);errors=[]
            class Finished:pid=900001
            def work(kind,index):
                try:
                    with pool.stage((kind,),Path(td)/f'{kind}{index}'):
                        pool.popen(['synthetic'],start_new_session=True)
                        with mutex:counts[kind]+=1;maximum[kind]=max(maximum[kind],counts[kind])
                        time.sleep(.05)
                        with mutex:counts[kind]-=1
                except BaseException as error:errors.append(error)
            with patch.object(R.subprocess,'Popen',return_value=Finished()),patch.object(R,'group_alive',return_value=False):
                for kind,limit in R.LIMITS.items():
                    ts=[threading.Thread(target=work,args=(kind,i))for i in range(limit+2)]
                    for t in ts:t.start()
                    for t in ts:t.join(timeout=8)
            self.assertFalse(errors,errors);self.assertEqual(maximum,R.LIMITS)


if __name__=='__main__':unittest.main()
