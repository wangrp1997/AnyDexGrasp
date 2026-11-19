"""Repeat only the unexecuted wave after the original pre-launch import error."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import author_collection_batch as B
import author_mjx_batch_collection_v9 as F
import author_mjx_cgr_fallback as fallback
from run_author_mjx_cgr_fallback import summarize
from run_author_mjx_expansion import dispatch


def freeze(root, failed):
    c = failed/'collection'
    source_protocol = c/'PROTOCOL.json'
    p = B.read(source_protocol)
    previous = B.read(c/'RESULTS.json')
    review = B.read(failed/'NO_SPAWN_REVIEW.json')
    B.verify(p['sources']); B.verify(p['inputs'])
    if (previous['status'] != 'failed' or previous['cost']['grasp_attempts'] != 0
            or "name 'subprocess' is not defined" not in previous['error']
            or review['status'] != 'prelaunch_failure_released'
            or review['result'] != B.binding(c/'RESULTS.json')):
        raise ValueError('Fresh recovery requires the verified zero-launch failure')
    root.mkdir(exist_ok=False)
    new = root/'collection'
    archive = new/'source_archive'
    archive.mkdir(parents=True)
    p['sources'][Path(__file__).name] = B.binding(__file__)
    for name, source in p['sources'].items():
        shutil.copyfile(source['path'], archive/name)
    p['inputs'].update(prior_protocol=B.binding(source_protocol), prior_result=B.binding(c/'RESULTS.json'),
                       prior_no_spawn_review=B.binding(failed/'NO_SPAWN_REVIEW.json'))
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT'] = str(new/'contact_spool')
    p['prior_native_environment_steps'] = previous['world']['native_environment_integrations']
    p['identical_layouts_after_dispatch_failure'] = True
    B.save(new/'PROTOCOL.json', p, exclusive=True)
    shutil.copytree(c/'warp_cache', new/'warp_cache')
    B.save(root/'READY.json', dict(status='frozen', protocol=B.binding(new/'PROTOCOL.json'),
        draws=len(p['rows']), ready=sum(bool(row['scene']) for row in p['rows']),
        prior_native_environment_steps=p['prior_native_environment_steps'], prior_grasps=0,
        max_grasps=p['max_grasp_attempts'], max_native_environment_steps=p['max_native_environment_steps']), exclusive=True)


def collect(root):
    p = B.read(root/'PROTOCOL.json')
    marker = B.sha(root/'PROTOCOL.json')
    if os.environ.get('ANYDEX_MJX_CGR_FALLBACK_RESUME_ENV') != marker:
        env = B._environment(p)
        env.update(PYTHONPATH=os.pathsep.join((str(F.WARP), str(F.DEPS), env['PYTHONPATH'])),
                   CUDA_VISIBLE_DEVICES=F.GPU, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   WARP_CACHE_PATH=str(root/'warp_cache'), ANYDEX_MJX_CGR_FALLBACK_RESUME_ENV=marker)
        os.execve(sys.executable, [sys.executable, '-B', str(Path(__file__).resolve()), 'collect', str(root)], env)
    from author_mjx_policy_compare_collection import run

    # Supply the missing dependency in this fresh controller only. The failed
    # run and the previously verified CPU planner retain their original source.
    with patch.object(fallback, 'subprocess', subprocess, create=True), patch.object(F, 'planning_batch', fallback.planning_batch):
        if run(root)['status'] != 'complete':
            raise SystemExit(1)


def run(root):
    ready = B.read(root/'READY.json')
    B.verify({'protocol':ready['protocol']})
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=B.ROOT).strip():
        raise ValueError('Commit the recovery entry before dispatch')
    B.save(root/'PIPELINE_STARTED.json', dict(timestamp=time.time(), git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=B.ROOT, text=True).strip()), exclusive=True)
    try:
        finish = dispatch([sys.executable, '-B', str(Path(__file__).resolve()), 'collect', str(root/'collection')], root/'timing', 3600)
        summarize(root)
        r = B.read(root/'BASELINE_RESULTS.json')
        B.save(root/'PIPELINE_FINISHED.json', dict(stage=finish, result=B.binding(root/'BASELINE_RESULTS.json'),
            prior_native_environment_steps=ready['prior_native_environment_steps'],
            total_native_environment_steps=ready['prior_native_environment_steps']+r['native_environment_steps']), exclusive=True)
    except BaseException:
        B.save(root/'PIPELINE_FAILURE.json', dict(error=traceback.format_exc()), exclusive=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for action in ('freeze', 'collect', 'run'):
        sub = commands.add_parser(action)
        sub.add_argument('root', type=Path)
        if action == 'freeze':
            sub.add_argument('failed', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.root.resolve(), args.failed.resolve())
    elif args.action == 'run':
        run(args.root.resolve())
    else:
        collect(args.root.resolve())
