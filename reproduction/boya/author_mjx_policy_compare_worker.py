"""Fixed-head top10 reranking plus the original measured-start planner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
import author_collection_batch as B
import author_training as T


def ranking_order(scores):
    values=np.asarray(scores,dtype=np.float32)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Finite one-dimensional scores required')
    return np.argsort(-values,kind='stable')


def semantic_candidate(candidate):
    keys=('source_row_index','source_top500_rank','collision_rank','gesture_id','score','source_if_flip',
        'actual_row15_camera','canonical_cgr480','raw_depth_index','extra_depth_index','slot_id','world_T_palm')
    return {k:candidate[k] for k in keys}


class Ranker:
    def __init__(self,binding):
        B.verify({'head':binding});self.binding=binding;torch.set_num_threads(1)
        _,cls,_,_=T.components();self.model=cls().eval()
        payload=torch.load(binding['path'],map_location='cpu',weights_only=True)
        if payload.get('gesture_id')!=1 or not payload.get('limited_data_pilot'):
            raise ValueError('Frozen g1 pilot checkpoint required')
        self.model.load_state_dict(payload['model_state_dict'],strict=True)

    def scores(self,candidates):
        if not candidates:return np.empty(0,np.float32)
        features=np.asarray([c['canonical_cgr480'] for c in candidates],np.float32)
        targets=np.asarray([c['raw_depth_index']*4+c['extra_depth_index'] for c in candidates],np.int64)
        if features.shape!=(len(candidates),480) or not np.isfinite(features).all() or not ((targets>=0)&(targets<20)).all():
            raise ValueError('Canonical480 and original executed-depth indices required')
        with torch.no_grad():output=self.model(torch.from_numpy(features))[0].numpy()
        if output.shape!=(len(candidates),1,20) or not np.isfinite(output).all():raise ValueError('Invalid published head output')
        return output[np.arange(len(candidates)),0,targets].copy()

    @contextmanager
    def installed(self,request):
        import author_collect as C
        original=C.prepare_collection;policy=request['comparison_policy'];directory=Path(request['output_dir'])
        if policy not in ('cgr','head'):raise ValueError('Explicit fixed policy required')
        def prepare(*args,**kwargs):
            result,arrays=original(*args,**kwargs)
            B.save(directory/'ORIGINAL_POOL.json',result,exclusive=True)
            candidates=result['candidates'];scores=self.scores(candidates);head_order=ranking_order(scores)
            order=np.arange(len(candidates)) if policy=='cgr' else head_order
            receipt=dict(policy=policy,checkpoint=self.binding,
                candidate_pool='Original CGR descending top10 after unchanged collision screening',
                head_combination='Head-only reranking of that top10; stable original-order ties',
                original_candidate_sha256=[c['candidate_sha256']for c in candidates],
                semantic_original_pool=[semantic_candidate(c)for c in candidates],
                selected_output_scores=scores.tolist(),head_order=head_order.tolist(),applied_order=order.tolist(),
                selected_original_rank=None if not len(order) else int(order[0]),
                head_used_for_selection=policy=='head',baseline_head_scores_are_shadow_only=policy=='cgr',
                selected_candidate_sha256=None if not len(order) else candidates[int(order[0])]['candidate_sha256'],
                physics_steps=0,training_updates=0)
            B.save(directory/'RANKING_RECEIPT.json',receipt,exclusive=True)
            if policy=='head':
                result=copy.deepcopy(result);result['candidates']=[result['candidates'][int(i)] for i in order]
                result['selected_candidate_sha256']=receipt['selected_candidate_sha256']
                result['selection']='Frozen pilot head descending score within original collision-filtered CGR top10; original order ties'
                result['learned_head_forward_count']=1 if candidates else 0
            return result,arrays
        with patch.object(C,'prepare_collection',prepare):yield


def run(requests,output):
    import author_resident_prepare as P
    import author_mjx_measured_planning as M
    if not requests:raise ValueError('Explicit requests required')
    binding=requests[0]['comparison_head'];ranker=Ranker(binding);original=P.prepare
    for request in requests:
        if request['comparison_head']!=binding:raise ValueError('No per-scene checkpoint changes')
    def prepare(request,session=None):
        with ranker.installed(request):return original(request,session)
    with patch.object(P,'prepare',prepare):M.run(requests,output)


def planning_batch(root,requests,p,pool):
    from author_collection_wave_resources import process_identity
    byindex={r['index']:r for r in p['rows']}
    requests=[(i,dict(request,comparison_policy=byindex[i]['comparison_policy'],comparison_head=p['inputs']['comparison_head']))for i,request in requests]
    groups=[requests[i::3]for i in range(3)];groups=[g for g in groups if g]
    def job(pair):
        number,items=pair;directory=root/f'worker_{number}';directory.mkdir(parents=True,exist_ok=False)
        request=directory/'REQUESTS.json';B.save(request,[r for _,r in items],exclusive=True)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--requests',str(request),'--output',str(directory/'RESULTS.json')]
        B.save(directory/'DISPATCH.json',dict(command=command,requests=B.binding(request)),exclusive=True)
        with pool.stage(('prepare',),directory):
            with (directory/'process.log').open('x')as log:
                process=pool.popen(command,cwd=B.ROOT,env=B._environment(p),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(process.pid);B.save(directory/'PROCESS.json',dict(identity=identity),exclusive=True)
                try:code=process.wait(timeout=p['stage_timeout_s'])
                except BaseException:
                    if process_identity(process.pid)==identity:
                        __import__('os').killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:__import__('os').killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
                    raise
                if code:raise RuntimeError('Comparison planning failed: '+str(directory))
        result=B.read(directory/'RESULTS.json')
        return [(index,row)for (index,_),row in zip(items,result['jobs'])]
    result={}
    with ThreadPoolExecutor(max_workers=3)as executor:
        for values in executor.map(job,enumerate(groups)):result.update(values)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(B.read(args.requests),args.output)
