"""Saved-certificate and export audit of the original15-scene tail; zero physics."""
import argparse
import ast
from pathlib import Path
import time

import author_production_tail as D
import author_contact_compressor as W

B=D.B
source=B.HERE/'audit_production_pilot.py'
if B.sha(source)!='132f9491b91551484d5f73ff02395708d69b99419c3f9f197ac726ba243a566c':
    raise ValueError('Reviewed independent production auditor changed')
tree=ast.parse(source.read_text());edits={32:0,128:0}


class Adapt(ast.NodeTransformer):
    def visit_Import(self,node):
        for alias in node.names:
            if alias.name=='author_production_collection':alias.name='author_production_tail'
        return node

    def visit_Constant(self,node):
        if type(node.value)is int and node.value in edits:
            edits[node.value]+=1
            return ast.copy_location(ast.Constant({32:15,128:60}[node.value]),node)
        return node


tree=Adapt().visit(tree)
if edits!={32:1,128:2}:raise ValueError('Independent tail-auditor budget sites changed')
namespace=dict(__name__='_anydex_saved_tail_audit',__file__=str(Path(__file__).resolve()))
exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),namespace)


def audit(root,observations,output):
    began=time.monotonic();root=Path(root).resolve();output=Path(output).resolve()
    intermediate=output.with_name(output.stem+'_BASE.json')
    namespace['audit'](root,Path(observations),intermediate)
    value=B.read(intermediate);p=D.verify(root);rows=D.ledger(root,p);checked={};seals=[]
    def verify(entries):
        for b in B.normalized_bindings(entries).values():
            if b['path']in checked:
                if checked[b['path']]!=b['sha256']:raise ValueError('Conflicting saved file binding')
            else:
                if B.sha(b['path'])!=b['sha256']:raise ValueError('Saved audit input changed: '+b['path'])
                checked[b['path']]=b['sha256']
    for row in rows:
        if row['label']is None:continue
        cert=B.read(row['artifact_bindings']['runtime_audit']['path']);verify(cert['audit_inputs_sha256'])
        artifacts=D.verified_contact_artifacts(row['trial']);verify(artifacts)
        prepared=Path(row['trial']['record']['path']).parent
        state=B.read(prepared/'execution/MULTI_OBJECT_CONTACTS.jsonl.gz.STATE.json')
        raw=Path(state['raw_spool']);expected=state['expected']
        if W.sha(raw)!=expected['sha256']or raw.stat().st_size!=expected['bytes']:
            raise ValueError('Retained raw spool changed')
        seals.append(dict(trial_id=row['trial_id'],seal=artifacts['contact_integrity_seal'],
                          raw_spool=str(raw),logical=expected,cost=artifacts['contact_storage_cost']))
    previous=B.read(B.HERE/'evidence/author_production_pilot_20261115_v1/PREFIX_AUDIT.json')
    verify({'prior_export':previous['export']})
    for entry in B.read(previous['export']['path'])['datasets'].values():verify({'dataset':entry['binding']})
    value.update(base_audit=B.binding(intermediate),contact_seals=seals,checked_files=checked,
                 checked_file_count=len(checked),previous_six_export_unchanged=previous['export'],
                 original_scene_indices=B.read(root.parent/'TAIL_PLAN.json')['original_scene_indices'],
                 full_point_audits_repeated=0,extra_Kit_launches=0,extra_physics_steps=0,
                 source=B.binding(__file__),audit_wall_s=time.monotonic()-began)
    B.save(output,value,exclusive=True)
    print({k:value[k]for k in ('status','new_labels','counts','physics_steps','checked_file_count')})
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();audit(a.run_dir,a.observations,a.output)
