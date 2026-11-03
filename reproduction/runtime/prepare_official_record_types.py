"""Apply the author's published hand-type migration to a derived JSON copy.

The public download is never edited. The original update_grasp_type function
executes unchanged except removal of imports of grasp_types, a name never read
by that function. This avoids importing robot/hardware packages offline.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def feature_sha(info):
    return hashlib.sha256(json.dumps(info['grasp_preds_features'],separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hand',choices=['allegro','inspire','dh3'],required=True)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():
        raise ValueError('Public original input must never be edited')
    if args.output.exists():
        raise FileExistsError('Derived output already exists; refusing overwrite')
    source=ROOT/'update_data.py'
    tree=ast.parse(source.read_text())
    func=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='update_grasp_type')
    assert not any(isinstance(node,ast.Name) and node.id=='grasp_types' and isinstance(node.ctx,ast.Load)
                   for node in ast.walk(func)), 'Imported name is used; cannot isolate safely'
    removed=[]
    class RemoveUnusedImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            assert all(alias.name=='grasp_types' for alias in node.names)
            removed.append(ast.unparse(node))
            return None
    func=RemoveUnusedImports().visit(func)
    ast.fix_missing_locations(func)
    scope={'json':json,'__name__':'official_record_type_overlay'}
    exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),'exec'),scope)
    original_sha=sha(args.input)
    before=json.loads(args.input.read_text())
    prefix={'allegro':'Allegro','inspire':'InspiredHandR','dh3':'DH3'}[args.hand]
    fingerprint={key:{'feature_sha256':feature_sha(info),'result':info['result']} for key,info in before.items()}
    before_types=dict(collections.Counter(str(info[prefix+'_pose_finger_type']) for info in before.values()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(args.input,args.output)
    scope['update_grasp_type'](str(args.output),args.hand)
    after=json.loads(args.output.read_text())
    assert sha(args.input)==original_sha
    assert all(feature_sha(info)==fingerprint[key]['feature_sha256'] and info['result']==fingerprint[key]['result']
               for key,info in after.items())
    report={
        'method':'original author update_data.py:update_grasp_type AST-isolated offline preprocessing',
        'official_source_revision':'c9c4a43df33e40860417c7e2dd02f5122d3b2da2',
        'function_source':str(source),'function_source_sha256':sha(source),
        'only_ast_change':'Removed unused grasp_types imports; mapping and mutation logic unchanged',
        'removed_unused_imports':removed,
        'original_input':str(args.input.resolve()),'original_input_sha256':original_sha,
        'original_input_unchanged':True,
        'derived_output':str(args.output.resolve()),'derived_output_sha256':sha(args.output),
        'hand':args.hand,'records_before':len(before),'records_after':len(after),
        'finger_types_before':before_types,
        'finger_types_after':dict(collections.Counter(str(info[prefix+'_pose_finger_type']) for info in after.values())),
        'discarded_original_ids':sorted(set(before)-set(after)),
        'all_retained_features_and_results_unchanged':True,
        'no_new_training':True,
    }
    args.output.with_name(args.output.name+'.preprocessing_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
