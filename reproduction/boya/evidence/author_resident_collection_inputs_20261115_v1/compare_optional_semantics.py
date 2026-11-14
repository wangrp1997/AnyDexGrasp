"""Read-only extension for controls that include an explicit no-candidate slot."""
from pathlib import Path
import argparse
import json

from compare_semantics import B, FIELDS, METADATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    snapshot = B.read(args.comparison)
    root = Path(snapshot['protocol']['path']).parent
    reference = Path(snapshot['reference_protocol']['path']).parent
    rows = []
    for scene in snapshot['scenes']:
        name = f'{scene["source_queue_index"]:07d}_{scene["scene_id"]}'
        for gesture, record in scene['gestures'].items():
            assert record['status'] == record['reference_status']
            current = root/'scenes'/name/f'g{gesture}/pool/selected_author_candidate.json'
            previous = reference/'scenes'/name/f'g{gesture}/pool/selected_author_candidate.json'
            row = dict(scene_id=scene['scene_id'], gesture_id=int(gesture), status=record['status'])
            if record['status'] == 'no_candidate':
                assert not current.exists() and not previous.exists()
                assert record['candidate_sha256'] is record['reference_candidate_sha256'] is None
                row.update(selected_candidate_exists=False, semantic_fields_exact=0)
            else:
                fresh, prior = B.read(current), B.read(previous)
                checks = {key: fresh[key] == prior[key] for key in FIELDS}
                assert all(checks.values()), checks
                differences = {key: [name for name, value in content['arrays'].items() if not value['exact']]
                    for key, content in record.items() if isinstance(content, dict) and 'arrays' in content}
                assert all(set(values) <= set(METADATA) for values in differences.values()), differences
                row.update(selected_candidate_exists=True, semantic_fields_exact=len(FIELDS),
                    semantic_checks=checks, nonexact_metadata_fields=differences,
                    current_candidate=B.binding(current), reference_candidate=B.binding(previous))
            rows.append(row)
    result = dict(status='pass', source=B.binding(__file__), semantic_definitions=B.binding(Path(__file__).with_name('compare_semantics.py')),
        comparison=B.binding(args.comparison), rows=rows, new_model_forwards=0, new_planning_calls=0, new_physics_steps=0)
    B.save(args.output, result, exclusive=True)
    print(json.dumps(dict(status='pass', slots=len(rows), selected_candidates=sum(r['selected_candidate_exists'] for r in rows))))


if __name__ == '__main__':
    main()
