#!/usr/bin/env python3
"""Publish an honest, non-destructive AnyDexGrasp intake inventory."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests

COMMIT = 'c9c4a43df33e40860417c7e2dd02f5122d3b2da2'
DRIVE = 'https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing to overwrite changed existing file: ' + str(path))
    else:
        path.write_bytes(data)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.urls.append(href)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--public-root', type=Path, default=Path('/mnt/public/datasets/AnyDexGrasp'))
    p.add_argument('--evidence-dir', type=Path, default=Path(__file__).resolve().parent/'evidence')
    args = p.parse_args()
    REPO, DEST, EVIDENCE = args.source_repo, args.public_root, args.evidence_dir
    existed = DEST.exists()
    usage = shutil.disk_usage(DEST.parent)
    for folder in ['metadata', 'source_tables', 'downloads', 'logs/model',
                   'logs/data/decision_model',
                   'logs/data/representation_model/graspnet_v1_newformat']:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    tables = []
    for relative in [
        'generate_mesh_and_pointcloud/allegro_urdf/width_16D_angle.json',
        'generate_mesh_and_pointcloud/dh3_urdf/width_12D_angle_2D_angle.json',
        'generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json',
    ]:
        data = (REPO / relative).read_bytes()
        value = json.loads(data)
        target = DEST / 'source_tables' / relative
        write_new(target, data)
        assert digest(target.read_bytes()) == digest(data)
        tables.append(dict(path=str(target), relative_source=relative, bytes=len(data),
                           sha256=digest(data), top_level_entries=len(value),
                           upstream_url=f'https://github.com/graspnet/AnyDexGrasp/blob/{COMMIT}/{relative}',
                           role='hand_mapping_lookup_table_not_training_data'))
    pages = []
    for name, url in [('project', 'https://www.graspnet.net/anydexgrasp/'),
                      ('graspnet_datasets', 'https://graspnet.net/datasets.html')]:
        try:
            r = requests.get(url, timeout=(10, 20))
            r.raise_for_status()
            write_new(DEST / 'metadata' / (name + '.html'), r.content)
            parser = Links()
            parser.feed(r.text)
            pages.append(dict(url=url, status=r.status_code, bytes=len(r.content),
                              sha256=digest(r.content),
                              links=[urljoin(r.url, x) for x in parser.urls]))
        except Exception as e:
            pages.append(dict(url=url, error=str(e)))
    existing = Path('/mnt/public/datasets/DexGraspNet2.0/extracted/scenes/scene_0000/realsense')
    reuse = dict(path=str(existing), exists=existing.exists(),
                 dataset='DexGraspNet 2.0 synthetic rendered scenes',
                 is_official_graspnet_rgbd=False,
                 reused_as_anydex_training_data=False,
                 reason='Different dataset/provenance; RGB and label paths required by the AnyDex loader are missing')
    for sub in ['depth/0000.png', 'camK.npy', 'rgb/0000.png', 'label/0000.png', 'meta/0000.mat']:
        reuse[sub] = (existing / sub).exists()
    receipt = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        upstream_commit=COMMIT, public_root=str(DEST), root_existed_before=existed,
        disk_free_bytes_before=usage.free, official_drive_folder=DRIVE,
        status='blocked_official_drive_network_access',
        official_drive_index_obtained=False,
        known_official_file_ids=[],
        representation_checkpoint={'downloaded':False,'bytes':0},
        hand_specific_checkpoints={'downloaded':False,'bytes':0},
        decision_training_data={'downloaded':False,'bytes':0},
        reannotated_representation_labels={'downloaded':False,'bytes':0},
        official_graspnet_rgbd={'downloaded':False,'bytes':0},
        official_minimal_input_sample={'downloaded':False,'bytes':0},
        source_tables=tables, official_pages=pages,
        local_related_dataset= reuse,
        training_or_inference_performed=False,
        installed_packages=False,
        limitations=[
            'Drive DNS returned anomalous 199.59.150.40; IPv4 folder requests timed out.',
            'HTTPS Google DNS unreachable; Cloudflare DNS reset; AliDNS returned anomalous 108.160.*.',
            'Four request-local Google IP alternatives also timed out with normal TLS verification.',
            'No file IDs or byte sizes could be read from the unavailable Drive folder.',
            'Downloaded project metadata and source tables are not model weights or a training dataset.',
            'No huge GraspNet RGBD download was started; official page lists train images totaling about 66.3 GB.'
        ])
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2) + '\n').encode()
    write_new(DEST / 'metadata' / 'intake_20261103.json', payload)
    write_new(EVIDENCE / 'intake_20261103.json', payload)
    readme = '''# AnyDexGrasp public data intake (2026-11-03)

Official upstream: https://github.com/graspnet/AnyDexGrasp
Pinned commit: c9c4a43df33e40860417c7e2dd02f5122d3b2da2
Official weights/data folder: https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo

STATUS: Official Drive cannot currently be reached from this server. No model
weights, decision training data, reannotated labels, or official RGB-D examples
have been downloaded. Empty logs/ and downloads/ folders are destination layout,
not evidence of data availability. See metadata/intake_20261103.json.

source_tables/ contains only the three hand mapping JSON files shipped in the
official source commit. Their byte counts and SHA-256 hashes have been checked.
They are lookup tables, not collected successful/failed grasp training data.

metadata/ holds the accessible official project/data pages and intake receipt.
The existing public DexGraspNet2.0 directory is a different synthetic dataset;
it has not been substituted for the missing GraspNet-1Billion RGB-D corpus.

The reproduction repository /home/wangrenpeng/AnyDexGrasp/reproduction/ holds
the resumable public-file download utility. It requires a real artifact URL or
FILE ID from the official folder and rejects HTML. No guessed file IDs or models
from other methods are presented as AnyDexGrasp artifacts.
'''
    write_new(DEST / 'README.md', readme.encode())
    print(json.dumps({'public_root': str(DEST), 'source_tables':len(tables),
                      'source_table_bytes':sum(x['bytes'] for x in tables),
                      'official_training_data_downloaded':False,
                      'official_checkpoint_downloaded':False}, indent=2))


if __name__ == '__main__':
    main()
