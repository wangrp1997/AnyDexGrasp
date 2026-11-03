# Official data intake, 2026-11-03

Public destination: `/mnt/public/datasets/AnyDexGrasp/`.

Actual available data: three official source lookup JSON files, **2,978,109 bytes**,
and fetched official project/GraspNet download pages. All lookup bytes have been
checked against pinned upstream commit
`c9c4a43df33e40860417c7e2dd02f5122d3b2da2` using SHA-256. These files describe hand
mapping and **are not model weights or grasp success/failure training data**.

The official Drive folder index cannot currently be reached. Model weights,
decision data, reannotated CGR labels, and official input examples are therefore
**not downloaded**, and their individual file IDs/sizes remain unknown. The public
manifest records this instead of treating the empty destination layout as data.
The server has approximately 11 TB free; capacity is not the blocker.

`intake_20261103.json` contains all source URLs, byte counts, hashes, the actual
public paths, and related-dataset checks. In particular, the local DexGraspNet 2.0
synthetic scenes are not substituted for GraspNet-1Billion RGB-D training data.

Network checks used normal certificate verification. Elevated direct requests,
explicit IPv4, public HTTPS DNS resolvers, and request-local alternative Google IPs
all failed for the Drive folder. No credentials, private proxies, or authentication
workarounds were used. README, project links, relevant official issues, and a
targeted official-domain search revealed no AnyDex-specific public mirror. The
generic GraspNet RGB-D page has Baidu/Jbox mirrors; these do not replace AnyDex's
reannotated labels or checkpoints. No 66+ GB raw training-image download was started.

## Resume after the official folder becomes reachable

First inspect the official folder and record real **file** IDs, names and sizes:

<https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo>

Download one explicitly selected artifact into the public folder:

```bash
python reproduction/data_intake/fetch_public_file.py \
  --file-id ACTUAL_FILE_ID \
  --output /mnt/public/datasets/AnyDexGrasp/downloads/OFFICIAL_FILENAME \
  --expected-size OFFICIAL_BYTE_COUNT
```

Add `--expected-sha256` when an authoritative hash is provided. Without that hash,
the receipt records a local content hash, not upstream checksum agreement. The
utility retains `.partial` files, validates resume offsets, restarts when a server
ignores Range, checks recognized file formats and ZIP CRCs, rejects HTML and refuses
to overwrite differing published files. It does not deserialize model objects,
authenticate to Drive, extract archives, install dependencies or claim model
compatibility. A Drive confirmation/login HTML response fails closed and needs
inspection before retrying. Three synthetic local checks passed: resume, HTML
rejection, and wrong-size rejection; these are not official model inference.

`publish_intake.py` is the non-destructive one-time publication script for this
snapshot. It refuses to overwrite differing receipts or table files. Re-run into
a new dated destination when changing the snapshot; do not erase this blocked
intake record.
