# AnyDexGrasp official data intake

Updated 2026-11-03. Read `/mnt/public/datasets/AnyDexGrasp/CURRENT_STATUS.json`
for the current snapshot; the old `intake_20261103.json` is historical evidence of
the initial direct-network failure.

## Available

- The coordinator's model campaign completed all 31 hand-specific obj140 model
  files (327,466,392 bytes), plus representation `checkpoint.tar.18`
  (307,636,549 bytes): 32 files, 635,102,941 bytes total.
- Eight official real decision JSON files are complete and JSON/SHA checked:
  7,403,397,972 bytes. This includes original Allegro, DH3 and Inspire test data,
  their obj140 train records, and DH3 obj30/Inspire obj40 records. Original names
  and bytes are preserved. Individual adjacent `.receipt.json` files record URLs,
  byte counts, and local SHA-256 (upstream did not provide reference hashes).
- Four official GraspNet demo inputs, pinned to upstream commit
  `280c215129f759ed8649cb4e89fc5dfee55f4f80`, are under
  `examples/graspnet_baseline/<commit>/`: `color.png`, `depth.png`,
  `workspace_mask.png`, `meta.mat`. PNG and MAT format checks passed. These are
  single-frame smoke inputs, not an AnyDex independent test set or physical result.
- Three source hand lookup tables are separately marked as lookup tables rather
  than collected training records.

## Remaining source limitations

- The DH3 obj40 JSON has all 1,322,043,333 advertised bytes, but contains invalid
  UTF-8/binary data at byte 1,308,622,852. A separate proxy's 96-byte range request
  returned exactly the same bytes. The raw `.partial` is retained for diagnosis;
  it is not counted among eight validated JSONs, modified, or silently repaired.
- All three official reannotated CGR ZIPs currently return Google Drive's
  **download quota exceeded** page. Their combined advertised size is
  55,835,206,232 bytes. No HTML is published as ZIP. The initial bounded data campaign finished; a
  separate low-frequency retry owner is now waiting for the quota to clear. This
  is an external file quota issue, not a missing proxy or confirmation form.
- The original GraspNet-1Billion RGB-D image archives were not downloaded. Labels
  alone would not suffice for representation-model retraining. The existing
  DexGraspNet2.0 synthetic scenes were not substituted for them.

## Low-frequency quota retry owner

A separate process was started: PID `2447202`, state
`metadata/quota_retry_campaign_20261103_v1.json`. Its first retry is scheduled for
**2026-11-03 22:30:38 Asia/Shanghai**, then six hours after each unsuccessful round,
with at most four rounds. Only the three quota-limited label archives are included;
each file gets one attempt per round. When Google permits a download, the normal
stream/partial/SHA/ZIP-CRC validation runs. The process uses the existing nros
proxy in its own environment, without cron, systemd changes, or global networking
changes. `next_retry_unix`, owner PID, round receipts, completion and remaining
files are persisted. Do not start a duplicate owner. The damaged DH3 obj40 JSON
is excluded from these retries.

## Supported network and retry

The server already provides shell functions in `/opt/nros_tools/nros_rc`; they are
not PATH executables. Use them only in the task's process environment:

```bash
source /opt/nros_tools/nros_rc
nros-proxy-sg
python reproduction/data_intake/fetch_public_file.py \
  --file-id ACTUAL_OFFICIAL_FILE_ID \
  --output /mnt/public/datasets/AnyDexGrasp/downloads/OFFICIAL_FILENAME \
  --expected-size OFFICIAL_BYTE_COUNT
```

`nros-proxy-sg2` also works. No global DNS/routes, shared environments, or
subscriptions were changed. The official source is
<https://drive.google.com/drive/folders/1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo>.

`fetch_public_file.py` retains partial bytes, validates resume ranges, handles
Google's ordinary public large-file confirmation form, rejects quota/login HTML,
checks JSON/ZIP formats and ZIP CRC, and never deserializes model objects. A fully
transferred partial is validated locally before issuing another request, avoiding
an unsatisfiable HTTP 416 range. Six synthetic downloader checks passed; these are
separate from actual official data/model verification.

`download_data_campaign.py` is a durable bounded-worker runner for an explicit
manifest. State updates are atomic every five seconds, a file lock prevents
duplicate owners, and published/partial bytes remain separate. After the present owner finishes, a future label
retry may use `metadata/label_retry_manifest_20261103.json` and a **new** state path;
retain the completed campaign and its failures. Do not re-download or rewrite the
eight good JSONs or repair the bad source record to inflate completeness.

All large files live in public and remain outside Git. Independent reproduction
repository: `/home/wangrenpeng/AnyDexGrasp/`.
