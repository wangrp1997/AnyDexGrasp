#!/usr/bin/env python3
"""Download one explicitly selected public artifact; never accept an HTML model.

The official Drive folder must first be inspected to obtain a file ID. This
utility does not guess filenames/file IDs, install packages, authenticate, or
deserialize PyTorch/pickle objects. A .partial file is retained on interruption.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tarfile
from urllib.parse import urlencode, urlparse, parse_qs, urljoin
import zipfile

import requests


class _DriveConfirmationForm(HTMLParser):
    """Parse Google's public large-file confirmation, never a login form."""
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'form':
            self.current = dict(action=attributes.get('action', ''),
                                method=attributes.get('method', 'get').lower(), fields={})
            self.forms.append(self.current)
        elif tag == 'input' and self.current is not None:
            name = attributes.get('name')
            if name in ['id', 'export', 'confirm', 'uuid', 'resourcekey']:
                self.current['fields'][name] = attributes.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.current = None


def _get_artifact_response(session, url, headers):
    response = session.get(url, headers=headers, stream=True, timeout=(15, 60))
    response.raise_for_status()
    if 'text/html' not in response.headers.get('Content-Type', '').lower():
        return response
    original = urlparse(url)
    if original.hostname not in ['drive.google.com', 'drive.usercontent.google.com']:
        response.close()
        raise ValueError('HTML response is not an artifact')
    page = response.text
    base_url = response.url
    response.close()
    if 'quota exceeded' in page.lower() or 'too many users have viewed or downloaded' in page.lower():
        raise ValueError('Official Google Drive file download quota exceeded; no artifact published')
    if len(page) > 2_000_000:
        raise ValueError('Unexpected large HTML response; no artifact published')
    parser = _DriveConfirmationForm()
    parser.feed(page)
    file_id = parse_qs(original.query).get('id', [None])[0]
    for form in parser.forms:
        action = urljoin(base_url, form['action'])
        parsed = urlparse(action)
        fields = form['fields']
        if (form['method'] != 'get' or parsed.scheme != 'https' or
                parsed.hostname != 'drive.usercontent.google.com' or
                parsed.path != '/download' or not file_id or
                fields.get('id') != file_id or not fields.get('confirm') or
                fields.get('export') != 'download'):
            continue
        confirmed = action.split('?')[0] + '?' + urlencode(fields)
        response = session.get(confirmed, headers=headers, stream=True, timeout=(15, 60))
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            response.close()
            raise ValueError('Drive returned HTML after public confirmation; no artifact published')
        return response
    raise ValueError('HTML is not the expected public Drive download confirmation; no artifact published')


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_file(path):
    with path.open("rb") as f:
        head = f.read(512)
    prefix = head.lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<head")):
        raise ValueError("HTML response is not an artifact")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                raise ValueError("ZIP CRC error: " + bad)
            return {"format": "zip", "members": len(z.infolist())}
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            return {"format": "tar", "members": sum(1 for _ in t)}
    if path.name.endswith((".json.partial", ".json")):
        with path.open() as f:
            value = json.load(f)
        return {"format": "json", "top_level_type": type(value).__name__}
    if head.startswith(b"\x93NUMPY"):
        return {"format": "npy", "unpickled": False}
    if head.startswith(b"\x80"):
        return {"format": "pickle_compatible", "unpickled": False}
    raise ValueError("Unrecognized artifact format; retain partial for inspection")


def fetch(url, destination, expected_size=None, expected_sha256=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = sha256(destination)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("Existing destination SHA differs; refusing overwrite")
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError("Existing destination size differs; refusing overwrite")
        return dict(path=str(destination), bytes=destination.stat().st_size,
                    sha256=digest, status="already_present", **inspect_file(destination))
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and offset == expected_size:
        # A prior connection may have completed before validation/receipt failed.
        # Validate those exact bytes without issuing an unsatisfiable Range.
        digest = sha256(partial)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError('Complete partial SHA differs from expected SHA')
        shape = inspect_file(partial)
        partial.replace(destination)
        result = dict(path=str(destination), bytes=offset, sha256=digest,
                      source=url, expected_sha256_provided=bool(expected_sha256),
                      status='completed_partial_content_checked', **shape)
        destination.with_name(destination.name + '.receipt.json').write_text(
            json.dumps(result, indent=2) + '\n')
        return result
    headers = {"Range": "bytes=%d-" % offset} if offset else {}
    with requests.Session() as session:
        response = _get_artifact_response(session, url, headers)
        if offset and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith("bytes %d-" % offset):
                raise ValueError("Resume Content-Range does not match partial size")
            mode = "ab"
        else:
            mode = "wb"  # A server may legally ignore Range; restart safely.
        with partial.open(mode) as f:
            for block in response.iter_content(1024 * 1024):
                if block:
                    f.write(block)
            f.flush()
            os.fsync(f.fileno())
        final_url = response.url
        content_type = response.headers.get("Content-Type")
    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError("Downloaded size differs from expected size")
    digest = sha256(partial)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Downloaded SHA differs from expected SHA")
    shape = inspect_file(partial)
    partial.replace(destination)
    result = dict(path=str(destination), bytes=size, sha256=digest,
                  source=url, final_url=final_url, content_type=content_type,
                  expected_sha256_provided=bool(expected_sha256),
                  status="downloaded_and_content_checked", **shape)
    destination.with_name(destination.name + ".receipt.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--file-id", help="A FILE ID, never the official folder ID")
    p.add_argument("--output", required=True)
    p.add_argument("--expected-size", type=int)
    p.add_argument("--expected-sha256")
    args = p.parse_args()
    if args.file_id == "1XfJmEkg29vq7swCndnS_B0Y4djwWhZRo":
        p.error("This is the folder ID; its public index is not yet available")
    url = args.url or ("https://drive.usercontent.google.com/download?" +
                       urlencode(dict(id=args.file_id, export="download", confirm="t")))
    if urlparse(url).scheme != "https":
        p.error("Use an HTTPS public artifact URL")
    print(json.dumps(fetch(url, args.output, args.expected_size,
                           args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()
