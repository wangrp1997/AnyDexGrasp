"""Encode existing simultaneously rendered frames; never replay simulation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'outputs/native_deps/python_overlay'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(run):
    import imageio_ffmpeg
    run = Path(run).resolve()
    manifest_path = run/'frames.json'
    manifest = json.loads(manifest_path.read_text())
    frames = manifest['frames']
    if not frames or not manifest['actual_simultaneous_render']:
        raise ValueError('Only existing simultaneous-render frames are supported')
    results = json.loads((run/'RESULTS.json').read_text())
    dt = results['physical_seconds']/results['physics_steps']
    fps = float(manifest['fps'])
    interval = int(manifest['every_physics_steps'])
    if abs(dt*fps*interval-1) > 1e-10:
        raise ValueError('Frame rate must match physical time')
    frame_hashes = []
    for i, frame in enumerate(frames):
        if frame['path'] != 'frames/frame_%05d.png' % i:
            raise ValueError('Unexpected frame order')
        if frame['physics_step'] != 1+i*interval:
            raise ValueError('Unexpected physics-step interval')
        if abs(frame['physical_time_s']-frame['physics_step']*dt) > 1e-10:
            raise ValueError('Frame timestamp differs from physical time')
        frame_hashes.append(sha(run/frame['path']))
    output = run/'grasp_normal_speed.mp4'
    if output.exists():
        raise FileExistsError(output)
    binary = imageio_ffmpeg.get_ffmpeg_exe()
    command = [binary, '-hide_banner', '-nostdin', '-n', '-framerate', str(fps),
               '-i', str(run/'frames/frame_%05d.png'), '-an', '-c:v', 'libx264',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    encoded = subprocess.run(command, capture_output=True, text=True, check=True)
    (run/'video_encode.log').write_text(encoded.stderr)
    decoded = subprocess.run([binary, '-hide_banner', '-nostdin', '-i', str(output),
                              '-progress', 'pipe:1', '-nostats', '-f', 'null', '-'],
                             capture_output=True, text=True, check=True)
    decoded_frames = int(re.findall(r'^frame=(\d+)$', decoded.stdout, re.M)[-1])
    if decoded_frames != len(frames):
        raise ValueError('Encoded frame count differs from recorded frames')
    receipt = {
        'run': str(run), 'scope': 'Normal-speed encoding of existing frames; zero new physics',
        'frames_manifest_sha256': sha(manifest_path),
        'ordered_frame_sha256_digest': hashlib.sha256('\n'.join(frame_hashes).encode()).hexdigest(),
        'source_frames': len(frames), 'decoded_frames': decoded_frames, 'fps': fps,
        'video_duration_s': len(frames)/fps,
        'first_frame_physical_time_s': frames[0]['physical_time_s'],
        'last_frame_physical_time_s': frames[-1]['physical_time_s'],
        'ffmpeg_binary': binary, 'ffmpeg_binary_sha256': sha(binary),
        'ffmpeg_version': subprocess.check_output([binary, '-version'], text=True).splitlines()[0],
        'encode_command': command, 'video': str(output), 'video_bytes': output.stat().st_size,
        'video_sha256': sha(output), 'new_physics_steps': 0,
        'result_at_capture': {'lift_observed': results['lift_observed'],
                              'port_stability_pass': results['port_stability_pass']},
    }
    (run/'VIDEO.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    encode(parser.parse_args().run)
