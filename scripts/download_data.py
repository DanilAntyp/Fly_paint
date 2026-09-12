"""Download only initial graph data. No account/token needed for public files."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(item, destination, metadata):
    target = destination / item['filename']
    prior = metadata.get(item['filename'])
    if target.exists():
        if prior and prior['url'] == item['url'] and checksum(target) == prior['sha256']:
            print(f'Already verified: {target.name}', flush=True)
            return prior
        raise ValueError(f'{target} exists without matching provenance; move it before retrying')
    partial = target.with_suffix(target.suffix + '.part')
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'FlyPaint/0.1'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get('Content-Length')
            expected = int(length) if length else None
            if expected and shutil.disk_usage(destination).free < expected * 2:
                raise OSError('Insufficient disk space for download and processing headroom')
            digest = hashlib.sha256()
            size = 0
            next_report = 64 * 1024 * 1024
            with partial.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size >= next_report:
                        print(f'{target.name}: {size / 1024**2:.0f} MiB', flush=True)
                        next_report += 64 * 1024 * 1024
            if expected is not None and size != expected:
                raise ValueError(f'Incomplete download: {size} of {expected} bytes')
            if size == 0:
                raise ValueError('Empty download')
            # Open Arrow metadata before accepting a downloaded file.
            import pyarrow as pa
            with pa.memory_map(str(partial), 'r') as source:
                reader = pa.ipc.open_file(source)
                if not reader.schema.names or reader.num_record_batches == 0:
                    raise ValueError('Empty Arrow dataset')
            result = {'url': item['url'], 'bytes': size, 'sha256': digest.hexdigest(),
                      'retrieved_at': datetime.now(timezone.utc).isoformat()}
            partial.replace(target)
            print(f'Downloaded and validated: {target.name}', flush=True)
            return result
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-neurotransmitters', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/raw')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/data_sources.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / 'downloads.json'
    metadata = json.loads(record.read_text()) if record.exists() else {}
    keys = ['annotations', 'weights']
    if args.include_neurotransmitters:
        keys.append('neurotransmitters')
    for key in keys:
        item = manifest['files'][key]
        metadata[item['filename']] = download(item, args.output, metadata)
        temp = record.with_suffix('.tmp')
        temp.write_text(json.dumps(metadata, indent=2) + '\n')
        temp.replace(record)


if __name__ == '__main__':
    main()
