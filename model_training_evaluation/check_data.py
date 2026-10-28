"""Check the presence and sizes of all input files; --sha256 additionally verifies their contents."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha256', action='store_true', help='Read all data and verify SHA-256 checksums (slower)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for entry in manifest['files']:
        path = root / entry['path']
        if not path.is_file():
            errors.append(f'Missing file: {entry["path"]}')
            continue
        if path.stat().st_size != entry['size_bytes']:
            errors.append(f'Size mismatch: {entry["path"]}')
            continue
        if args.sha256:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                errors.append(f'SHA-256 mismatch: {entry["path"]}')
                continue
        print(f'OK {entry["path"]}', flush=True)
    if errors:
        for error in errors:
            print(error, flush=True)
        raise SystemExit(1)
    mode = 'size and SHA-256' if args.sha256 else 'existence and size'
    print(f'All {len(manifest["files"])} data files passed the {mode} checks.')


if __name__ == '__main__':
    main()
