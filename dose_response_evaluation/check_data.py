import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Check input file existence and sizes, with optional full checksum verification')
    parser.add_argument('--sha256', action='store_true', help='Read complete files and verify SHA-256 checksums')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    entries = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))['files']
    errors = []
    for entry in entries:
        path = root / entry['path']
        if not path.is_file():
            errors.append(f'Missing file: {entry["path"]}')
            continue
        if path.stat().st_size != entry['size_bytes']:
            errors.append(f'Size mismatch: {entry["path"]}')
            continue
        if args.sha256:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                errors.append(f'SHA-256 mismatch: {entry["path"]}')
                continue
        print(f'OK {entry["path"]}', flush=True)
    if errors:
        print('\n'.join(errors), flush=True)
        raise SystemExit(1)
    print(f'All {len(entries)} input files passed verification checks.')


if __name__ == '__main__':
    main()
