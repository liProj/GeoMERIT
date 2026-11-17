"""Verify the dated publication files against MANIFEST.sha256 (standard library)."""
from pathlib import Path
import hashlib
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = root / 'MANIFEST.sha256'
    failures = []
    checked = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f'Missing or invalid file: {relative}')
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            failures.append(f'SHA256 mismatch: {relative}')
        checked += 1
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        return 1
    print(f'PASS: {checked} published files match MANIFEST.sha256.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
