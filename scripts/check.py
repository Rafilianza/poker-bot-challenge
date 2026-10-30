"""Run isolated core checks, or all learning/PHH checks with --full."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='also test learning and PHH workflows')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'unittest']
    if args.full:
        command.extend(['discover', '-s', 'arena/tests', '-q'])
    else:
        command.extend([
            'arena.tests.test_arena', 'arena.tests.test_study',
            'arena.tests.test_datasets', 'arena.tests.test_recording',
            'arena.tests.test_release', '-q',
        ])
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return subprocess.call(command, cwd=root, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
