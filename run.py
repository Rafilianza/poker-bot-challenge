"""Start the local dashboard from any working directory: python run.py."""

from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys


def main():
    if sys.version_info < (3, 12):
        raise SystemExit('Use Python 3.12 or newer. Python 3.12 is the tested target.')
    missing = [name for name in ('macpoker', 'numpy') if find_spec(name) is None]
    if missing:
        raise SystemExit('Missing dependencies: ' + ', '.join(missing)
                         + '. Run: python -m pip install -r requirements.txt')

    # Relative data paths belong to the checkout, and Unicode works on Windows.
    folder = Path(__file__).resolve().parent
    environment = dict(os.environ, PYTHONUTF8='1')
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'arena.server', '--hold-queued']
    command.extend(sys.argv[1:])
    try:
        return subprocess.call(command, cwd=folder, env=environment)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
