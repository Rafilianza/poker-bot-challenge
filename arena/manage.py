"""CLI for bulk registration and queuing through the running local dashboard."""
import argparse
import base64
import glob
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    commands = parser.add_subparsers(dest='command', required=True)
    upload = commands.add_parser('import', help='Register Python files or ZIPs')
    upload.add_argument('files', nargs='+', help='Python/zip paths or glob patterns')
    upload.add_argument('--prefix', default='')
    queue = commands.add_parser('run', help='Queue a benchmark, tournament or study')
    queue.add_argument('--mode', choices=['benchmark', 'tournament', 'study'], default='benchmark')
    preview = commands.add_parser('preview', help='Preview a study without queuing games')
    preview.set_defaults(mode='study')

    # Study previews use the same settings as the eventual run.
    for command in (queue, preview):
        command.add_argument('--agents', nargs='+', required=True, help='Agent IDs from import output or state')
        command.add_argument('--seeds', nargs='+', default=['arena-1', 'arena-2', 'arena-3'])
        command.add_argument('--deals', type=int, default=100, help='Hands per game')
        command.add_argument('--table-sizes', nargs='+', type=int, choices=[4, 5, 6], default=[4, 5, 6], help='Study table sizes')
        command.add_argument('--batches', type=int, default=1, help='Study batches per seed, from 1 to 20')
        command.add_argument('--record-training', action='store_true', help='Save teacher decisions for behavioural cloning')
        command.add_argument('--teachers', nargs='+', help='Teacher agent IDs; defaults to all selected agents when recording')

    for action in ('pause', 'resume'):
        command = commands.add_parser(action, help=f'{action.capitalize()} a saved table-size study')
        command.add_argument('run_id')
    commands.add_parser('state', help='Show registered agents and saved runs')
    args = parser.parse_args()

    def call(path, payload=None):
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
        request = Request(args.url.rstrip('/') + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            message = error.read().decode('utf-8', errors='replace')
            parser.exit(1, f'Server rejected the request ({error.code}): {message}\n')
        except URLError as error:
            parser.exit(1, f'Cannot reach the dashboard at {args.url}: {error.reason}\n')

    if args.command == 'import':
        files = []
        for pattern in args.files:
            files.extend(glob.glob(pattern))
        files = sorted(set(files))
        if not files:
            parser.error('No matching files')
        for name in files:
            path = Path(name)
            content = base64.b64encode(path.read_bytes()).decode()
            result = call('/api/agents', {'name': args.prefix + path.stem, 'filename': path.name, 'content': content})
            print(json.dumps(result))
    elif args.command in ('run', 'preview'):
        if args.teachers and not args.record_training:
            parser.error('--teachers requires --record-training')
        teachers = []
        if args.record_training:
            teachers = args.teachers if args.teachers is not None else args.agents
        payload = {
            'mode': args.mode, 'agents': args.agents, 'seeds': args.seeds,
            'deals': args.deals, 'record_training': args.record_training,
            'teachers': teachers,
        }
        if args.mode == 'study':
            payload.update(table_sizes=args.table_sizes, batches=args.batches)
        endpoint = '/api/study/preview' if args.command == 'preview' else '/api/runs'
        print(json.dumps(call(endpoint, payload), indent=2))
    elif args.command in ('pause', 'resume'):
        run_id = quote(args.run_id, safe='')
        print(json.dumps(call(f'/api/runs/{run_id}/{args.command}', {}), indent=2))
    else:
        print(json.dumps(call('/api/state'), indent=2))


if __name__ == '__main__':
    main()
