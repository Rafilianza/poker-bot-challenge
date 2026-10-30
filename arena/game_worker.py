"""One official-engine game. Run outside the web server process."""
import hashlib
import json
from pathlib import Path
import sys
from macpoker.match import MatchConfig, MatchRunner
from macpoker.transport import SubprocessTransport

if __package__:
    from .recording import DecisionRecorder, RecordingTransport
else:
    from recording import DecisionRecorder, RecordingTransport


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    cfg = MatchConfig(**request['config'])
    output_path = Path(sys.argv[2])
    recorder = None
    if request.get('training'):
        recorder = DecisionRecorder(output_path, request['training'], cfg)

    transports = []
    for player, spec in enumerate(request['specs']):
        command = [
            sys.executable, '-u', str(Path(__file__).with_name('bot_runner.py')),
            spec, f'{cfg.seed}:policy:{player}',
        ]
        if recorder is not None and player in recorder.teachers:
            transport = RecordingTransport(command, recorder, recorder.teachers[player])
        else:
            transport = SubprocessTransport(command)
        transports.append(transport)

    try:
        result = MatchRunner(cfg, transports).run()
        data = {**result.to_dict(), 'hands': result.hands, 'logs': [t.stderr_tail() for t in transports]}
        if recorder is not None:
            data['training'] = recorder.finish(result.verdicts)
        if request.get('checkpoint'):
            data['request_sha256'] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        temporary_path = output_path.with_suffix('.json.tmp')
        temporary_path.write_text(json.dumps(data), encoding='utf-8')
        temporary_path.replace(output_path)
    finally:
        for transport in transports:
            transport.close()
        if recorder is not None:
            recorder.close()

if __name__ == '__main__':
    main()
