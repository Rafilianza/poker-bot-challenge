"""Record teachers' decisions using only the information sent to those bots."""

from copy import deepcopy
import json
from pathlib import Path

from macpoker.transport import SubprocessTransport


class DecisionRecorder:
    """Spool decisions, then remove records from teachers that failed the game."""

    def __init__(self, output_path, training, config):
        self.training = training
        self.config = config
        self.teachers = {teacher['player']: dict(teacher) for teacher in training['teachers']}
        self.output_path = Path(output_path).with_suffix('.decisions.jsonl')
        self.pending_path = self.output_path.with_suffix('.pending.jsonl')
        self.pending_file = self.pending_path.open('w', encoding='utf-8')

    def record(self, teacher, observation, requested_action, executed_action, elapsed_ms):
        # The observation was copied before asking the bot to act. Do not read
        # spectator histories, hand results or later broadcasts into this record.
        decision_id = (
            f"{self.training['run_id']}:{self.training['game_id']}:"
            f"{observation['hand']}:{len(observation['history'])}"
        )
        record = {
            'schema_version': 1,
            'run_id': self.training['run_id'],
            'game_id': self.training['game_id'],
            'seed_group': self.training['seed_group'],
            'game_seed': self.config.seed,
            'offset': self.config.offset,
            'decision_id': decision_id,
            'teacher': teacher,
            'observation': observation,
            'requested_action': requested_action,
            'executed_action': executed_action,
            'elapsed_ms': elapsed_ms,
            'verdict': 'OK',
        }
        for field in ('table_size', 'lineup_id', 'lineup', 'batch'):
            if field in self.training:
                record[field] = self.training[field]
        self.pending_file.write(json.dumps(record) + '\n')

    def finish(self, verdicts):
        self.pending_file.close()
        failed_teachers = []
        teacher_counts = {}
        for player, teacher in self.teachers.items():
            teacher_counts[teacher['id']] = 0
            if verdicts[player] != 'OK':
                failed_teachers.append(teacher['id'])

        records = 0
        excluded_records = 0
        # Filter line by line so recording does not keep the dataset in memory.
        temporary_path = self.output_path.with_suffix('.tmp')
        try:
            with self.pending_path.open(encoding='utf-8') as source:
                with temporary_path.open('w', encoding='utf-8') as destination:
                    for line in source:
                        record = json.loads(line)
                        teacher = record['teacher']
                        if verdicts[teacher['player']] != 'OK':
                            excluded_records += 1
                            continue
                        destination.write(line)
                        records += 1
                        teacher_counts[teacher['id']] += 1
            temporary_path.replace(self.output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
            self.pending_path.unlink(missing_ok=True)

        return {
            'file': self.output_path.name,
            'records': records,
            'teacher_counts': teacher_counts,
            'excluded_records': excluded_records,
            'failed_teachers': failed_teachers,
        }

    def close(self):
        self.pending_file.close()
        self.pending_path.unlink(missing_ok=True)


class RecordingTransport(SubprocessTransport):
    """Pair a teacher's response with the engine's resulting public action."""

    def __init__(self, command, recorder, teacher):
        super().__init__(command)
        self.recorder = recorder
        self.teacher = teacher
        self.pending_decision = None

    def act(self, view, timeout_ms):
        self.pending_decision = None
        observation = deepcopy(view)
        action, elapsed_ms = super().act(view, timeout_ms)
        self.pending_decision = (observation, action.to_wire(), elapsed_ms)
        return action, elapsed_ms

    def send(self, message):
        super().send(message)
        if message.get('type') != 'action' or self.pending_decision is None:
            return

        observation, requested_action, elapsed_ms = self.pending_decision
        # Seats rotate every hand. Match the pending seat, hand and street,
        # rather than assuming that a teacher's player ID is its current seat.
        if (
            message['seat'] != observation['seat']
            or message['hand'] != observation['hand']
            or message['street'] != observation['street']
        ):
            return

        executed_action = {'action': message['action'], 'amount': message['amount']}
        self.recorder.record(
            self.teacher, observation, requested_action, executed_action, elapsed_ms
        )
        self.pending_decision = None
