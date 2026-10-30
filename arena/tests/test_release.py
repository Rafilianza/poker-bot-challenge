"""Check public examples, source-preserving exports, and the 80-agent boundary."""

import base64
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arena import server


ROOT = Path(__file__).resolve().parents[2]


class ReleaseTests(unittest.TestCase):
    def test_examples_complete_a_real_benchmark_in_a_fresh_workspace(self):
        with tempfile.TemporaryDirectory(prefix='poker-arena-demo-') as directory:
            with patch.object(server, 'DATA', Path(directory)):
                server.STOP.clear()
                server.CANCEL.clear()
                server.init()
                identifiers = []
                for path in sorted((ROOT / 'examples').glob('*.py')):
                    agent = server.register({
                        'name': path.stem, 'filename': path.name,
                        'content': base64.b64encode(path.read_bytes()).decode(),
                    })
                    identifiers.append(agent['id'])
                self.assertEqual(len(identifiers), 3)
                run = server.queue_run({'mode': 'benchmark', 'agents': identifiers,
                                        'seeds': ['release-demo'], 'deals': 1})
                server.execute(run)
                self.assertEqual(run['status'], 'completed')
                games = [game for table in run['results'] for game in table['games']]
                self.assertEqual(len(games), 15)
                for game in games:
                    self.assertTrue(all(verdict == 'OK' for verdict in game['verdicts']))

    def test_eighty_agents_are_accepted_and_study_coverage_is_balanced(self):
        identifiers = [f'agent-{number}' for number in range(80)]

        def registered_agent(table, identifier):
            return {'id': identifier, 'name': identifier, 'hash': identifier}

        # Scheduling only: no matches are played or saved in the real workspace.
        with patch.object(server, 'get', side_effect=registered_agent), patch.object(server, 'save'):
            for mode in ('benchmark', 'tournament', 'study'):
                settings = {'mode': mode, 'agents': identifiers, 'seeds': ['limit'], 'deals': 1}
                run = server.queue_run(settings)
                self.assertEqual(len(run['config']['agents']), 80)
                if mode == 'study':
                    for size in (4, 5, 6):
                        appearances = Counter()
                        for table in run['schedule']:
                            if table['table_size'] == size:
                                appearances.update(table['agent_ids'])
                        self.assertEqual(len(appearances), 80)
                        self.assertEqual(set(appearances.values()), {size})
                settings['agents'] = identifiers + ['agent-80']
                with self.assertRaisesRegex(ValueError, '80'):
                    server.queue_run(settings)

    def test_historical_manifest_is_separate_from_local_jobs(self):
        research = json.loads((ROOT / 'docs/experiments.json').read_text(encoding='utf-8'))
        sources = json.loads((ROOT / 'docs/model-sources.json').read_text(encoding='utf-8'))
        methods = {method['id'] for method in research['methods']}
        self.assertIn(research['latest_experiment']['method'], methods)
        for experiment in research['history']:
            self.assertIn(experiment['method'], methods)
        self.assertTrue(all(model['bundle_status'] == 'reference_only' for model in sources['models']))


if __name__ == '__main__':
    unittest.main()
