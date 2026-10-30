import base64
import ast
import errno
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile
from arena import server
from arena.scoring import points, table_sizes
from arena.export import lean_public_files


class ArenaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = server.DATA
        server.DATA = Path(self.temp.name)
        server.CANCEL.clear()
        server.STOP.clear()
        server.init()

    def tearDown(self):
        server.DATA = self.previous
        server.STOP.clear()
        self.temp.cleanup()

    def agent(self, action='return state.call()', name='Test'):
        source = 'from macpoker import Bot\nclass Test(Bot):\n    def act(self, state):\n        ' + action + '\n'
        return server.register({'name':name,'filename':'test.py','content':base64.b64encode(source.encode()).decode()})

    def run_for(self, agents, mode='benchmark', deals=3):
        return server.queue_run({'agents':[a['id'] for a in agents], 'seeds':['test-fixed'], 'mode':mode, 'deals':deals})

    def test_scoring_and_partitioning(self):
        self.assertEqual(points([10,8,8,4,0]), [5,3.5,3.5,2,1])
        self.assertEqual(points([0,0]), [1.5,1.5])
        self.assertEqual(table_sizes(6), [6])
        self.assertEqual(table_sizes(7), [4,3])
        for n in range(2,61):
            sizes=table_sizes(n)
            self.assertEqual(sum(sizes),n)
            self.assertTrue(all(2<=x<=6 for x in sizes))
            if n>=8:
                self.assertTrue(all(4<=x<=6 for x in sizes))

    def test_download_exact_snapshot(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('main.py','from helper import value\n')
            z.writestr('helper.py','value = 1\n')
            z.writestr('LICENSE','test license')
        agent=server.register({'name':'ZIP test','filename':'a.zip','content':base64.b64encode(stream.getvalue()).decode()})
        raw=server.agent_archive(agent['id'])
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            self.assertEqual(set(z.namelist()),{'main.py','helper.py','LICENSE'})
            self.assertEqual(z.read('helper.py'),b'value = 1\n')
        self.assertEqual(raw,server.agent_archive(agent['id']))
        again=server.register({'name':'Export roundtrip','filename':'a.zip','content':base64.b64encode(raw).decode()})
        self.assertEqual(agent['hash'],again['hash'])
        (server.DATA/'agents'/agent['id']/'helper.py').write_text('changed')
        with self.assertRaises(ValueError):
            server.agent_archive(agent['id'])

    def test_public_export_preserves_source_and_required_attribution(self):
        files={'main.py':b'"""Adapted from example; see ADAPTATION.md."""\n# explanation\nvalue = 1\n',
               'PUBLIC_BASELINES.md':b'long reference', 'ADAPTATION.md':b'Source: example\n',
               'original_reference.txt':b'not executable', 'LICENSE':b'required licence'}
        result=lean_public_files(files)
        self.assertEqual(result, files)
        self.assertEqual(result['LICENSE'], b'required licence')
        self.assertIn(b'Adapted from example', result['main.py'])

    def test_public_export_executable_code_unchanged(self):
        def executable(source):
            tree=ast.parse(source)
            for node in ast.walk(tree):
                body=getattr(node,'body',None)
                if isinstance(body,list) and body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
                    body.pop(0)
            return ast.dump(tree)
        # The executable-code guarantee must not depend on someone else's checkout.
        files = {
            'PUBLIC_BASELINES.md': b'Public baseline source reference',
            'ADAPTATION.md': b'Adaptation notes',
            'main.py': (
                '"""Example public bot.\nIncludes a multiline module docstring.\n"""\n'
                'from helpers.policy import choose_action\n'
                '# Module comment to remove.\n'
                'LABEL = "keep # inside strings"\n'
                'class ExampleBot:\n'
                '    """A class docstring."""\n'
                '    def act(self, state):\n'
                '        """Select a legal action."""\n'
                '        return choose_action(state)  # Trailing comment.\n'
            ).encode(),
            'helpers/policy.py': (
                '\ufeff"""Helper module with a UTF-8 byte-order mark."""\n'
                'def choose_action(state):\n'
                '    """Preserve the actual decision logic."""\n'
                '    if state.to_call:\n'
                '        return state.call()\n'
                '    return state.check()\n'
            ).encode(),
        }
        exported = lean_public_files(files)
        self.assertEqual(set(exported), set(files))
        for name, data in exported.items():
            if not name.endswith('.py'):
                continue
            original = files[name].decode('utf-8-sig')
            self.assertEqual(executable(original), executable(data.decode('utf-8-sig')), name)
        self.assertIn(b'keep # inside strings', exported['main.py'])
        self.assertIn(b'Trailing comment', exported['main.py'])

    def test_public_export_docstring_only_function_still_runs(self):
        files = {
            'PUBLIC_BASELINES.md': b'Source reference',
            'main.py': b'def optional_hook():\n    """No action needed."""\n',
        }
        exported = lean_public_files(files)
        namespace = {}
        exec(exported['main.py'], namespace)
        self.assertIsNone(namespace['optional_hook']())

    def test_upload_limits_and_traversal(self):
        for path in ['../main.py','C:/main.py','main.py/../../bad','CON.py','main.py.']:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w') as z:
                z.writestr(path,'x')
            with self.assertRaises(ValueError):
                server.register({'name':'x','filename':'a.zip','content':base64.b64encode(stream.getvalue()).decode()})
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:
            z.writestr('main.py','x')
            z.writestr('MAIN.py','y')
        with self.assertRaises(ValueError):
            server.register({'name':'x','filename':'a.zip','content':base64.b64encode(stream.getvalue()).decode()})

    def test_real_engine_repeatability_and_profile_isolation(self):
        agent=self.agent()
        run=self.run_for([agent])
        server.execute(run)
        games=run['results'][0]['games']
        self.assertEqual(len(games),5)
        self.assertEqual(run['leaderboard'][0]['hands'],15)
        for g in games:
            self.assertEqual(sum(g['chips']),0)
            self.assertEqual(g['verdicts'],['OK']*5)
        again=self.run_for([agent])
        server.execute(again)
        self.assertEqual([g['chips'] for g in games],[g['chips'] for g in again['results'][0]['games']])
        board=server.state(run['profile'])['leaderboard']
        self.assertEqual(len(board),1)
        self.assertEqual(board[0]['hands'],15)
        other=self.run_for([agent],deals=4)
        self.assertNotEqual(other['profile'],run['profile'])
        self.assertEqual(server.state(other['profile'])['leaderboard'],[])

    def test_crashed_bot_scored_not_dropped(self):
        bad=self.agent('raise RuntimeError("test crash")')
        run=self.run_for([bad])
        server.execute(run)
        self.assertEqual(run['status'],'completed')
        self.assertEqual(run['leaderboard'][0]['failures'],5)
        self.assertTrue(all(g['verdicts'][0]=='RTE' for g in run['results'][0]['games']))

    def test_tournament_four_rounds_and_tie_playoff(self):
        agents=[self.agent(name=str(i)) for i in range(2)]
        run=self.run_for(agents,mode='tournament',deals=2)
        server.execute(run)
        self.assertEqual([t['round'] for t in run['results'][:4]],[1,2,3,4])
        self.assertEqual(run['results'][-1]['round'],'playoff')
        self.assertEqual([r['placement_points'] for r in run['leaderboard']],[6,6])

    def test_cancelled_and_interrupted_excluded(self):
        run=self.run_for([self.agent()])
        server.CANCEL.add(run['id'])
        with self.assertRaises(server.Cancelled):
            server.execute(run)
        run['status']='running'
        server.save('runs',run)
        server.init()
        self.assertEqual(server.get('runs',run['id'])['status'],'interrupted')
        self.assertEqual(server.state()['leaderboard'],[])

    def test_old_queue_is_held_only_when_requested(self):
        agent = self.agent()
        queued = self.run_for([agent])
        running = self.run_for([agent])
        running['status'] = 'running'
        server.save('runs', running)

        # The normal Python entry point keeps the existing queue behavior.
        server.init()
        self.assertEqual(server.get('runs', queued['id'])['status'], 'queued')
        self.assertEqual(server.get('runs', running['id'])['status'], 'interrupted')

        # The Mac launcher's option prevents inherited experiments from starting.
        server.init(hold_queued=True)
        held = server.get('runs', queued['id'])
        self.assertEqual(held['status'], 'interrupted')
        self.assertIn('Queued before this launch', held['error'])
        self.assertEqual(server.state()['leaderboard'], [])

        new_run = self.run_for([agent])
        self.assertEqual(new_run['status'], 'queued')

    def test_cli_uses_selected_data_directory_and_optional_browser(self):
        data_folder = Path(self.temp.name) / 'separate arena data'
        arguments = ['arena.server', '--data-dir', str(data_folder), '--port', '0', '--open-browser']

        # Replace networking and the background worker; database initialization is real.
        with (
            mock.patch('sys.argv', arguments),
            mock.patch('sys.stdout', new_callable=io.StringIO),
            mock.patch.object(server, 'ThreadingHTTPServer') as http_server,
            mock.patch.object(server.threading, 'Thread') as worker_thread,
            mock.patch.object(server.webbrowser, 'open') as open_browser,
        ):
            http_server.return_value.server_port = 9123
            server.main()

        self.assertEqual(server.DATA, data_folder.resolve())
        self.assertTrue((data_folder / 'arena.sqlite').is_file())
        self.assertEqual(server.all_rows('agents'), [])
        http_server.assert_called_once_with(('127.0.0.1', 0), server.Handler)
        http_server.return_value.serve_forever.assert_called_once()
        worker_thread.return_value.start.assert_called_once()
        open_browser.assert_called_once_with('http://127.0.0.1:9123')

    def test_cli_does_not_open_browser_without_flag(self):
        arguments = ['arena.server', '--data-dir', self.temp.name, '--hold-queued']
        queued = self.run_for([self.agent()])
        with (
            mock.patch('sys.argv', arguments),
            mock.patch('sys.stdout', new_callable=io.StringIO),
            mock.patch.object(server, 'ThreadingHTTPServer'),
            mock.patch.object(server.threading, 'Thread'),
            mock.patch.object(server.webbrowser, 'open') as open_browser,
        ):
            server.main()

        open_browser.assert_not_called()
        self.assertEqual(server.get('runs', queued['id'])['status'], 'interrupted')

    def test_busy_port_does_not_change_existing_run(self):
        queued = self.run_for([self.agent()])
        arguments = ['arena.server', '--data-dir', self.temp.name, '--hold-queued']
        busy_port = OSError(errno.EADDRINUSE, 'Address already in use')
        with (
            mock.patch('sys.argv', arguments),
            mock.patch('sys.stderr', new_callable=io.StringIO) as error_output,
            mock.patch.object(server, 'ThreadingHTTPServer', side_effect=busy_port),
            mock.patch.object(server, 'init') as initialize,
            self.assertRaises(SystemExit) as stopped,
        ):
            server.main()

        self.assertEqual(stopped.exception.code, 1)
        self.assertIn('Port 8765 is already in use', error_output.getvalue())
        initialize.assert_not_called()
        self.assertEqual(server.get('runs', queued['id'])['status'], 'queued')

if __name__=='__main__':
    unittest.main()
