"""Checks for fair table-size schedules and resumable study results."""

import base64
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import zipfile

from arena import server, study


def fake_agents(count):
    return [{"id": f"agent-{number}", "name": f"Agent {number}"} for number in range(count)]


def example_run(agent_count=4, sizes=None, seeds=None, batches=1, deals=5):
    agents = fake_agents(agent_count)
    identifiers = [agent["id"] for agent in agents]
    sizes = [4] if sizes is None else sizes
    seeds = ["study-test"] if seeds is None else seeds
    plan = study.build_plan(identifiers, sizes, seeds, deals=deals, batches=batches)
    run = {
        "mode": "study",
        "config": {"agents": identifiers, "table_sizes": sizes, "seeds": seeds, "batches": batches, "deals": deals},
        "profile_config": {"blinds": [1, 2], "deals": deals},
        "schedule": plan["schedule"],
        "results": [],
    }
    return run, agents, plan


def completed_table(scheduled, agents, game_count=None, deals=5, winner_chips=10):
    """Create simple zero-sum results in the scheduled player's order."""
    agents_by_id = {agent["id"]: agent for agent in agents}
    participants = [agents_by_id[identifier] for identifier in scheduled["agent_ids"]]
    seat_count = scheduled["table_size"]
    if game_count is None:
        game_count = seat_count
    games = []
    for offset in range(game_count):
        chips = [0] * seat_count
        chips[0] = winner_chips
        chips[1] = -winner_chips
        games.append({
            "chips": chips,
            "num_hands": deals,
            "verdicts": ["OK"] * seat_count,
            "training": {"teacher_counts": {participants[0]["id"]: offset + 1}},
        })
    return {**deepcopy(scheduled), "agents": participants, "games": games}


class StudyScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_with_unique_deck_seeds(self):
        identifiers = [agent["id"] for agent in fake_agents(7)]
        arguments = (identifiers, [4, 5, 6], ["seed-a", "seed-b"])
        first = study.build_schedule(*arguments, batches=2)
        second = study.build_schedule(*arguments, batches=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7 * 3 * 2 * 2)
        self.assertEqual(len({table["seed"] for table in first}), len(first))
        self.assertEqual({table["seed_group"] for table in first}, {"seed-a", "seed-b"})
        self.assertTrue(all(table["seed"] != table["seed_group"] for table in first))
        self.assertEqual(len({table["table"] for table in first}), len(first))

    def test_each_bot_has_equal_exposure_at_each_size_in_each_batch(self):
        identifiers = [agent["id"] for agent in fake_agents(7)]
        schedule = study.build_schedule(identifiers, [4, 5, 6], ["first", "second"], batches=2)
        for seed in ("first", "second"):
            for batch in (1, 2):
                for table_size in (4, 5, 6):
                    tables = [
                        table for table in schedule
                        if table["seed_group"] == seed and table["batch"] == batch
                        and table["table_size"] == table_size
                    ]
                    counts = Counter()
                    self.assertEqual(len(tables), 7)
                    for table in tables:
                        self.assertEqual(len(table["agent_ids"]), table_size)
                        self.assertEqual(len(set(table["agent_ids"])), table_size)
                        counts.update(table["agent_ids"])
                    self.assertEqual(counts, Counter({identifier: table_size for identifier in identifiers}))

    def test_exact_roster_has_one_full_lineup_and_all_seat_games(self):
        for table_size in (4, 5, 6):
            identifiers = [agent["id"] for agent in fake_agents(table_size)]
            plan = study.build_plan(identifiers, [table_size], ["full-roster"], deals=3)
            self.assertEqual(len(plan["schedule"]), 1)
            self.assertCountEqual(plan["schedule"][0]["agent_ids"], identifiers)
            self.assertEqual(plan["total_games"], table_size)
            self.assertEqual(plan["total_hands"], table_size * 3)
            for row in plan["coverage"]:
                self.assertEqual(row["sets"], 1)
                self.assertEqual(row["games"], table_size)
                self.assertEqual(row["hands"], table_size * 3)

    def test_plan_counts_physical_games_and_each_bot_separately(self):
        _, _, plan = example_run(7, [4, 5, 6], ["one", "two"], batches=2, deals=3)
        self.assertEqual(plan["total_tables"], 84)
        self.assertEqual(plan["total_games"], 420)
        self.assertEqual(plan["total_hands"], 1260)
        self.assertEqual(len(plan["coverage"]), 21)
        for row in plan["coverage"]:
            size = row["table_size"]
            self.assertEqual(row["sets"], size * 4)
            self.assertEqual(row["games"], size * size * 4)
            self.assertEqual(row["hands"], size * size * 4 * 3)

    def test_invalid_schedule_inputs_are_rejected(self):
        identifiers = [agent["id"] for agent in fake_agents(6)]
        invalid_arguments = [
            (identifiers[:3], [4], ["seed"], 1),
            (identifiers[:4], [6], ["seed"], 1),
            (identifiers + [identifiers[0]], [4], ["seed"], 1),
            (identifiers, [], ["seed"], 1),
            (identifiers, [3], ["seed"], 1),
            (identifiers, [4, 4], ["seed"], 1),
            (identifiers, ["4"], ["seed"], 1),
            (identifiers, [4], [], 1),
            (identifiers, [4], ["seed", "seed"], 1),
            (identifiers, [4], ["seed"], 0),
            (identifiers, [4], ["seed"], True),
        ]
        for agents, sizes, seeds, batches in invalid_arguments:
            with self.subTest(agents=agents, sizes=sizes, seeds=seeds, batches=batches):
                with self.assertRaises(ValueError):
                    study.build_schedule(agents, sizes, seeds, batches=batches)


class StudySummaryTests(unittest.TestCase):
    def test_empty_summary_has_no_rate_or_completed_work(self):
        run, agents, plan = example_run()
        summary = study.summarize(run, agents)
        self.assertEqual(summary["completed_games"], 0)
        self.assertEqual(summary["completed_tables"], 0)
        self.assertEqual(summary["completed_hands"], 0)
        self.assertEqual(summary["total_games"], plan["total_games"])
        self.assertEqual(len(summary["rows"]), 4)
        for row in summary["rows"]:
            self.assertEqual(row["hands"], 0)
            self.assertIsNone(row["bb_per_100"])

    def test_rate_uses_actual_hands_and_failed_games_still_count(self):
        run, agents, _ = example_run()
        table = completed_table(run["schedule"][0], agents)
        table["games"][-1]["verdicts"][0] = "RTE"
        table["games"][-1]["training"]["teacher_counts"] = {}
        run["results"] = [table]
        summary = study.summarize(run, agents)
        rows = {row["id"]: row for row in summary["rows"]}
        winner = rows[table["agents"][0]["id"]]
        self.assertEqual(winner["chips"], 40)
        self.assertEqual(winner["hands"], 20)
        self.assertEqual(winner["games"], 4)
        self.assertEqual(winner["completed_sets"], 1)
        self.assertEqual(winner["failures"], 1)
        self.assertEqual(winner["decisions"], 6)
        self.assertAlmostEqual(winner["bb_per_100"], 100)
        self.assertEqual(summary["completed_games"], 4)
        self.assertEqual(summary["completed_hands"], 20)
        self.assertEqual(sum(row["chips"] for row in summary["rows"]), 0)

    def test_partial_tables_count_games_but_not_sets_or_finished_batches(self):
        run, agents, _ = example_run()
        run["results"] = [completed_table(run["schedule"][0], agents, game_count=2)]
        summary = study.summarize(run, agents)
        self.assertEqual(summary["completed_games"], 2)
        self.assertEqual(summary["completed_tables"], 0)
        for row in summary["rows"]:
            self.assertEqual(row["games"], 2)
            self.assertEqual(row["completed_sets"], 0)
            self.assertEqual(row["completed_batches"], 0)
            self.assertIsNone(row["batch_min_bb_per_100"])
            self.assertIsNone(row["batch_max_bb_per_100"])

    def test_batch_ranges_exclude_partial_batches_and_summary_does_not_accumulate(self):
        run, agents, _ = example_run(batches=3)
        first = completed_table(run["schedule"][0], agents, winner_chips=10)
        second = completed_table(run["schedule"][1], agents, winner_chips=-5)
        # Fix the recipient of each game's positive/negative chips by identity,
        # because the scheduler can change player order in the next batch.
        target_id = first["agents"][0]["id"]
        opposite_id = first["agents"][1]["id"]
        for game in second["games"]:
            game["chips"] = [
                -5 if agent["id"] == target_id else 5 if agent["id"] == opposite_id else 0
                for agent in second["agents"]
            ]
        partial = completed_table(run["schedule"][2], agents, game_count=1, winner_chips=1000)
        run["results"] = [first, second, partial]
        summary = study.summarize(run, agents)
        self.assertEqual(summary, study.summarize(run, agents))
        row = next(row for row in summary["rows"] if row["id"] == target_id)
        self.assertEqual(row["completed_batches"], 2)
        self.assertAlmostEqual(row["batch_min_bb_per_100"], -50)
        self.assertAlmostEqual(row["batch_max_bb_per_100"], 100)
        self.assertEqual(row["games"], 9)
        self.assertEqual(row["hands"], 45)

    def test_each_size_finishes_batches_independently_and_rates_are_not_pooled(self):
        run, agents, _ = example_run(6, [4, 6], batches=2)
        for scheduled in run["schedule"]:
            if scheduled["table_size"] == 4:
                run["results"].append(completed_table(scheduled, agents))
            else:
                run["results"].append(completed_table(scheduled, agents, game_count=1))
        summary = study.summarize(run, agents)
        self.assertEqual(len(summary["rows"]), 12)
        for row in summary["rows"]:
            if row["table_size"] == 4:
                self.assertEqual(row["games"], 32)
                self.assertEqual(row["completed_batches"], 2)
                self.assertIsNotNone(row["batch_min_bb_per_100"])
            else:
                self.assertEqual(row["games"], 2)
                self.assertEqual(row["completed_batches"], 0)
                self.assertIsNone(row["batch_min_bb_per_100"])
            self.assertAlmostEqual(row["bb_per_100"], row["chips"] / 2 / row["hands"] * 100)


class StudyServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_data = server.DATA
        server.DATA = Path(self.temporary_directory.name)
        server.CANCEL.clear()
        server.PAUSE.clear()
        server.STOP.clear()
        server.init()
        self.agents = []
        source = "from macpoker import Bot\nclass CallingBot(Bot):\n    def act(self, state):\n        return state.call()\n"
        for number in range(6):
            self.agents.append(server.register({
                "name": f"Study bot {number}", "filename": "main.py",
                "content": base64.b64encode(source.encode()).decode(),
            }))

    def tearDown(self):
        server.DATA = self.previous_data
        server.CANCEL.clear()
        server.PAUSE.clear()
        server.STOP.clear()
        self.temporary_directory.cleanup()

    def queue_study(self, size=4, **overrides):
        body = {
            "mode": "study", "agents": [agent["id"] for agent in self.agents[:size]],
            "table_sizes": [size], "seeds": ["study-server-test"], "deals": 1,
        }
        body.update(overrides)
        return server.queue_run(body)

    def game_result(self, agents, deals=1):
        return {
            "chips": [10, -10] + [0] * (len(agents) - 2),
            "num_hands": deals, "verdicts": ["OK"] * len(agents),
        }

    def test_queue_saves_schedule_and_rejects_invalid_study_settings(self):
        run = self.queue_study(6, table_sizes=[4, 5, 6], batches=2)
        saved = server.get("runs", run["id"])
        self.assertEqual(saved["schedule"], run["schedule"])
        self.assertEqual(saved["study"]["completed_games"], 0)
        self.assertEqual(saved["study"]["total_games"], 120)
        self.assertEqual(saved["config"]["batches"], 2)
        for change in (
            {"table_sizes": [6]}, {"table_sizes": []}, {"table_sizes": [4, 4]},
            {"table_sizes": [3]}, {"table_sizes": ["4"]}, {"table_sizes": "4"},
            {"batches": 0}, {"batches": 21}, {"batches": True}, {"batches": 1.5},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.queue_study(4, **change)

    def test_pause_finishes_current_game_and_resume_keeps_completed_offsets(self):
        run = self.queue_study()
        run["status"] = "running"
        server.save("runs", run)
        offsets = []

        def play_and_pause(current_run, agents, seed, offset, number, seed_group=None):
            offsets.append(offset)
            self.assertEqual(seed_group, "study-server-test")
            server.pause_run(current_run["id"])
            return self.game_result(agents)

        with mock.patch.object(server, "game", side_effect=play_and_pause):
            server.execute(run)
        paused = server.get("runs", run["id"])
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["study"]["completed_games"], 1)
        self.assertEqual(len(paused["results"][0]["games"]), 1)
        saved_schedule = deepcopy(paused["schedule"])

        def play_remaining(current_run, agents, seed, offset, number, seed_group=None):
            stored = server.get("runs", current_run["id"])
            # Every previous game is durable before the next one starts.
            self.assertEqual(stored["study"]["completed_games"], len(offsets))
            offsets.append(offset)
            return self.game_result(agents)

        resumed = server.resume_run(run["id"])
        self.assertEqual(resumed["status"], "queued")
        self.assertEqual(resumed["schedule"], saved_schedule)
        resumed["status"] = "running"
        server.save("runs", resumed)
        with mock.patch.object(server, "game", side_effect=play_remaining):
            server.execute(resumed)
        completed = server.get("runs", run["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(offsets, [0, 1, 2, 3])
        self.assertEqual(completed["study"]["completed_games"], 4)
        self.assertTrue(all(row["games"] == 4 for row in completed["study"]["rows"]))

    def test_pause_cannot_overwrite_a_concurrent_worker_completion(self):
        run = self.queue_study()
        run["status"] = "running"
        server.save("runs", run)
        finished_run = deepcopy(run)
        finished_run.update(status="completed", progress="Complete", finished=server.now())

        pause_has_read = threading.Event()
        worker_attempted_save = threading.Event()
        worker_finished_save = threading.Event()
        thread_errors = []
        original_get = server.get

        def pause_read(table, key):
            value = original_get(table, key)
            if threading.current_thread().name == "study-pause-regression":
                # Hold the request's lock after reading its old RUNNING snapshot
                # while the worker attempts to persist a newer COMPLETED state.
                pause_has_read.set()
                if not worker_attempted_save.wait(timeout=2):
                    raise AssertionError("Worker did not attempt its final save")
                worker_finished_save.wait(timeout=0.25)
            return value

        def pause_request():
            try:
                # Match Handler.do_POST's read/modify/write transaction.
                with server.LOCK:
                    server.pause_run(run["id"])
            except BaseException as error:
                thread_errors.append(error)

        def worker_completion():
            try:
                if not pause_has_read.wait(timeout=2):
                    raise AssertionError("Pause request did not read the run")
                worker_attempted_save.set()
                server.save("runs", finished_run)
                worker_finished_save.set()
            except BaseException as error:
                thread_errors.append(error)

        pause_thread = threading.Thread(target=pause_request, name="study-pause-regression", daemon=True)
        worker_thread = threading.Thread(target=worker_completion, daemon=True)
        with mock.patch.object(server, "get", side_effect=pause_read):
            pause_thread.start()
            worker_thread.start()
            pause_thread.join(timeout=3)
            worker_thread.join(timeout=3)

        self.assertFalse(pause_thread.is_alive(), "Pause transaction did not finish")
        self.assertFalse(worker_thread.is_alive(), "Worker final save did not finish")
        self.assertEqual(thread_errors, [])
        self.assertTrue(worker_finished_save.is_set())
        self.assertEqual(server.get("runs", run["id"])["status"], "completed")

    def test_restart_after_exception_resumes_from_last_saved_game(self):
        run = self.queue_study()
        run["status"] = "running"
        server.save("runs", run)
        calls = []

        def interrupt_second_game(current_run, agents, seed, offset, number, seed_group=None):
            calls.append(offset)
            if offset == 1:
                raise RuntimeError("simulated worker interruption")
            return self.game_result(agents)

        with mock.patch.object(server, "game", side_effect=interrupt_second_game):
            with self.assertRaisesRegex(RuntimeError, "simulated worker interruption"):
                server.execute(run)
        server.init()
        interrupted = server.get("runs", run["id"])
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual(interrupted["study"]["completed_games"], 1)

        def complete(current_run, agents, seed, offset, number, seed_group=None):
            calls.append(offset)
            return self.game_result(agents)

        resumed = server.resume_run(run["id"])
        resumed["status"] = "running"
        server.save("runs", resumed)
        with mock.patch.object(server, "game", side_effect=complete):
            server.execute(resumed)
        self.assertEqual(calls, [0, 1, 1, 2, 3])
        self.assertEqual(server.get("runs", run["id"])["study"]["completed_games"], 4)

    def test_queued_pause_and_supported_resume_statuses_preserve_results(self):
        run = self.queue_study()
        paused = server.pause_run(run["id"])
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["results"], [])
        for status in ("paused", "interrupted", "cancelled", "failed"):
            paused["status"] = status
            paused["error"] = "previous interruption"
            server.save("runs", paused)
            server.CANCEL.add(paused["id"])
            server.PAUSE.add(paused["id"])
            resumed = server.resume_run(paused["id"])
            self.assertEqual(resumed["status"], "queued")
            self.assertEqual(resumed["schedule"], paused["schedule"])
            self.assertEqual(resumed["results"], paused["results"])
            self.assertNotIn(paused["id"], server.CANCEL)
            self.assertNotIn(paused["id"], server.PAUSE)

    def test_real_recorded_studies_run_at_four_five_and_six_seats(self):
        for table_size in (4, 5, 6):
            teacher = self.agents[0]
            run = self.queue_study(
                table_size, record_training=True, teachers=[teacher["id"]],
            )
            server.execute(run)
            completed = server.get("runs", run["id"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["study"]["completed_games"], table_size)
            self.assertEqual(completed["study"]["completed_hands"], table_size)
            games = completed["results"][0]["games"]
            self.assertEqual(len(games), table_size)
            self.assertTrue(all(game["verdicts"] == ["OK"] * table_size for game in games))
            self.assertTrue(all(sum(game["chips"]) == 0 for game in games))
            self.assertGreater(completed["training"]["records"], 0)

            archive_path = server.training_archive(run["id"], table_size=table_size)
            with zipfile.ZipFile(archive_path) as archive:
                record_count = 0
                for filename in archive.namelist():
                    if not filename.endswith(".jsonl"):
                        continue
                    for line in archive.read(filename).decode().splitlines():
                        record = json.loads(line)
                        self.assertEqual(record["table_size"], table_size)
                        self.assertEqual(record["seed_group"], "study-server-test")
                        self.assertEqual(record["teacher"]["id"], teacher["id"])
                        self.assertEqual(record["verdict"], "OK")
                        self.assertEqual(len(record["observation"]["stacks"]), table_size)
                        record_count += 1
                self.assertEqual(record_count, completed["training"]["records"])

    def test_completed_worker_file_is_recovered_without_replaying_the_game(self):
        teacher = self.agents[0]
        run = self.queue_study(record_training=True, teachers=[teacher["id"]])
        planned = run["schedule"][0]
        registered = {agent["id"]: agent for agent in self.agents}
        participants = [registered[identifier] for identifier in planned["agent_ids"]]
        arguments = (run, participants, planned["seed"], 0, 1)
        first = server.game(*arguments, seed_group=planned["seed_group"])
        self.assertGreater(first["training"]["records"], 0)

        folder = server.DATA / "runs" / run["id"]
        stale_work = folder / "work-1"
        (stale_work / "leftover").mkdir(parents=True)
        with mock.patch.object(server.subprocess, "Popen", side_effect=AssertionError("Completed game must not replay")):
            recovered = server.game(*arguments, seed_group=planned["seed_group"])
        self.assertEqual(recovered, first)
        self.assertFalse(stale_work.exists())

        # A result without its recorded examples is not a complete checkpoint.
        dataset = folder / first["training"]["file"]
        dataset.unlink()
        real_popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen", wraps=real_popen) as start_worker:
            repaired = server.game(*arguments, seed_group=planned["seed_group"])
        start_worker.assert_called_once()
        self.assertTrue(dataset.is_file())
        self.assertEqual(repaired["training"]["records"], first["training"]["records"])
        self.assertEqual(repaired["chips"], first["chips"])


if __name__ == "__main__":
    unittest.main()
