"""Dataset collection, checkpoint and immutable export integration checks."""

import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from arena import datasets, server


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_data = server.DATA
        server.DATA = Path(self.temporary_directory.name)
        server.STOP.clear()
        server.PAUSE.clear()
        server.CANCEL.clear()
        server.init()
        source = "from macpoker import Bot\nclass Caller(Bot):\n    def act(self, state):\n        return state.call()\n"
        self.agents = []
        for number in range(6):
            self.agents.append(server.register({
                "name": f"Dataset bot {number}", "filename": "main.py",
                "content": base64.b64encode(source.encode()).decode(),
            }))
        self.calls = []

    def tearDown(self):
        server.DATA = self.previous_data
        server.STOP.clear()
        server.PAUSE.clear()
        server.CANCEL.clear()
        self.temporary_directory.cleanup()

    def body(self, **changes):
        identifiers = [agent["id"] for agent in self.agents[:4]]
        body = {
            "name": "Small dataset", "agents": identifiers, "teachers": identifiers,
            "table_sizes": [4], "deals": 1, "target_records": 1,
            "minimum_seed_groups": 3, "seed_prefix": "dataset-test", "max_disk_gb": 1,
        }
        body.update(changes)
        return body

    def episode_body(self, target_episodes=1, **changes):
        body = self.body(**changes)
        body.pop("target_records")
        body["target_episodes"] = target_episodes
        return body

    def fake_game(self, run, agents, seed, offset, number, seed_group=None):
        """Write one valid example per selected teacher without starting bots."""
        self.calls.append((seed_group, seed, offset, number))
        folder = server.DATA / "runs" / run["id"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"game-{number}.decisions.jsonl"
        counts = {}
        with path.open("w", encoding="utf-8") as destination:
            for player, agent in enumerate(agents):
                if agent["id"] not in run["config"]["teachers"]:
                    continue
                counts[agent["id"]] = 1
                record = {
                    "schema_version": 1, "run_id": run["id"], "game_id": number,
                    "seed_group": seed_group, "game_seed": seed, "offset": offset,
                    "decision_id": f"{run['id']}:{number}:0:{player}",
                    "teacher": {"player": player, **{key: agent[key] for key in ("id", "name", "hash")}},
                    "table_size": len(agents), "verdict": "OK",
                    "observation": {"street": "preflop", "hole": ["As", "Kd"], "board": [],
                                    "history": [], "stacks": [200] * len(agents), "seat": player},
                    "requested_action": {"action": "call"},
                    "executed_action": {"action": "call", "amount": 2},
                }
                destination.write(json.dumps(record) + "\n")
        return {
            "chips": [0] * len(agents), "num_hands": 1, "verdicts": ["OK"] * len(agents),
            "training": {"file": path.name, "records": sum(counts.values()),
                         "teacher_counts": counts, "excluded_records": 0, "failed_teachers": []},
        }

    def finish_fake(self, job):
        with mock.patch.object(server, "game", side_effect=self.fake_game):
            datasets.execute(server, job)
        return server.get("datasets", job["id"])

    def read_records(self, job):
        records = []
        folder = server.DATA / "datasets" / job["id"]
        for shard in datasets.shards(server, job["id"]):
            with gzip.open(folder / shard["name"], "rt", encoding="utf-8") as source:
                records.extend(json.loads(line) for line in source)
        return records

    def test_invalid_collection_settings_leave_no_saved_job(self):
        invalid_settings = [
            {"name": ""}, {"target_records": 0}, {"target_records": True},
            {"target_records": 1.5}, {"target_records": 100000001},
            {"minimum_seed_groups": 2}, {"minimum_seed_groups": True},
            {"minimum_seed_groups": 1001}, {"max_disk_gb": 0},
            {"max_disk_gb": float("nan")}, {"max_disk_gb": float("inf")},
            {"max_disk_gb": True}, {"teachers": []},
            {"teachers": [{}]}, {"teachers": [[self.agents[0]["id"]]]},
            {"teachers": [self.agents[5]["id"]]},
            {"teachers": [self.agents[0]["id"], self.agents[0]["id"]]},
            {"agents": [agent["id"] for agent in self.agents[:3]]},
            {"table_sizes": [6]}, {"table_sizes": []}, {"deals": 0},
            {"seed_prefix": " "},
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    datasets.queue(server, self.body(**settings))
                self.assertEqual(server.all_rows("datasets"), [])

    def test_collection_finishes_balanced_blocks_and_uses_fresh_seed_groups(self):
        first = self.finish_fake(datasets.queue(server, self.body()))
        second = self.finish_fake(datasets.queue(server, self.body()))
        for job in (first, second):
            self.assertEqual(job["status"], "ready")
            self.assertEqual(job["games"], 12)
            self.assertEqual(job["completed_blocks"], 3)
            self.assertEqual(job["block_game"], 0)
            self.assertEqual(job["records"], 48)
            self.assertEqual(len(job["seed_groups"]), 3)
            self.assertEqual(set(job["teacher_counts"].values()), {12})
            self.assertEqual(job["table_size_counts"], {"4": 48})
        self.assertTrue(set(first["seed_groups"]).isdisjoint(second["seed_groups"]))

    def test_episode_target_is_the_default_and_invalid_or_mixed_targets_are_rejected(self):
        for target in (0, -1, True, 1.5, "100", None, 100000001):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    datasets.queue(server, self.episode_body(target))
                self.assertEqual(server.all_rows("datasets"), [])
        with self.assertRaises(ValueError):
            datasets.queue(server, self.body(target_episodes=10))
        self.assertEqual(server.all_rows("datasets"), [])

        body = self.body()
        body.pop("target_records")
        job = datasets.queue(server, body)
        self.assertEqual(job["target_unit"], "episodes")
        self.assertEqual(job["target_episodes"], 100000)
        self.assertIsNone(job.get("target_records"))
        self.assertEqual(job["episodes"], 0)

    def test_episode_goal_uses_actual_hand_counts_and_finishes_a_balanced_block(self):
        teacher = self.agents[0]["id"]
        job = datasets.queue(server, self.episode_body(37, deals=50, teachers=[teacher]))

        def play_three_hands(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            result["num_hands"] = 3
            return result

        with mock.patch.object(server, "game", side_effect=play_three_hands):
            datasets.execute(server, job)
        ready = server.get("datasets", job["id"])
        # Each four-game block contributes 12 physical table hands. There is
        # only one saved decision per game in this fixture.
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["episodes"], 48)
        self.assertEqual(ready["records"], 16)
        self.assertEqual(ready["games"], 16)
        self.assertEqual(ready["completed_blocks"], 4)
        self.assertGreater(ready["episodes_per_second"], 0)
        self.assertEqual(ready["target_unit"], "episodes")
        self.assertEqual(ready["target_episodes"], 37)
        self.assertEqual(sum(shard["episodes"] for shard in datasets.shards(server, job["id"])), 48)

        manifest_path = server.DATA / "datasets" / job["id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["episodes"], 48)
        self.assertEqual(manifest["records"], 16)
        self.assertEqual(manifest["target_unit"], "episodes")
        self.assertEqual(manifest["target_episodes"], 37)
        self.assertIsNone(manifest.get("target_records"))

    def test_episode_target_pause_resume_preserves_completed_hands_once(self):
        job = datasets.queue(server, self.episode_body(20, deals=2))

        def play_two_hands(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            result["num_hands"] = 2
            if len(self.calls) == 1:
                datasets.control(server, job["id"], "pause")
            return result

        with mock.patch.object(server, "game", side_effect=play_two_hands):
            datasets.execute(server, job)
            paused = server.get("datasets", job["id"])
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["episodes"], 2)
            self.assertEqual(paused["records"], 4)
            datasets.execute(server, datasets.control(server, job["id"], "resume"))
        ready = server.get("datasets", job["id"])
        self.assertEqual(ready["episodes"], 24)
        self.assertEqual(ready["records"], 48)
        self.assertEqual(ready["games"], 12)
        self.assertEqual([call[3] for call in self.calls], list(range(1, 13)))

    def test_invalid_game_hand_counts_do_not_commit_records_or_episodes(self):
        for hand_count in (-1, True, 1.5):
            with self.subTest(hand_count=hand_count):
                job = datasets.queue(server, self.episode_body())

                def invalid_game(*arguments, **keywords):
                    result = self.fake_game(*arguments, **keywords)
                    result["num_hands"] = hand_count
                    return result

                with mock.patch.object(server, "game", side_effect=invalid_game):
                    with self.assertRaisesRegex(ValueError, "hand count"):
                        datasets.execute(server, job)
                saved = server.get("datasets", job["id"])
                self.assertEqual(saved["episodes"], 0)
                self.assertEqual(saved["records"], 0)
                self.assertEqual(saved["games"], 0)
                self.assertEqual(datasets.shards(server, job["id"]), [])

    def test_legacy_collection_resumes_with_inferred_hands_and_original_decision_goal(self):
        teacher = self.agents[0]["id"]
        job = datasets.queue(server, self.body(target_records=13, deals=3, teachers=[teacher]))

        def play_three_hands(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            result["num_hands"] = 3
            if len(self.calls) == 1:
                datasets.control(server, job["id"], "pause")
            return result

        with mock.patch.object(server, "game", side_effect=play_three_hands):
            datasets.execute(server, job)
        legacy = server.get("datasets", job["id"])
        for field in ("episodes", "episodes_per_second", "target_unit", "target_episodes"):
            legacy.pop(field, None)
        server.save("datasets", legacy)
        self.assertEqual(datasets.detail(server, job["id"])["episodes"], 3)
        self.assertEqual(server.get("datasets", job["id"]), legacy)

        with mock.patch.object(server, "game", side_effect=play_three_hands):
            datasets.execute(server, datasets.control(server, job["id"], "resume"))
        ready = server.get("datasets", job["id"])
        self.assertEqual(ready["target_unit"], "decisions")
        self.assertEqual(ready["target_records"], 13)
        self.assertEqual(ready["records"], 16)
        self.assertEqual(ready["episodes"], 48)
        self.assertEqual(ready["games"], 16)

    def test_table_size_and_teacher_counts_match_the_saved_records(self):
        teacher = self.agents[0]["id"]
        job = datasets.queue(server, self.body(
            agents=[agent["id"] for agent in self.agents], teachers=[teacher], table_sizes=[4, 6],
        ))
        ready = self.finish_fake(job)
        records = self.read_records(ready)
        self.assertEqual({record["teacher"]["id"] for record in records}, {teacher})
        self.assertEqual(len(records), ready["records"])
        self.assertEqual(ready["teacher_counts"], {teacher: len(records)})
        expected_sizes = dict(Counter(str(record["table_size"]) for record in records))
        self.assertEqual(ready["table_size_counts"], expected_sizes)
        self.assertEqual(ready["teacher_size_counts"], {teacher: expected_sizes})
        self.assertEqual(expected_sizes, {"4": 48, "6": 18})

    def test_pause_after_a_game_and_resume_never_counts_it_twice(self):
        job = datasets.queue(server, self.body())

        def play_then_pause(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            datasets.control(server, job["id"], "pause")
            return result

        with mock.patch.object(server, "game", side_effect=play_then_pause):
            datasets.execute(server, job)
        paused = server.get("datasets", job["id"])
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["games"], 1)
        self.assertEqual(paused["records"], 4)
        self.assertEqual(paused["block_game"], 1)

        resumed = datasets.control(server, job["id"], "resume")
        ready = self.finish_fake(resumed)
        self.assertEqual(ready["status"], "ready")
        self.assertEqual([call[3] for call in self.calls], list(range(1, 13)))
        self.assertEqual(len(self.read_records(ready)), 48)
        self.assertEqual(len(datasets.shards(server, ready["id"])), 12)

    def test_finish_request_waits_for_all_seats_in_the_current_block(self):
        job = datasets.queue(server, self.body(target_records=1000000))

        def play_then_finish(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            if len(self.calls) == 1:
                datasets.control(server, job["id"], "finish")
            return result

        with mock.patch.object(server, "game", side_effect=play_then_finish):
            datasets.execute(server, job)
        ready = server.get("datasets", job["id"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["games"], 4)
        self.assertEqual([call[2] for call in self.calls], [0, 1, 2, 3])
        self.assertEqual(ready["completed_blocks"], 1)
        self.assertEqual(ready["records"], 16)

    def test_restart_after_compression_before_commit_recovers_exact_counts(self):
        job = datasets.queue(server, self.body())
        with mock.patch.object(server, "game", side_effect=self.fake_game):
            with mock.patch.object(datasets, "commit_shard", side_effect=RuntimeError("interrupted before commit")):
                with self.assertRaisesRegex(RuntimeError, "before commit"):
                    datasets.execute(server, job)
        saved = server.get("datasets", job["id"])
        self.assertEqual(saved["games"], 0)
        self.assertEqual(saved["records"], 0)
        self.assertEqual(saved["episodes"], 0)
        self.assertEqual(datasets.shards(server, job["id"]), [])
        folder = server.DATA / "datasets" / job["id"]
        self.assertTrue((folder / "part-00000001.jsonl.gz").is_file())
        self.assertTrue((server.DATA / "runs" / job["id"] / "game-1.decisions.jsonl").is_file())

        server.init()
        self.assertEqual(server.get("datasets", job["id"])["status"], "interrupted")
        ready = self.finish_fake(datasets.control(server, job["id"], "resume"))
        records = self.read_records(ready)
        self.assertEqual(ready["records"], 48)
        self.assertEqual(ready["episodes"], 12)
        self.assertEqual(len(records), 48)
        self.assertEqual(len({record["decision_id"] for record in records}), 48)
        self.assertEqual(len(datasets.shards(server, job["id"])), 12)

    def test_restart_after_commit_keeps_the_completed_game(self):
        job = datasets.queue(server, self.body())
        with mock.patch.object(server, "game", side_effect=self.fake_game):
            with mock.patch.object(datasets, "clean_committed_game", side_effect=RuntimeError("interrupted after commit")):
                with self.assertRaisesRegex(RuntimeError, "after commit"):
                    datasets.execute(server, job)
        saved = server.get("datasets", job["id"])
        self.assertEqual(saved["games"], 1)
        self.assertEqual(saved["records"], 4)
        self.assertEqual(saved["episodes"], 1)
        self.assertEqual(len(datasets.shards(server, job["id"])), 1)
        server.init()
        ready = self.finish_fake(datasets.control(server, job["id"], "resume"))
        self.assertEqual([call[3] for call in self.calls], list(range(1, 13)))
        self.assertEqual(ready["records"], 48)
        self.assertEqual(ready["episodes"], 12)
        self.assertEqual(list((server.DATA / "runs" / job["id"]).glob("game-*")), [])

    def test_real_worker_checkpoint_is_recovered_without_replaying_the_game(self):
        job = datasets.queue(server, self.body())
        with mock.patch.object(datasets, "commit_shard", side_effect=RuntimeError("interrupted before commit")):
            with self.assertRaisesRegex(RuntimeError, "before commit"):
                datasets.execute(server, job)
        server.init()
        # Finish one full block so recovery needs the cached first game and
        # exactly three new seat rotations.
        resumed = datasets.control(server, job["id"], "finish")
        real_popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen", wraps=real_popen) as start_worker:
            datasets.execute(server, resumed)
        self.assertEqual(start_worker.call_count, 3)
        ready = server.get("datasets", job["id"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["games"], 4)
        records = self.read_records(ready)
        self.assertEqual(len(records), ready["records"])
        self.assertEqual(len({record["decision_id"] for record in records}), len(records))
        self.assertEqual(list((server.DATA / "runs" / job["id"]).glob("game-*")), [])

    def test_real_engine_collection_contains_only_teacher_observations(self):
        teacher = self.agents[0]
        job = datasets.queue(server, self.episode_body(1, teachers=[teacher["id"]]))
        datasets.execute(server, job)
        ready = server.get("datasets", job["id"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["failures"], 0)
        self.assertEqual(ready["episodes"], 12)
        self.assertEqual(len(ready["seed_groups"]), 3)
        records = self.read_records(ready)
        self.assertGreater(len(records), 0)
        self.assertEqual(len(records), ready["records"])
        self.assertEqual({record["teacher"]["id"] for record in records}, {teacher["id"]})
        for record in records:
            self.assertEqual(record["verdict"], "OK")
            self.assertEqual(record["table_size"], 4)
            self.assertEqual(len(record["observation"]["hole"]), 2)
            self.assertNotIn("holes", record["observation"])
            self.assertNotIn("deltas", record["observation"])
            self.assertEqual(record["teacher"]["hash"], teacher["hash"])
        self.assertEqual(len({record["decision_id"] for record in records}), len(records))
        self.assertEqual(list((server.DATA / "runs" / job["id"]).glob("game-*")), [])

    def test_manifest_archive_counts_hashes_and_files_agree(self):
        ready = self.finish_fake(datasets.queue(server, self.body()))
        archive_path = datasets.archive(server, ready["id"])
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["records"], ready["records"])
            self.assertEqual(manifest["seed_groups"], ready["seed_groups"])
            self.assertEqual(manifest["teachers"], ready["teachers"])
            count = 0
            total_bytes = 0
            for shard in manifest["files"]:
                raw = archive.read(shard["name"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), shard["sha256"])
                self.assertEqual(len(raw), shard["bytes"])
                lines = gzip.decompress(raw).decode().splitlines()
                self.assertEqual(len(lines), shard["records"])
                count += len(lines)
                total_bytes += len(raw)
            self.assertEqual(count, ready["records"])
            self.assertEqual(total_bytes, ready["bytes"])
            self.assertEqual(len(archive.namelist()), len(manifest["files"]) + 1)
        detail = datasets.detail(server, ready["id"])
        self.assertEqual(detail["seed_group_count"], 3)
        self.assertTrue(Path(detail["folder"]).is_dir())
        self.assertTrue(detail["download_url"].endswith("/download"))

    def test_archive_refuses_a_modified_shard(self):
        ready = self.finish_fake(datasets.queue(server, self.body()))
        shard = datasets.shards(server, ready["id"])[0]
        path = server.DATA / "datasets" / ready["id"] / shard["name"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            datasets.archive(server, ready["id"])

    def test_import_preserves_source_groups_and_refuses_duplicate_run(self):
        settings = self.body()
        run = server.queue_run({
            "mode": "study", "agents": settings["agents"], "teachers": settings["teachers"],
            "record_training": True, "table_sizes": [4], "seeds": ["original-seed"], "deals": 1,
        })
        with mock.patch.object(server, "game", side_effect=self.fake_game):
            server.execute(run)
        imported = datasets.import_run(server, {"name": "Imported study", "run_id": run["id"]})
        with mock.patch.object(server, "game", side_effect=AssertionError("Import must not replay games")):
            datasets.execute(server, imported)
        ready = server.get("datasets", imported["id"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["records"], 16)
        self.assertEqual(ready["episodes"], 4)
        self.assertEqual(ready["seed_groups"], ["original-seed"])
        records = self.read_records(ready)
        self.assertEqual({record["run_id"] for record in records}, {run["id"]})
        self.assertTrue((server.DATA / "runs" / run["id"] / "game-1.decisions.jsonl").is_file())
        with self.assertRaisesRegex(ValueError, "already"):
            datasets.import_run(server, {"name": "Duplicate", "run_id": run["id"]})

    def test_legacy_import_resumes_with_actual_source_hand_counts(self):
        settings = self.body()
        run = server.queue_run({
            "mode": "study", "agents": settings["agents"], "teachers": settings["teachers"],
            "record_training": True, "table_sizes": [4], "seeds": ["import-hand-counts"], "deals": 100,
        })
        hand_counts = [2, 3, 5, 7]

        def play_varied_hands(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            result["num_hands"] = hand_counts[len(self.calls) - 1]
            return result

        with mock.patch.object(server, "game", side_effect=play_varied_hands):
            server.execute(run)
        imported = datasets.import_run(server, {"name": "Hand counts", "run_id": run["id"]})
        real_commit = datasets.commit_shard

        def commit_then_pause(*arguments, **keywords):
            real_commit(*arguments, **keywords)
            if arguments[1]["games"] == 1:
                datasets.control(server, imported["id"], "pause")

        with mock.patch.object(datasets, "commit_shard", side_effect=commit_then_pause):
            datasets.execute(server, imported)
        paused = server.get("datasets", imported["id"])
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["episodes"], 2)
        self.assertEqual(paused["games"], 1)

        # Older imports have no per-job hand counter and no deals in config.
        # Recover only the source games already committed to the import.
        paused.pop("episodes")
        paused.pop("episodes_per_second", None)
        server.save("datasets", paused)
        self.assertEqual(datasets.detail(server, imported["id"])["episodes"], 2)
        self.assertEqual(server.get("datasets", imported["id"]), paused)
        datasets.execute(server, datasets.control(server, imported["id"], "resume"))
        ready = server.get("datasets", imported["id"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["episodes"], sum(hand_counts))
        self.assertEqual(ready["records"], 16)
        self.assertEqual(ready["games"], 4)
        manifest_path = server.DATA / "datasets" / imported["id"] / "manifest.json"
        self.assertEqual(json.loads(manifest_path.read_text())["episodes"], 17)

    def test_legacy_ready_import_remains_readable_when_source_run_is_missing(self):
        settings = self.body()
        run = server.queue_run({
            "mode": "study", "agents": settings["agents"], "teachers": settings["teachers"],
            "record_training": True, "table_sizes": [4], "seeds": ["old-import"], "deals": 1,
        })
        with mock.patch.object(server, "game", side_effect=self.fake_game):
            server.execute(run)
        imported = datasets.import_run(server, {"name": "Old import", "run_id": run["id"]})
        datasets.execute(server, imported)

        legacy = server.get("datasets", imported["id"])
        legacy.pop("episodes")
        legacy.pop("episodes_per_second", None)
        server.save("datasets", legacy)
        manifest_path = server.DATA / "datasets" / imported["id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("episodes", None)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        frozen_bytes = manifest_path.read_bytes()
        with server.connect() as database:
            database.execute("DELETE FROM runs WHERE id=?", (run["id"],))

        detail = datasets.detail(server, imported["id"])
        self.assertEqual(detail["status"], "ready")
        self.assertEqual(detail["records"], 16)
        self.assertIsNone(detail["episodes"])
        self.assertIsNone(detail["episodes_per_second"])
        library = server.lab_state()
        self.assertEqual(len(library["datasets"]), 1)
        self.assertEqual(library["datasets"][0]["id"], imported["id"])
        self.assertIsNone(library["datasets"][0]["episodes"])
        self.assertEqual(server.get("datasets", imported["id"]), legacy)
        self.assertEqual(manifest_path.read_bytes(), frozen_bytes)

    def test_import_rejects_an_unfinished_run(self):
        run = server.queue_run({
            "mode": "study", "agents": self.body()["agents"], "table_sizes": [4],
            "seeds": ["unfinished"], "deals": 1, "record_training": True,
            "teachers": self.body()["teachers"],
        })
        with self.assertRaisesRegex(ValueError, "completed"):
            datasets.import_run(server, {"name": "Too early", "run_id": run["id"]})
        self.assertEqual(server.all_rows("datasets"), [])

    def test_empty_teacher_batches_stop_instead_of_looping_forever(self):
        job = datasets.queue(server, self.body())

        def empty_game(*arguments, **keywords):
            result = self.fake_game(*arguments, **keywords)
            path = server.DATA / "runs" / job["id"] / result["training"]["file"]
            path.write_text("", encoding="utf-8")
            result["training"].update(records=0, teacher_counts={}, excluded_records=4)
            result["verdicts"] = ["RTE"] * 4
            return result

        with mock.patch.object(server, "game", side_effect=empty_game):
            with self.assertRaisesRegex(ValueError, "Three batches"):
                datasets.execute(server, job)
        saved = server.get("datasets", job["id"])
        self.assertEqual(saved["records"], 0)
        self.assertEqual(saved["games"], 12)
        self.assertEqual(saved["empty_blocks"], 3)
        self.assertEqual(saved["excluded_records"], 48)
        self.assertEqual(saved["failures"], 48)
        self.assertFalse((server.DATA / "datasets" / job["id"] / "manifest.json").exists())

    def test_pause_before_start_and_ready_dataset_controls(self):
        job = datasets.queue(server, self.body())
        paused = datasets.control(server, job["id"], "pause")
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["games"], 0)
        with self.assertRaisesRegex(ValueError, "Finish collection"):
            datasets.archive(server, job["id"])
        ready = self.finish_fake(datasets.control(server, job["id"], "resume"))
        for action in ("pause", "resume"):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    datasets.control(server, ready["id"], action)
        self.assertEqual(datasets.control(server, ready["id"], "finish"), ready)

    def test_resume_can_raise_storage_limit_and_rejects_invalid_limits_atomically(self):
        job = datasets.queue(server, self.body())
        paused = datasets.control(server, job["id"], "pause")
        for invalid_limit in (-1, 0.099, 1001, True, float("inf"), "10"):
            with self.subTest(limit=invalid_limit):
                with self.assertRaisesRegex(ValueError, "Storage limit"):
                    datasets.control(server, job["id"], "resume", {"max_disk_gb": invalid_limit})
                self.assertEqual(server.get("datasets", job["id"]), paused)

        resumed = datasets.control(server, job["id"], "resume", {"max_disk_gb": 2.5})
        self.assertEqual(resumed["status"], "queued")
        self.assertEqual(resumed["config"], {**paused["config"], "max_disk_gb": 2.5})
        self.assertEqual(resumed["records"], paused["records"])
        self.assertEqual(resumed["teacher_counts"], paused["teacher_counts"])
        self.assertEqual(server.get("datasets", job["id"]), resumed)


if __name__ == "__main__":
    unittest.main()
