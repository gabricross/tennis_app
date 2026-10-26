from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_database = storage.DATABASE
        storage.DATABASE = Path(self.temporary.name) / "test.sqlite3"
        storage.initialize()
        self.clip_id = str(uuid.uuid4())
        storage.create_clip(self.clip_id, "Mi vídeo.mp4", {"duration": 10})

    def tearDown(self):
        storage.DATABASE = self.previous_database
        self.temporary.cleanup()

    def test_create_and_reopen_job(self):
        job = storage.create_job(self.clip_id, {"mode": "players", "start": 0})
        loaded = storage.get_job(job["id"])
        self.assertEqual(loaded["clip_id"], self.clip_id)
        self.assertEqual(loaded["config"]["mode"], "players")
        storage.update_job(job["id"], status="completed", summary={"coverage": 0.5})
        self.assertEqual(storage.get_job(job["id"])["summary"], {"coverage": 0.5})

    def test_restart_marks_unfinished_jobs_interrupted(self):
        pending = storage.create_job(self.clip_id, {"mode": "players"})
        finished = storage.create_job(self.clip_id, {"mode": "players"})
        storage.update_job(finished["id"], status="completed", summary={"ok": True})
        storage.initialize()
        self.assertEqual(storage.get_job(pending["id"])["status"], "interrupted")
        self.assertEqual(storage.get_job(finished["id"])["status"], "completed")

    def test_clip_deletion_removes_job_records(self):
        job = storage.create_job(self.clip_id, {"mode": "players"})
        storage.remove_clip(self.clip_id)
        self.assertIsNone(storage.get_clip(self.clip_id))
        self.assertIsNone(storage.get_job(job["id"]))

    def test_resource_paths_reject_traversal(self):
        with self.assertRaises(ValueError):
            storage.resource_dir("../models")
        with self.assertRaises(ValueError):
            storage.job_dir(self.clip_id, "../../secrets")


if __name__ == "__main__":
    unittest.main()
