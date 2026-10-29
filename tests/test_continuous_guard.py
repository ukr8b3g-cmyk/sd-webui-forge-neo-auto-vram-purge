"""Run the real extension with a virtual clock and Forge/accelerator stubs.

No server, browser, GPU, dependencies or real background workers are started.
"""
import asyncio
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "auto_vram_purge.py"


class Harness:
    def __init__(self):
        self.now = 100.0
        self.on_sleep = None
        self.sleeps = 0
        self.workers = []
        self.current = None
        self.unloads = []
        self.recoveries = []
        self.gc_calls = []
        self.lock_entries = 0
        self.on_queue_enter = None
        self.in_queue = False
        self.options = types.SimpleNamespace(data={})
        self.state = types.SimpleNamespace(job="")
        outer = self

        class Worker:
            def __init__(self, target, args, **kwargs):
                self.target, self.args = target, args
                self.started = False
                outer.workers.append(self)

            def start(self):
                self.started = True

            def run(self):
                previous = outer.current
                outer.current = self
                try:
                    self.target(*self.args)
                finally:
                    outer.current = previous

        class QueueLock:
            def __enter__(self):
                outer.in_queue = True
                outer.lock_entries += 1
                if outer.on_queue_enter:
                    outer.on_queue_enter()

            def __exit__(self, *args):
                outer.in_queue = False

        callbacks = types.SimpleNamespace(
            on_ui_settings=lambda *a, **k: None,
            on_app_started=lambda *a, **k: None,
            on_script_unloaded=lambda *a, **k: None,
        )
        shared = types.SimpleNamespace(opts=self.options, state=self.state)
        scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
        memory = types.SimpleNamespace(
            unload_all_models=self.unload,
            soft_empty_cache=lambda **kw: self.recoveries.append(kw),
        )
        self.memory = memory
        stubs = {
            "modules": types.SimpleNamespace(script_callbacks=callbacks, scripts=scripts, shared=shared),
            "modules.call_queue": types.SimpleNamespace(queue_lock=QueueLock()),
            "backend": types.SimpleNamespace(memory_management=memory),
            "gradio": types.SimpleNamespace(Dropdown=object),
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            "threading": types.SimpleNamespace(
                Lock=threading.Lock, RLock=threading.RLock, Thread=Worker,
                current_thread=lambda: self.current,
            ),
            "time": types.SimpleNamespace(monotonic=lambda: self.now, perf_counter=lambda: self.now,
                                          sleep=self.sleep),
            "gc": types.SimpleNamespace(collect=lambda: self.gc_calls.append(True) or 0),
        }
        self.ns = {"__name__": "purge_test", "__file__": str(SOURCE)}
        with patch.dict(sys.modules, stubs):
            exec(compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"), self.ns)
        self.imports = stubs
        self.options.forge_neo_auto_vram_purge_mode = self.ns["MODE_UNLOAD_GPU"]

    def unload(self):
        assert self.in_queue, "Unloading must hold Forge's queue lock"
        self.unloads.append(self.now)

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps > 10000:
            raise AssertionError("Worker did not finish")
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()

    def send(self, active, client="A", seq=1):
        return self.ns["_set_continuous_generation"](active, client, seq)

    def active(self):
        return self.ns["_continuous_generation_active"]()

    def run(self, worker=None):
        with patch.dict(sys.modules, self.imports):
            (worker or self.ns["_pending_worker"]).run()


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def test_late_on_cannot_override_newer_off(self):
        h = self.h
        h.send(True, seq=1)
        h.send(False, seq=3)
        worker = h.ns["_pending_worker"]
        self.assertFalse(h.send(True, seq=2))
        self.assertFalse(h.active())
        self.assertIs(h.ns["_pending_worker"], worker)
        h.run()
        self.assertEqual(len(h.unloads), 1)

    def test_other_tab_off_preserves_active_tab_until_expiry(self):
        h = self.h
        h.send(True, "A", 1)
        h.send(True, "B", 1)
        h.send(False, "B", 2)
        self.assertTrue(h.active())
        h.run()
        self.assertEqual(len(h.unloads), 1)
        self.assertGreaterEqual(h.unloads[0], 107.5)

    def test_ttl_expiry_retains_waiter_without_off(self):
        h = self.h
        h.send(True)
        self.assertIsNotNone(h.ns["_pending_worker"])
        h.run()
        self.assertEqual(len(h.unloads), 1)
        self.assertGreaterEqual(h.unloads[0], 107.5)
        self.assertIsNone(h.ns["_pending_worker"])

    def test_new_on_after_stop_keeps_the_existing_ttl_waiter(self):
        h = self.h
        h.send(True, seq=1)
        h.send(False, seq=2)
        worker = h.ns["_pending_worker"]
        h.send(True, seq=3)
        self.assertIs(h.ns["_pending_worker"], worker)
        h.run()
        self.assertEqual(len(h.unloads), 1)
        self.assertGreaterEqual(h.unloads[0], 107.5)

    def test_repeated_off_and_postprocess_share_one_worker(self):
        h = self.h
        for i in range(20):
            h.send(False, seq=i)
        h.ns["AutoVRAMPurgeScript"]().postprocess(types.SimpleNamespace(), None)
        self.assertEqual(len(h.workers), 1)
        h.run()
        self.assertEqual(len(h.unloads), 1)

    def test_thread_start_failure_can_be_retried(self):
        h = self.h
        worker_type = h.ns["threading"].Thread
        with patch.object(worker_type, "start", side_effect=RuntimeError("start failed")):
            with self.assertRaisesRegex(RuntimeError, "start failed"):
                h.send(False, seq=1)
        self.assertIsNone(h.ns["_pending_worker"])
        h.send(False, seq=2)
        h.run()
        self.assertEqual(len(h.unloads), 1)

    def test_heartbeats_extend_ttl_without_churning_worker(self):
        h = self.h
        h.send(True, seq=1)
        worker = h.ns["_pending_worker"]
        token = h.ns["_pending_token"]
        h.now += 5
        h.send(True, seq=2)
        h.now += 2
        self.assertTrue(h.active())
        self.assertEqual(h.ns["_pending_token"], token)
        self.assertIs(h.ns["_pending_worker"], worker)

    def test_duplicate_sequence_does_not_refresh_ttl(self):
        h = self.h
        h.send(True, seq=5)
        h.now += 5
        self.assertFalse(h.send(True, seq=5))
        h.now += 2
        self.assertFalse(h.active())

    def test_stale_off_does_not_stop_newer_on(self):
        h = self.h
        h.send(False, seq=1)
        h.send(True, seq=3)
        self.assertFalse(h.send(False, seq=2))
        self.assertTrue(h.active())

    def test_stop_history_survives_guard_ttl(self):
        h = self.h
        h.send(False, seq=10)
        h.now += 10
        self.assertFalse(h.send(True, seq=9))
        self.assertFalse(h.active())

    def test_inactive_history_is_reclaimed(self):
        h = self.h
        h.send(False)
        h.now += h.ns["CONTINUOUS_HISTORY_SECONDS"] + 1
        h.send(True, "B")
        self.assertNotIn("A", h.ns["_continuous_clients"])

    def test_legacy_off_cannot_clear_identified_tab(self):
        h = self.h
        h.send(True)
        self.assertTrue(h.ns["_set_continuous_generation"](True))
        self.assertTrue(h.ns["_set_continuous_generation"](False))
        self.assertTrue(h.active())

    def test_incomplete_sender_parameters_are_rejected(self):
        h = self.h
        self.assertFalse(h.send(True, "A", -1))
        self.assertFalse(h.send(True, "", 1))
        self.assertFalse(h.send(True, "A" * 129, 1))
        self.assertFalse(h.active())
        self.assertEqual(h.workers, [])

    def test_one_shot_waits_for_job_then_idle_and_native_unload(self):
        h = self.h
        h.state.job = "txt2img"
        h.ns["AutoVRAMPurgeScript"]().postprocess(types.SimpleNamespace(), None)
        h.on_sleep = lambda: setattr(h.state, "job", "") if h.now >= 102 else None
        h.run()
        self.assertGreaterEqual(h.unloads[0], 103.5)
        self.assertEqual(h.recoveries, [])
        self.assertEqual(h.gc_calls, [])

    def test_new_generation_cancels_old_waiter(self):
        h = self.h
        h.send(False)
        old = h.ns["_pending_worker"]
        script = h.ns["AutoVRAMPurgeScript"]()
        script.before_process(types.SimpleNamespace())
        script.postprocess(types.SimpleNamespace(), None)
        h.run(old)
        self.assertEqual(h.unloads, [])
        h.run()
        self.assertEqual(len(h.unloads), 1)

    def test_adetailer_hooks_do_not_cancel_or_reschedule(self):
        h = self.h
        h.send(False)
        worker = h.ns["_pending_worker"]
        script = h.ns["AutoVRAMPurgeScript"]()
        inner = types.SimpleNamespace(_ad_inner=True)
        script.before_process(inner)
        script.postprocess(inner, None)
        self.assertIs(h.ns["_pending_worker"], worker)
        self.assertEqual(len(h.workers), 1)

    def test_forge_default_has_no_extension_unload(self):
        h = self.h
        h.options.forge_neo_auto_vram_purge_mode = h.ns["MODE_FORGE_DEFAULT"]
        h.send(True)
        h.send(False, seq=2)
        h.ns["AutoVRAMPurgeScript"]().postprocess(types.SimpleNamespace(), None)
        self.assertEqual(h.workers, [])
        self.assertEqual(h.unloads, [])

    def test_mode_change_during_heartbeat_stops_waiter(self):
        h = self.h
        h.send(True)
        h.on_sleep = lambda: setattr(h.options, "forge_neo_auto_vram_purge_mode",
                                    h.ns["MODE_FORGE_DEFAULT"])
        h.run()
        self.assertEqual(h.unloads, [])
        self.assertIsNone(h.ns["_pending_worker"])

    def test_shutdown_clears_clients_and_invalidates_waiter(self):
        h = self.h
        h.send(True)
        old = h.ns["_pending_worker"]
        h.ns["_shutdown_extension"]()
        self.assertFalse(h.active())
        h.run(old)
        self.assertEqual(h.unloads, [])

    def test_heartbeat_at_queue_boundary_does_not_lose_waiter(self):
        h = self.h
        h.send(False)
        def enter():
            if h.lock_entries == 1:
                h.send(True, seq=2)
        h.on_queue_enter = enter
        h.run()
        self.assertEqual(len(h.workers), 1)
        self.assertEqual(len(h.unloads), 1)
        self.assertGreaterEqual(h.unloads[0], 109.0)

    def test_normal_unload_and_abnormal_recovery_stay_separate(self):
        h = self.h
        def fail():
            raise RuntimeError("intentional native unload failure")
        h.memory.unload_all_models = fail
        h.send(False)
        h.run()
        self.assertEqual(len(h.gc_calls), 1)
        self.assertEqual(h.recoveries, [{"force": True}])

    def test_api_passes_sender_and_sequence_and_accepts_legacy(self):
        h = self.h
        routes = {}
        class App:
            def post(self, path, **kwargs):
                def register(fn):
                    routes[path] = fn
                    return fn
                return register
        h.ns["_register_api"](None, App())
        call = routes["/forge-neo-auto-vram-purge/continuous"]
        self.assertTrue(asyncio.run(call(True, "A", 2))["ok"])
        self.assertFalse(asyncio.run(call(False, "A", 1))["ok"])
        self.assertTrue(h.active())
        self.assertTrue(asyncio.run(call(False))["ok"])
        self.assertTrue(h.active())

    def test_existing_mode_migration_and_settings_contract(self):
        h = self.h
        for old in ("OFF", "Cache Only"):
            self.assertEqual(h.ns["_normalize_mode"](old), "Forge Default")
        h.options.data[h.ns["OPTION_KEY"]] = "OFF"
        options = []
        h.options.add_option = lambda *args: options.append(args)
        h.ns["shared"].OptionInfo = lambda *args, **kwargs: (args, kwargs)
        h.ns["on_ui_settings"]()
        self.assertEqual(h.options.data[h.ns["OPTION_KEY"]], "Forge Default")
        self.assertEqual(options[0][0], "forge_neo_auto_vram_purge_mode")
        args, kwargs = options[0][1]
        self.assertEqual(args[0], "Unload GPU Models")
        self.assertEqual(args[3]["choices"], ["Forge Default", "Unload GPU Models"])


if __name__ == "__main__":
    unittest.main()
