import gc
import logging
import threading
import time

import gradio as gr
import torch

from modules import script_callbacks, scripts, shared

logger = logging.getLogger("forge_neo_auto_vram_purge")
try:
    from backend.logging import setup_logger

    setup_logger(logger)
except Exception:
    # Keep the extension usable on compatible forks without Forge Neo's logger helper.
    logger.setLevel(logging.INFO)

MODE_FORGE_DEFAULT = "Forge Default"
MODE_UNLOAD_GPU = "Unload GPU Models"
DEFAULT_MODE = MODE_UNLOAD_GPU
VALID_MODES = {MODE_FORGE_DEFAULT, MODE_UNLOAD_GPU}
LEGACY_FORGE_DEFAULT_MODES = {"OFF", "Cache Only"}
OPTION_KEY = "forge_neo_auto_vram_purge_mode"
VERSION = "1.2.6"

# Browser-side Generate Forever detection is authoritative while its heartbeat is
# alive. If the browser disappears without sending a stop signal, the TTL lets the
# backend recover automatically instead of keeping models resident forever.
IDLE_UNLOAD_DELAY_SECONDS = 1.5
JOB_STATE_POLL_SECONDS = 0.05
CONTINUOUS_GUARD_TTL_SECONDS = 6.0

_purge_lock = threading.RLock()
_worker_lock = threading.Lock()
_pending_token = 0
_pending_worker = None
_continuous_lock = threading.Lock()
_continuous_guard_until = 0.0


def _normalize_mode(mode):
    """Map old saved settings to the current two-mode policy."""
    if mode in LEGACY_FORGE_DEFAULT_MODES:
        return MODE_FORGE_DEFAULT
    if mode in VALID_MODES:
        return mode

    logger.warning("Unknown Auto VRAM Purge mode %r; using %s", mode, DEFAULT_MODE)
    return DEFAULT_MODE


def _get_forge_memory_manager():
    try:
        from backend import memory_management

        return memory_management
    except Exception:
        logger.exception("Forge Neo memory manager is unavailable")
        return None


def _fallback_empty_accelerator_cache():
    """Best-effort allocator cleanup when Forge's helper is unavailable."""
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug("CUDA synchronize failed during fallback cleanup", exc_info=True)

            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                logger.debug("CUDA IPC cleanup is unavailable", exc_info=True)
            return
    except Exception:
        logger.debug("CUDA cache cleanup failed", exc_info=True)

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            try:
                xpu.synchronize()
            except Exception:
                logger.debug("XPU synchronize failed during fallback cleanup", exc_info=True)
            xpu.empty_cache()
            return
    except Exception:
        logger.debug("XPU cache cleanup failed", exc_info=True)

    try:
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except Exception:
        logger.debug("MPS cache cleanup failed", exc_info=True)


def _recovery_cache_cleanup(memory_management):
    """Force allocator cleanup only on the abnormal unload fallback path."""
    soft_empty_cache = getattr(memory_management, "soft_empty_cache", None) if memory_management else None

    if callable(soft_empty_cache):
        try:
            soft_empty_cache(force=True)
            return
        except TypeError:
            # Compatibility with forks that expose soft_empty_cache() without `force`.
            try:
                soft_empty_cache()
                return
            except Exception:
                logger.exception("Forge cache cleanup failed")
        except Exception:
            logger.exception("Forge cache cleanup failed")

    _fallback_empty_accelerator_cache()


def _cuda_memory_snapshot():
    """Return lightweight CUDA memory stats for diagnostics, or None."""
    try:
        if not torch.cuda.is_available():
            return None

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "reserved": torch.cuda.memory_reserved(),
            "free": free_bytes,
            "total": total_bytes,
        }
    except Exception:
        logger.debug("Could not read CUDA memory statistics", exc_info=True)
        return None


def _format_mib(value):
    return value / (1024 * 1024)


def _log_unload_result(elapsed, before, after, collected):
    gc_result = "skipped" if collected is None else str(collected)

    if before and after:
        freed = after["free"] - before["free"]
        logger.info(
            "Auto VRAM Purge completed: %s | %.3fs | GC=%s | "
            "CUDA free %.0f -> %.0f MiB (%+.0f MiB) | reserved %.0f -> %.0f MiB",
            MODE_UNLOAD_GPU,
            elapsed,
            gc_result,
            _format_mib(before["free"]),
            _format_mib(after["free"]),
            _format_mib(freed),
            _format_mib(before["reserved"]),
            _format_mib(after["reserved"]),
        )
    else:
        logger.info(
            "Auto VRAM Purge completed: %s | %.3fs | GC=%s",
            MODE_UNLOAD_GPU,
            elapsed,
            gc_result,
        )


def _perform_unload():
    """Run the actual Forge-managed model unload."""
    with _purge_lock:
        start = time.perf_counter()
        before = _cuda_memory_snapshot()
        memory_management = _get_forge_memory_manager()
        collected = None
        unload_ok = False

        unload_all_models = getattr(memory_management, "unload_all_models", None) if memory_management else None
        if callable(unload_all_models):
            try:
                # Forge's unload_all_models() routes through free_memory(), which
                # performs model bookkeeping and allocator cleanup when models unload.
                unload_all_models()
                unload_ok = True
            except Exception:
                logger.exception("Failed to unload Forge-managed GPU models")
        else:
            logger.warning(
                "Forge Neo unload_all_models() is unavailable; falling back to GC and cache cleanup"
            )

        # Normal path: do not repeat Forge's allocator cleanup or full Python GC.
        # Abnormal path: retain stronger best-effort recovery.
        if not unload_ok:
            collected = gc.collect()
            _recovery_cache_cleanup(memory_management)

        after = _cuda_memory_snapshot()
        _log_unload_result(time.perf_counter() - start, before, after, collected)


def _token_is_current(token):
    with _worker_lock:
        return token == _pending_token


def _cancel_pending_unload():
    """Invalidate any worker waiting to unload after the current/previous job."""
    global _pending_token, _pending_worker

    with _worker_lock:
        _pending_token += 1
        _pending_worker = None


def _continuous_generation_active():
    with _continuous_lock:
        return time.monotonic() < _continuous_guard_until


def _clear_continuous_generation():
    global _continuous_guard_until

    with _continuous_lock:
        _continuous_guard_until = 0.0


def _set_continuous_generation(active):
    """Update the frontend-reported Generate Forever guard."""
    global _continuous_guard_until

    now = time.monotonic()
    with _continuous_lock:
        was_active = now < _continuous_guard_until
        if active:
            _continuous_guard_until = now + CONTINUOUS_GUARD_TTL_SECONDS
        else:
            _continuous_guard_until = 0.0

    # Only invalidate the current worker when Generate Forever first becomes
    # active. Heartbeats merely extend the guard and do not churn worker tokens.
    if active and not was_active:
        _cancel_pending_unload()
        logger.info("Generate Forever guard active; GPU model unload suspended")
        return

    if not active:
        if was_active:
            logger.info("Generate Forever guard inactive; idle unload resumed")

        # Cancel can happen after the last postprocess worker was superseded by a
        # heartbeat/new job. Arm a fresh worker so the final VRAM release is not lost.
        mode = _normalize_mode(getattr(shared.opts, OPTION_KEY, DEFAULT_MODE))
        if mode == MODE_UNLOAD_GPU:
            _schedule_unload()


def _wait_for_current_job_to_end(token):
    """Wait until Forge's shared state says the active generation has ended."""
    while True:
        if not _token_is_current(token):
            return False

        if not getattr(shared.state, "job", ""):
            return True

        time.sleep(JOB_STATE_POLL_SECONDS)


def _wait_idle_grace(token):
    """Wait for true idle time; Generate Forever heartbeats pause the countdown."""
    deadline = None

    while True:
        if not _token_is_current(token):
            return False

        if getattr(shared.state, "job", ""):
            return False

        # While Generate Forever is active, do not count any idle gap toward the
        # unload delay. This avoids depending on browser/Gradio response timing.
        if _continuous_generation_active():
            deadline = None
            time.sleep(JOB_STATE_POLL_SECONDS)
            continue

        if deadline is None:
            deadline = time.monotonic() + IDLE_UNLOAD_DELAY_SECONDS

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True

        time.sleep(min(JOB_STATE_POLL_SECONDS, remaining))


def _scheduled_unload_worker(token):
    """Unload only after the outer Forge job has ended and remained truly idle."""
    global _pending_worker

    try:
        if not _wait_for_current_job_to_end(token):
            return

        if not _wait_idle_grace(token):
            return

        # The Web UI and API share this queue lock. If a generation starts at the
        # grace boundary, wait for it to finish and then reject this stale token.
        from modules.call_queue import queue_lock

        with queue_lock:
            if not _token_is_current(token):
                return

            if getattr(shared.state, "job", ""):
                return

            if _continuous_generation_active():
                return

            mode = _normalize_mode(getattr(shared.opts, OPTION_KEY, DEFAULT_MODE))
            if mode != MODE_UNLOAD_GPU:
                return

            _perform_unload()
    except Exception:
        # Delayed cleanup must never affect generation or server stability.
        logger.exception("Scheduled Auto VRAM Purge failed")
    finally:
        with _worker_lock:
            if token == _pending_token and _pending_worker is threading.current_thread():
                _pending_worker = None


def _schedule_unload():
    """Schedule one idle-aware unload worker."""
    global _pending_token, _pending_worker

    with _worker_lock:
        _pending_token += 1
        token = _pending_token

        worker = threading.Thread(
            target=_scheduled_unload_worker,
            args=(token,),
            name="ForgeNeoAutoVRAMPurge",
            daemon=True,
        )
        _pending_worker = worker

    worker.start()
    logger.debug(
        "Auto VRAM Purge armed; unload after %.1fs true idle",
        IDLE_UNLOAD_DELAY_SECONDS,
    )


def _register_api(_, app):
    """Receive the browser-side Generate Forever state/heartbeat."""

    @app.post("/forge-neo-auto-vram-purge/continuous", include_in_schema=False)
    async def forge_neo_auto_vram_purge_continuous(active: bool):
        _set_continuous_generation(bool(active))
        return {"ok": True, "active": bool(active)}


def on_ui_settings():
    # Migrate old in-memory saved values so the two-choice dropdown is valid.
    if shared.opts.data.get(OPTION_KEY) in LEGACY_FORGE_DEFAULT_MODES:
        shared.opts.data[OPTION_KEY] = MODE_FORGE_DEFAULT

    section = ("forge_neo_auto_vram_purge", "Forge Neo Auto VRAM Purge")
    shared.opts.add_option(
        OPTION_KEY,
        shared.OptionInfo(
            DEFAULT_MODE,
            "After-generation memory policy",
            gr.Dropdown,
            {"choices": [MODE_FORGE_DEFAULT, MODE_UNLOAD_GPU]},
            section=section,
        ),
    )


def _shutdown_extension():
    _clear_continuous_generation()
    _cancel_pending_unload()


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(_register_api, name="forge_neo_auto_vram_purge_api")
script_callbacks.on_script_unloaded(_shutdown_extension)
logger.info("Forge Neo Auto VRAM Purge v%s loaded", VERSION)


class AutoVRAMPurgeScript(scripts.Script):
    def title(self):
        return "Forge Neo Auto VRAM Purge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        # Any new outer generation invalidates a pending idle unload. The browser
        # guard independently keeps Generate Forever protected across UI gaps.
        if getattr(p, "_ad_inner", False):
            return
        _cancel_pending_unload()

    def postprocess(self, p, processed, *args):
        # ADetailer creates internal img2img jobs. Scheduling from those nested
        # jobs could unload between passes, so only the outer job may schedule.
        if getattr(p, "_ad_inner", False):
            return

        mode = _normalize_mode(getattr(shared.opts, OPTION_KEY, DEFAULT_MODE))

        if mode == MODE_FORGE_DEFAULT:
            _cancel_pending_unload()
            return

        try:
            _schedule_unload()
        except Exception:
            # Scheduling must never turn an otherwise successful generation into a failed job.
            logger.exception("Auto VRAM Purge scheduling failed")
