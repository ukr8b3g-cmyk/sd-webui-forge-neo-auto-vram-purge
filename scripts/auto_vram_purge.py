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
VERSION = "1.2.3"

_purge_lock = threading.RLock()


def _normalize_mode(mode):
    """Map old saved settings to the v1.2.3 two-mode policy."""
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


def purge_memory(mode):
    mode = _normalize_mode(mode)

    # Forge Default intentionally does nothing here. Forge Neo already runs
    # devices.torch_gc() / soft_empty_cache() in its normal generation lifecycle.
    if mode == MODE_FORGE_DEFAULT:
        return

    with _purge_lock:
        start = time.perf_counter()
        before = _cuda_memory_snapshot()
        memory_management = _get_forge_memory_manager()
        collected = None
        unload_ok = False

        unload_all_models = getattr(memory_management, "unload_all_models", None) if memory_management else None
        if callable(unload_all_models):
            try:
                # Forge's unload_all_models() already routes through free_memory(),
                # which performs model bookkeeping and cache cleanup when models unload.
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


script_callbacks.on_ui_settings(on_ui_settings)
logger.info("Forge Neo Auto VRAM Purge v%s loaded", VERSION)


class AutoVRAMPurgeScript(scripts.Script):
    def title(self):
        return "Forge Neo Auto VRAM Purge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def postprocess(self, p, processed, *args):
        # ADetailer creates internal img2img jobs. Purging inside those nested jobs
        # would force unnecessary model reloads before the outer generation finishes.
        if getattr(p, "_ad_inner", False):
            return

        mode = getattr(shared.opts, OPTION_KEY, DEFAULT_MODE)
        try:
            purge_memory(mode)
        except Exception:
            # Cleanup must never turn an otherwise successful generation into a failed job.
            logger.exception("Auto VRAM Purge failed")
