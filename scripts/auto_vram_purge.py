import gc
import logging

import gradio as gr
import torch

from modules import script_callbacks, scripts, shared

logger = logging.getLogger("forge_neo_auto_vram_purge")

MODE_OFF = "OFF"
MODE_CACHE_ONLY = "Cache Only"
MODE_UNLOAD_GPU = "Unload GPU Models"
OPTION_KEY = "forge_neo_auto_vram_purge_mode"


def _empty_accelerator_cache():
    """Release unused allocator cache without unloading Forge-managed models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass

    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def purge_memory(mode):
    if mode == MODE_OFF:
        return

    if mode == MODE_UNLOAD_GPU:
        try:
            from backend import memory_management

            memory_management.unload_all_models()
        except Exception:
            logger.exception("Failed to unload Forge-managed GPU models")

    gc.collect()
    _empty_accelerator_cache()
    logger.info("Auto VRAM Purge completed: %s", mode)


def on_ui_settings():
    section = ("forge_neo_auto_vram_purge", "Forge Neo Auto VRAM Purge")
    shared.opts.add_option(
        OPTION_KEY,
        shared.OptionInfo(
            MODE_CACHE_ONLY,
            "Memory cleanup after generation",
            gr.Dropdown,
            {"choices": [MODE_OFF, MODE_CACHE_ONLY, MODE_UNLOAD_GPU]},
            section=section,
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)


class AutoVRAMPurgeScript(scripts.Script):
    def title(self):
        return "Forge Neo Auto VRAM Purge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def postprocess(self, p, processed, *args):
        mode = getattr(shared.opts, OPTION_KEY, MODE_CACHE_ONLY)
        try:
            purge_memory(mode)
        except Exception:
            # Cleanup must never turn an otherwise successful generation into a failed job.
            logger.exception("Auto VRAM Purge failed")
