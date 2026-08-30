# Forge Neo Auto VRAM Purge

A small Forge Neo extension that automatically cleans GPU memory after a generation job finishes.

## Features

The extension adds one setting under:

**Settings → Forge Neo Auto VRAM Purge**

`Memory cleanup after generation`

- **OFF** — Do nothing after generation.
- **Cache Only** — Keep Forge-managed models loaded, run Python garbage collection, and force Forge Neo's own allocator cache cleanup. This is the default and recommended mode for normal use.
- **Unload GPU Models** — Ask Forge Neo's memory manager to unload GPU-resident models, then run garbage collection and force allocator cache cleanup. This releases substantially more VRAM, but the next generation may take longer because models must be moved back to the GPU.

No controls are added to txt2img or img2img.

## Installation

Clone this repository into Forge Neo's `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-auto-vram-purge.git
```

## Design

The extension uses Forge Neo's normal post-processing lifecycle and memory-management APIs rather than replacing the allocator or modifying Forge Neo core files.

### Cache Only

`Cache Only` intentionally does **not** unload the checkpoint, UNet, VAE, or text encoder. Because those objects remain resident, Task Manager or `nvidia-smi` may show only a small VRAM decrease. That is expected.

Cleanup order:

1. Python garbage collection.
2. Forge Neo `soft_empty_cache(force=True)` when available.
3. Compatibility fallback to native CUDA/XPU/MPS cache cleanup if Forge's helper is unavailable.

On CUDA, Forge Neo's native cache cleanup synchronizes the device, clears the PyTorch allocator cache, and performs CUDA IPC cleanup.

### Unload GPU Models

Cleanup order:

1. Forge Neo `backend.memory_management.unload_all_models()`.
2. Python garbage collection.
3. Forge Neo forced allocator cache cleanup.

This mode uses Forge Neo's own model manager so its loaded-model bookkeeping stays consistent.

## Reliability improvements

v1.1 adds:

- Forge Neo native `soft_empty_cache()` is preferred over direct allocator calls.
- Forced cache cleanup after generation.
- CUDA synchronization through Forge's memory manager.
- Safe fallback for Forge variants where a helper is missing or has a different signature.
- Serialized purge execution to prevent two cleanup operations from running at the same time.
- Cleanup continues even if the model-unload stage fails.
- Unknown/invalid saved setting values safely fall back to `Cache Only`.
- Diagnostic logging shows cleanup duration, garbage-collected object count, CUDA free memory, and allocator-reserved memory before/after cleanup.
- Cleanup errors are logged and never turn an otherwise successful generation into a failed job.

Example log:

```text
Auto VRAM Purge completed: Cache Only | 0.084s | GC=31 | CUDA free 4210 -> 4375 MiB (+165 MiB) | reserved 1080 -> 915 MiB
```

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API. The extension contains a best-effort fallback for allocator cache cleanup, but `Unload GPU Models` requires Forge-compatible model-management functions for full behavior.

## Version

v1.1.0
