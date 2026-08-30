# Forge Neo Auto VRAM Purge

A small Forge Neo extension that automatically cleans GPU memory after a generation job finishes.

## Features

The extension adds one setting under:

**Settings → Forge Neo Auto VRAM Purge**

`Memory cleanup after generation`

- **OFF** — Do nothing after generation.
- **Cache Only** — Run Python garbage collection and release unused accelerator allocator cache. Forge-managed models stay loaded. This is the default and recommended mode for normal use.
- **Unload GPU Models** — Ask Forge Neo's memory manager to unload GPU-resident models, then run garbage collection and release allocator cache. The next generation may take longer because models must be moved back to the GPU.

No controls are added to txt2img or img2img.

## Installation

Clone this repository into Forge Neo's `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-auto-vram-purge.git
```

## Design

The extension uses Forge Neo's normal post-processing lifecycle and memory manager rather than replacing or patching Forge Neo's allocator logic.

`Cache Only` does not intentionally unload the checkpoint, UNet, VAE, or text encoder. It only releases memory that is no longer actively referenced.

`Unload GPU Models` calls Forge Neo's own `backend.memory_management.unload_all_models()` before clearing allocator cache.

Cleanup failures are logged and are not allowed to turn an otherwise successful generation into a failed job.

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API.

## Version

v1.0.0
