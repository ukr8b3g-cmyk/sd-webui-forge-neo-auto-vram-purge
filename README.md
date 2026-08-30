# Forge Neo Auto VRAM Purge
<img width="299" height="170" alt="{CF43CBAA-6A3C-4286-8335-296344AF214E}" src="https://github.com/user-attachments/assets/4ec31864-5ddc-4685-a228-f5dc99d65492" />


A small Forge Neo extension that automatically cleans GPU memory after a generation job finishes.

## Recommended default

**`Unload GPU Models` is the default mode from v1.2.0.**

In practical testing, unloading Forge-managed GPU models released several GiB of VRAM while adding only a small amount of cleanup time. On systems where model movement is fast, the difference can be difficult to notice during normal generation.

If the next generation becomes noticeably slower on your system, switch to **`Cache Only`**.

> Existing installations may keep a previously saved setting. Changing the extension default does not intentionally overwrite a user's saved Forge Neo setting.

## Settings

The extension adds one setting under:

**Settings → Forge Neo Auto VRAM Purge**

`Memory cleanup after generation`

- **OFF** — Do nothing after generation.
- **Cache Only** — Keep Forge-managed models loaded, run Python garbage collection, and force Forge Neo's allocator cache cleanup. Lowest reload overhead, but often releases little or no additional VRAM because Forge already manages its allocator aggressively.
- **Unload GPU Models** — Ask Forge Neo's memory manager to unload GPU-resident models, then run garbage collection and force allocator cache cleanup. This is the default mode from v1.2.0 and usually releases substantially more VRAM.

No controls are added to txt2img or img2img.

## Which mode should I use?

### Unload GPU Models — recommended

Use this when you want VRAM returned after each generation, especially when:

- another GPU application may run alongside Forge Neo;
- you use larger models or workflows where free VRAM is valuable;
- your system can move models back to the GPU quickly;
- you prefer predictable low idle VRAM usage after generation.

A 12 GB VRAM-class GPU can also benefit from this mode, but suitability depends on the model, resolution, quantization, extensions, system RAM, and workflow. It is not a guarantee that every 12 GB configuration will behave the same way.

### Cache Only — compatibility / minimum reload overhead

Use this when model reload or transfer is noticeably slow on your machine.

`Cache Only` intentionally does **not** unload the checkpoint, UNet, VAE, or text encoder. Because those objects stay resident, Task Manager or `nvidia-smi` may show only a small decrease — or **0 MiB** of additional VRAM released. This is normal and does not mean the extension failed.

Forge Neo already performs active memory management during generation, so by the time a job finishes there may simply be no unused CUDA allocator cache left to return.

## SSD, RAM, and reload speed

Fast storage can help when Forge Neo actually needs to read model data from disk, such as startup, model switching, or a true model reload.

However, `Unload GPU Models` uses Forge Neo's own memory manager and normally removes GPU residency rather than deleting the entire model from host memory. Therefore, the next-generation cost can be dominated by **RAM/CPU-to-VRAM transfer, PCIe bandwidth, memory bandwidth, model size, and Forge's offload state**, not SSD speed alone.

This is why a system with adequate RAM and fast model transfer may show very little practical difference between keeping models resident and unloading them after every generation. If your machine shows a large delay, use `Cache Only` instead.

## Informal performance test

The following results are from a simple real-world test on one Forge Neo setup. They are **not a universal benchmark**; model size, GPU, RAM, storage, PCIe link, Forge settings, and workflow will change the result.

| Mode | Purge runs | Average purge time | Additional VRAM released | Observed next KModel move |
|---|---:|---:|---:|---:|
| Cache Only | 4 | ~0.277 s | 0 MiB in all 4 runs | No forced unload |
| Unload GPU Models | 2 | ~0.556 s | ~4720 MiB average | ~1.01 s |

Observed individual unload results:

| Purge time | CUDA free before → after | VRAM released | Reserved before → after |
|---:|---:|---:|---:|
| 0.727 s | 9314 → 14882 MiB | +5568 MiB | 5792 → 224 MiB |
| 0.384 s | 11042 → 14914 MiB | +3872 MiB | 4064 → 192 MiB |

Observed `Cache Only` runs were approximately `0.263–0.304 s`, with `+0 MiB` additional CUDA free memory. Python garbage collection still ran, but there was no unused CUDA allocator memory left for Forge to return.

Total generation progress in this test remained around **12–13 seconds** in both usage patterns. The exact end-to-end difference was therefore small relative to total generation time.

## Design

The extension uses Forge Neo's normal post-processing lifecycle and memory-management APIs rather than replacing the allocator or modifying Forge Neo core files.

### Cache Only cleanup order

1. Python garbage collection.
2. Forge Neo `soft_empty_cache(force=True)` when available.
3. Compatibility fallback to native CUDA/XPU/MPS cache cleanup if Forge's helper is unavailable.

On CUDA, Forge Neo's native cache cleanup synchronizes the device, clears the PyTorch allocator cache, and performs CUDA IPC cleanup.

### Unload GPU Models cleanup order

1. Forge Neo `backend.memory_management.unload_all_models()`.
2. Python garbage collection.
3. Forge Neo forced allocator cache cleanup.

This mode uses Forge Neo's own model manager so its loaded-model bookkeeping stays consistent.

## Diagnostics

On startup you should see:

```text
Forge Neo Auto VRAM Purge v1.2.0 loaded
```

After a generation with `Cache Only` or `Unload GPU Models`, a completion line reports cleanup time and CUDA memory changes, for example:

```text
Auto VRAM Purge completed: Unload GPU Models | 0.384s | GC=13865 | CUDA free 11042 -> 14914 MiB (+3872 MiB) | reserved 4064 -> 192 MiB
```

`OFF` intentionally produces no completion line.

## Reliability

- Forge Neo native `soft_empty_cache()` is preferred over direct allocator calls.
- Forced cache cleanup runs after generation.
- CUDA synchronization follows Forge's memory manager.
- A compatibility fallback is available when a cache helper is missing or has a different signature.
- Purge execution is serialized to prevent simultaneous cleanup operations.
- Cleanup continues even if the model-unload stage fails.
- Unknown or invalid saved setting values safely fall back to the current default.
- Diagnostic logging reports cleanup duration, garbage-collected object count, CUDA free memory, and allocator-reserved memory before/after cleanup.
- Cleanup errors are logged and never turn an otherwise successful generation into a failed job.

## Installation

Clone this repository into Forge Neo's `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-auto-vram-purge.git
```

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API. The extension contains a best-effort fallback for allocator cache cleanup, but `Unload GPU Models` requires Forge-compatible model-management functions for full behavior.

## Version

v1.2.0
