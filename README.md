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
- **Cache Only** — Keep Forge-managed models loaded and force Forge Neo's allocator cache cleanup. Full Python garbage collection is intentionally skipped for minimum latency.
- **Unload GPU Models** — Ask Forge Neo's memory manager to unload GPU-resident models, then force allocator cache cleanup. This is the default mode and usually releases substantially more VRAM. From v1.2.2, a successful Forge Neo unload no longer runs a second full Python GC.

No controls are added to txt2img or img2img.

## Which mode should I use?

### Unload GPU Models — recommended

Use this when you want VRAM returned after each generation, especially when:

- another GPU application may run alongside Forge Neo;
- you use larger models or workflows where free VRAM is valuable;
- your system can move models back to the GPU quickly;
- you prefer predictable low idle VRAM usage after generation.

A 12 GB VRAM-class GPU can also benefit from this mode, but suitability depends on the model, resolution, quantization, extensions, system RAM, and workflow. It is not a guarantee that every 12 GB configuration will behave the same way.

### Cache Only — fastest / minimum reload overhead

Use this when model reload or transfer is noticeably slow on your machine, or when you only want Forge's unused allocator cache returned.

`Cache Only` intentionally does **not** unload the checkpoint, UNet, VAE, or text encoder. Because those objects stay resident, Task Manager or `nvidia-smi` may show only a small decrease — or **0 MiB** of additional VRAM released. This is normal and does not mean the extension failed.

Forge Neo already performs active memory management during generation, so by the time a job finishes there may simply be no unused CUDA allocator cache left to return.

From **v1.2.1**, `Cache Only` no longer runs a full `gc.collect()`. In testing, full Python GC collected many objects but released **0 MiB** of additional CUDA memory while adding about **0.26–0.30 seconds** per cleanup. Removing that full GC reduced measured Cache Only cleanup to approximately **0.000–0.001 seconds**.

## SSD, RAM, and reload speed

Fast storage can help when Forge Neo actually needs to read model data from disk, such as startup, model switching, or a true model reload.

However, `Unload GPU Models` uses Forge Neo's own memory manager and normally removes GPU residency rather than deleting the entire model from host memory. Therefore, the next-generation cost can be dominated by **RAM/CPU-to-VRAM transfer, PCIe bandwidth, memory bandwidth, model size, and Forge's offload state**, not SSD speed alone.

This is why a system with adequate RAM and fast model transfer may show very little practical difference between keeping models resident and unloading them after every generation. If your machine shows a large delay, use `Cache Only` instead.

## Informal performance test

The following results are from a simple real-world test on one Forge Neo setup. They are **not a universal benchmark**; model size, GPU, RAM, storage, PCIe link, Forge settings, and workflow will change the result.

### Cache Only optimization

Pre-v1.2.1:

| Mode | Runs | Cleanup time | Additional VRAM released |
|---|---:|---:|---:|
| Cache Only with full GC | 4 | ~0.263–0.304 s, ~0.277 s average | 0 MiB in all runs |

v1.2.1+:

| Mode | Runs | Cleanup time | Additional VRAM released |
|---|---:|---:|---:|
| Cache Only without full GC | 4 | 0.000–0.001 s | 0 MiB in all runs |

The measured cleanup overhead therefore dropped by roughly **99%+** while preserving the same observed VRAM result.

### Unload GPU Models — v1.2.1 baseline

Representative runs:

| Purge time | CUDA free before → after | VRAM released | Reserved before → after |
|---:|---:|---:|---:|
| 0.734 s | 9314 → 14882 MiB | +5568 MiB | 5792 → 224 MiB |
| 0.401 s | 11002 → 14842 MiB | +3840 MiB | 4096 → 256 MiB |
| 0.392 s | 11034 → 14810 MiB | +3776 MiB | 4064 → 288 MiB |
| 0.400 s | 11034 → 14810 MiB | +3776 MiB | 4064 → 288 MiB |
| 0.395 s | 11034 → 14810 MiB | +3776 MiB | 4064 → 288 MiB |

The stable unload runs were around **0.39–0.40 seconds**, with the first heavier run taking 0.734 seconds. Observed next KModel moves were about **0.96–1.09 seconds**.

Total generation progress in the shown test was roughly **10 seconds with Cache Only** and **11 seconds with Unload GPU Models**, so the practical end-to-end difference remained small relative to generation time.

### v1.2.2 unload optimization

From v1.2.2, a successful Forge Neo `unload_all_models()` path skips the extension's second full `gc.collect()`. Forge Neo already performs model bookkeeping and memory cleanup during its unload path, so the extra full Python GC appeared redundant.

The normal v1.2.2 unload sequence is now:

1. Forge Neo `backend.memory_management.unload_all_models()`.
2. Skip extension-level full Python GC when unload succeeds.
3. Forge Neo forced allocator cleanup.

If Forge's unload function is unavailable or throws an error, the extension falls back to full Python GC before allocator cleanup. This preserves a stronger recovery path for abnormal cases.

The v1.2.2 unload timing and VRAM-release figures should be re-measured after updating; the v1.2.1 numbers above remain the baseline for comparison.

## Design

The extension uses Forge Neo's normal post-processing lifecycle and memory-management APIs rather than replacing the allocator or modifying Forge Neo core files.

### Cache Only cleanup order

1. Skip full Python garbage collection.
2. Forge Neo `soft_empty_cache(force=True)` when available.
3. Compatibility fallback to native CUDA/XPU/MPS cache cleanup if Forge's helper is unavailable.

On CUDA, Forge Neo's native cache cleanup synchronizes the device, clears the PyTorch allocator cache, and performs CUDA IPC cleanup.

The completion log shows `GC=skipped` in this mode.

### Unload GPU Models cleanup order — v1.2.2+

Normal successful path:

1. Forge Neo `backend.memory_management.unload_all_models()`.
2. Skip the extension's redundant full Python GC.
3. Forge Neo forced allocator cache cleanup.

Fallback path if Forge model unloading is unavailable or fails:

1. Log the unload problem.
2. Run full Python `gc.collect()`.
3. Run allocator cache cleanup.

This keeps Forge Neo's own model manager authoritative while avoiding an unnecessary full GC in the common path.

## Diagnostics

On startup you should see:

```text
Forge Neo Auto VRAM Purge v1.2.2 loaded
```

After a generation with `Cache Only` or a successful `Unload GPU Models`, the completion line should normally show `GC=skipped`.

`Cache Only` example:

```text
Auto VRAM Purge completed: Cache Only | 0.001s | GC=skipped | CUDA free ...
```

Successful v1.2.2 unload example format:

```text
Auto VRAM Purge completed: Unload GPU Models | 0.xxxs | GC=skipped | CUDA free ...
```

If host unloading fails, the log will instead show the number of objects collected by the fallback GC.

`OFF` intentionally produces no completion line.

## Reliability

- Forge Neo native `soft_empty_cache()` is preferred over direct allocator calls.
- CUDA synchronization follows Forge's memory manager.
- `Cache Only` avoids full Python GC for lower latency.
- Successful `Unload GPU Models` avoids a redundant second full Python GC from v1.2.2.
- Full Python GC is retained as an abnormal-path fallback if Forge model unloading fails or is unavailable.
- A compatibility fallback is available when a cache helper is missing or has a different signature.
- Purge execution is serialized to prevent simultaneous cleanup operations.
- Unknown or invalid saved setting values safely fall back to the current default.
- Diagnostic logging reports cleanup duration, GC state/count, CUDA free memory, and allocator-reserved memory before/after cleanup.
- Cleanup errors are logged and never turn an otherwise successful generation into a failed job.

## Installation

Clone this repository into Forge Neo's `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-auto-vram-purge.git
```

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API. The extension contains best-effort fallbacks for cleanup, but full `Unload GPU Models` behavior requires Forge-compatible model-management functions.

## Version

v1.2.2
