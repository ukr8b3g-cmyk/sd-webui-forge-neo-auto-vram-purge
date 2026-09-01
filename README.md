# Forge Neo Auto VRAM Purge
<img width="299" height="170" alt="{CF43CBAA-6A3C-4286-8335-296344AF214E}" src="https://github.com/user-attachments/assets/4ec31864-5ddc-4685-a228-f5dc99d65492" />

A small Forge Neo extension that can automatically unload GPU-resident models after generation to return VRAM to the system.

## Recommended default

**`Unload GPU Models` is the default mode.**

In practical testing, it returned roughly **3.8 GiB of VRAM** after normal runs while adding only a small unload cost. On systems with fast RAM/PCIe model transfer, the next generation delay was about one second in the tested setup.

If model transfer is noticeably slow on your machine, use **`Forge Default`** instead.

> Existing installations that previously saved `OFF` or `Cache Only` are treated as `Forge Default` in v1.2.3.

## Settings

The extension adds one setting under:

**Settings → Forge Neo Auto VRAM Purge**

`After-generation memory policy`

- **Forge Default** — The extension performs no extra purge. Forge Neo continues to run its own normal cache cleanup.
- **Unload GPU Models** — Uses Forge Neo's own memory manager to unload GPU-resident models after generation. This is the default and usually returns substantially more VRAM.

No controls are added to txt2img or img2img.

## Why `Cache Only` was removed in v1.2.3

Forge Neo already performs cache cleanup as part of its normal generation lifecycle.

Before extension `postprocess()` runs, Forge Neo calls:

```text
devices.torch_gc()
→ backend.memory_management.soft_empty_cache()
```

Forge Neo also runs `devices.torch_gc()` again when the generation job ends. The Gradio generation wrapper contains another cleanup path as well.

Therefore the old extension-side `Cache Only` mode was duplicating cleanup that Forge Neo already performs automatically.

This also explains the test result where `Cache Only` repeatedly showed:

```text
CUDA free 11002 -> 11002 MiB (+0 MiB)
reserved 4096 -> 4096 MiB
```

The extension had nothing additional to release because Forge Neo had already cleaned its allocator cache.

### Important

`Forge Default` does **not** mean GPU cache cleanup is disabled.

It means:

```text
Extension extra purge: OFF
Forge Neo native cleanup: ON
```

This is the correct replacement for the previous `OFF` and `Cache Only` choices.

## Which mode should I use?

### Unload GPU Models — recommended

Use this when:

- you want VRAM returned after every generation;
- another GPU application may run alongside Forge Neo;
- you use larger models or workflows where free VRAM is valuable;
- your system can move models back to the GPU quickly;
- you prefer predictable low idle VRAM usage.

A 12 GB VRAM-class GPU can also benefit, but suitability depends on model size, quantization, resolution, extensions, system RAM, and workflow.

### Forge Default — minimum overhead

Use this when:

- model transfer back to the GPU is slow;
- you do not need VRAM returned between generations;
- you prefer Forge Neo's normal memory behavior without extra model unloading.

Forge Neo still performs its standard allocator cleanup automatically.

## SSD, RAM, and reload speed

Fast storage helps when Forge Neo must actually read model data from disk, such as startup, model switching, or a true model reload.

However, `Unload GPU Models` normally removes GPU residency through Forge Neo's model manager rather than deleting the entire model from host memory. The next-generation cost is therefore often dominated by:

- RAM/CPU-to-VRAM transfer;
- PCIe bandwidth;
- system memory bandwidth;
- model size;
- Forge Neo's current offload state.

This is why a system with adequate RAM and fast model transfer can show little practical difference between keeping models GPU-resident and unloading them after each generation.

## Informal performance tests

These measurements are from one real Forge Neo setup and are **not universal benchmarks**.

### Old Cache Only path

Before v1.2.1, the extension also ran full Python garbage collection:

| Mode | Runs | Cleanup time | Additional VRAM released |
|---|---:|---:|---:|
| Cache Only + full GC | 4 | ~0.263–0.304 s, ~0.277 s average | 0 MiB |

After removing redundant full GC in v1.2.1:

| Mode | Runs | Cleanup time | Additional VRAM released |
|---|---:|---:|---:|
| Cache Only | 4 | 0.000–0.001 s | 0 MiB |

The test confirmed that the extension-side Cache Only path did not provide additional VRAM release. In v1.2.3 it was removed completely because Forge Neo already performs the same allocator cleanup itself.

### Unload GPU Models — v1.2.1 baseline

Representative runs:

| Purge time | CUDA free before → after | VRAM released | Reserved before → after |
|---:|---:|---:|---:|
| 0.734 s | 9314 → 14882 MiB | +5568 MiB | 5792 → 224 MiB |
| 0.401 s | 11002 → 14842 MiB | +3840 MiB | 4096 → 256 MiB |
| 0.392 s | 11034 → 14810 MiB | +3776 MiB | 4064 → 288 MiB |
| 0.400 s | 11034 → 14810 MiB | +3776 MiB | 4064 → 288 MiB |
| 0.395 s | 11034 → 14810 MiB | +3776 MiB | 4064 → 288 MiB |

Stable runs were around **0.39–0.40 seconds**.

### Unload GPU Models — v1.2.2

After skipping redundant extension-level full Python GC, the measured runs were:

| Run | Purge time | VRAM released | Reserved after |
|---|---:|---:|---:|
| First/heavier state | 0.468 s | +5632 MiB | 192 MiB |
| Stable 1 | 0.139 s | +3808 MiB | 256 MiB |
| Stable 2 | 0.123 s | +3808 MiB | 256 MiB |
| Stable 3 | 0.133 s | +3776 MiB | 288 MiB |
| Stable 4 | 0.118 s | +3776 MiB | 288 MiB |

Stable four-run average: **~0.128 seconds**.

Compared with the previous stable ~0.397-second baseline, this was roughly a **68% reduction in unload cleanup time** while preserving VRAM release.

Observed next KModel moves were about **0.93–1.08 seconds**.

### v1.2.3 optimization

v1.2.3 removes one more redundant normal-path cleanup:

```text
v1.2.2
unload_all_models()
→ extension soft_empty_cache(force=True)

v1.2.3
unload_all_models()
→ finish
```

Forge Neo's `unload_all_models()` routes through its own `free_memory()` logic, which already performs allocator cleanup when models are unloaded. The extension now only runs explicit GC/cache recovery if Forge's unload API is unavailable or throws an error.

The v1.2.3 timing should be re-measured after updating; the v1.2.2 values above are the comparison baseline.

## ADetailer compatibility

ADetailer can create internal img2img generations and marks those internal jobs with `_ad_inner = True`.

From v1.2.3, Auto VRAM Purge skips those internal jobs so it does not unload GPU models between ADetailer passes.

The purge runs only on the outer generation job.

## Design

The extension intentionally uses Forge Neo's own memory-management APIs instead of patching allocator internals.

### Forge Default

```text
No extension-side memory operation
→ Forge Neo native cleanup continues normally
```

### Unload GPU Models — normal path

```text
Forge Neo backend.memory_management.unload_all_models()
→ log result
```

No redundant extension-level full GC or second allocator cleanup is performed after a successful unload.

### Recovery path

If Forge's model-unload API is unavailable or fails:

```text
gc.collect()
→ Forge soft_empty_cache(force=True), if available
→ native CUDA/XPU/MPS fallback if necessary
```

This keeps the normal path minimal while retaining a stronger abnormal-path recovery mechanism.

## Diagnostics

On startup:

```text
Forge Neo Auto VRAM Purge v1.2.3 loaded
```

Successful unload example:

```text
Auto VRAM Purge completed: Unload GPU Models | 0.xxxs | GC=skipped | CUDA free ...
```

`Forge Default` intentionally produces no per-generation extension cleanup line because no extra extension-side cleanup is performed.

## Reliability

- Uses Forge Neo's native `unload_all_models()` rather than manipulating model objects directly.
- Does not duplicate Forge Neo's standard cache cleanup in `Forge Default` mode.
- Avoids redundant full Python GC on successful unloads.
- Avoids redundant second allocator cleanup after successful unloads from v1.2.3.
- Skips ADetailer internal `_ad_inner` generations.
- Full Python GC and forced cache cleanup remain available as an abnormal-path fallback.
- Cleanup execution is serialized to avoid simultaneous unload operations.
- Legacy `OFF` and `Cache Only` saved values map safely to `Forge Default`.
- Cleanup errors are logged and never turn an otherwise successful generation into a failed job.

## Installation

Clone this repository into Forge Neo's `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-auto-vram-purge.git
```

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API. The extension retains best-effort recovery fallbacks, but full `Unload GPU Models` behavior requires a Forge-compatible `unload_all_models()` implementation.

## Version

v1.2.3
