# Forge Neo Auto VRAM Purge
<img width="228" height="149" alt="{9917F30E-10C1-4624-89D2-915A07150E58}" src="https://github.com/user-attachments/assets/53284edf-a2cf-4ebf-b527-da69b0754f01" />

A small Forge Neo extension that can automatically unload GPU-resident models after generation to return VRAM to the system.

## Recommended default

**`Unload GPU Models` is the default mode.**

In practical testing, it returned roughly **3.7–3.8 GiB of VRAM** after normal steady-state runs. On the tested system, moving the main KModel back to the GPU took roughly **0.9–1.1 seconds**.

From **v1.2.5**, model unloading is idle-aware: the extension first waits until the Forge generation job has genuinely ended, then requires **3.0 seconds of idle time** before unloading. If another generation starts during that grace period, the pending unload is cancelled. This avoids needless unload/reload cycles during Forge Neo's **Generate forever** mode.

If model transfer is noticeably slow on your machine, use **`Forge Default`** instead.

> Existing installations that previously saved `OFF` or `Cache Only` are treated as `Forge Default`.

## Settings

The extension adds one setting under:

**Settings → Forge Neo Auto VRAM Purge**

`After-generation memory policy`

- **Forge Default** — The extension performs no extra purge. Forge Neo continues to run its own normal cache cleanup.
- **Unload GPU Models** — Uses Forge Neo's own memory manager to unload GPU-resident models after a confirmed idle grace period. This is the default and usually returns substantially more VRAM.

No controls are added to txt2img or img2img.

## Generate forever / continuous generation — v1.2.5

Forge Neo's built-in **Generate forever** implementation checks every **500 ms** for the previous generation to finish and then starts another generation.

Before the idle guard, `Unload GPU Models` ran immediately after every generation. That meant a continuous loop could become:

```text
Generation 1
→ unload GPU models
→ next generation starts
→ reload KModel / VAE
→ Generation 2
→ unload again
→ ...
```

In the tested workflow, the next KModel move alone took about **0.98–1.08 seconds**, with the VAE move adding roughly another **0.10 seconds**. This made Generate forever feel as though it had a noticeable pause between images even though the actual purge itself was only around 0.13–0.15 seconds.

### Why v1.2.4 was not sufficient

v1.2.4 started its 1.5-second grace period from extension `postprocess()`. That hook runs before Forge Neo has completely ended the outer job and before the browser necessarily receives and applies the final Gradio response. In testing, the grace period could therefore expire before Generate forever submitted the next request, so models were still unloaded between generations.

v1.2.5 changes the timing model:

```text
Extension postprocess()
→ wait until shared Forge job state is actually idle
→ start 3.0 s idle grace
→ if another generation starts: invalidate pending unload
→ otherwise unload GPU models once
```

When Generate forever is cancelled and no new generation starts:

```text
Last generation genuinely ends
→ 3.0 s idle grace
→ unload GPU models once
```

So continuous generation can keep model residency and speed, while VRAM is still returned shortly after generation activity stops.

### Concurrency safety

The delayed unload is protected by:

- a generation token that invalidates stale workers;
- cancellation from the next outer `before_process()` call;
- direct checks of Forge Neo's shared job state while waiting;
- Forge Neo's shared generation `queue_lock`, used by both the Web UI and API;
- daemon workers that are invalidated when the extension is unloaded/reloaded.

A worker that reaches the grace boundary cannot unload models in the middle of an active queued generation.

## Why `Cache Only` was removed in v1.2.3

Forge Neo already performs cache cleanup as part of its normal generation lifecycle.

Before extension `postprocess()` runs, Forge Neo calls:

```text
devices.torch_gc()
→ backend.memory_management.soft_empty_cache()
```

Forge Neo also runs `devices.torch_gc()` again when the generation job ends, and the Gradio generation wrapper contains another cleanup path.

Therefore the old extension-side `Cache Only` mode duplicated cleanup that Forge Neo already performs automatically.

This also explains repeated test results such as:

```text
CUDA free 11002 -> 11002 MiB (+0 MiB)
reserved 4096 -> 4096 MiB
```

The extension had nothing additional to release because Forge Neo had already cleaned its allocator cache.

### Important

`Forge Default` does **not** mean GPU cache cleanup is disabled.

```text
Extension extra purge: OFF
Forge Neo native cleanup: ON
```

## Which mode should I use?

### Unload GPU Models — recommended

Use this when:

- you want VRAM returned after generation activity stops;
- another GPU application may run alongside Forge Neo;
- you use larger models or workflows where free VRAM is valuable;
- your system can move models back to the GPU quickly;
- you use Generate forever and still want VRAM released after cancelling it.

A 12 GB VRAM-class GPU can also benefit, but suitability depends on model size, quantization, resolution, extensions, system RAM, and workflow.

### Forge Default — minimum overhead

Use this when:

- model transfer back to the GPU is slow;
- you do not need model VRAM returned between sessions;
- you prefer Forge Neo's normal memory behavior without extra model unloading.

Forge Neo still performs its standard allocator cleanup automatically.

## SSD, RAM, and reload speed

Fast storage helps when Forge Neo actually needs to read model data from disk, such as startup, model switching, or a true model reload.

However, `Unload GPU Models` normally removes GPU residency through Forge Neo's model manager rather than deleting the entire model from host memory. The next-generation cost is therefore often dominated by:

- RAM/CPU-to-VRAM transfer;
- PCIe bandwidth;
- system memory bandwidth;
- model size;
- Forge Neo's current offload state.

This is why a system with adequate RAM and fast model transfer can show little practical difference between keeping models GPU-resident and unloading them after an idle period.

## Informal performance tests

These measurements are from one real Forge Neo setup and are **not universal benchmarks**.

### Old Cache Only path

| Version / path | Cleanup time | Additional VRAM released |
|---|---:|---:|
| Cache Only + full GC | ~0.263–0.304 s, ~0.277 s average | 0 MiB |
| Cache Only without full GC | 0.000–0.001 s | 0 MiB |

The test confirmed that extension-side Cache Only did not provide additional VRAM release, so it was removed in v1.2.3.

### Unload optimization history

v1.2.1 representative steady runs were around **0.39–0.40 seconds**.

After removing redundant extension-level full Python GC in v1.2.2:

| Run | Purge time | VRAM released | Reserved after |
|---|---:|---:|---:|
| First/heavier state | 0.468 s | +5632 MiB | 192 MiB |
| Stable 1 | 0.139 s | +3808 MiB | 256 MiB |
| Stable 2 | 0.123 s | +3808 MiB | 256 MiB |
| Stable 3 | 0.133 s | +3776 MiB | 288 MiB |
| Stable 4 | 0.118 s | +3776 MiB | 288 MiB |

Stable four-run average: **~0.128 seconds**, roughly **68% faster** than the previous ~0.397-second baseline while preserving VRAM release.

### v1.2.3 steady-state test

A later four-run test produced:

| Purge time | VRAM released |
|---:|---:|
| 0.126 s | +3712 MiB |
| 0.149 s | +3712 MiB |
| 0.153 s | +3680 MiB |
| 0.133 s | +3712 MiB |

Average purge time was about **0.140 seconds**. Observed KModel moves between immediate-unload runs were about **0.98–1.08 seconds**, plus roughly **0.10 seconds** for the VAE. Those repeated model moves, rather than the ~0.14-second purge itself, were the main reason continuous generation felt slower.

v1.2.5 is designed specifically to prevent those repeated unload/reload cycles by measuring the idle grace only after the Forge job has genuinely ended. Its continuous-generation behavior should be re-measured after updating.

## ADetailer compatibility

ADetailer can create internal img2img generations and marks those internal jobs with `_ad_inner = True`.

Auto VRAM Purge skips those internal jobs so it does not schedule an unload between ADetailer passes. Only the outer generation job may schedule the idle unload.

## Design

The extension intentionally uses Forge Neo's own memory-management APIs instead of patching allocator internals.

### Forge Default

```text
No extension-side memory operation
→ Forge Neo native cleanup continues normally
```

### Unload GPU Models — v1.2.5 normal path

```text
Outer postprocess finishes
→ wait until Forge shared job state is empty
→ start 3.0 s idle grace
→ next generation starts before timeout: invalidate worker
OR
→ grace expires while queue remains idle
→ acquire Forge queue lock
→ Forge Neo backend.memory_management.unload_all_models()
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

## Diagnostics

On startup:

```text
Forge Neo Auto VRAM Purge v1.2.5 loaded
```

With `Unload GPU Models`, a successful one-shot generation should show the unload completion line roughly **3 seconds after the Forge job becomes genuinely idle**:

```text
Auto VRAM Purge completed: Unload GPU Models | 0.xxxs | GC=skipped | CUDA free ...
```

During a working Generate forever loop, you should normally **not** see that line between each generation. After cancelling Generate forever, one unload line should appear after the final idle grace period.

`Forge Default` intentionally produces no per-generation extension cleanup line.

## Reliability

- Uses Forge Neo's native `unload_all_models()` rather than manipulating model objects directly.
- Does not duplicate Forge Neo's standard cache cleanup in `Forge Default` mode.
- Starts the idle grace only after Forge's shared job state reports that the outer generation has ended.
- Uses a 3.0-second idle grace to cover browser/Gradio response time before Generate forever submits the next request.
- Uses Forge Neo's shared queue lock so delayed unload does not race active Web UI/API generation.
- Uses token invalidation so stale workers cannot unload after a newer generation starts.
- Invalidates pending workers when the extension is unloaded/reloaded.
- Skips ADetailer internal `_ad_inner` generations.
- Avoids redundant full Python GC and second allocator cleanup on successful unloads.
- Full Python GC and forced cache cleanup remain available as an abnormal-path fallback.
- Legacy `OFF` and `Cache Only` saved values map safely to `Forge Default`.
- Cleanup errors are logged and never turn an otherwise successful generation into a failed job.

## Installation

Clone this repository into Forge Neo's `extensions` directory, then restart Forge Neo.

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-auto-vram-purge.git
```

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API or shared queue behavior. The extension retains best-effort recovery fallbacks, but full `Unload GPU Models` behavior requires a Forge-compatible `unload_all_models()` implementation.

## Version

v1.2.5
