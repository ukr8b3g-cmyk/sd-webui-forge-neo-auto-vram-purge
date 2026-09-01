# Forge Neo Auto VRAM Purge
<img width="228" height="149" alt="{9917F30E-10C1-4624-89D2-915A07150E58}" src="https://github.com/user-attachments/assets/53284edf-a2cf-4ebf-b527-da69b0754f01" />

A small Forge Neo extension that can automatically unload GPU-resident models after generation to return VRAM to the system.

## Recommended default

**`Unload GPU Models` is the default mode.**

In practical testing, it returned roughly **3.7–3.8 GiB of VRAM** after normal steady-state runs. On the tested system, moving the main KModel back to the GPU took roughly **0.9–1.1 seconds**.

From **v1.2.6**, Forge Neo's **Generate forever** state is detected directly in the browser. While Generate forever is active, model unloading is suspended completely. After **Cancel generate forever**, the extension waits for the final Forge job to finish plus **1.5 seconds of true idle time**, then unloads once.

If model transfer is noticeably slow on your machine, use **`Forge Default`** instead.

> Existing installations that previously saved `OFF` or `Cache Only` are treated as `Forge Default`.

## Settings

The extension adds one setting under:

**Settings → Forge Neo Auto VRAM Purge**

`After-generation memory policy`

- **Forge Default** — The extension performs no extra purge. Forge Neo continues to run its own normal cache cleanup.
- **Unload GPU Models** — Uses Forge Neo's own memory manager to unload GPU-resident models after generation becomes truly idle. This is the default and usually returns substantially more VRAM.

No controls are added to txt2img or img2img.

## Generate forever / continuous generation — v1.2.6

Forge Neo's built-in **Generate forever** implementation checks every **500 ms** for the previous generation to finish and then starts another generation.

With immediate unloading, a continuous loop becomes:

```text
Generation 1
→ unload GPU models
→ next generation starts
→ reload KModel / VAE
→ Generation 2
→ unload again
→ ...
```

In the tested workflow, the KModel move alone took about **0.94–1.08 seconds**, with the VAE move adding roughly another **0.10 seconds**. Those repeated model transfers, rather than the ~0.13–0.20-second unload itself, were the main cause of the visible pause between Generate forever runs.

### Why the v1.2.4 / v1.2.5 timing guards were not enough

v1.2.4 used a 1.5-second delay from extension `postprocess()`. That hook occurs before the browser necessarily receives and applies the final Gradio response, so the timer could expire before Generate forever submitted the next request.

v1.2.5 improved this by waiting until Forge's shared job state became idle and then requiring 3.0 seconds of idle time. Real testing still showed some browser/UI gaps longer than that, so occasional unload/reload cycles remained.

The conclusion was that **backend timing alone cannot reliably determine whether Generate forever is still enabled**.

### v1.2.6 direct browser guard

v1.2.6 adds a small internal JavaScript helper under `javascript/` and uses Forge Neo's normal extension JavaScript loading system.

When the user chooses:

```text
Generate forever
```

the browser reports that state directly to the extension backend and sends a lightweight heartbeat every **2 seconds**.

While that heartbeat is active:

```text
Generate forever ON
→ GPU model unload blocked
→ Generation 1
→ Generation 2
→ Generation 3
→ ...
```

When the user chooses:

```text
Cancel generate forever
```

the browser immediately reports the OFF state:

```text
Generate forever OFF
→ wait until final Forge job ends
→ 1.5 s true idle
→ unload GPU models once
```

This removes the need to guess continuous-generation state from the time gap between requests.

### Heartbeat fail-safe

The frontend heartbeat has a backend TTL of **6 seconds**.

If the browser closes, reloads, or loses the state request unexpectedly, the guard expires automatically instead of keeping models resident forever. Once the guard expires and Forge is idle, normal unload behavior resumes.

### Concurrency safety

The delayed unload is protected by:

- direct browser Generate forever ON/OFF reporting;
- a 2-second heartbeat with a 6-second backend TTL;
- generation-token invalidation for stale workers;
- cancellation from the next outer `before_process()` call;
- direct checks of Forge Neo's shared job state;
- Forge Neo's shared generation `queue_lock`, used by both Web UI and API;
- worker invalidation when the extension is unloaded/reloaded.

The final unload cannot run in the middle of an active queued generation.

## Why `Cache Only` was removed in v1.2.3

Forge Neo already performs cache cleanup as part of its normal generation lifecycle.

Before extension `postprocess()` runs, Forge Neo calls:

```text
devices.torch_gc()
→ backend.memory_management.soft_empty_cache()
```

Forge Neo also performs native cleanup again around job completion / Gradio processing.

Therefore the old extension-side `Cache Only` mode duplicated cleanup Forge Neo already performs automatically.

Repeated tests showed:

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

Average purge time was about **0.140 seconds**. Observed KModel moves between immediate-unload runs were about **0.98–1.08 seconds**, plus roughly **0.10 seconds** for the VAE.

### v1.2.5 Generate forever test

Testing the 3-second backend-only idle heuristic still showed intermittent unloads between continuous generations:

- unload examples around **0.161–0.196 seconds**;
- VRAM release remained about **3.7–3.8 GiB**;
- subsequent KModel reloads were about **0.94–1.07 seconds**.

This confirmed that varying frontend/Gradio gaps can exceed a fixed backend idle threshold and motivated the direct browser state guard in v1.2.6.

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

### Unload GPU Models — v1.2.6 normal Web UI path

```text
Browser selects Generate forever
→ backend continuous guard ON
→ heartbeats keep guard alive
→ model unload suspended across continuous runs

Browser selects Cancel generate forever
→ backend continuous guard OFF
→ wait for active Forge job to finish
→ 1.5 s true idle
→ acquire Forge queue lock
→ backend.memory_management.unload_all_models()
→ log result
```

For ordinary one-shot generation, no Generate forever heartbeat is active, so the extension waits for the job to end plus 1.5 seconds of idle time and then unloads normally.

### Recovery path

If Forge's model-unload API is unavailable or fails:

```text
gc.collect()
→ Forge soft_empty_cache(force=True), if available
→ native CUDA/XPU/MPS fallback if necessary
```

No redundant extension-level full GC or second allocator cleanup is performed after a successful Forge unload.

## Diagnostics

On startup:

```text
Forge Neo Auto VRAM Purge v1.2.6 loaded
```

When Generate forever is enabled, the console should show once:

```text
Generate Forever guard active; GPU model unload suspended
```

When it is cancelled:

```text
Generate Forever guard inactive; idle unload resumed
```

During Generate forever, you should **not** see this between every generation:

```text
Auto VRAM Purge completed: Unload GPU Models
Requested to load KModel
```

After cancelling Generate forever and the final generation ends, one unload line should appear after roughly **1.5 seconds of true idle**:

```text
Auto VRAM Purge completed: Unload GPU Models | 0.xxxs | GC=skipped | CUDA free ...
```

`Forge Default` intentionally produces no per-generation extension cleanup line.

## Reliability

- Uses Forge Neo's native `unload_all_models()` rather than manipulating model objects directly.
- Does not duplicate Forge Neo's standard cache cleanup in `Forge Default` mode.
- Detects Generate forever state directly in the browser instead of relying only on request timing.
- Uses a 2-second heartbeat with a 6-second fail-safe TTL.
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

**Restart Forge Neo after updating to v1.2.6.** The new browser-side JavaScript and backend API endpoint are loaded at startup.

## Compatibility

Designed for Forge Neo. Other A1111/Forge forks are not guaranteed to expose the same backend memory-management API, JavaScript extension loading, or shared queue behavior. The extension retains best-effort recovery fallbacks, but full `Unload GPU Models` behavior requires a Forge-compatible `unload_all_models()` implementation.

## Version

v1.2.6
