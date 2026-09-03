# PyNvVideoCodec / CV-CUDA feasibility — dense keyframe extraction

**Date:** 2026-09-01
**Trigger:** Gavin: "I want to sample 100% of frames, and I want to sample large corpuses of large videos. I don't see any reason to leave a performance stone unturned."

**Update 2026-09-02:** the "100% of frames" goal is now the research direction (subprocess-per-batch, in a sibling branch); this feasibility doc and the pipeline it informs were reframed as the **dense** keyframe extraction path. See `docs/adr/0001-reframe-exhaustive-as-dense-and-research.md` for the decision and `docs/release-bug-investigation.md` for the upstream-bug investigation that informed it. The pipeline (`concourse/frame-flay/flay_video_dense.py`, formerly `flay_video_exhaustive.py`) and the underlying technical analysis below are unchanged — only the framing changed.

## Summary

**Feasible, with one real open bug.** The zero-copy NVDEC → CV-CUDA → CLIP pipeline works correctly — decode, color conversion, and CLIP embedding all verified against real video. But a native-library bug in the current `PyNvVideoCodec`/`cvcuda-cu12` versions crashes the Python interpreter the moment any of their objects are released, which currently caps real usable scale well short of "100% of frames of a multi-hour source." See `concourse/frame-flay/flay_video_dense.py`'s module docstring for the full technical writeup; this doc is the narrative version.

## What's confirmed working

- `PyNvVideoCodec` 2.2.2 and `cvcuda-cu12` 0.17.0 both pip-install cleanly against this box's driver (595.84), no version-floor conflicts.
- Zero-copy DLPack handoff is real: decoded NV12 frames go from `PyNvVideoCodec.SimpleDecoder` → `torch.from_dlpack()` → `cvcuda.as_tensor()` (after an NHWC reshape `cvtcolor` requires) → `cvcuda.cvtcolor(..., YUV2RGB_NV12)` → `.cuda()` (a `cvcuda.Tensor` doesn't itself support `from_dlpack` directly, its `.cuda()`-converted `ExternalBuffer` does) → `torch.from_dlpack()` again → a real CUDA tensor, correct shape, no host round-trip.
- That RGB tensor feeds a real CLIP ViT-L/14 forward pass (`CLIPVisionModelWithProjection`) directly as `pixel_values`, preprocessed manually with torch ops (resize + CLIP's standard normalization constants) rather than routing through `CLIPProcessor` (which expects host-side PIL/numpy input and would defeat the whole point).
- End-to-end verified for real: a 10-second real clip, decoded via NVDEC, converted, embedded, HDBSCAN-clustered, and the winning frames re-decoded and saved as real JPEG keyframes with a correct `manifest.json` — no synthetic data anywhere in this test.
- Earlier single-frame throughput smoke test (before this session's deeper debugging): ~400fps for decode+convert+resize combined on 1920×1080 source, single RTX 3090. Multithreaded CPU decode on this box's 16 cores actually beat that (~696fps) — so the case for NVDEC here isn't raw fps, it's eliminating the disk I/O and PCIe round-trips a million-plus-frame corpus would otherwise need for the CPU-decode-then-reload-into-CLIP path.

## The known blocking bug

`PyNvVideoCodec`'s `DecodedFrame` objects and `cvcuda`'s `Tensor`/`ExternalBuffer` objects crash the whole Python interpreter the instant any of them is released — confirmed via extensive isolated `fly execute` testing, not a guess:

```
Fatal Python error: PyThreadState_Get: the function must be called with
the GIL held, but the GIL is released (the current Python thread state is NULL)
```

Isolated and ruled out one at a time:
- **Not just cyclic GC** — `gc.disable()` doesn't prevent it.
- **Not just crossing a function-return boundary** — inlining everything at module scope still crashes.
- **Not tied to any specific extra computation** (quality-filter ops, `torch.no_grad()`, calling `get_stream_metadata()` first) — each tested in isolation, none triggered or prevented it on their own.
- **The actual trigger:** any release at all, including a plain `keepalive = []` reassignment between batches in the exact same scope with no function boundary crossed.

**Current workaround:** a process-lifetime `KEEPALIVE` list in `flay_video_dense.py` that never releases anything — objects only actually get freed when the whole process exits via `os._exit(0)` (a normal `sys.exit()`/return also crashes at interpreter finalization for the same underlying reason, independent of whether `dec.stop()` is called — which is itself separately broken in this package version, raising `AttributeError` unconditionally).

**What this means practically:** the script is correct and crash-free at bounded scale (verified: 24 real frames, 3 batches, full pipeline including clustering and keyframe writing). It is **not yet** the true "process 100% of frames of an arbitrarily large corpus" capability — holding every processed frame's GPU buffers alive simultaneously for a 13-hour/~1.12M-frame source would need roughly 10TB of VRAM. `--max-frames` (wired into the Concourse job as `flay_dense_max_frames`, default 1300 = the production-path VRAM ceiling on 24 GB) is a required safety valve, not a tuning knob, until this is actually resolved upstream.

## Source of the throughput numbers

The original smoke-test script behind the ~400fps figure above was never preserved (`fly execute` doesn't archive local input dirs server-side, and local `/tmp` was wiped after the investigation). It's been reconstructed from build `1261330`'s preserved stdout log and re-executed live (build `1430505`) to confirm it's real, working code — see `docs/spike/nvdec_cvcuda_smoke_test.py`. Re-run reproduced the same shape of result (zero-copy confirmed, `cvtcolor`/`resize` completing on all 500 frames); it does not include the CLIP step, matching what the original run actually tested.

## Not yet attempted

~~- Reporting the bug upstream to the PyNvVideoCodec or CV-CUDA projects.~~
  **Resolved 2026-09-02** — filed as
  [CVCUDA/CV-CUDA#298](https://github.com/CVCUDA/CV-CUDA/issues/298).
  PyNvVideoCodec has no public tracker (source on NVIDIA NGC), but
  the crash signature is from cvcuda's object chain so the CV-CUDA
  report is the right one. Repro is preserved at
  [gavmor/cvcuda-bug-repros](https://github.com/gavmor/cvcuda-bug-repros).
  Full investigation results (including prior-reports search
  revealing the bug is a recurring class previously "fixed" in #72,
  #188, and #208's v0.11 notes) are in
  `docs/release-bug-investigation.md`.

~~- `PyNvVideoCodec.ThreadedDecoder` instead of `SimpleDecoder` (a different class in the same package — untested here, might have different buffer-lifetime semantics).~~
  **Resolved 2026-09-02** — investigated, does not avoid the bug.
  Same SIGSEGV on the second batch's cvcuda call with the same
  trigger. The bug is in cvcuda's release path, not the decoder
  class. See `docs/release-bug-investigation.md`.

- Different package versions, if/when ones without this bug become available.
  **Partially resolved 2026-09-02** — every version combination on
  PyPI was tested (cvcuda 0.15/0.16/0.17 × PyNvVideoCodec 2.0/2.0.5/2.1/2.2.2);
  all reproduce. No version upgrade path exists. The bug predates
  all publicly available versions. See
  `docs/release-bug-investigation.md`.

- A batched (not single-frame-at-a-time) throughput benchmark at real scale — the ~400fps number above predates this session's bug-hunting and used a simpler, non-batched test loop.
