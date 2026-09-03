# ADR 0001: Reframe the exhaustive-frame job; ship dense, parallelize the rest

- **Status:** proposed
- **Date:** 2026-09-02
- **Branch:** `exhaustive-frame`

## Context

The original goal of the `exhaustive-frame` work (per `docs/pynvvideocodec-feasibility.md`) was:

> "I want to sample 100% of frames, and I want to sample large corpuses of large videos. I don't see any reason to leave a performance stone unturned."

That goal required running a zero-copy NVDEC → CV-CUDA → CLIP pipeline that processes every decoded frame of a multi-hour source. The pipeline was built and works correctly (`concourse/frame-flay/flay_video_exhaustive.py`), but is blocked by a real upstream bug in `cvcuda` / `PyNvVideoCodec` that crashes the Python interpreter whenever any native object from the decode chain is released. The currently-shipped workaround is a process-lifetime `KEEPALIVE` list + `os._exit(0)`, which caps any single process to ~1300 frames of in-VRAM pixel buffers regardless of `--max-frames`.

The upstream bug was filed as [CV-CUDA issue #298](https://github.com/CVCUDA/CV-CUDA/issues/298). Investigation in `docs/release-bug-investigation.md` established that:

- Every version combination on PyPI reproduces the bug
- `ThreadedDecoder` does not avoid it
- The bug is a recurring class the maintainers have "fixed" at least twice (issues #72, #188, plus the v0.11 release note in #208), with both fixes appearing to regress
- The maintainers' track record does not support blocking the project on a fix landing

This leaves the original "100% of frames" goal contingent on an unbounded-timeline upstream fix, which is a bad place for a project to be.

## Decision

We reframe the work as two parallel paths with different success criteria:

1. **Production path (immediate):** ship a *dense* variant of the sparse-sampling `flay_video.py` — same algorithm, but cap frame count per process to the VRAM ceiling (~1300 frames on 24 GB) and reframe the job as "denser-than-sparse keyframe extraction," not "exhaustive." This gives a real, useful intermediate result (1 frame per ~36 seconds on a 13-hour source, vs. the sparse pipeline's 1 per 60) with zero new code, zero new failure surface, and zero dependency on the upstream fix.

2. **Research path (parallel, in the same branch):** pursue the original "100% of frames" goal via a subprocess-per-batch orchestrator (Option A in `docs/exhaustive-scaling-options.md`) that breaks the VRAM ceiling by giving each child process a fresh interpreter. The orchestrator is the only architecture that delivers the original goal without depending on the upstream fix landing. Treat this as a research direction, not a production capability, until it's built, tested, and the upstream fix question is no longer load-bearing.

The CV-CUDA #298 issue remains filed and the `gavmor/cvcuda-bug-repros` repro remains published. If the upstream fix lands and sticks, the research path's "depends on the upstream fix" condition collapses and the research path can become the production path. If it doesn't, the research path is the right answer regardless.

The original goal is **not abandoned** — it is moved from "success criterion" to "research direction." The dense variant is a real deliverable in its own right, not a consolation prize.

## Consequences

**What this gets us:**

- A shippable, useful job today, with no upstream dependency and no new code (just renaming + reframing the existing `flay_video_exhaustive.py` as `flay_video_dense.py` or similar).
- The original goal stays alive in a research direction, with a clear architectural path (subprocess-per-batch) that doesn't depend on the upstream fix.
- A future state where the upstream fix is a *bonus*, not a *blocker*: if it lands and sticks, the research path's complexity drops substantially; if it doesn't, the research path is the right answer.
- The branch's complexity matches the reality of the constraint: the dense variant is simple and bounded; the orchestrator is complex and uncovers its own issues. Two separate code paths, two separate goals.

**What this costs:**

- The `exhaustive-frame` job's marketing changes. Calling it "exhaustive" when it's bounded to 1300 frames was misleading; this ADR agrees the misleading name should be retired, and the dense variant should be named honestly.
- The research path's subprocess-per-batch work is real, ~1-2 hours of careful implementation, with its own failure surface (child lifecycle, partial-failure recovery, embedding shard reassembly).
- The "100% of frames" goal is no longer a near-term deliverable. If the user actually needs exhaustive coverage for a specific use case, this ADR should be revised — the decision is contingent on "denser-than-sparse is useful enough for the use cases we have."

**Operational changes:**

- `concourse/frame-flay/flay_video_exhaustive.py` should be renamed (or copied and renamed) to reflect the dense framing. The `--max-frames` safety valve stays.
- The new `exhaustive-frame-flay` job in `concourse/pipeline.yml` should be retitled. Its `flay_exhaustive_max_frames` var should be set to ~1300 on the production path, and `--max-frames` documentation should explain the bound.
- The research path lives in a sibling branch (suggested: `exhaustive-frame-orchestrator`) until the orchestrator exists and is tested.
- `docs/pynvvideocodec-feasibility.md`'s "Not yet attempted" section is now misleading (two of three items have been investigated and ruled out — see `docs/release-bug-investigation.md`). A follow-up edit should mark them resolved.

## Alternatives considered

**Option A (subprocess-per-batch) as the production path immediately, no dense variant.** Pursues the original goal directly. Cost: ~1-2 hours of new code, new failure surface, and the user gets no working job in the meantime. The dense variant is essentially free by comparison and gets a real, useful intermediate result into the user's hands today. Rejected.

**Option B (cap `--max-frames`) as the only path; abandon the original goal.** Cheapest option. The user's original goal wasn't met but a real artifact ships. Cost: abandons a valid research direction with a clear architectural path. The user has not asked to abandon the original goal; they've asked what to do given the constraint. Rejected.

**Option C (chase upstream) as the only path; wait indefinitely.** Originally the user's lean, until the investigation showed prior fixes have regressed. The maintainers' track record does not support gating the project on a fix landing. Rejected as the *sole* path; still in scope as a parallel (low-EV) track.

**Option D (defer; current state).** This is the implicit baseline. The current state is honest about its bound, but it's not actually shipping any job. The dense variant is a strict improvement over this baseline at near-zero cost. Rejected as the final state; partially adopted as the starting point for the production path.

**Combine all four options (ship dense now, do orchestrator in parallel, file upstream, document the decision).** This is the decision this ADR records. The only thing it adds beyond this combination is the explicit "100% of frames is now a research direction, not a success criterion" framing — which is the part the user said was missing from the handoff doc.
