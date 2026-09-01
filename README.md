# flay-video

Semantic keyframe extraction from video: CPU-decode ingestion (`ffmpeg`) → CLIP ViT-L/14 embeddings → HDBSCAN clustering, orchestrated as a standalone Concourse pipeline.

Given a raw video source, this produces a small set of representative keyframes — one per visually-distinct scene, with pans/crossfades/near-duplicate frames automatically dropped as HDBSCAN "noise" rather than needing to be hand-detected.

## Why this exists

Split out of `gavmor/comfyui-workflows` (the blades68-lora repo) on 2026-08-31 — this is a generic, reusable capability that works on any video source, not tied to Blade '68 content. Same reasoning that put `gavmor/T2VA` in its own repo rather than folding it into a project-specific one.

It runs on the same shared Concourse instance and GPU box as blades68-lora and T2VA (same `gpu-lock` pool for cross-pipeline GPU exclusivity, same self-hosted MinIO) — those are genuinely shared infra, reused via matching config values, not duplicated.

## Architecture

Two Concourse jobs:

1. **`extract-candidates`** — pure CPU work (`ffmpeg`, no GPU touched), decodes a staged raw video source into candidate JPEG stills at a configurable sample rate.
2. **`flay-video`** — the GPU-touching job. Embeds surviving candidates with CLIP, clusters with HDBSCAN, keeps the frame closest to each cluster's centroid.

Both jobs' resource `get` steps use `trigger: true`, so staging a new source and re-setting the pipeline vars is genuinely hands-off from there.

GPU work is a plain inline Concourse task, not a persistent HTTP service — this Concourse deployment's task containers have real GPU compute access (confirmed directly: `nvidia-smi` and CUDA/PyTorch inference both work in a bare task) but no host-filesystem access, so inputs are staged via MinIO (`s3` resource) rather than a bind-mounted persistent container. See `docs/nvdec-investigation.md` for why hardware-accelerated *decode* (NVDEC) isn't wired into `extract-candidates` yet even though it's technically available.

## Setting the pipeline

```bash
fly -t blades68 set-pipeline -p flay-video \
  -c concourse/pipeline.yml \
  -l concourse/vars.default.yml -l concourse/.secrets/vars.yml
fly -t blades68 unpause-pipeline -p flay-video
```

## Ingesting a new source

1. Tar the source's raw video file(s) into one archive (the `s3` resource type tracks versions of a single logical object, not a directory listing — many loose files under one prefix won't work).
2. Upload to `s3://<bucket>/frame-flay/raw-video/<name>-<version>.tar.gz`.
3. Re-set the pipeline with `-v flay_raw_video_regexp`, `-v flay_source_name`, and `-v flay_candidate_regexp` all pointed at `<name>` (see `concourse/vars.default.yml` for the pattern).
4. Concourse takes it from there — `extract-candidates` fires, then `flay-video` fires off its output.

## Tuning

`flay_blur_threshold`/`flay_dark_threshold` are calibrated against 720p JPEG-compressed candidate stills (median variance-of-Laplacian ~39 in the first real source). Recalibrate if a new source's resolution or compression differs meaningfully — the textbook variance-of-Laplacian threshold of 100 assumes full-resolution, uncompressed frames and will reject nearly everything against compressed stills.
