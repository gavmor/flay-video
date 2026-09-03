# AGENTS.md — flay-video

Operational guide for agents working in this repo. Focuses on non-obvious knowledge (gotchas, implicit conventions, surprising wiring) rather than facts trivially readable from any single file.

## What this repo is

Standalone Concourse pipeline that turns a raw video source into a small set of representative keyframes:

1. `extract-candidates` (CPU, no GPU) — `ffmpeg` decodes the raw video into candidate JPEG stills at a configurable sample rate, filtered down by quality.
2. `flay-video` (GPU, behind `serial_groups: [gpu]` + `gpu-lock` pool) — quality-filters candidates further, embeds survivors with CLIP ViT-L/14, clusters with HDBSCAN, keeps the frame closest to each cluster's centroid.

Pans, crossfades, and near-duplicate frames naturally fall out as HDBSCAN's `-1` "noise" label — no hand-tuned scene-detection needed.

Split out of `gavmor/comfyui-workflows` on 2026-08-31; it is generic and reusable, not tied to Blade '68 content.

## Layout

```
concourse/
  pipeline.yml                 -- the two-job pipeline definition
  vars.default.yml             -- non-secret, environment-specific config (committed on purpose)
  frame-flay/
    flay_video.py              -- the Python script the GPU task runs (sparse-sampling, main branch)
concourse/.secrets/            -- gitignored; holds vars.yml with actual secrets
docs/
  nvdec-investigation.md       -- root-cause + fix report for NVDEC-in-Concourse (verified, not yet wired in)
  pynvvideocodec-feasibility.md -- feasibility report + open-bug writeup for the exhaustive NVDEC+cvcuda+CLIP path
  spike/
    nvdec_cvcuda_smoke_test.py -- throughput smoke test for the NVDEC/cvcuda path (~400fps, real measurement)
README.md                      -- high-level architecture + setup + ingest steps
```

This is a config-driven repo. There is no `make` target, no test suite, no linter, no package manager, no language build system. The "code" on main is one Python script (`concourse/frame-flay/flay_video.py`, ~150 lines) and two YAML files.

## Essential commands

### Setting / updating the pipeline

```bash
fly -t blades68 set-pipeline -p flay-video \
  -c concourse/pipeline.yml \
  -l concourse/vars.default.yml -l concourse/.secrets/vars.yml
fly -t blades68 unpause-pipeline -p flay-video
```

`blades68` is the shared fly target — same instance used by `comfyui-workflows` (blades68-lora) and `T2VA`. Re-run `set-pipeline` whenever `vars.default.yml` changes, or to point at a new source via `-v` overrides.

### Running flay_video.py locally

The script is run inside a Concourse task, but it works fine locally:

```bash
pip install transformers scikit-learn opencv-python-headless torch pillow numpy
python3 concourse/frame-flay/flay_video.py \
  --input-dir /path/to/candidate-stills \
  --output-dir /path/to/output \
  --min-cluster-size 3 --blur-threshold 15.0 --dark-threshold 15.0 --max-edge 768
```

Note: the Concourse task installs `transformers scikit-learn opencv-python-headless` at task-run time on top of `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (which already provides `torch`). No `requirements.txt` exists; the task's `pip install` line is the source of truth for the script's Python deps.

### Ingesting a new source

1. Tar raw video files (even if multiple) into one archive.
2. Upload to `s3://<bucket>/frame-flay/raw-video/<name>-<version>.tar.gz`.
3. Re-set the pipeline with `-v flay_raw_video_regexp`, `-v flay_source_name`, `-v flay_candidate_regexp` all pointed at `<name>`.

Both `get` steps in the job chain use `trigger: true`, so from there it's hands-off — `extract-candidates` fires, then `flay-video` consumes its output.

## Critical gotchas

These are the traps that already cost time. Read before touching the pipeline.

### The `s3` resource type tracks ONE object per regexp, not a directory

It uses the regexp's capture group as the version of a single logical object. Multiple loose files under one prefix do **not** work — see the comment in `concourse/pipeline.yml` on `flay-raw-video-s3` ("learned the hard way on the first pass: 775 loose .jpg keys couldn't be fetched as a set"). Always tar inputs into one archive per source.

### `flay-result-s3`'s regexp requires a non-empty capture group

A bare `flay-result.tar.gz` fails the `s3` `put` with "regex does not match provided version" — after the real work is done. The pipeline works around this by stamping the filename with a timestamp (`flay-result-$(date +%Y%m%d%H%M%S).tar.gz`). Don't "simplify" that pattern.

### `tar` with a bare glob fails when the glob matches zero files

`extract-candidates`'s tar line uses `$(ls *.jpg 2>/dev/null)` rather than a bare `*.jpg` so a legitimately-empty source (very short clips, very sparse fps) produces an empty archive instead of crashing on a literal-glob argument to `tar`. Don't "clean this up" by removing the `ls`.

### `flay_video.py` exits non-zero if quality filtering drops everything

```python
if not frames:
    raise SystemExit("no frames survived quality filtering")
```

This is intentional. Be aware when tuning `flay_blur_threshold` / `flay_dark_threshold` — a too-aggressive threshold turns the job red, doesn't just silently produce an empty result.

### GPU lock is a per-deployment singleton pool

`gpu-lock` is the *same physical pool* every other GPU-touching pipeline on this box uses (`comfyui-workflows` / blades68-lora, `T2VA`). Cross-pipeline GPU exclusivity requires sharing the one pool, not standing up a second one. `serial_groups: [gpu]` is set on `flay-video` for the same reason. Don't fork either into a per-pipeline pool.

The `extract-candidates` job deliberately has **no** `serial_groups` / `gpu-lock` — it's pure CPU work, so gating it would only serialize against the real GPU job needlessly.

### `blend_thresholds` were calibrated against THIS specific source

`flay_blur_threshold: 15.0` and `flay_dark_threshold: 15.0` in `vars.default.yml` were measured directly against the 720p JPEG-compressed "parts-series" candidate stills (median sharpness ~39, 95th percentile ~75). The textbook variance-of-Laplacian threshold of 100 assumes full-resolution uncompressed frames and would reject nearly everything here. **Recalibrate** for any new source whose resolution or compression differs meaningfully. There is no automatic calibration pass — it's a manual measurement + a manual vars edit.

### `NVIDIA_DRIVER_CAPABILITIES: compute,utility,video` is not currently set

`extract-candidates` does not yet use GPU-accelerated decode. NVDEC-in-Concourse is verified working (see `docs/nvdec-investigation.md`) but not wired in. If you add it, the minimal change is two env vars on the task's `params:` block:

```yaml
params:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility,video
```

No change to `docker-compose.yml`, no `concourse-concourse-1` recreation, no risk to other pipelines — the OCI prestart hook is per-task. There is a separate, secondary finding in `docs/nvdec-investigation.md` (`jrottenberg/ffmpeg` nvcodec-header mismatch during actual multi-frame decode) — if you wire NVDEC in for real, use a more current NVDEC-capable ffmpeg image.

### Caches rely on a single-worker deployment

`flay-video` task's caches (`hf-cache`, `pip-cache`) assume one Concourse worker. This is currently true (`fly workers` confirms), so the ~1.7 GB CLIP checkpoint and pip deps persist across runs. If the deployment ever scales to multiple workers, this assumption breaks — task containers may land on a fresh worker without the cache, and downloads happen on every run.

### `concourse/.secrets/` gitignore uses directory form, not trailing-slash

`.gitignore` comment explains this in full — it matters because gitignore pattern matching treats symlinks as files, not directories. If this repo ever grows worktrees that symlink back to this directory (incident documented in `gavmor/comfyui-workflows` docs/adr/0008), a trailing-slash pattern would silently fail to match.

### No persistent GPU service — flush-vram intentionally absent

GPU work runs as an inline Concourse task, not as a curl to a host service. Verified directly that this deployment's task containers have GPU compute access but no host filesystem. No `flush-vram` ensure step exists because an ephemeral task's GPU memory is reclaimed on container exit. Don't add one "to be safe" — it'd be cargo-culting.

## Output structure

`flay-result/` after a successful run:

```
flay-result/
  manifest.json                  -- inputs, params, frame counts, cluster list, elapsed time
  cluster0000_<original>.jpg     -- keyframe per cluster
  cluster0001_<original>.jpg
  ...
```

`manifest.json` is the canonical record of what happened. The pipeline tars `manifest.json` + all `*.jpg` into one archive under `frame-flay/results/` (regexp `flay-result-(.*)\.tar\.gz`).

## `exhaustive-frame` branch (pushed, NOT merged, NOT registered on Concourse)

A second pipeline mode exists on the `origin/exhaustive-frame` branch. Status as of 2026-09-02: pushed for review, not yet `fly set-pipeline`d, not yet triggered against a real corpus. Treat it as a **production-capable bounded pipeline**, not a "100% of frames" capability — the original goal was reframed per `docs/adr/0001-reframe-exhaustive-as-dense-and-research.md` after the upstream-bug investigation showed the maintainers' fix track is unreliable. Don't merge to main or trigger it without explicit owner go-ahead.

What it adds (see `concourse/frame-flay/flay_video_dense.py` on the branch):

- NVDEC hardware decode via PyNvVideoCodec → CV-CUDA color convert → CLIP embed, all VRAM-resident via DLPack. No frame touches host memory or disk until the final ~handful of cluster winners get re-decoded and saved as JPEGs.
- Two-pass design: pass 1 decodes + embeds every frame the script accepts and discards pixels immediately (only embedding vectors + `(source_file, frame_index)` survive); pass 2 HDBSCAN-clusters the embeddings, then re-decodes only the cluster winners.
- A new `dense-keyframe-flay` job in `pipeline.yml` (on the branch), behind the same `gpu-lock` pool + `serial_groups: [gpu]`, with `ensure: release`. Separate S3 output prefix (`frame-flay/results-dense/`) so its runs never collide with the sparse-sampling `flay-video` job's results.
- `docs/spike/nvdec_cvcuda_smoke_test.py` is the throughput smoke test for this path (~400fps measured, real). Preserved on the branch because the original got lost when local scratch was wiped — a subagent rebuilt it from a build log and re-ran it live to confirm the numbers are real.

### VRAM ceiling (this is the bound, not a bug)

PyNvVideoCodec 2.2.2 / `cvcuda-cu12` 0.17.0 native objects (`DecodedFrame`, `cvcuda.Tensor`, `cvcuda.ExternalBuffer` chain from `nv12_frame_to_rgb_nhwc`) crash the whole Python interpreter — `Fatal Python error: PyThreadState_Get: the function must be called with the GIL held, but the GIL is released` — the instant any of them is released. The branch author isolated this multiple ways and it is **not**:

- Python's cyclic GC (reproduces with `gc.disable()`)
- a function-return boundary (reproduces with everything inlined at module scope)
- a specific op (reproduces from a plain list reassignment between batches in the same scope)

The only mitigation that has held up in testing is never releasing any of these objects until the process calls `os._exit(0)`. Consequence: the current implementation is verified-correct at bounded scale (tested: 24 real frames, full pipeline, no crash) but bounded to ~1300 frames per process (24 GB ÷ ~18 MB/frame in KEEPALIVE) on the standard 24 GB GPU. For a 13-hour / ~1.12M-frame source, this gives ~1 frame per 36 seconds — denser than the sparse pipeline's 1 per 60, not exhaustive.

This is why `--max-frames` (driven by `flay_dense_max_frames`, default 1300) is wired in as a **required safety valve** — the script will OOM rather than silently run away. The original "100% of frames" goal is a research direction in a sibling branch pursuing a subprocess-per-batch architecture; see ADR 0001 for the decision and the tradeoffs.

### What's done (logged in `docs/release-bug-investigation.md`)

- Reporting the bug upstream: done — [CVCUDA/CV-CUDA#298](https://github.com/CVCUDA/CV-CUDA/issues/298), with prior instances #72, #188, #208 cited.
- `ThreadedDecoder` instead of `SimpleDecoder`: investigated 2026-09-02, same bug, ruled out.
- Other package versions: tested cvcuda 0.15/0.16/0.17 × PyNvVideoCodec 2.0/2.0.5/2.1/2.2.2, all reproduce. No version upgrade path exists on PyPI.

### What the branch is NOT

- It is not on `main`. `git log main..origin/exhaustive-frame` shows the two new commits.
- It is not registered on Concourse (`fly -t blades68 pipelines` won't show it). No `set-pipeline` has been run.
- It has not been run against the real parts-series corpus.
- The smoke test on the branch is the throughput measurement, not a full end-to-end run on real data.

If you're working on main: ignore the branch entirely. If you're reviewing the branch: read the script's docstring and `docs/pynvvideocodec-feasibility.md` first — the bug is documented honestly there, not glossed over.

## Style conventions observed in the repo

- Python: type hints absent; functions are short and pure-ish; one helper per concern (`load_and_filter`, `embed_frames`, `save_keyframe`); `print()` with `---` delimiters is the logging convention (Concourse task stdout is the log).
- YAML: every resource / job / non-obvious step has a multi-line `#`-prefixed comment explaining **why** it's configured that way — failures, calibrations, blast-radius notes. Match this style when adding new resources or jobs. A silent change in this repo is a bad change.
- Secrets vs non-secrets: `vars.default.yml` is committed on purpose — it's facts about the local network topology, not credentials. Real secrets live in `concourse/.secrets/vars.yml`, gitignored. Don't move anything from `.secrets/` into the committed file "for convenience".

## Where to look first when something breaks

| Symptom | Look here |
|---|---|
| Job fails to start / `get` step errors | Concourse UI → resource version + `fly workers` to confirm the one-worker assumption still holds |
| "regex does not match provided version" on `put: flay-result-s3` | Pipeline's tar filename pattern — must include a non-empty version capture (`flay-result-$(date +%Y%m%d%H%M%S).tar.gz`) |
| `flay_video.py` exits "no frames survived quality filtering" | Tune `flay_blur_threshold` / `flay_dark_threshold` — current values calibrated against parts-series only |
| NVDEC-related failure in a new ffmpeg task | `docs/nvdec-investigation.md` — capability fix is per-task `NVIDIA_DRIVER_CAPABILITIES`; secondary finding is image nvcodec-header mismatch |
| Job blocked waiting for `gpu-lock` | Other GPU pipeline is running (`comfyui-workflows`, `T2VA`); expected behavior, not a bug |
| `tar` step crashes on literal `*.jpg` | Restore the `$(ls *.jpg 2>/dev/null)` form — empty input is legitimate |