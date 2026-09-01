# NVDEC in Concourse — investigation report

**Date:** 2026-08-31
**Trigger:** ongoing video ingestion for the `frame-flay` pipeline (CLIP+HDBSCAN keyframe selection) — CPU decode works today but GPU-accelerated decode was worth checking before it becomes a real bottleneck.

## The question

Why does `ffmpeg -hwaccel cuda -c:v h264_cuvid -i input.mp4 ...` fail inside a Concourse task container on this box (`Cannot load libnvcuvid.so.1`, followed by a segfault), when GPU *compute* (CUDA, PyTorch/CLIP inference) works fine in the exact same kind of task container?

## Root cause (confirmed empirically, not theorized)

This Concourse deployment's GPU access is **not** the standard `docker run --gpus` mechanism. The container's config (`docker inspect concourse-concourse-1` → `Runtime: runc`, `DeviceRequests: null`) initially looked inconsistent with GPU access working at all — but that's because it isn't using that mechanism in the first place.

The real, git-tracked source of truth is `~/code/blades68-lora/concourse/docker-compose.yml` (main branch, no drift from a stray `/tmp/docker-compose.gpu.yml` file that turned out to be an unrelated leftover). It's a bespoke setup:

- The Concourse container (Wolfi-based, no NVIDIA toolkit of its own) **bind-mounts the host's** `nvidia-container-toolkit`/`nvidia-container-cli` userland, plus custom `host-shims/` wrapper scripts.
- A custom **OCI prestart hook** (`oci-hooks/nvidia-container-toolkit.json`, `"when": {"always": true}`) fires on *every* task container Concourse's containerd-based worker spawns, and injects whatever NVIDIA devices/libraries that specific task's own `NVIDIA_DRIVER_CAPABILITIES` environment variable asks for.
- The host itself has no restriction: `/etc/nvidia-container-runtime/config.toml` already lists `video` under `supported-driver-capabilities`, and the CDI spec (`/var/run/cdi/nvidia.yaml`) already maps `libnvcuvid.so.*` / `libnvidia-encode.so.*` under the video capability class.

**So the failure had nothing to do with a missing library, a broken driver, or a restrictive Concourse/host configuration.** The one task that tried `-hwaccel cuda` simply never set `NVIDIA_DRIVER_CAPABILITIES`, so it inherited the default (`compute,utility`) — which is exactly enough for CUDA/PyTorch but excludes the video-codec libraries entirely.

## The fix

Add two lines to the relevant Concourse task's `params:` block:

```yaml
params:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility,video
```

This pattern already existed, unused, in `concourse/spike/gpu-test-task.yml` (which sets `NVIDIA_DRIVER_CAPABILITIES: all`) — nobody had wired it into a real ffmpeg-decode task before.

**Verification:** ran a scratch, one-off `fly execute` task (same throwaway mechanism used earlier this session to confirm GPU compute access) with this param set, on the `jrottenberg/ffmpeg:6.1-nvidia2204` image:
- `libnvcuvid.so.1` and `libnvidia-encode.so.1` both resolved and loaded inside the task container.
- `ffmpeg -hwaccel cuda -c:v h264_cuvid ...` no longer crashed or segfaulted — the original reported failure is fully resolved.

**Blast radius: none.** This is a per-task parameter. It doesn't touch `docker-compose.yml`, doesn't require recreating the shared `concourse-concourse-1` container, and can't affect any other pipeline (comfyui-local, T2VA, blades68-charref, etc.) running concurrently on the same box.

**Confidence:** high enough to apply directly to a real task. No further testing of the *capability* fix itself is needed.

## Secondary finding (separate issue, not fixed, flagged only)

Once past the library-load stage, the same scratch test hit a different, unrelated error during actual multi-frame decode:

```
[h264_cuvid @ ...] cuvid decode callback error
[vist#0:0/h264 @ ...] Decoding error: Generic error in an external library
```

Checked whether this was another device-exposure gap — it wasn't. All three device majors NVDEC/CUVID need (195: `nvidia0`/`nvidiactl`/`nvidia-modeset`; 510: `nvidia-uvm`/`nvidia-uvm-tools`; 235: `nvidia-caps`) are already covered by the `--worker-containerd-allowed-device` flags in `docker-compose.yml`.

Most likely explanation: `jrottenberg/ffmpeg` is an old, infrequently-rebuilt image, and its bundled nvcodec headers are probably mismatched against the host's quite-new driver (595.84). Not chased further since it's outside the scope of the root-cause question — worth a follow-up with a more current NVDEC-capable ffmpeg image (or a real, non-synthetic input) only if/when actual working NVDEC decode output is needed, not just the capability fix.

## Bottom line

- The capability fix is real, verified, and safe to apply to any task that needs it — no infra change, no shared-container risk.
- Actually getting a clean multi-frame NVDEC decode still needs a better base image than the one used for this scratch test.
- The `frame-flay` pipeline's `extract-candidates` job currently uses plain CPU `ffmpeg` decode, which is already proven adequate (~13 hours of source video extracted in ~25 minutes wall-clock for the first real batch). Switching it to NVDEC is a genuinely available option now, but not yet done — pending a decision on whether the ingestion volume justifies it, plus picking/validating a properly NVDEC-capable ffmpeg image.
