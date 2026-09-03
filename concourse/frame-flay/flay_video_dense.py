#!/usr/bin/env python3
"""flay-video exhaustive mode: process every decoded frame, not a sparse sample.

Zero-copy GPU-resident pipeline: PyNvVideoCodec (NVDEC hardware decode) hands
decoded NV12 frames to CV-CUDA (color conversion + resize) via DLPack, which
hands the result straight to CLIP as a CUDA tensor -- no frame ever touches
host memory or disk until after clustering, when only the handful of selected
keyframes get re-decoded and saved as JPEGs.

*** KNOWN BLOCKING BUG (2026-09-01, not yet resolved) ***
The zero-copy decode->convert->embed chain itself is verified correct
(confirmed via multiple isolated `fly execute` smoke tests: real frames
decode, convert to RGB with correct shape/values, and feed a real CLIP
forward pass). But PyNvVideoCodec 2.2.2 / cvcuda-cu12 0.17.0's native
objects (the DecodedFrame, cvcuda.Tensor, and cvcuda.ExternalBuffer chain
built in nv12_frame_to_rgb_nhwc) crash the whole Python interpreter
("Fatal Python error: PyThreadState_Get: the function must be called with
the GIL held, but the GIL is released") the moment ANY of them is released
-- confirmed this is not limited to Python's cyclic garbage collector
(gc.disable() does not prevent it), not limited to crossing a function
-return boundary (inlining everything at module scope does not prevent
it), and happens even from a plain list reassignment (`keepalive = []`)
between batches in the exact same scope. The only thing that has worked in
isolated testing is never releasing ANY of these objects until the whole
process calls os._exit(0) -- which does not scale to real "exhaustive,
large corpus" processing: holding every frame's raw NV12 + converted RGB
buffers alive simultaneously for a 13-hour/~1.12M-frame source would need
roughly 10TB of VRAM.

This script currently keeps every batch's intermediate objects in a single
process-lifetime `KEEPALIVE` list (never cleared) as the least-bad known
mitigation -- meaning it can process a bounded number of frames/batches
correctly (bounded by real VRAM, use --max-frames to cap this explicitly)
but will eventually OOM on a genuinely large corpus, not just slow down.
This is NOT yet the production-ready "sample 100% of frames of large
corpuses" capability that was the actual goal -- it's the verified-correct
core algorithm with an open, well-characterized upstream library bug
blocking real scale. Options for resolving this, not yet attempted:
report upstream to the PyNvVideoCodec/cvcuda projects, try the
ThreadedDecoder class instead of SimpleDecoder, or try different package
versions once ones are available that don't have this bug.

Two passes, deliberately:
  1. Decode + embed every frame, discard pixels immediately after embedding
     (only the embedding vector + (source file, frame index) survive) --
     this is what makes exhaustive processing of a multi-hour, multi-file
     corpus tractable in VRAM: embeddings for 1M+ frames are a few GB in
     host RAM, but 1M+ raw 1080p frames are not. (The KNOWN BLOCKING BUG
     above currently defeats this VRAM-tractability goal for the pixel
     buffers themselves, even though the embeddings-only design is sound.)

  Note on naming: per docs/adr/0001-reframe-exhaustive-as-dense-and-research.md,
  this script is the foundation of the "dense" production variant
  (VRAM-bounded, ~1300 frames on 24GB) and the "100% of frames" research
  direction (subprocess-per-batch orchestrator, separate branch). The
  file name still says "exhaustive" for git-history continuity; the
  job that runs it has been retitled accordingly. Do not interpret
  the file name as a promise of full-corpus coverage in a single process.

  2. HDBSCAN-cluster the full embedding set, then re-seek and re-decode only
     the winning frame per cluster to save as a keyframe JPEG -- cheaper
     than holding every frame's pixels around for the whole run.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

# Must be set before importing transformers/huggingface_hub: disables the
# background tqdm progress-bar monitor thread that from_pretrained() spawns
# during weight download. Confirmed directly (via a crash stack trace) that
# thread is still alive -- waiting, not working, but alive -- at the exact
# moment the fatal PyThreadState_Get/GIL-released crash first hit, on the
# very first NVDEC->CV-CUDA->torch frame conversion. Plausible GIL-release
# race between that thread and the native decode/cvcuda call chain, not
# confirmed root cause but a real, testable, low-risk thing to eliminate.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import cvcuda
import numpy as np
import PyNvVideoCodec as nvc
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN
from transformers import CLIPVisionModelWithProjection

DEVICE = "cuda"

# Process-lifetime, never-cleared -- see the KNOWN BLOCKING BUG note above.
# Every intermediate cvcuda/PyNvVideoCodec object gets appended here and
# NEVER removed; the list (and the VRAM it pins) is only actually freed by
# the process exiting. This bounds real usable input size to whatever fits
# in VRAM simultaneously -- use --max-frames to cap this explicitly rather
# than let it OOM uncontrolled.
KEEPALIVE: list = []

# Standard openai/clip-vit-large-patch14 preprocessing constants -- doing this
# manually with torch ops (resize + normalize, both GPU-resident) rather than
# routing through CLIPProcessor, which expects PIL/numpy CPU input and would
# force a host round-trip that defeats the whole point of this script.
CLIP_INPUT_SIZE = 224
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)


def nv12_frame_to_rgb_nhwc(frame) -> torch.Tensor:
    """DecodedFrame (NV12, packed HxWx1.5) -> (1, H, W, 3) uint8 CUDA tensor.

    cvcuda.cvtcolor rejects the raw (H*1.5, W) tensor as-is -- it needs an
    explicit NHWC reshape to (1, H*1.5, W, 1) first. Found by inspecting
    cvcuda.cvtcolor.__doc__ directly (never call help() in a non-interactive
    Concourse task -- it hangs forever on pydoc's pager waiting for stdin
    that will never arrive).

    Every intermediate object below is appended to the module-level
    KEEPALIVE list and NEVER released for the life of the process -- see
    the KNOWN BLOCKING BUG note at the top of this file. Confirmed directly
    (extensive isolated testing, not a guess) that releasing ANY of these
    -- via cyclic GC, a function return, or even a same-scope list
    reassignment -- crashes the whole interpreter with "Fatal Python
    error: PyThreadState_Get: the function must be called with the GIL
    held, but the GIL is released", a real bug in the native cvcuda/
    PyNvVideoCodec cleanup path, not a usage error on this script's part.
    """
    raw = torch.from_dlpack(frame)  # (H*1.5, W) uint8, NV12 packed
    h15, w = raw.shape
    reshaped = raw.view(1, h15, w, 1).contiguous()
    cv_in = cvcuda.as_tensor(reshaped, "NHWC")
    cv_out = cvcuda.cvtcolor(cv_in, cvcuda.ColorConversion.YUV2RGB_NV12)
    # cvcuda.Tensor itself doesn't expose a usable __dlpack__ (torch.from_dlpack
    # on it directly raises "invalid capsule... already consumed", verified by
    # isolating this exact call) -- .cuda() converts it to a cvcuda.ExternalBuffer
    # first, which does support both __cuda_array_interface__ and __dlpack__.
    ext = cv_out.cuda()
    result = torch.from_dlpack(ext)  # (1, H, W, 3) uint8 CUDA
    KEEPALIVE.extend([raw, reshaped, cv_in, cv_out, ext])
    return result


def rgb_nhwc_to_clip_input(rgb_nhwc: torch.Tensor) -> torch.Tensor:
    """(1, H, W, 3) uint8 CUDA -> (1, 3, 224, 224) float32 CUDA, CLIP-normalized."""
    chw = rgb_nhwc.permute(0, 3, 1, 2).float() / 255.0  # (1, 3, H, W)
    resized = torch.nn.functional.interpolate(
        chw, size=(CLIP_INPUT_SIZE, CLIP_INPUT_SIZE), mode="bilinear", align_corners=False
    )
    return (resized - CLIP_MEAN) / CLIP_STD


def process_source(video_path: Path, model, batch_size: int, blur_threshold: float, dark_threshold: float, max_frames: int):
    """Decode + embed every frame of one source video. Returns embeddings + metadata, drops pixels."""
    dec = nvc.SimpleDecoder(str(video_path), use_device_memory=True)
    meta = dec.get_stream_metadata()
    total_frames = min(meta.num_frames, max_frames) if max_frames else meta.num_frames
    print(f"--- {video_path.name}: {total_frames}/{meta.num_frames} frames, "
          f"{meta.width}x{meta.height} @ {meta.average_fps:.2f}fps ---")

    embeddings = []
    frame_indices = []
    dropped_dark, dropped_blur = 0, 0

    idx = 0
    with torch.no_grad():
        while idx < total_frames:
            n = min(batch_size, total_frames - idx)
            frames = dec.get_batch_frames(n)
            if not frames:
                break

            batch_clip_inputs = []
            batch_kept_indices = []
            for i, f in enumerate(frames):
                rgb = nv12_frame_to_rgb_nhwc(f)  # (1, H, W, 3) uint8 CUDA
                # GPU-side quality filter: mean brightness (dark) + a cheap
                # Laplacian-variance proxy (blur) via a fixed conv kernel,
                # same semantics as the sparse pipeline's cv2-based filter
                # (flay_video.py) but computed without leaving the GPU.
                gray = rgb.float().mean(dim=3, keepdim=True)  # (1, H, W, 1)
                brightness = gray.mean().item()
                if brightness <= dark_threshold:
                    dropped_dark += 1
                    continue
                lap_kernel = torch.tensor(
                    [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=DEVICE, dtype=torch.float32
                ).view(1, 1, 3, 3)
                gray_chw = gray.permute(0, 3, 1, 2)  # (1, 1, H, W)
                lap = torch.nn.functional.conv2d(gray_chw, lap_kernel, padding=1)
                sharpness = lap.var().item()
                if sharpness <= blur_threshold:
                    dropped_blur += 1
                    continue

                batch_clip_inputs.append(rgb_nhwc_to_clip_input(rgb))
                batch_kept_indices.append(idx + i)

            if batch_clip_inputs:
                clip_batch = torch.cat(batch_clip_inputs, dim=0)
                out = model(pixel_values=clip_batch).image_embeds
                out = out / out.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(out.cpu().numpy())
                frame_indices.extend(batch_kept_indices)

            idx += n
            if idx % (batch_size * 20) == 0 or idx >= total_frames:
                print(f"    {idx}/{total_frames} decoded, {len(frame_indices)} embedded so far")

    # Deliberately NOT calling dec.stop() -- confirmed broken in
    # PyNvVideoCodec 2.2.2 (raises AttributeError: '_PyNvVideoCodec.
    # SimpleDecoder' object has no attribute 'stop', a real bug in the
    # package itself, not a usage error). The decoder object is left for
    # normal Python GC; see main()'s os._exit(0) for why that's fine here.
    print(f"--- {video_path.name}: done, {len(frame_indices)} embedded, "
          f"{dropped_dark} dropped dark, {dropped_blur} dropped blurry ---")
    if not embeddings:
        return np.empty((0, 768), dtype=np.float32), [], total_frames
    return np.vstack(embeddings), frame_indices, total_frames


def save_keyframe(video_path: Path, frame_index: int, max_edge: int, dest: Path) -> None:
    """Re-seek and re-decode a single winning frame to save as a JPEG."""
    dec = nvc.SimpleDecoder(str(video_path), use_device_memory=True)
    frame = dec.get_batch_frames_by_index([frame_index])[0]
    rgb_nhwc = nv12_frame_to_rgb_nhwc(frame)  # (1, H, W, 3) uint8 CUDA
    arr = rgb_nhwc[0].cpu().numpy()  # only host round-trip in the whole script, and only for the ~50-150 winners
    # No dec.stop() -- see process_source's comment, same broken API.

    img = Image.fromarray(arr)
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=92)


def main():
    # gc.disable(): belt-and-suspenders alongside the module-level KEEPALIVE
    # list (see the KNOWN BLOCKING BUG note at the top of this file) --
    # PyNvVideoCodec/cvcuda's native objects crash the interpreter the
    # moment any of them is released, whether that release comes from
    # cyclic GC, a function return, or a same-scope reassignment. KEEPALIVE
    # already prevents all three by never releasing anything; gc.disable()
    # additionally stops Python's cyclic collector from running at all,
    # removing one more path by which something could get swept
    # unexpectedly. Not a leak risk beyond what KEEPALIVE already accepts:
    # the process exits via os._exit(0) below rather than relying on any
    # GC-driven cleanup.
    gc.disable()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path, help="dir of raw video files (all *.mp4 processed)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch-size", type=int, default=16, help="decode+embed batch size")
    ap.add_argument("--blur-threshold", type=float, default=15.0)
    ap.add_argument("--dark-threshold", type=float, default=15.0)
    ap.add_argument("--max-edge", type=int, default=768)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=0,
                     help="cap frames processed per source (0 = no cap). Real safety valve "
                          "given the KNOWN BLOCKING BUG documented at the top of this file -- "
                          "KEEPALIVE grows unboundedly per frame processed, so an uncapped run "
                          "against a large source will OOM rather than complete. Note: on a 24GB "
                          "card the actual VRAM ceiling (~1300 frames at 1080p) is lower than "
                          "any --max-frames value you might pick; this knob is a refinement on "
                          "top of the VRAM ceiling, not a way around it. See "
                          "docs/exhaustive-scaling-options.md for the math.")
    args = ap.parse_args()

    start = time.time()
    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 files found in {args.input_dir}")

    model = CLIPVisionModelWithProjection.from_pretrained(args.model_id).to(DEVICE)
    model.eval()
    # Confirmed directly (isolated repro, not a guess): without this sync,
    # the first PyNvVideoCodec/cvcuda call after CLIP model loading crashes
    # the interpreter ("Fatal Python error: PyThreadState_Get... GIL
    # released") -- pending async CUDA work from model loading (weight
    # transfer, kernel init) appears to race with the native decoder/cvcuda
    # extensions' own CUDA calls. Forcing full completion of the model-load
    # work before touching either library removes the crash entirely.
    torch.cuda.synchronize()

    all_embeddings = []
    all_refs = []  # (video_path, frame_index) per embedding, same order as all_embeddings
    total_sampled = 0
    for video_path in videos:
        emb, indices, n_frames = process_source(
            video_path, model, args.batch_size, args.blur_threshold, args.dark_threshold, args.max_frames
        )
        total_sampled += n_frames
        all_embeddings.append(emb)
        all_refs.extend((video_path, i) for i in indices)

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768), dtype=np.float32)
    print(f"--- {len(embeddings)}/{total_sampled} total frames embedded across {len(videos)} source(s) ---")
    if len(embeddings) == 0:
        raise SystemExit("no frames survived quality filtering across all inputs")

    hdbscan_kwargs = {"min_cluster_size": args.min_cluster_size, "metric": "euclidean"}
    if args.min_samples is not None:
        hdbscan_kwargs["min_samples"] = args.min_samples
    print(f"--- clustering with HDBSCAN({hdbscan_kwargs}) ---")
    labels = HDBSCAN(**hdbscan_kwargs).fit_predict(embeddings)

    unique_clusters = sorted(c for c in set(labels) if c != -1)
    noise_count = int(np.sum(labels == -1))
    print(f"--- {len(unique_clusters)} clusters, {noise_count} frames dropped as noise ---")

    clusters = []
    for cluster_id in unique_clusters:
        idxs = np.where(labels == cluster_id)[0]
        centroid = embeddings[idxs].mean(axis=0)
        distances = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
        winner = int(idxs[int(np.argmin(distances))])
        video_path, frame_index = all_refs[winner]

        out_name = f"cluster{cluster_id:04d}_{video_path.stem}_f{frame_index:06d}.jpg"
        out_path = args.output_dir / out_name
        save_keyframe(video_path, frame_index, args.max_edge, out_path)

        clusters.append({
            "cluster_id": int(cluster_id),
            "size": int(len(idxs)),
            "source_video": video_path.name,
            "frame_index": frame_index,
            "output_path": str(out_path),
        })

    manifest = {
        # See docs/adr/0001-reframe-exhaustive-as-dense-and-research.md.
        # "dense" = the bounded VRAM ceiling path; "exhaustive" was
        # honest about the algorithm but misleading about the scale it
        # can actually run at without depending on the upstream cvcuda
        # release-bug fix. Keep this in sync if the script's job title
        # changes again.
        "mode": "dense",
        "sources": [v.name for v in videos],
        "params": vars(args) | {"input_dir": str(args.input_dir), "output_dir": str(args.output_dir)},
        "frames_sampled": total_sampled,
        "frames_after_quality_filter": len(embeddings),
        "frames_dropped_as_noise": noise_count,
        "clusters": clusters,
        "elapsed_s": time.time() - start,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"--- wrote {len(clusters)} keyframes + manifest.json to {args.output_dir} "
          f"in {manifest['elapsed_s']:.1f}s ---")

    # os._exit(0) rather than a normal return: PyNvVideoCodec/cvcuda's native
    # objects crash the interpreter during ordinary shutdown/finalization
    # (confirmed directly, reproducible, independent of whether dec.stop()
    # is called -- "Fatal Python error: PyThreadState_Get... GIL released").
    # All real output (manifest.json + keyframe JPEGs) is already flushed to
    # disk above by this point, so os._exit(0) -- which skips Python's
    # normal cleanup/finalization entirely -- is the correct fix here, not
    # a workaround masking a real failure: Concourse's `set -e` on the task
    # script needs a clean exit code, and a SIGABRT-crashing interpreter
    # during finalization would fail the whole task despite the actual work
    # having already succeeded.
    os._exit(0)


if __name__ == "__main__":
    main()
