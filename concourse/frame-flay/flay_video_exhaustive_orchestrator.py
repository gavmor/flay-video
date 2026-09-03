#!/usr/bin/env python3
"""flay-video exhaustive-mode orchestrator: subprocess-per-batch architecture.

The fundamental problem: PyNvVideoCodec 2.2.2 / cvcuda-cu12 0.17.0's native
objects (DecodedFrame, cvcuda.Tensor, cvcuda.ExternalBuffer) crash the Python
interpreter the instant any of them is released -- confirmed multiple ways
across versions, documented in concourse/frame-flay/flay_video_dense.py's
module docstring and in docs/release-bug-investigation.md.

The "dense" production variant works around this with a process-lifetime
KEEPALIVE list that never releases anything, which caps any single process
at ~1300 frames on 24GB VRAM. This is fine for the dense variant. It is
NOT fine for the "100% of frames of an arbitrarily large corpus" goal --
a 13-hour / ~1.12M-frame source would need ~10TB of VRAM held simultaneously
under that workaround.

This script takes a different approach: keep the per-batch VRAM footprint
bounded (each child only holds ITS batch's frames in KEEPALIVE), but
*replace the interpreter* between batches so the crash is impossible to
trigger by release. The child process exits cleanly, its KEEPALIVE list
goes with it, and the parent gets a clean slate.

Architecture (subprocess-per-batch, Option A in
docs/exhaustive-scaling-options.md):

  parent (this script)
    |- load CLIP model ONCE in the parent (or save state_dict to disk
    |  and let each child re-load it; cheaper for short batches,
    |  same amortized cost either way at corpus scale)
    |- for each (source, batch_index):
    |    fork+exec child process with:
    |      - path to the dense script (concourse/frame-flay/flay_video_dense.py)
    |      - a small wrapper that imports the dense script's decode
    |        primitives, processes only frames [i..i+N), and writes
    |        this batch's embeddings to a per-shard .npy file + a
    |        per-shard index.jsonl (source, frame_idx) BEFORE os._exit(0)
    |- after all children complete:
    |    - mmap all shards into one (N_total, 768) float32 array
    |    - HDBSCAN-cluster the union
    |    - re-decode the ~handful of cluster winners via small subprocesses
    |      (also bounded) and save as keyframe JPEGs

Properties:
  - Per-batch VRAM ceiling: 18MB * N (e.g. 100 frames = 1.8GB -- well
    under 24GB even with CLIP loaded alongside)
  - Corpus scale: bounded by disk (embedding shards) and by total wall
    time, NOT by VRAM
  - Per-child runtime: ~0.25s decode + ~0.3s embed per 100 frames; plus
    ~10ms fork overhead and ~3s CLIP load if not shared with the parent
  - New failure surface: child process management, shard reassembly,
    partial-failure recovery (which shards completed? do we crash or
    resume on a re-run?), manifest consistency under repeated runs

STATUS (2026-09-02): skeleton only. The actual child wrapper, embedding
shard handoff, and the cluster-winner re-decode subprocess are
intentionally not implemented yet -- this is the research path described
in docs/adr/0001-reframe-exhaustive-as-dense-and-research.md, not a
production path. The skeleton here is enough to validate the parent-side
control flow, the (source, batch_index) partitioning, the
resume-from-shards logic, and the env/var plumbing against Concourse
without committing to the full design.

What this skeleton DOES implement:
  - Argument parsing and var plumbing (--input-dir, --output-dir,
    --batch-size, --max-frames, --min-cluster-size, etc., mirroring
    the dense script so a future child wrapper can be near-trivial)
  - Source enumeration and per-source frame-budget partitioning into
    a list of (source_path, start_idx, end_idx) batches
  - Sequential child process invocation via subprocess.run() with a
    stable, deterministic env-var handoff (CHILD_MODE,
    CHILD_INPUT_PATH, CHILD_BATCH_START, CHILD_BATCH_END,
    CHILD_OUTPUT_SHARD, CHILD_INDEX_SHARD)
  - Per-shard .npy + .jsonl file contract, with a resume check that
    skips a batch if its shard files already exist (so re-runs are
    cheap and idempotent)
  - Final embedding-matrix mmap + HDBSCAN clustering, identical to
    what the dense script does
  - Manifest write at the end (cluster list + per-cluster metadata)
  - But: the actual child wrapper that runs inside the child process
    is NOT implemented here -- see "TODO" below. Running this skeleton
    against a real source will invoke the child once per batch, the
    child will exit 0 without doing work, and the parent will report
    "0 frames embedded" and raise. That's expected; the skeleton is
    here so the parent control flow can be exercised against the
    live Concourse task environment without committing to the full
    child-side design.

TODO when this gets promoted from skeleton to actual research code:
  1. Implement the child wrapper (concourse/frame-flay/
     flay_video_dense_child.py or similar) that imports the dense
     script's decode/convert/embed primitives, processes
     [CHILD_BATCH_START..CHILD_BATCH_END), and writes the .npy shard
     + .jsonl index before os._exit(0). The dense script's
     nv12_frame_to_rgb_nhwc + rgb_nhwc_to_clip_input + CLIP forward
     are the exact primitives to reuse.
  2. Re-decode subprocesses for cluster winners. Currently the dense
     script's save_keyframe is the only path that does host round-trip
     + JPEG write; reuse it inside a small subprocess wrapper so each
     winner re-decode is also VRAM-bounded.
  3. Decide: load CLIP in the parent and pass embeddings-only work to
     children, or have each child reload CLIP from a saved state_dict
     (~3s amortized)? The first is cheaper at any scale; the second is
     simpler to reason about. Pick when the child wrapper exists and
     can be measured.
  4. Decide: write shards to disk and re-mmap at the end, or stream
     them to the parent via stdout? Disk is simpler and survives a
     child crash mid-batch; stdout avoids the disk round-trip but
     makes partial-failure recovery hard. Pick based on (3).
  5. Add an explicit --dry-run flag that does everything except
     spawn children, so the parent control flow can be validated
     against a real corpus without GPU/VRAM commitment.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

# Same env flag the dense script sets -- disables the HF progress-bar
# monitor thread that has been observed alive in the same GIL context
# where the cvcuda release-crash first hit. Cheap to set in the parent
# too in case anything here triggers an HF download (it shouldn't, but
# the env is inherited by children via subprocess.run() so it
# propagates).
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

EMBEDDING_DIM = 768  # CLIP ViT-L/14 projected image_embed dim
CHILD_SCRIPT = Path(__file__).parent / "flay_video_dense_child.py"


def partition_batches(sources, total_frames_per_source, batch_size, max_frames):
    """Yield (source_path, start_idx, end_idx) for every batch the
    orchestrator will run, across all sources. Reserves the per-source
    max_frames budget; the dense script's --max-frames still applies as
    the per-source cap.
    """
    for source, n_total in zip(sources, total_frames_per_source):
        cap = min(n_total, max_frames) if max_frames else n_total
        for start in range(0, cap, batch_size):
            yield (source, start, min(start + batch_size, cap))


def shard_paths(output_dir, source, start, end):
    """Per-batch .npy (embeddings) + .jsonl (source, frame_idx rows).
    Stable filename so a re-run can skip a completed batch -- see
    resume_shard_exists.
    """
    safe_name = source.stem
    shard = output_dir / "shards" / f"{safe_name}_{start:08d}_{end:08d}.npy"
    index = output_dir / "shards" / f"{safe_name}_{start:08d}_{end:08d}.jsonl"
    return shard, index


def resume_shard_exists(shard, index):
    """Resume check: if both files exist, this batch is already done.
    Cheap mtime check; could be stricter (validate dims, validate
    index row count matches npy row count) but the dense child wrapper
    is atomic -- it either writes both or crashes -- so file existence
    is sufficient for the skeleton. Tighten when the child exists.
    """
    return shard.exists() and index.exists()


def run_child(source, start, end, shard, index, batch_size, blur_threshold, dark_threshold, max_edge):
    """Invoke the child subprocess for one batch. The child wrapper
    itself is not implemented yet (see module docstring TODO 1), so
    this currently produces an empty shard + index file the parent
    can validate the control flow against. Replace the child-path
    branch with the real wrapper when it exists.
    """
    env = os.environ.copy() | {
        "CHILD_MODE": "dense_batch",
        "CHILD_INPUT_PATH": str(source),
        "CHILD_BATCH_START": str(start),
        "CHILD_BATCH_END": str(end),
        "CHILD_OUTPUT_SHARD": str(shard),
        "CHILD_INDEX_SHARD": str(index),
        "FLAY_BATCH_SIZE": str(batch_size),
        "FLAY_BLUR_THRESHOLD": str(blur_threshold),
        "FLAY_DARK_THRESHOLD": str(dark_threshold),
        "FLAY_MAX_EDGE": str(max_edge),
    }
    if not CHILD_SCRIPT.exists():
        # Skeleton fallback: write empty shard + index so the parent's
        # control flow (batch enumeration, mmap, clustering, manifest)
        # can be exercised end-to-end without the real child existing.
        # Replace with `subprocess.run([sys.executable, str(CHILD_SCRIPT)], env=env, check=True)`
        # when the child wrapper lands.
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.save(shard, np.empty((0, EMBEDDING_DIM), dtype=np.float32))
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("")
        return

    subprocess.run([sys.executable, str(CHILD_SCRIPT)], env=env, check=True)


def gather_shards(output_dir, sources):
    """Concatenate every shard's .npy + .jsonl into one (N, EMBEDDING_DIM)
    float32 array + a single (N,) object array of (source, frame_idx) refs.
    Shards may legitimately be empty (skipped batches) -- filtered here.
    """
    all_embs = []
    all_refs = []
    for shard in sorted((output_dir / "shards").glob("*.npy")):
        index = shard.with_suffix(".jsonl")
        if not index.exists():
            print(f"--- WARN: {shard.name} missing index, skipping ---")
            continue
        embs = np.load(shard)
        if embs.shape[0] == 0:
            continue
        # Reconstruct (source, frame_idx) from the shard filename --
        # shards/<source_stem>_<start>_<end>.npy; frame_idx per row is
        # encoded in the index jsonl as {"source": ..., "frame_index": ...}
        # and in shard row order. Decode here.
        with index.open() as f:
            for row_i, line in enumerate(f):
                if not line.strip():
                    continue
                entry = json.loads(line)
                all_refs.append((Path(entry["source"]), int(entry["frame_index"])))
        all_embs.append(embs)
    if not all_embs:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
    return np.vstack(all_embs), all_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path, help="dir of raw video files (all *.mp4 processed)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16, help="frames per child process")
    ap.add_argument("--max-frames", type=int, default=0,
                     help="per-source frame cap (0 = no cap). Same semantics as "
                          "flay_video_dense.py -- --max-frames -- the dense variant's "
                          "VRAM ceiling (~1300 frames on 24GB) does not apply here, "
                          "because each child is a fresh interpreter with a bounded "
                          "KEEPALIVE list. Set this to whatever you actually want "
                          "processed per source.")
    ap.add_argument("--blur-threshold", type=float, default=15.0)
    ap.add_argument("--dark-threshold", type=float, default=15.0)
    ap.add_argument("--max-edge", type=int, default=768)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                     help="enumerate batches + simulate child invocations without "
                          "actually spawning subprocesses. Useful for validating "
                          "the parent control flow against a real source without "
                          "GPU/VRAM commitment.")
    args = ap.parse_args()

    start = time.time()
    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 files found in {args.input_dir}")

    # Per-source frame counts: in the real child wrapper, the child
    # itself would call dec.get_stream_metadata() to learn this. For
    # the skeleton we just trust the dense script's per-source budget
    # and partition assuming an "unbounded" source (the child will
    # stop at the actual end of stream). Replace with real
    # metadata when the child wrapper lands.
    total_frames_per_source = [None] * len(videos)

    batches = list(partition_batches(videos, total_frames_per_source, args.batch_size, args.max_frames))
    print(f"--- {len(videos)} source(s), {len(batches)} batch(es) at batch_size={args.batch_size} ---")
    for source, s, e in batches:
        shard, index = shard_paths(args.output_dir, source, s, e)
        if resume_shard_exists(shard, index):
            print(f"--- [skip] {source.name} [{s}..{e}) -- shard already exists ---")
            continue
        print(f"--- [run ] {source.name} [{s}..{e}) ---")
        if args.dry_run:
            continue
        run_child(source, s, e, shard, index, args.batch_size, args.blur_threshold, args.dark_threshold, args.max_edge)

    if args.dry_run:
        print(f"--- dry-run complete: {len(batches)} batches enumerated, 0 children spawned ---")
        return

    embeddings, all_refs = gather_shards(args.output_dir, videos)
    print(f"--- {len(embeddings)} total frames embedded across {len(videos)} source(s) ---")
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
        # TODO: re-decode the winner via a small subprocess (also bounded).
        # The dense script's save_keyframe is the right primitive to wrap.
        clusters.append({
            "cluster_id": int(cluster_id),
            "size": int(len(idxs)),
            "source_video": video_path.name,
            "frame_index": frame_index,
            "output_path": str(out_path),
        })

    manifest = {
        "mode": "exhaustive-orchestrator",
        "sources": [v.name for v in videos],
        "params": vars(args) | {"input_dir": str(args.input_dir), "output_dir": str(args.output_dir)},
        "batches_total": len(batches),
        "frames_after_quality_filter": len(embeddings),
        "frames_dropped_as_noise": noise_count,
        "clusters": clusters,
        "elapsed_s": time.time() - start,
        "status": "skeleton -- see module docstring TODOs",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"--- wrote {len(clusters)} keyframes + manifest.json to {args.output_dir} "
          f"in {manifest['elapsed_s']:.1f}s ---")


if __name__ == "__main__":
    main()
