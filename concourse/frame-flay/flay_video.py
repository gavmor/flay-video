#!/usr/bin/env python3
"""frame-flay: semantic keyframe selection from a folder of candidate stills.

Quality-filters candidate frames (drops blurry/near-black shots), embeds
survivors with CLIP ViT-L/14, clusters the embeddings with HDBSCAN
(density-based, so pans/crossfades/near-duplicates fall out as the -1
"noise" label instead of needing to be hand-detected), then keeps the
frame closest to each cluster's centroid as that scene's keyframe.

Runs as a plain inline Concourse task step (no persistent service, no
HTTP) -- input stills arrive via an s3 `get`, output lands in the task's
own output directory for a subsequent `put`.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_filter(input_dir: Path, blur_threshold: float, dark_threshold: float):
    frames, paths = [], []
    dropped_blur, dropped_dark = 0, 0
    for f in sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.jpeg")) + sorted(input_dir.glob("*.png")):
        img_bgr = cv2.imread(str(f))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = gray.mean()
        if sharpness <= blur_threshold:
            dropped_blur += 1
            continue
        if brightness <= dark_threshold:
            dropped_dark += 1
            continue
        frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        paths.append(f)
    return frames, paths, dropped_blur, dropped_dark


def embed_frames(frames, model_id: str, batch_size: int):
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = [Image.fromarray(f) for f in frames[i:i + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(embeds.cpu().numpy())
    return np.vstack(embeddings)


def save_keyframe(frame: np.ndarray, max_edge: int, dest: Path) -> None:
    img = Image.fromarray(frame)
    if max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--blur-threshold", type=float, default=100.0)
    ap.add_argument("--dark-threshold", type=float, default=15.0)
    ap.add_argument("--max-edge", type=int, default=768)
    ap.add_argument("--min-cluster-size", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    start = time.time()
    print(f"--- device: {DEVICE} ---")

    frames, paths, dropped_blur, dropped_dark = load_and_filter(
        args.input_dir, args.blur_threshold, args.dark_threshold
    )
    sampled_total = len(frames) + dropped_blur + dropped_dark
    print(f"--- {sampled_total} candidates, {len(frames)} survived quality filter "
          f"({dropped_blur} dropped blurry, {dropped_dark} dropped dark) ---")
    if not frames:
        raise SystemExit("no frames survived quality filtering")

    print(f"--- embedding {len(frames)} frames with {args.model_id} ---")
    embeddings = embed_frames(frames, args.model_id, args.batch_size)

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
        original_idx = int(idxs[int(np.argmin(distances))])

        src_path = paths[original_idx]
        out_name = f"cluster{cluster_id:04d}_{src_path.stem}.jpg"
        out_path = args.output_dir / out_name
        save_keyframe(frames[original_idx], args.max_edge, out_path)

        clusters.append({
            "cluster_id": int(cluster_id),
            "size": int(len(idxs)),
            "source_still": src_path.name,
            "output_path": str(out_path),
        })

    manifest = {
        "input_dir": str(args.input_dir),
        "params": vars(args) | {"input_dir": str(args.input_dir), "output_dir": str(args.output_dir)},
        "frames_sampled": sampled_total,
        "frames_after_quality_filter": len(frames),
        "frames_dropped_as_noise": noise_count,
        "clusters": clusters,
        "elapsed_s": time.time() - start,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"--- wrote {len(clusters)} keyframes + manifest.json to {args.output_dir} "
          f"in {manifest['elapsed_s']:.1f}s ---")


if __name__ == "__main__":
    main()
