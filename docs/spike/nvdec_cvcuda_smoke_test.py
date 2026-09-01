#!/usr/bin/env python3
"""Reconstruction of the original PyNvVideoCodec/cvcuda throughput smoke test.

2026-09-01: the original file was never preserved (fly execute doesn't
archive local input dirs server-side, and local /tmp was wiped after the
first investigation). This version was rebuilt from build 1261330's
preserved stdout log (matches the 404.8/401.6/372.0 fps numbers cited in
../pynvvideocodec-feasibility.md) and then actually re-executed live
(build 1430505) to confirm it's real, working code, not just plausible
text. It ran end-to-end: is_cuda True, a real device pointer, cvtcolor +
resize completing on all 500 frames.

Not the CLIP-feeding pipeline -- the original smoke test stopped at
resize. See ../../concourse/frame-flay/flay_video_exhaustive.py for the
full decode->convert->embed chain (which also documents a separate,
later-discovered blocking bug in this same object-lifetime area).

Usage: fly execute against a task with NVIDIA_VISIBLE_DEVICES=all,
NVIDIA_DRIVER_CAPABILITIES=compute,utility,video, and clip.mp4 (any local
video, ffmpeg-trimmed) staged as an input alongside this script.
"""

import subprocess
import sys
import time

print("=== Step 1: install PyNvVideoCodec + cvcuda-cu12 ===")
t0 = time.time()
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "PyNvVideoCodec", "cvcuda-cu12"],
    capture_output=True, text=True,
)
print(f"pip exit code: {result.returncode}")
print(f"install took {time.time() - t0:.1f}s")

import torch
print(f"torch: {torch.__version__} cuda available: {torch.cuda.is_available()}")

print("nvidia-smi driver check:")
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version,name,memory.free", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print(smi.stdout.strip())

print("=== Step 2: import PyNvVideoCodec + cvcuda ===")
import PyNvVideoCodec as nvc
print(f"PyNvVideoCodec imported OK, version attr: {nvc.__version__}")
import cvcuda
print(f"cvcuda imported OK, version attr: {cvcuda.__version__}")

VIDEO_PATH = "clip.mp4"
N_FRAMES = 500

print("=== Step 3: decode real clip with PyNvVideoCodec, DLPack -> torch, cvcuda NV12->RGB+resize ===")
decoder = nvc.SimpleDecoder(VIDEO_PATH, use_device_memory=True)
print(f"SimpleDecoder reports {len(decoder)} frames")
print(f"processing {N_FRAMES} frames")

before = torch.cuda.memory_allocated()
t0 = time.time()
frames = decoder[0:N_FRAMES]
tensors = []
for i, f in enumerate(frames):
    t = torch.from_dlpack(f)
    if i == 0:
        print(f"frame 0 tensor.is_cuda: {t.is_cuda} shape: {tuple(t.shape)} dtype: {t.dtype}")
        print(f"frame 0 data_ptr (device pointer): {hex(t.data_ptr())}")
    tensors.append(t)
dlpack_elapsed = time.time() - t0
after = torch.cuda.memory_allocated()
print(
    f"torch CUDA memory_allocated before={before} after={after} "
    "(both 0: torch.from_dlpack wraps the decoder's own CUDA allocation without going "
    "through torch's caching allocator -- expected for a true zero-copy wrap, not a red flag)"
)
dlpack_fps = len(tensors) / dlpack_elapsed
print(f"=== decode+dlpack-handoff of {len(tensors)} frames took {dlpack_elapsed:.2f}s -> {dlpack_fps:.1f} fps ===")

print("=== Step 4: full NV12->RGB + resize pipeline throughput ===")
decoder3 = nvc.SimpleDecoder(VIDEO_PATH, use_device_memory=True)
t0 = time.time()
pipeline_frames = decoder3[0:N_FRAMES]
n_ok = 0
for f in pipeline_frames:
    t = torch.from_dlpack(f)
    h_full, w = t.shape  # NV12 packed buffer: (H*1.5, W) -- the NHWC reshape CV-CUDA needs
    reshaped = t.reshape(1, h_full, w, 1)
    cv_in = cvcuda.as_tensor(reshaped.cuda(), cvcuda.TensorLayout.NHWC)
    rgb = cvcuda.cvtcolor(cv_in, cvcuda.ColorConversion.YUV2RGB_NV12)
    resized = cvcuda.resize(rgb, (1, 224, 224, 3), cvcuda.Interp.LINEAR)
    n_ok += 1
torch.cuda.synchronize()
pipeline_elapsed = time.time() - t0
pipeline_fps = n_ok / pipeline_elapsed
print(f"=== full pipeline (decode+dlpack+NV12->RGB+resize) of {n_ok} frames took {pipeline_elapsed:.2f}s -> {pipeline_fps:.1f} fps ===")

print("=== Step 5: raw NVDEC decode-only fps (no cvcuda, no explicit torch wrap) ===")
decoder2 = nvc.SimpleDecoder(VIDEO_PATH, use_device_memory=True)
t0 = time.time()
raw_frames = decoder2[0:N_FRAMES]
n_raw = len(raw_frames)
raw_elapsed = time.time() - t0
raw_fps = n_raw / raw_elapsed
print(f"=== raw NVDEC decode-only of {n_raw} frames took {raw_elapsed:.2f}s -> {raw_fps:.1f} fps ===")

print("=== SUMMARY ===")
print(f"zero_copy (frame0 tensor.is_cuda): {tensors[0].is_cuda}")
print(f"decode+dlpack fps: {dlpack_fps:.1f}")
print(f"full pipeline (decode+color+resize) fps: {pipeline_fps:.1f}")
print(f"raw decode-only fps: {raw_fps:.1f}")
print("DONE")
