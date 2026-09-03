#!/usr/bin/env python3
"""Multi-batch release-bug repro for cvcuda + PyNvVideoCodec native objects.

Trigger: builds the cvcuda chain for batch 1, holds in a module-scope
list, then builds batch 2's chain and reassigns the list -- a plain
in-scope `keepalive = []` between batches. If the bug exists, the
second batch's cvcuda call dereferences a freed native pointer and
crashes the interpreter.

Each test is a self-contained subprocess with a known exit code.
Crashes surface as one of (depending on platform / Python / driver):
  - SIGABRT (rc=-6) with "Fatal Python error: PyThreadState_Get: the
    function must be called with the GIL held, but the GIL is released"
    in stderr (Python's signal handler runs before the SIGSEGV is
    delivered)
  - SIGSEGV (rc=-11) with no Python error message (raw signal delivery
    before Python's signal handler runs)
Clean runs exit 0 with a PASS line on stdout.

Originally written 2026-09-02 to investigate a known recurring bug
class in cvcuda (the same class as #72 and #188 on the CV-CUDA
tracker, both of which were "fixed" via Pybind11 upgrades but appear
to have regressed -- this repro triggers the same error class with a
different surface condition: single-threaded in-scope list
reassignment, vs. those reports' multithreading + cvcuda.resize).

Usage: python3 multi_batch_repro.py <decoder_class> [video_path] [n_batches]
  decoder_class: SimpleDecoder | ThreadedDecoder (default SimpleDecoder)
  video_path:    default ./clip.mp4 (any H.264 mp4 ffmpeg can produce works)
  n_batches:     default 2 (one reassignment is enough to trigger the crash)
"""

import os
import subprocess
import sys
import textwrap

VIDEO_PATH = sys.argv[2] if len(sys.argv) > 2 else "clip.mp4"
DECODER = sys.argv[1] if len(sys.argv) > 1 else "SimpleDecoder"
N_BATCHES = int(sys.argv[3]) if len(sys.argv) > 3 else 2
# ThreadedDecoder requires a buffer_size; SimpleDecoder doesn't accept one.
# 10 is a small enough buffer to test reassignment-driven release.
THREADED_BUFFER_SIZE = 10


def build_script():
    # Plain string (not f-string) because the embedded script body has its own
    # {}-braces that should be passed through to the child Python unmodified.
    # DECODER and VIDEO_PATH are substituted via .replace() below.
    return (
        textwrap.dedent(
            """
            import os, sys, gc
            # Same env as flay_video_dense.py -- disable HF progress
            # bar thread and torch progress monitoring that could race with
            # the native decoder cleanup.
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            import torch
            import PyNvVideoCodec as nvc
            import cvcuda

            DecoderClass = getattr(nvc, DECODER_NAME)
            if DECODER_NAME == "ThreadedDecoder":
                decoder = DecoderClass(VIDEO_PATH_PLACEHOLDER, THREADED_BUFFER_SIZE, use_device_memory=True)
            else:
                decoder = DecoderClass(VIDEO_PATH_PLACEHOLDER, use_device_memory=True)

            # Module-scope list, never reassigned before the test point.
            # This mirrors flay_video_dense.py's KEEPALIVE pattern exactly.
            keepalive = []
            try:
                for batch_idx in range(N_BATCHES):
                    frame = decoder.get_batch_frames(1)[0]
                    raw = torch.from_dlpack(frame)
                    h15, w = raw.shape
                    reshaped = raw.view(1, h15, w, 1).contiguous()
                    cv_in = cvcuda.as_tensor(reshaped, "NHWC")
                    cv_out = cvcuda.cvtcolor(cv_in, cvcuda.ColorConversion.YUV2RGB_NV12)
                    ext = cv_out.cuda()
                    result = torch.from_dlpack(ext)
                    keepalive.append((raw, reshaped, cv_in, cv_out, ext, result))
                    print(f"BEFORE_REASSIGN: batch={batch_idx} keepalive_len={len(keepalive)}")
                    # The documented trigger: plain list reassignment in scope.
                    keepalive = []
                    gc.collect()
                    print(f"AFTER_REASSIGN: batch={batch_idx} clean")
            except Exception as e:
                print(f"PYTHON_EXCEPTION: {type(e).__name__}: {e}", file=sys.stderr)
                sys.exit(2)
            torch.cuda.synchronize()
            print("PASS: keepalive reassignment between batches did not crash")
            """
        )
        .replace("DECODER_NAME", repr(DECODER))
        .replace("VIDEO_PATH_PLACEHOLDER", repr(VIDEO_PATH))
        .replace("N_BATCHES", str(N_BATCHES))
        .replace("THREADED_BUFFER_SIZE", str(THREADED_BUFFER_SIZE))
    )


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"FATAL: {VIDEO_PATH} not found", file=sys.stderr)
        sys.exit(2)
    if DECODER not in ("SimpleDecoder", "ThreadedDecoder"):
        print(f"FATAL: decoder must be SimpleDecoder or ThreadedDecoder, got {DECODER!r}", file=sys.stderr)
        sys.exit(2)

    print(f"=== {DECODER} × {N_BATCHES} batches on {VIDEO_PATH} ===")
    completed = subprocess.run(
        [sys.executable, "-u", "-c", build_script()],
        capture_output=True, text=True, timeout=60,
    )
    print(f"--- subprocess exit code: {completed.returncode} ---")
    if completed.stdout:
        print(f"--- stdout ---\n{completed.stdout.rstrip()}")
    if completed.stderr:
        print(f"--- stderr ---\n{completed.stderr.rstrip()}")

    combined = completed.stdout + completed.stderr
    if completed.returncode == 0 and "PASS:" in completed.stdout:
        print(f"=== {DECODER}: NO BUG -- reassignment between batches clean ===")
        sys.exit(0)
    # Crashes can surface as either:
    # - SIGABRT (rc=-6) with "Fatal Python error: PyThreadState_Get" / "GIL is released"
    # - SIGSEGV (rc=-11) when the dangling-reference segfaults before the
    #   Python error path can fire
    # Both count as a successful repro: the chain broke on release.
    if (
        "GIL is released" in combined
        or "PyThreadState_Get" in combined
        or (completed.returncode < 0 and "BEFORE_REASSIGN" in completed.stdout)
    ):
        sig = -completed.returncode
        sig_name = {6: "SIGABRT", 11: "SIGSEGV"}.get(sig, f"signal {sig}")
        print(f"=== {DECODER}: BUG REPRODUCED -- {sig_name} after release ===")
        sys.exit(1)
    print(f"=== {DECODER}: INCONCLUSIVE (rc={completed.returncode}) ===")
    sys.exit(3)


if __name__ == "__main__":
    main()
