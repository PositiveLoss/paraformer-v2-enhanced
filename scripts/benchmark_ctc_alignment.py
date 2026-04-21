from __future__ import annotations

import argparse
import json
import time

import torch

from paraformer_v2.ctc_alignment import (
    CYTHON_BACKEND_AVAILABLE,
    TRITON_BACKEND_AVAILABLE,
    batch_ctc_viterbi_alignments,
    batch_ctc_viterbi_alignments_python,
)


def build_inputs(
    batch_size: int,
    time_steps: int,
    vocab_size: int,
    target_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator().manual_seed(17)
    blank_id = vocab_size - 1
    log_probs = torch.randn(
        batch_size,
        time_steps,
        vocab_size,
        generator=generator,
    ).log_softmax(dim=-1).to(device)
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
    targets = torch.randint(
        low=0,
        high=vocab_size - 1,
        size=(batch_size, target_len),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    target_lengths = torch.full((batch_size,), target_len, dtype=torch.long, device=device)
    return log_probs, input_lengths, targets, target_lengths, blank_id


def time_backend(
    backend: str,
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
    repeats: int,
) -> tuple[torch.Tensor, float]:
    if backend != "python":
        _ = batch_ctc_viterbi_alignments(
            log_probs,
            input_lengths,
            targets,
            target_lengths,
            blank_id,
            backend=backend,
        )
        if log_probs.is_cuda:
            torch.cuda.synchronize()
    if log_probs.is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    out = None
    for _ in range(repeats):
        if backend == "python":
            out = batch_ctc_viterbi_alignments_python(
                log_probs,
                input_lengths,
                targets,
                target_lengths,
                blank_id,
            )
        elif backend == "cython":
            out = batch_ctc_viterbi_alignments(
                log_probs,
                input_lengths,
                targets,
                target_lengths,
                blank_id,
                backend="cython",
            )
        else:
            out = batch_ctc_viterbi_alignments(
                log_probs,
                input_lengths,
                targets,
                target_lengths,
                blank_id,
                backend="triton",
            )
    if log_probs.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return out, elapsed / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=33)
    parser.add_argument("--target-len", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not CYTHON_BACKEND_AVAILABLE:
        raise RuntimeError("Cython backend is not available; build the extension first.")

    inputs = build_inputs(
        args.batch_size,
        args.time_steps,
        args.vocab_size,
        args.target_len,
        device,
    )
    log_probs, input_lengths, targets, target_lengths, blank_id = inputs

    python_out, python_seconds = time_backend(
        "python",
        log_probs,
        input_lengths,
        targets,
        target_lengths,
        blank_id,
        args.repeats,
    )
    cython_out, cython_seconds = time_backend(
        "cython",
        log_probs,
        input_lengths,
        targets,
        target_lengths,
        blank_id,
        args.repeats,
    )
    triton_out = None
    triton_seconds = None
    if device.type == "cuda" and TRITON_BACKEND_AVAILABLE:
        triton_out, triton_seconds = time_backend(
            "triton",
            log_probs,
            input_lengths,
            targets,
            target_lengths,
            blank_id,
            args.repeats,
        )

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "vocab_size": args.vocab_size,
        "target_len": args.target_len,
        "repeats": args.repeats,
        "outputs_match": bool(torch.equal(python_out.cpu(), cython_out.cpu())),
        "python_seconds_per_run": python_seconds,
        "cython_seconds_per_run": cython_seconds,
        "speedup": python_seconds / cython_seconds if cython_seconds > 0 else None,
    }
    if triton_seconds is not None:
        result["triton_outputs_match"] = bool(torch.equal(python_out.cpu(), triton_out.cpu()))
        result["triton_seconds_per_run"] = triton_seconds
        result["triton_speedup_vs_python"] = python_seconds / triton_seconds if triton_seconds > 0 else None
        result["triton_speedup_vs_cython"] = cython_seconds / triton_seconds if triton_seconds > 0 else None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
