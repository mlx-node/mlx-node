#!/usr/bin/env python3
"""Export exact eval-cache IDs, or compare Splash full logits to that cache.

No torch/safetensors installation is needed; the Splash venv supplies numpy.
The top-K-plus-tail KL is an explicit lower bound on full-vocabulary KL.
Argmax agreement and next-token NLL use the full vocabulary in both engines.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def tensors(path):
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        payload = stream.read()
    types = {"U32": "<u4", "I32": "<i4", "F32": "<f4", "F64": "<f8"}
    return {
        key: np.frombuffer(payload[value["data_offsets"][0]:value["data_offsets"][1]], dtype=types[value["dtype"]]).reshape(value["shape"])
        for key, value in header.items() if key != "__metadata__"
    }


def log_softmax(row):
    row = row.astype(np.float64)
    row -= row.max()
    return row - np.log(np.exp(row).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_row", type=Path, help="mlx eval 0000000000.safetensors")
    parser.add_argument("--tokens", type=Path, help="export its token IDs as little-endian uint32")
    parser.add_argument("--splash-logits", type=Path, help="teacher-logits OUTPUT-END.f32")
    parser.add_argument("--window", help="the same END:ROWS passed to Splash")
    parser.add_argument("--vocabulary", type=int, default=248320)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    teacher = tensors(args.cache_row)
    if args.tokens:
        teacher["tokens"].astype("<u4").tofile(args.tokens)
        print(f"Exported {teacher['tokens'].size} exact IDs to {args.tokens}")
    if not args.splash_logits:
        if not args.tokens:
            parser.error("provide --tokens or --splash-logits")
        return
    if not args.window:
        parser.error("--splash-logits requires --window END:ROWS")
    end, rows = map(int, args.window.split(":"))
    if rows <= 0 or rows % 8 or rows >= end or end >= teacher["tokens"].size:
        parser.error("window needs a positive multiple of 8 rows, below END, with the next token present")
    logits = np.fromfile(args.splash_logits, dtype="<f4")
    if logits.size != (rows + 1) * args.vocabulary:
        raise ValueError("Splash dump size does not match window/vocabulary")
    logits = logits.reshape(rows + 1, args.vocabulary)
    if not np.isfinite(logits).all():
        raise ValueError("Splash dump contains non-finite logits")
    for key in ["logits", "lse", "target_logprob"]:
        if not np.isfinite(teacher[key]).all():
            raise ValueError(f"Native teacher cache contains non-finite {key}")
    if teacher["indices"].max() >= args.vocabulary or teacher["tokens"].max() >= args.vocabulary:
        raise ValueError("Teacher cache contains token IDs outside the compared vocabulary")
    metrics = []
    for offset, candidate in enumerate(logits):
        position = end - rows - 1 + offset
        indices = teacher["indices"][position].astype(np.int64)
        reference_logp = teacher["logits"][position].astype(np.float64) - float(teacher["lse"][position])
        candidate_logp = log_softmax(candidate)
        p, q = np.exp(reference_logp), np.exp(candidate_logp[indices])
        # The cache's lse/logits are f32, so roundoff can put summed mass a
        # few ulps above 1. Normalize only that roundoff, retaining the tail.
        if p.sum() > 1:
            p /= p.sum()
        p_tail, q_tail = max(0.0, 1.0 - p.sum()), max(0.0, 1.0 - q.sum())
        kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - candidate_logp[indices])))
        if p_tail > 0:
            kl += float(p_tail * np.log(p_tail / max(q_tail, 1e-300)))
        following = int(teacher["tokens"][position + 1])
        metrics.append({
            "position": position,
            "reference_argmax": int(teacher["argmax"][position]),
            "candidate_argmax": int(candidate.argmax()),
            "bucketed_kl": max(0.0, kl),
            "reference_topk_mass": float(p.sum()),
            "reference_nll": -float(teacher["target_logprob"][position]),
            "candidate_nll": -float(candidate_logp[following]),
        })
    divergences = np.array([entry["bucketed_kl"] for entry in metrics])
    summary = {
        "reference": str(args.cache_row), "candidate": str(args.splash_logits), "window": args.window,
        "positions": len(metrics), "top_k": int(teacher["indices"].shape[1]),
        "top1_agreement": sum(entry["reference_argmax"] == entry["candidate_argmax"] for entry in metrics) / len(metrics),
        "mean_reference_nll": float(np.mean([entry["reference_nll"] for entry in metrics])),
        "mean_candidate_nll": float(np.mean([entry["candidate_nll"] for entry in metrics])),
        "mean_nll_change": float(np.mean([entry["candidate_nll"] - entry["reference_nll"] for entry in metrics])),
        "mean_topk_mass": float(np.mean([entry["reference_topk_mass"] for entry in metrics])),
        "mean_bucketed_kl_lower_bound": float(divergences.mean()),
        "p99_bucketed_kl_lower_bound": float(np.percentile(divergences, 99)),
        "records": metrics,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
