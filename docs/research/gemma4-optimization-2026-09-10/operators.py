"""Release-build projection replay using captured real agent activations."""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / ".cache/benchmarks/gemma4-optimization-2026-09-10"
BASE_ENV = dict(os.environ,
    GEMMA4_QMV_MODEL=str(ROOT / ".cache/benchmarks/gemma4-path-audit-2026-09-10/hoisted"),
    GEMMA4_QMV_INPUTS=str(DATA / "captured-qmv"),
    MLX_GEMMA4_MIXED_QMV="1")
COMMAND = ["cargo", "test", "-p", "mlx-core", "--release", "--target", "aarch64-apple-darwin", "--lib", "recorded_gemma4_qmv_bandwidth"]

def main():
    # Finish compilation before any timed sample. No parallel inference work.
    with (DATA / "release-probe-build.log").open("w") as log:
        subprocess.run(COMMAND + ["--no-run"], cwd=ROOT, env=BASE_ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
    samples = []
    for repeat in range(1, 4):
        order = ["baseline", "hoisted", "symmetric"]
        if repeat % 2 == 0:
            order.reverse()
        for variant in order:
            path = DATA / f"release-probe-{variant}-{repeat}.log"
            with path.open("w") as log:
                subprocess.run(COMMAND + ["--", "--ignored", "--nocapture", "--test-threads=1"],
                    cwd=ROOT, env=dict(BASE_ENV, GEMMA4_QMV_VARIANT=variant),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            record = next(line.split("RECORDED_QMV ", 1)[1] for line in path.read_text().splitlines() if "RECORDED_QMV " in line)
            sample = json.loads(record)
            sample["repeat"] = repeat
            sample["medianMs"] = median(sample["elapsedMs"])
            sample["logicalGBps"] = (sample["weightBytes"] + sample["sidecarBytes"]) / sample["medianMs"] / 1e6
            samples.append(sample)
            print(variant, repeat, sample["medianMs"], sample["logicalGBps"], flush=True)
            time.sleep(2)
    executable = re.search(r"Executable .* \(([^)]+)\)", (DATA / "release-probe-build.log").read_text()).group(1)
    source_paths = ["crates/mlx-core/src/models/gemma4/recorded_qmv.rs",
        "crates/mlx-sys/src/mlx_affine_qmv.cpp", "crates/mlx-sys/src/metal/affine_qmv_bf16.metal.inc"]
    report = {"profile": "release",
        "executableSha256": hashlib.sha256((ROOT / executable).read_bytes()).hexdigest(),
        "sourceFileSha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}, "warmupPasses": 10, "measuredPassesPerProcess": 20,
        "sourceCaptureSha256": hashlib.sha256((DATA / "capture-provenance.json").read_bytes()).hexdigest(),
        "samples": samples,
        "note": "Logical operand traffic divided by wall time, not DRAM counters. Baseline excludes extra cast reads/writes from this byte count. All operand representations are resident before timing. Outputs checked against stock promoted QMM outside timing."}
    (HERE / "operators.json").write_text(json.dumps(report, indent=2) + "\n")

if __name__ == "__main__":
    main()
