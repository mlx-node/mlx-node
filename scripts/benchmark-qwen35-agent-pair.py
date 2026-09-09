"""Alternate identical long agent turns between two resident processes.

Usage: python3 scripts/benchmark-qwen35-agent-pair.py EXPERIMENT.json
See docs/research/qwen35-agent-mlxfast/README.md for the configuration.
Only one child performs inference at a time. Raw outputs stay in outputDir.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time


def manifest(directory):
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Snapshot must contain regular files, not symlinks")
        if path.is_file():
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            entries.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size, "sha256": digest})
    return entries


def main(config):
    repo = Path(__file__).resolve().parents[1]
    output = Path(config["outputDir"]).resolve()
    cold = Path(config["coldCacheDir"]).resolve()
    warm_turns = config.get("warmTurns", 6)
    if type(warm_turns) is not int or not 2 <= warm_turns <= 12:
        raise ValueError("warmTurns must be an integer from 2 to 12")
    first = config.get("first", "control")
    if first not in ("control", "candidate"):
        raise ValueError("first must be control or candidate")
    order = [first, "candidate" if first == "control" else "control"]
    reserved = {"MLX_COLD_CACHE_DIR", "MLX_AGENT_BENCH_GATE_DIR", "MLX_AGENT_BENCH_WARM_TURNS", "MLX_CACHE_LIMIT_GB", "MLX_PAGED_PREFILL_CHUNK_SIZE"}
    for flags in (config.get("common", {}), config["control"], config["candidate"]):
        if not isinstance(flags, dict) or any(not key.startswith("MLX_") or key in reserved or not isinstance(value, str) for key, value in flags.items()):
            raise ValueError("Experiment flags must be MLX_ string values and must not override controller/cache settings")
    if output.exists() or cold.exists():
        raise FileExistsError("Use new outputDir and coldCacheDir; existing experiments are never overwritten")
    output.mkdir(parents=True)
    cold.mkdir(parents=True)
    snapshot = Path(config["snapshot"]).resolve() if config.get("snapshot") else None
    original = manifest(snapshot) if snapshot else None
    if snapshot and not original:
        raise ValueError("Empty or missing SSD snapshot")
    runs = {}
    report = {"config": config, "controllerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "events": [], "pairs": [], "initialPrefillsAreSetup": True}

    def save():
        (output / "pair.json").write_text(json.dumps(report, indent=2) + "\n")

    def start(arm):
        gate = output / (arm + "-gate")
        gate.mkdir()
        cache = cold / arm
        if snapshot:
            shutil.copytree(snapshot, cache / "mlx-paged-v1")
            copied = manifest(cache / "mlx-paged-v1")
            if copied != original:
                raise RuntimeError("SSD snapshot copy changed")
            (output / (arm + "-snapshot.json")).write_text(json.dumps(copied, indent=2) + "\n")
        env = {key: value for key, value in os.environ.items() if not key.startswith("MLX_")}
        env.update(config.get("common", {}))
        env.update(config[arm])
        env.update(MLX_COLD_CACHE_DIR=str(cache), MLX_PAGED_PREFILL_CHUNK_SIZE="2048", MLX_CACHE_LIMIT_GB="2", MLX_AGENT_BENCH_GATE_DIR=str(gate), MLX_AGENT_BENCH_WARM_TURNS=str(warm_turns))
        result = output / (arm + ".json")
        args = ["yarn", "oxnode", "scripts/benchmark-qwen35-agent-session.ts", str(Path(config["model"]).resolve()), str(Path(config["session"]).resolve()), str(result), "restore" if snapshot else "capture", config["entryId"]]
        log = (output / (arm + ".log")).open("w")
        try:
            process = subprocess.Popen(args, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        runs[arm] = {"process": process, "gate": gate, "result": result, "log": log}

    def turn(arm, index):
        run = runs[arm]
        (run["gate"] / f"turn-{index}.go").write_text("go\n")
        deadline = time.monotonic() + 540
        while not (run["gate"] / f"turn-{index}.done").exists():
            for name, other in runs.items():
                if other["process"].poll() is not None:
                    raise RuntimeError(f"{name} exited before the experiment finished; see its local log")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{arm} turn {index} timed out")
            time.sleep(0.25)
        row = json.loads(run["result"].read_text())["rows"][index]
        event = {"arm": arm, "turn": index, "wallMs": row["wallMs"], "time": time.time()}
        report["events"].append(event)
        save()
        print(json.dumps(event), flush=True)

    def compare(index):
        data = [json.loads(runs[arm]["result"].read_text()) for arm in ("control", "candidate")]
        for key in ("contextSha256", "addonSha256", "checkpointConfigSha256", "harnessSha256"):
            if data[0][key] != data[1][key]:
                raise RuntimeError(f"Mismatched {key}")
        rows = [value["rows"][index] for value in data]
        final = [row["telemetry"]["final"] for row in rows]
        for key in ("rawText", "numTokens", "promptTokens", "cachedTokens"):
            if final[0].get(key) != final[1].get(key):
                raise RuntimeError(f"Work/output mismatch at turn {index}: {key}")
        if index:
            report["pairs"].append({"turn": index, "controlMs": rows[0]["wallMs"], "candidateMs": rows[1]["wallMs"], "speedRatio": rows[0]["wallMs"] / rows[1]["wallMs"], "outputSha256": hashlib.sha256(final[0]["rawText"].encode()).hexdigest(), "mtpCycles": [value["performance"].get("mtpCycles") for value in final]})
        save()

    try:
        for arm in order:
            start(arm)
            turn(arm, 0)
        compare(0)
        for index in range(1, warm_turns + 1):
            for arm in order if index % 2 else list(reversed(order)):
                turn(arm, index)
            compare(index)
        for run in runs.values():
            (run["gate"] / "finish.go").write_text("go\n")
        for arm, run in runs.items():
            if run["process"].wait(timeout=60):
                raise RuntimeError(f"{arm} failed during final cleanup")
            data = json.loads(run["result"].read_text())
            if not all(data["drains"]):
                raise RuntimeError(f"{arm} did not drain its SSD writer")
            if data["cold"]["writeErrors"] or data["cold"]["corruptions"]:
                raise RuntimeError(f"{arm} reported an SSD write error or corruption")
        report["geometricMeanSpeedRatio"] = statistics.geometric_mean(pair["speedRatio"] for pair in report["pairs"])
        report["complete"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        for run in runs.values():
            if run["process"].poll() is None:
                try:
                    os.killpg(run["process"].pid, signal.SIGTERM)
                    try:
                        run["process"].wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(run["process"].pid, signal.SIGKILL)
                        run["process"].wait()
                except ProcessLookupError:
                    pass
            run["log"].close()
        save()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Expected one experiment JSON path")
    # Convert SIGTERM into an exception so cleanup also reaches detached children.
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Experiment interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    main(json.loads(Path(sys.argv[1]).read_text()))
