"""Summarize measured samples; exclude pilots and keep session content private."""

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / ".cache/benchmarks/gemma4-optimization-2026-09-10"
ORIGINAL = HERE.parent / "gemma4-agent-benchmark-2026-09-10"
CASES = ["diff-review", "test-review", "final-review"]
METRICS = ["prefillMs", "prefillTps", "decodeTps", "wallMs", "generatedTokens"]


def main():
    baseline = json.loads((ORIGINAL / "results.json").read_text())
    fixture = baseline["fixture"]
    environment = json.loads((DATA / "final-provenance.json").read_text())
    samples = []
    for name in CASES:
        expected = next(x for x in fixture["cases"] if x["name"] == name)
        for runtime in ["optimized", "llama"]:
            for run in range(1, 4):
                path = DATA / "raw" / f"{runtime}-{name}-{run}.json"
                sample = json.loads(path.read_text())
                assert sample["cachedTokens"] == 0, path
                assert sample["promptTokens"] == expected["promptTokens"], path
                assert sample["inputSha256"] == expected["sha256"], path
                assert sample["generatedTokens"] == 256, path
                if runtime == "optimized":
                    assert sample["addonSha256"] == environment["addonSha256"], path
                assert abs(sample["wallMs"] - sample["accountedMs"]) < 250, path
                result = sample.pop("result")
                sample["finishReason"] = result.get("finishReason", result.get("stop_type"))
                sample["timingOverheadMs"] = sample["wallMs"] - sample["accountedMs"]
                sample.pop("args", None)
                samples.append(sample)

    rows = []
    for name in CASES:
        row = {"name": name}
        for runtime in ["optimized", "llama"]:
            group = [s for s in samples if s["name"] == name and s["runtime"] == runtime]
            row[runtime] = {
                metric: {
                    "median": median(s[metric] for s in group),
                    "min": min(s[metric] for s in group),
                    "max": max(s[metric] for s in group),
                }
                for metric in METRICS
            }
            row[runtime]["distinctOutputHashes"] = len({s["textSha256"] for s in group})
            row[runtime]["finishReasons"] = sorted({s["finishReason"] for s in group})
        row["llamaOverMlxPrefill"] = row["llama"]["prefillTps"]["median"] / row["optimized"]["prefillTps"]["median"]
        row["llamaOverMlxDecode"] = row["llama"]["decodeTps"]["median"] / row["optimized"]["decodeTps"]["median"]
        row["mlxOverLlamaLatency"] = row["optimized"]["wallMs"]["median"] / row["llama"]["wallMs"]["median"]
        row["matchingOutputPairs"] = sum(
            next(s for s in samples if s["name"] == name and s["runtime"] == "optimized" and s["run"] == str(run))["textSha256"]
            == next(s for s in samples if s["name"] == name and s["runtime"] == "llama" and s["run"] == str(run))["textSha256"]
            for run in range(1, 4)
        )
        row["original"] = next(r for r in baseline["rows"] if r["name"] == name)["mlx"]
        row["decodeSpeedupOverOriginal"] = row["optimized"]["decodeTps"]["median"] / row["original"]["decodeTps"]["median"]
        row["requestSpeedupOverOriginal"] = row["original"]["wallMs"]["median"] / row["optimized"]["wallMs"]["median"]
        old_hashes = {s["textSha256"] for s in baseline["samples"] if s["name"] == name and s["runtime"] == "mlx"}
        new_hashes = {s["textSha256"] for s in samples if s["name"] == name and s["runtime"] == "optimized"}
        row["matchesOriginalOutputHashes"] = old_hashes == new_hashes
        rows.append(row)

    # These files contain provenance and measurements, not prompts or generated text.
    summary = {"environment": environment, "fixture": fixture, "rows": rows, "samples": samples}
    (HERE / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("| Real session boundary | Input tokens | Prompt tok/s MLX / llama | Decode tok/s MLX / llama | Request seconds MLX / llama |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        n = next(x["promptTokens"] for x in fixture["cases"] if x["name"] == row["name"])
        cells = []
        for metric in ["prefillTps", "decodeTps", "wallMs"]:
            scale = 1000 if metric == "wallMs" else 1
            cells.append(" / ".join(f"{row[r][metric]['median'] / scale:,.2f}" for r in ["optimized", "llama"]))
        print(f"| {row['name']} | {n:,} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
