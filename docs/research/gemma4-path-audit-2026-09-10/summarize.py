"""Validate the saved real-fixture experiments and export numeric evidence."""

import json
import math
from pathlib import Path
from statistics import geometric_mean, median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / ".cache/benchmarks/gemma4-path-audit-2026-09-10"
INPUTS = {
    x["name"]: x
    for x in json.loads((ROOT / ".cache/benchmarks/gemma4-agent-2026-09-10/inputs.json").read_text())
}


def read_sample(filename):
    sample = json.loads((DATA / "raw" / filename).read_text())
    fixture = INPUTS[sample["name"]]
    assert sample["inputSha256"] == fixture["sha256"]
    assert sample["promptTokens"] == fixture["promptTokens"]
    assert sample["generatedTokens"] == 256
    assert sample["cachedTokens"] == 0
    # The first five short samples predate the explicit label. That harness
    # always reset caches before its one measured request; counters confirm it.
    sample.setdefault("cachePolicy", "cold")
    assert sample["cachePolicy"] == "cold"
    assert all(math.isfinite(sample[k]) and sample[k] > 0 for k in ("prefillTps", "decodeTps", "wallMs"))
    profile = sample.pop("profiles")["generations"][0]
    sample["memoryBefore"] = profile["memoryBefore"]
    sample["memoryAfter"] = profile["memoryAfter"]
    sample["hostPhaseTimings"] = profile["phases"]
    sample["rawFile"] = filename
    return sample


pairs = [
    {variant: read_sample(f"{variant}-diff-review-{i}.json") for variant in ("baseline", "hoisted")}
    for i in range(1, 4)
]
assert len({s["outputSha256"] for pair in pairs for s in pair.values()}) == 1
ratios = [pair["hoisted"]["decodeTps"] / pair["baseline"]["decodeTps"] for pair in pairs]
summary = {}
for variant in ("baseline", "hoisted"):
    samples = [pair[variant] for pair in pairs]
    summary[variant] = {
        key: {
            "median": median(s[key] for s in samples),
            "min": min(s[key] for s in samples),
            "max": max(s[key] for s in samples),
        }
        for key in ("prefillTps", "decodeTps", "wallMs")
    }

routes = [read_sample(f"hoisted-final-review-{route}-long.json") for route in ("auto", "sdpa", "grouped")]
reverse_routes = [read_sample(f"hoisted-final-review-{route}-long-2.json") for route in ("grouped", "auto")]
grouped_ratios = [
    routes[2]["decodeTps"] / routes[0]["decodeTps"],
    reverse_routes[0]["decodeTps"] / reverse_routes[1]["decodeTps"],
]
result = {
    "date": "2026-09-10",
    "provenance": json.loads((DATA / "provenance.json").read_text()),
    "sourceEvidence": json.loads((DATA / "source-evidence.json").read_text()),
    "recordedCloseoutChecks": json.loads((DATA / "closeout.json").read_text()) if (DATA / "closeout.json").exists() else None,
    "castHoist": {
        "summary": summary,
        "pairedDecodeRatios": ratios,
        "pairedDecodeGeometricMeanRatio": geometric_mean(ratios),
        "outputHashesIdentical": True,
        "pairs": pairs,
    },
    "longContextRouteScreen": {
        "protocol": "Cold-cache 66904-token real-fixture samples, 256 output tokens, identical hoisted weights and binary. Initial order auto/sdpa/grouped, then grouped/auto. Two auto/grouped comparisons, one exploratory SDPA sample; not a full production crossover benchmark.",
        "samples": routes + reverse_routes,
        "groupedPairedDecodeRatios": grouped_ratios,
        "groupedPairedDecodeGeometricMeanRatio": geometric_mean(grouped_ratios),
        "outputHashesIdentical": len({s["outputSha256"] for s in routes + reverse_routes}) == 1,
    },
    "excludedSamples": [{
        "file": "hoisted-final-review-auto-long-reuse1.json",
        "reason": "The attempted warm-prefix replay reported cachedTokens=0. The guard aborted the warm-screen controller; this request is excluded from route and paired statistics.",
    }],
    "limits": [
        "All inputs are saved real agent histories; no synthetic operator input is measured.",
        "Host phase timings are not GPU per-operator measurements; asynchronous work crosses phase boundaries.",
        "Logical tensor byte counts are not measured physical DRAM traffic.",
        "No production runtime or source checkpoint was modified during this audit.",
    ],
}
(HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"castHoist": summary, "pairedDecodeGeomean": geometric_mean(ratios), "routes": [{k:s[k] for k in ("run","decodeTps","prefillTps","wallMs","outputSha256")} for s in routes + reverse_routes]}, indent=2))
