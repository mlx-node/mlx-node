let meta = #{
    name: "windows-feature-parity",
    description: "Audit Windows-gated features in mlx-node, implement real Windows support instead of stubs, and verify on-device",
    phases: [
        #{ title: "Inventory", detail: "classify each Windows-gated feature by platform and effort" },
        #{ title: "Implement", detail: "implement the features that belong on Windows" },
        #{ title: "Verify", detail: "build and test on Windows ARM64" },
    ],
};

let inventory_schema = #{
    "type": "object", "required": ["area", "features"],
    "properties": #{
        "area": #{ "type": "string" },
        "features": #{ "type": "array", "maxItems": 8, "items": #{
            "type": "object",
            "required": ["name", "windows_status", "platform_specific", "should_implement", "effort", "rationale"],
            "properties": #{
                "name": #{ "type": "string" },
                "windows_status": #{ "type": "string", "description": "stub / cfg-gated / partial / implemented" },
                "platform_specific": #{ "type": "boolean", "description": "true if the feature is inherently macOS/Metal-only" },
                "should_implement": #{ "type": "boolean" },
                "effort": #{ "type": "string", "description": "small / medium / large / multi-week" },
                "rationale": #{ "type": "string" },
            },
        }},
    },
};

let report_schema = #{
    "type": "object", "required": ["summary", "implemented", "platform_boundaries"],
    "properties": #{
        "summary": #{ "type": "string" },
        "implemented": #{ "type": "array", "items": #{ "type": "string" } },
        "platform_boundaries": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

fn join_lines(items) {
    let out = "";
    for it in items {
        if out.len() > 0 { out += "\n"; }
        out += it;
    }
    out
}

let root = "C:/Users/lynwe/workspace/github/mlx-node";
let common = "Repository: " + root + " (Windows ARM64 + NVIDIA CUDA). Use grep/read_file to inspect the real code before answering. "
    + "The goal is NOT to silence warnings with cfg; it is to understand whether each Windows-gated item is a real missing Windows feature or an inherently macOS/Metal-only path. "
    + "CUDA in this repo is an inference-only preview with paged-attention forced OFF (see CLAUDE.md Known limitations), so a full CUDA paged-attention kernel is out of scope for this audit. ";

phase("Inventory");
let areas = [
    #{
        label: "paged-attention",
        focus: "crates/mlx-core/src/transformer/paged_kv_cache_adapter.rs. Warnings: dead_code on KvInputInfo, KvTensorMeta, QueryInputInfo, validate_kv_input, validate_query_input, decode_read_span, build_prefill_block_ids_for_total; unused tracing imports (Instant, elapsed_ms, inference_trace_enabled, write_inference_trace); unused_mut new_num_blocks. Determine which of these are used only by #[cfg(target_os = \"macos\")] production paths vs genuine Windows features.",
    },
    #{
        label: "asr-capture",
        focus: "crates/mlx-core/src/models/qwen3_asr/capture.rs and model.rs. Warnings: unused ThreadsafeFunctionCallMode and StreamFeedSource imports; dead_code Capture/PrepareCapture/ReleaseCapture variants. The capture platform module is #[cfg(target_os = \"macos\")] Core Audio with a non-macOS stub. Assess what a real Windows implementation (e.g. WASAPI/cpal) would take.",
    },
    #{
        label: "disk-tier",
        focus: "crates/mlx-core/src/disk_tier.rs. Warnings: dead_code temp_root/cache_root methods and PersistentIdentity fields. open_disk_tier is #[cfg(unix)] with a Windows stub returning unsupported. Assess what a real Windows SSD tier would take (file locking, mmap, temp/cache dirs).",
    },
    #{
        label: "misc",
        focus: "crates/mlx-core/src/utils/gguf_read.rs. Warnings: dead_code RESIDENT_PROBES/RESIDENT_PROBE_PAGES used only by #[cfg(unix)] sampled_resident_fraction (mincore). Determine whether a Windows equivalent (e.g. QueryWorkingSet) is worth implementing or whether this is a unix-only diagnostic.",
    },
];

let jobs = [];
for a in areas {
    jobs.push(#{
        prompt: common + "Audit this area: " + a.focus + " Report each feature with windows_status, whether it is inherently platform_specific, whether we should_implement it on Windows now, the effort, and a concrete rationale grounded in the code you read. Be honest about large efforts; do not claim a CUDA paged-attention kernel or a full audio capture stack is trivial.",
        label: "inventory:" + a.label,
        capability_mode: "read-only",
        output_schema: inventory_schema,
    });
}
let inventories = parallel(jobs);

let implement_list = [];
let boundary_list = [];
for inv in inventories {
    if inv != () && inv.success && inv.output.features != () {
        for f in inv.output.features {
            if f.should_implement == true && f.platform_specific == false && (f.effort == "small" || f.effort == "medium") {
                implement_list.push(f.name + " — " + f.rationale);
            } else {
                boundary_list.push(f.name + " [" + f.effort + "] — " + f.rationale);
            }
        }
    }
}

log("Candidates to implement: " + implement_list.len().to_string() + "; platform boundaries/large efforts: " + boundary_list.len().to_string());
let inventory_report = "IMPLEMENT NOW:\n" + join_lines(implement_list) + "\n\nPLATFORM BOUNDARIES / LARGE:\n" + join_lines(boundary_list);
write_scratch_file("inventory.md", inventory_report);

await_user("user", "Review the inventory in scratch/inventory.md. Implement only the small/medium non-platform-specific features now? (The large CUDA paged-attention kernel and full WASAPI capture stay out of scope.)");

phase("Implement");
if implement_list.len() > 0 {
    let impl_schema = #{
        "type": "object", "required": ["summary", "files"],
        "properties": #{
            "summary": #{ "type": "string" },
            "files": #{ "type": "array", "items": #{ "type": "string" } },
        },
    };
    let impl_jobs = [];
    for item in implement_list {
        impl_jobs.push(#{
            prompt: common + "Implement this Windows feature for real (no stub, no cfg-only suppression): " + item + ". Make the minimal correct code change in the repo, keep macOS behavior intact, and add or update tests where practical. Report the files changed and a short summary. If the item is actually platform-specific on closer reading, say so instead of forcing an implementation.",
            label: "implement",
            capability_mode: "read-write",
            output_schema: impl_schema,
        });
    }
    let impl_results = parallel(impl_jobs);
    let done = [];
    for r in impl_results {
        if r != () && r.success { done.push(r.output.summary); }
    }
    write_scratch_file("implemented.md", join_lines(done));
}

phase("Verify");
let verify = agent(
    common + "Run the native Windows build for the current changes: `& target/windows-tools/build.ps1 node .yarn/releases/yarn-4.18.1.cjs build:native` (this takes several minutes), and report the remaining cargo warnings verbatim. Then run `& node_modules/.bin/vp.cmd run build:ts`. Report whether the build succeeds and which of the 19 original warnings remain. Do not edit code in this step.",
    #{ label: "verify-build", capability_mode: "execute", output_schema: report_schema },
);

let summary = if verify != () && verify.success { verify.output.summary } else { "verification did not complete" };
complete(#{ summary: summary, inventory: "scratch/inventory.md", implemented: "scratch/implemented.md" });
