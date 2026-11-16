let meta = #{
    name: "windows-cuda-release",
    description: "Add a Windows ARM64 CUDA MLX prebuilt target and a Windows desktop packaging path, verified locally before any CI publish",
    phases: [
        #{ title: "Research", detail: "map prebuilt, runner/CUDA, and desktop packaging gaps" },
        #{ title: "Implement", detail: "prebuilt target + Windows packaging" },
        #{ title: "Verify", detail: "local build and packaging checks" },
    ],
};

fn join_lines(items) {
    let out = "";
    for it in items {
        if out.len() > 0 { out += "\n"; }
        out += it;
    }
    out
}

let research_schema = #{
    "type": "object", "required": ["area", "findings", "plan"],
    "properties": #{
        "area": #{ "type": "string" },
        "findings": #{ "type": "array", "maxItems": 10, "items": #{
            "type": "object", "required": ["topic", "status", "gap", "action"],
            "properties": #{
                "topic": #{ "type": "string" },
                "status": #{ "type": "string" },
                "gap": #{ "type": "string" },
                "action": #{ "type": "string" },
            },
        }},
        "plan": #{ "type": "array", "items": #{ "type": "string" } },
    },
};

let root = "C:/Users/lynwe/workspace/github/mlx-node";
let common = "Repository: " + root + " (Windows ARM64 + NVIDIA CUDA, RTX Spark). Use read_file/grep on the real files. "
    + "Existing release flow is macOS-only: .github/workflows/mlx-prebuilt.yml builds only aarch64-apple-darwin, and .github/workflows/desktop-release.yml uses codesign/notarytool/DMG. "
    + "We must NOT publish or push anything; this run only prepares code and local verification. macOS flow must stay intact. ";

phase("Research");
let topics = [
    #{
        label: "prebuilt-buildrs",
        focus: "crates/mlx-sys/build.rs prebuilt logic (MlxSource, PrebuiltPin, prebuilt_cache_root, release_tag_for, portable_cmake_defines) and crates/mlx-sys/mlx-prebuilt.json. The prebuilt artifact is currently lib/libmlx.a + lib/libjaccl.a + lib/mlx.metallib + include/**. Determine exactly what a Windows ARM64 CUDA prebuilt must contain (static .lib vs .a, whether mlx.metallib is needed given NVRTC JIT), which parts of build.rs are macOS-specific, and what target triple to use (aarch64-pc-windows-msvc).",
    },
    #{
        label: "ci-runner-cuda",
        focus: ".github/workflows/mlx-prebuilt.yml matrix and probe/build/package/publish steps. Determine how to add an aarch64-pc-windows-msvc job: which GitHub runner provides Windows on ARM64 (windows-11-arm), how to install the CUDA toolkit + cuDNN on it, and how the macOS-specific steps (metallib, MACOSX_DEPLOYMENT_TARGET, COPYFILE_DISABLE, shasum) need Windows equivalents (PowerShell Get-FileHash). Note we cannot test GitHub-hosted runners locally; produce the exact yaml changes.",
    },
    #{
        label: "desktop-package",
        focus: "packages/desktop/scripts/package.ts, stage-app.ts, payload.ts, verify-bundle.ts and packages/desktop/package.json. Determine how macOS-only the packaging is (codesign, DMG, install-name rewrite) and what a Windows ARM64 NSIS/zip packaging path needs: where the native .node, CUDA/cuDNN/OpenBLAS DLLs, and models live, and how to produce an installable artifact without Apple signing. Inspect packages/desktop/src/main/paths.ts for how the packaged app locates the native addon on Windows.",
    },
    #{
        label: "release-workflow",
        focus: ".github/workflows/desktop-release.yml and packages/desktop/scripts/publish-release.ts + release-version.ts + check-release.ts. Determine what a Windows release job should reuse (version resolve, artifact checks) and what must be replaced (codesign, notarytool, DMG, Homebrew cask). Propose whether to add a separate windows job in desktop-release.yml or a new workflow, and how update manifests work for Windows (NSIS vs Squirrel).",
    },
];

let jobs = [];
for t in topics {
    jobs.push(#{
        prompt: common + "Research this area and produce a concrete implementation plan: " + t.focus + " Report findings as {topic, status, gap, action} and a step-by-step plan. Read the actual files; do not guess from memory.",
        label: "research:" + t.label,
        capability_mode: "read-only",
        output_schema: research_schema,
    });
}
let research = parallel(jobs);

let plan_lines = [];
for r in research {
    if r != () && r.success && r.output.plan != () {
        plan_lines.push("## " + r.output.area);
        for step in r.output.plan { plan_lines.push("- " + step); }
    }
}
write_scratch_file("research.md", join_lines(plan_lines));

await_user("user", "Review scratch/research.md. Proceed to implement the prebuilt target and Windows packaging (local verification only, no CI publish)?");

phase("Implement");
let impl_schema = #{
    "type": "object", "required": ["summary", "files"],
    "properties": #{
        "summary": #{ "type": "string" },
        "files": #{ "type": "array", "items": #{ "type": "string" } },
    },
};
let impl_jobs = [
    #{
        prompt: common + "Implement the Windows ARM64 CUDA prebuilt support based on scratch/research.md: extend crates/mlx-sys/build.rs so a Windows prebuilt artifact (static libraries + headers, no Metal metallib) is produced and consumed, and extend .github/workflows/mlx-prebuilt.yml with an aarch64-pc-windows-msvc job (windows-11-arm runner, CUDA+cuDNN setup, PowerShell hashing, correct package/publish steps). Keep the macOS path byte-for-byte compatible. Do not run CI; just make the code/yaml changes and report files.",
        label: "impl:prebuilt",
        capability_mode: "read-write",
        output_schema: impl_schema,
    },
    #{
        prompt: common + "Implement Windows ARM64 desktop packaging based on scratch/research.md: add a packaging path that stages the native .node plus CUDA/cuDNN/OpenBLAS runtime DLLs and produces an installable Windows artifact (NSIS or zip) without Apple signing, and wire a Windows release job into the release workflow reusing the existing version/artifact checks. Keep macOS packaging intact. Do not publish; make the code/yaml changes and report files.",
        label: "impl:desktop",
        capability_mode: "read-write",
        output_schema: impl_schema,
    },
];
let impl_results = parallel(impl_jobs);
let impl_lines = [];
for r in impl_results {
    if r != () && r.success { impl_lines.push(r.output.summary); }
}
write_scratch_file("implemented.md", join_lines(impl_lines));

phase("Verify");
let verify = agent(
    common + "Verify the current working tree on this Windows ARM64 device. Run `& target/windows-tools/build.ps1 node .yarn/releases/yarn-4.18.1.cjs build:native` (several minutes) and `& node_modules/.bin/vp.cmd run build:ts`. If a packaging script was added, run it in a non-publishing/dry-run mode if one exists. Report whether the native and TS builds succeed, any new warnings/errors, and whether the packaging script produces an artifact. Do not edit code in this step; just report.",
    #{ label: "verify", capability_mode: "execute", output_schema: research_schema },
);
let summary = if verify != () && verify.success { "verification completed; see scratch/verify.md" } else { "verification incomplete" };
complete(#{ summary: summary, research: "scratch/research.md", implemented: "scratch/implemented.md" });
