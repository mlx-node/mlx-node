# mlx-node for macOS

Install the signed DMG from [GitHub Releases](https://github.com/mlx-node/mlx-node/releases)
by dragging mlx-node to Applications. The app requires Apple Silicon and macOS 26 or newer.

## Coding Agents

Open **Coding Agents** in the control panel to let other agents delegate GitHub
work to `mlx delegate github`. The app includes the CLI and runs it with Electron's
bundled Node runtime; users do not need npm or a separate Node installation. At
launch it creates `~/.mlx-node/bin/mlx` (`mlx.cmd` on Windows), verifies delegation
support, and repairs its target after the app moves or updates. It never overwrites a foreign command
at that location or changes shell startup files or an existing global `mlx`.

Setup writes the launcher's quoted absolute path into the routing prompt, so
coding agents can invoke it regardless of their terminal/editor `PATH`. The
launcher executes in the caller's process tree and preserves its permissions.
Authenticate the [GitHub CLI](https://cli.github.com/) before delegating GitHub work.

Successful command checks persist in `~/.mlx-node/cli-verification.json` across
app restarts. Unchanged runtime files use a cheap metadata check; changed metadata
triggers a content hash, and only different content or permissions require another
command probe. Failed probes remain retryable. The cache stores hashes in an
owner-only file and never skips checking that the launcher and runtime still exist.

`mlx delegate` uses the same prompt-and-exit runtime as `mlx agent --print`,
including its model settings, inference cache, metrics, and saved sessions.
It has a focused worker prompt, read/bash tools, and inherits Codex's process
permissions when launched by Codex. Permission or authentication blockers return
a handoff without recursive subagents. Delegated sessions appear on **Sessions**.
The CLI owns its model process.

The page reads each global instruction file and looks for this block:

```
<!-- portato managed instructions: begin -->
…delegation instructions…
<!-- portato managed instructions: end -->
```

| File contents                                         | Status        |
| ----------------------------------------------------- | ------------- |
| No markers                                            | Not installed |
| Block matches the current instructions                | Installed     |
| Block has other content                               | Needs update  |
| Incomplete or repeated markers, or an unreadable file | Error         |

**Install** appends the block after a blank line and keeps the rest of the file.
**Update** replaces only the text between the markers. Text outside the markers
is never changed, and delegation wording there does not count as installed. Fix
an error by hand, then check again.
Start a new coding-agent session after setup.

The page needs an installed local model and a working app command. Without a
model it links to **Models**. If command setup fails, it shows the reason and
**Retry setup**. Opening or focusing the page reads the files again, and an idle
page refreshes every 30 seconds.

Supported native global files:

| Agent       | File                                                                              |
| ----------- | --------------------------------------------------------------------------------- |
| Claude Code | `~/.claude/CLAUDE.md` (`CLAUDE_CONFIG_DIR` supported)                             |
| Codex       | `~/.codex/AGENTS.md`, or a nonempty `AGENTS.override.md` (`CODEX_HOME` supported) |
| Grok        | `~/.grok/AGENTS.md` (`GROK_HOME` supported)                                       |

Project instructions, imported files and Grok's optional Claude compatibility
sources are outside this check. Grok may already read a Claude installation
through that compatibility layer; in that case a second native installation is
unnecessary. See [Grok's instruction discovery rules](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/12-project-rules.md).

Instruction contents stay on this Mac.

## Updates

Signed stable builds check for updates at launch and every six hours. Updates download
in the background. Use **Restart to Update…** in the tray or application menu when
you are ready to stop inference and restart. A downloaded update also applies the
next time you launch the app after quitting normally.

The same menu offers **Check for Updates…**, download progress, and a retry action if
an update fails. Models, sessions, and settings stay in their existing data directories.
Development, unsigned, and prerelease builds do not check the public update feed.

Versions released before this updater was added need one manual installation of a
release that includes it.

## Release artifacts

The [desktop release workflow](../../.github/workflows/desktop-release.yml) produces:

- `mlx-node-<version>-arm64.dmg` for installation.
- `mlx-node-<version>-darwin-arm64.zip` for automatic updates.
- `mlx-node-<version>-darwin-arm64.zip.blockmap` for differential downloads.
- `latest-mac.yml` with the update version, archive size/hash, minimum macOS, and rollout percentage.
- `mlx-node-<version>-win32-arm64.zip`, a portable Windows ARM64 app with the
  native addon, CUDA/cuDNN/OpenBLAS DLLs, and NVRTC headers. Extract it and run
  `mlx-node.exe`; no automatic update feed is used. When the `ES_USERNAME`,
  `ES_PASSWORD`, `CREDENTIAL_ID` and `ES_TOTP_SECRET` secrets are set, every
  `.exe`, `.dll` and `.node` in it without an existing valid, timestamped
  signature (NVIDIA's DLLs already carry one) is Authenticode-signed and
  timestamped through SSL.com eSigner (via pinned `CodeSignTool` v1.3.2) before
  the ZIP is built; without them the ZIP ships unsigned, with a warning.

Stable Windows ARM64 builds also produce two workflow artifacts (kept off GitHub Releases):

- `mlx-node-desktop-windows-arm64-msix`: unsigned `mlx-node-<version>-win32-arm64.msix`
  for Microsoft Store Partner Center ingestion.
- `mlx-node-desktop-winget-manifests`: WinGet 1.12.0 multi-file manifests pointing at
  the GitHub Release `win32-arm64.zip` archive.

## Microsoft Store (Windows ARM64 MSIX)

Stable `package:windows` runs invoke the Windows SDK's `MakePri.exe` and `MakeAppx.exe`
to produce `mlx-node-<version>-win32-arm64.msix` using the Partner Center Store identity
`LongYinan.Portato` (reserved Store display name `Portato`, `runFullTrust`, `arm64`,
`Windows.Desktop` `MinVersion="10.0.26100.0"`, and `Microsoft.VCLibs.140.00.UWPDesktop`).
Because Microsoft Store Partner Center rejects packages whose `<Identity Version>` starts
with `0.` (Major must be `1..65535`, Minor/Build `0..65535`, Revision `0`), each stable
semver `MAJOR.MINOR.PATCH` (`MAJOR <= 65534`) is mapped to Store identity version
`(MAJOR + 1).MINOR.PATCH.0` (for example, `0.0.21` → `1.0.21.0`, `0.1.0` → `1.1.0.0`,
`1.0.0` → `2.0.0.0`) so Store versions stay strictly monotonic across `0.x` and `1.x+`.
Prerelease and `+build` versions skip MSIX packaging because the Microsoft Store forbids
prerelease identifiers and build metadata. Pass `--skip-msix` for local portable-only
builds, or override identity fields via `MSIX_IDENTITY_NAME`, `MSIX_PUBLISHER`,
`MSIX_PUBLISHER_DISPLAY_NAME`, and `MSIX_DISPLAY_NAME`.

To submit a release to the Microsoft Store:

1. Download the `mlx-node-desktop-windows-arm64-msix` artifact from the tagged
   **Desktop Release** workflow run.
2. Upload the `.msix` directly to Partner Center for **Portato**; Partner Center signs
   the package with Microsoft's Store certificate during ingestion. The unsigned `.msix`
   is intentionally excluded from GitHub Releases because Windows refuses to sideload
   unsigned packages.
3. Disclose the runtime system requirements in the Partner Center Store listing:
   Windows 11 24H2 (OS build `26100` or newer) on ARM64 and an NVIDIA RTX Spark GPU
   with NVIDIA R615+ drivers (required by CUDA 13.4 on Windows on Arm).

When running inside a Store MSIX container (`process.windowsStore === true`), the app
disables in-app auto-updates, hides update menu items, and disables external `mlx agent`
binary downloads/updates so updates flow exclusively through the Microsoft Store.

## WinGet (Windows Package Manager)

WinGet distributes the portable `mlx-node-<version>-win32-arm64.zip` release asset as a
nested portable package (`ArchiveBinariesDependOnPath: true`, `mlx-node` command alias).

For the one-time initial submission to `microsoft/winget-pkgs`:

1. Download the `mlx-node-desktop-winget-manifests` artifact from a stable release run,
   or generate the manifests locally (pass `--tag <release-tag>` if the GitHub release
   tag omits the default `v` prefix):

   ```bash
   yarn workspace @mlx-node/desktop winget:manifests \
     --version <version> \
     --zip packages/desktop/out/mlx-node-<version>-win32-arm64.zip
   ```

2. Copy the generated `manifests/<letter>/<Publisher>/<App>/<version>/` directory into
   a fork of `microsoft/winget-pkgs`, verify with `winget validate --manifest <dir>`,
   and open the initial PR.

For automatic WinGet PRs on every subsequent stable release, configure the repository
secret `WINGET_TOKEN` (a GitHub PAT with `public_repo` scope) and optionally set the
`WINGET_PACKAGE_ID` repository variable (defaults to `mlx-node.mlx-node`). Once the
GitHub release is published (even if the subsequent Homebrew cask bump fails), the
workflow's `winget` job installs a pinned, SHA-256-verified Komac binary in a
credential-free step, then runs `komac update <package-id> --version <version> --urls <exact-release-tag-zip-url> --submit`
with `WINGET_TOKEN` scoped strictly to that submission step via `GITHUB_TOKEN`.

The macOS DMG and update ZIP contain the same Developer ID signed, notarized, and stapled app. The ZIP is
created after stapling, then extracted to verify its signature, ticket, and version
before publication. Dry runs build and verify the macOS and Windows archives but
skip notarization and release uploads.

The app uses [electron-updater](https://www.electron.build/v26/docs/features/auto-update/)
with public GitHub Releases. macOS installation still uses Squirrel.Mac. Keep the
archive naming, bundle ID `ai.mlxnode.desktop`, and Developer ID signing identity
consistent across releases. No additional update server, client token, or signing
secret is needed. Packaging includes `Contents/Resources/app-update.yml` before signing.

The release workflow generates blockmaps from the final ZIP using electron-builder's
build-time library; it does not replace our packager or ship that library in the app.
The updater reuses a previously cached ZIP where possible and falls back to a full
download when the cache or blockmaps are unavailable. The first update normally
downloads the complete ZIP. Retain old ZIPs and blockmaps on their original releases.

Updates default to a 100% rollout. Set the repository variable
`DESKTOP_UPDATE_STAGING_PERCENTAGE` to an integer from 0 to 100 before a release
to limit automatic offers; the manual workflow has a matching input. To expand or
pause an existing rollout, edit only `stagingPercentage` in that release's
`latest-mac.yml` and replace the manifest asset. Keep all archive hashes and URLs
unchanged. Clients keep a persistent rollout ID, so increasing the percentage
includes the earlier group. A rollout change does not undo installed updates;
publish a higher fixed version to repair a bad release. Beta-channel selection is
not enabled in the desktop UI.

Packaging stamps update eligibility into the staged app manifest before signing;
the source manifest does not enable updates. The release tag, desktop manifest,
bundle version, DMG filename, ZIP filename, and update metadata must agree. The
workflow verifies that the ZIP, blockmap, DMG, and manifest are fully uploaded
before publishing the draft release.

Push the version tag to start a release. The workflow creates or resumes a draft,
builds from the tagged commit, and publishes only after signing, notarization,
all uploads, and main-branch CI for that exact commit succeed. The publication
step waits up to three hours for the latest `ci.yml` push run on `main`; failed,
cancelled, or missing CI cannot publish an update. Keep release notes in a draft;
do not publish through the GitHub release editor first. A failed build leaves the draft hidden
from updater clients, and existing published releases are never overwritten.

To retry a failed release, rerun its workflow or dispatch **Desktop Release** with
`dry_run=false` and its existing `release_tag`. The dispatch checks out that tag.
Dry runs default to `true`, build the selected ref, and make no release changes;
an optional tag input validates the planned version without requiring the tag yet.

Before the first production rollout, verify an upgrade between signed releases,
including normal quit without relaunch, **Restart to Update…**, active inference,
and a failed differential download falling back to the full ZIP.
