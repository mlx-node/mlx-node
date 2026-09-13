# mlx-node for macOS

Install the signed DMG from [GitHub Releases](https://github.com/mlx-node/mlx-node/releases)
by dragging mlx-node to Applications. The app requires Apple Silicon and macOS 26 or newer.

## Coding Agents

Open **Coding Agents** in the control panel to let other agents delegate GitHub
work to `mlx delegate github`. The app includes the CLI and runs it with Electron's
bundled Node runtime; users do not need npm or a separate Node installation. At
launch it creates `~/.mlx-node/bin/mlx`, verifies delegation support, and repairs
its target after the app moves or updates. It never overwrites a foreign command
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
a handoff without recursive subagents. Delegated sessions appear on **Sessions**. The CLI owns its model
process; the desktop inference service is used separately for installation checks.

The page uses the installed default local model to read the listed global
instruction file and recognize active delegation instructions, including manual
wording. **Install…** appends a short plain-text instruction, preserves existing
content, and verifies it with that model. No markers are added. Without an
installed local model, checks and installation are disabled and the page links
to **Models**. Start a new coding-agent session after setup.

Opening the page never starts inference. **Check status** checks only new or
changed nonempty files; missing and empty files are recognized without loading
the model. Checks run sequentially, with **Waiting…** shown for queued rows.
The Installed menu offers **Recheck with model** to bypass a cached result.

Semantic results persist in the owner-only `~/.mlx-node/coding-agents.json` file,
keyed by the instruction content, path, selected model, app command and detection prompt.
Only hashes, verdicts and check times are stored there, not instruction text.
Both installed and not-installed results survive restarts and identical file rewrites.
Active checks poll in-memory status. Idle pages refresh metadata every 30 seconds
and when focused; model discovery for this page skips recursive directory sizing.

Supported native global files:

| Agent       | File                                                                              |
| ----------- | --------------------------------------------------------------------------------- |
| Claude Code | `~/.claude/CLAUDE.md` (`CLAUDE_CONFIG_DIR` supported)                             |
| Codex       | `~/.codex/AGENTS.md`, or a nonempty `AGENTS.override.md` (`CODEX_HOME` supported) |
| Grok        | `~/.grok/AGENTS.md` (`GROK_HOME` supported)                                       |

**Installed** requires both a working app command and a model-verified prompt
that uses it. Older prompts using bare `mlx` show **Update…**; the exact previous
template is upgraded in place without duplicate instructions. Custom wording is
preserved, with the current routing instruction appended when needed. If command
setup fails, the page shows the reason and **Retry setup**; it cannot install or
report a working integration until the command is available. Project
instructions, imported files and Grok's optional Claude compatibility sources
are outside this check. Grok may already read a Claude installation through that
compatibility layer; in that case a second native installation is unnecessary.
See [Grok's instruction discovery rules](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/12-project-rules.md).

Instruction contents stay on this Mac. The control panel requests inference
credentials over a private process channel; credentials never reach the page.

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

Both contain the same Developer ID signed, notarized, and stapled app. The ZIP is
created after stapling, then extracted to verify its signature, ticket, and version
before publication. Dry runs build and verify both archives but skip notarization
and release uploads.

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
