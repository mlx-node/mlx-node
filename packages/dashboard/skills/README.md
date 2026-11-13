# Skills bundled with the Control Panel

Skills ship inside `@mlx-node/dashboard` and are never fetched at run time. `manifest.json` lists every file of every
bundle with its sha256 and size. The Control Panel copies a bundle into `~/.mlx-node/chat/plugins/<id>` only after
every file matches the manifest, and hands that folder to the agent as a plugin directory.

| Bundle      | Root               | Used by                                                                                                                                 |
| ----------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `chat-web`  | `own/chat-web`     | Chat: the `web-tools` skill. Its body (after the frontmatter) is also the end of the Chat system prompt                                 |
| `exa`       | `vendor/exa`       | Work, when Exa is the search provider: `exa-search`, `exa-contents` (`exa-labs/agent-skills`, MIT)                                      |
| `deepseek`  | `own/deepseek`     | Work, when DeepSeek is the search provider: `deepseek-search` (written by mlx-node; DeepSeek has no skill)                              |
| `brave`     | `vendor/brave`     | Work, when Brave is the search provider: `web-search`, `llm-context`, `news-search` (`brave/brave-search-skills`, MIT)                  |
| `firecrawl` | `vendor/firecrawl` | Work, when Firecrawl reads pages and the `firecrawl` CLI is found: the CLI, search, scrape, crawl, map skills (`firecrawl/skills`, ISC) |

Only the search and read subset of each upstream repository is shipped. Not shipped: Brave `bx` (its install step
pipes a download into a shell), Exa `lead-generation` / `company-research` / `build-with-exa`, Firecrawl `interact`,
`agent`, `monitor` and the other core skills, and the `build/` and `workflows/` groups.

A bundle root holds `LICENSE`, `PROVENANCE.md` (vendored bundles only) and `skills/<name>/SKILL.md`, optionally
`skills/<name>/references/<file>.md`. Path segments use `a-z`, `0-9`, `-` and `.` only. No symlinks, each file at
most 64 KiB, at most 40 files per bundle.

## Changing a bundle

1. Edit or replace the files under the bundle root.
2. Recompute each file's sha256 (`shasum -a 256 <file>`) and byte size, and update `manifest.json`.
3. Run the dashboard tests: the bundle tests refuse any file that is not in the manifest, a hash that does not match,
   and a Chat prompt that no longer ends with the `web-tools` skill body.

## Review checklist for a vendored bundle update

Read the full diff of every file, then check each point:

- No new file types: Markdown, `LICENSE` and `PROVENANCE.md` only.
- No `allowed-tools` line in any frontmatter (local change 1): in Work every command must pass the permission prompt.
- No pointer to a file that is not shipped; a pointer to an install or security file is replaced by the fixed
  sentence of local change 2, and both changes are listed line by line in `PROVENANCE.md`.
- No instruction to send data anywhere but the provider's own API host.
- No install step that pipes a download into a shell, no `npm install -g`, no `@latest`, no `--api-key` on a
  command line, no `firecrawl login` / `init` / `setup`.
- `PROVENANCE.md` names the upstream repository, the full 40-character commit (equal to the manifest's), the date
  fetched, the files taken and every local change.
- Known upstream lines kept as written, answered by the Work rules (never create a Firecrawl monitor, never send
  Firecrawl feedback, never pass a key with `-k` or `--api-key`): the `firecrawl` skill's "Prefer setting a monitor"
  and `firecrawl-monitor` link, its "Alexandria feedback" calls, its "firecrawl-workflows skills (already installed
  alongside this CLI skill)" line (they are not shipped), and `firecrawl-search`'s "`-k` is the API-key option". Check
  that an update adds no new line of these kinds.
