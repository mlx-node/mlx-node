# Provenance: firecrawl

- Upstream: https://github.com/firecrawl/skills
- Commit: 667d3f7fe804f3f95d375fb0da24802f83a115ab
- Commit date: 2026-11-09T18:07:22Z
- Fetched: 2026-11-12
- License: ISC (`LICENSE`, copied unchanged from the repository root)

## Files taken

| Here | Upstream |
|---|---|
| `LICENSE` | `LICENSE` |
| `skills/firecrawl/SKILL.md` | `skills/core/firecrawl/SKILL.md` |
| `skills/firecrawl-search/SKILL.md` | `skills/core/firecrawl-search/SKILL.md` |
| `skills/firecrawl-scrape/SKILL.md` | `skills/core/firecrawl-scrape/SKILL.md` |
| `skills/firecrawl-crawl/SKILL.md` | `skills/core/firecrawl-crawl/SKILL.md` |
| `skills/firecrawl-map/SKILL.md` | `skills/core/firecrawl-map/SKILL.md` |
| `skills/firecrawl-scrape/references/large-results.md` | `skills/core/firecrawl-scrape/references/large-results.md` |

## Local changes

Only the two changes the app allows. Line numbers are those of the upstream file.

1. The `allowed-tools` frontmatter entry is removed (it pre-approves shell commands; in Work every command must pass the permission prompt).
2. A line that points to a file that is not shipped (install, security or setup instructions) is replaced by the fixed sentence: "The firecrawl CLI is already installed and signed in through FIRECRAWL_API_KEY. Never install it. Never run firecrawl login, init or setup."

| File | Upstream line | Change | Upstream text |
|---|---|---|---|
| `skills/firecrawl/SKILL.md` | 5 | removed | `allowed-tools:` |
| `skills/firecrawl/SKILL.md` | 6 | removed | `  - Bash(firecrawl *)` |
| `skills/firecrawl/SKILL.md` | 7 | removed | `  - Bash(npx firecrawl-cli *)` |
| `skills/firecrawl/SKILL.md` | 18 | replaced by the fixed sentence | `Check with `firecrawl --status` (shows auth state, concurrency limit, and remaining credits). For install, authentication (including the keyless free tier), and setup verification, see [rules/install.md](rules/install.md). For output handling guidelines, see [rules/security.md](rules/security.md).` |
| `skills/firecrawl/SKILL.md` | 90 | replaced by the fixed sentence | `- **Install, auth, or setup problems** -> [rules/install.md](rules/install.md)` |
| `skills/firecrawl/SKILL.md` | 91 | replaced by the fixed sentence | `- **Output handling and safe file-reading patterns** -> [rules/security.md](rules/security.md)` |
| `skills/firecrawl/SKILL.md` | 92 | replaced by the fixed sentence | `- **Integrating Firecrawl into an app, adding `FIRECRAWL_API_KEY` to `.env`, or choosing endpoint usage in product code** -> the [firecrawl-build skills](https://github.com/firecrawl/skills/tree/main/skills/build) (`firecrawl-build-onboarding`, `-scrape`, `-search`, `-interact`). They live in a separate repo; install with `firecrawl setup build`.` |
| `skills/firecrawl-search/SKILL.md` | 4 | removed | `allowed-tools:` |
| `skills/firecrawl-search/SKILL.md` | 5 | removed | `  - Bash(firecrawl *)` |
| `skills/firecrawl-search/SKILL.md` | 6 | removed | `  - Bash(npx firecrawl-cli *)` |
| `skills/firecrawl-scrape/SKILL.md` | 4 | removed | `allowed-tools:` |
| `skills/firecrawl-scrape/SKILL.md` | 5 | removed | `  - Bash(firecrawl *)` |
| `skills/firecrawl-scrape/SKILL.md` | 6 | removed | `  - Bash(npx firecrawl-cli *)` |
| `skills/firecrawl-crawl/SKILL.md` | 5 | removed | `allowed-tools:` |
| `skills/firecrawl-crawl/SKILL.md` | 6 | removed | `  - Bash(firecrawl *)` |
| `skills/firecrawl-crawl/SKILL.md` | 7 | removed | `  - Bash(npx firecrawl-cli *)` |
| `skills/firecrawl-map/SKILL.md` | 5 | removed | `allowed-tools:` |
| `skills/firecrawl-map/SKILL.md` | 6 | removed | `  - Bash(firecrawl *)` |
| `skills/firecrawl-map/SKILL.md` | 7 | removed | `  - Bash(npx firecrawl-cli *)` |

Links to upstream skills that are not shipped (for example `firecrawl-interact`, `firecrawl-download`) are left as they are: they point to no install step, and the agent does not find those skills.
