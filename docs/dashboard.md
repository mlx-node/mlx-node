# Dashboard (Control Panel window)

Browse local models, `mlx agent` sessions, inference metrics, and the
paged-attention cold cache. It ships inside `@mlx-node/dashboard` and is the Control Panel
window of **Portato**, the desktop app — open it from the tray.

**There is no `mlx dashboard` command and no HTTP server.** Both were removed once
the app replaced them: the SPA is served from a custom `app://` scheme and speaks to
the runtime over a MessagePort. That deletes a whole class of exposure rather than
guarding it — there is no port to bind, nothing to firewall, and no `--host` that can
put an unauthenticated control panel API on a LAN.

The dashboard still never links the native addon (no Metal init, instant start), and
all data still comes from disk under `~/.mlx-node`.

[Portato](../packages/desktop/README.md) downloads app updates in the
background. Use **Restart to Update…** from the tray when ready, or **Check for Updates…**
to check manually.

## Where it runs

```
CONTROL PANEL  utilityProcess               no native addon
  createDashboardRuntime()
    main thread   DownloadManager + the RPC transport      async only
                  ChatService: one mlx-agent per mode (ACP over stdio)
                    ├─ Chat agent ─ mlx_web (stdio MCP: web_search, web_fetch)
                    └─ Work agent ─ shell, files, the user's MCP servers and hooks
    worker thread DashboardDb + every synchronous FS walk
                  (both session homes: ~/.mlx-agent and ~/.mlx-node/chat/chat-agent)
  ⟵ MessagePort ⟶ Control Panel renderer   (MAIN brokers the port once, then leaves)
```

`DashboardDb` and the synchronous filesystem walks own a `node:worker_threads`
worker, because `node:sqlite` and drizzle are synchronous: a trivial call issued
during a 1.5 s query used to wait 1449 ms. With the split, a download-progress call
made during a heavy query returns in 0.2 ms instead of 296 ms.

Route ownership is data on the route (`mainRoute` / `workerRoute`), so a new route
cannot silently land on the wrong thread. The transport thread keeps the five download
routes (the Update plan is one), the catalog update check, the two coding-agent action
routes and `/api/runtime/live`: network I/O, progress events and the live snapshot must never
queue behind a synchronous query. `/api/runtime/live` lists the runtime directory and
reads each small heartbeat file and `supervisor.json` with async fs; none of the others
touches the database.

Chat runs on the transport thread too: `ChatService` owns the agent processes, their
stdio and the 33 ms stream batching, so every Chat route but the conversation list is
a transport route. The list reads session files and stays on the worker. The agent is
the bundled `mlx-agent`, started on demand, one process per mode (`chat`, `work`), and
stopped with the runtime. The Chat agent gets one MCP server, `mlx_web`
(`dist/chat/web/server.js`, run with the Control Panel's own Node), and the bundled
skills from `skills/` (checked against `skills/manifest.json`, copied to
`~/.mlx-node/chat/plugins`). The app writes only metadata there; prompt, tool and model
text is never written, logged or rendered as HTML by the app.

What the Control Panel spawns, and nothing else:

| Process    | Started                        | cwd                                                            | `HOME` / `GROK_HOME`                                | Environment                                                                                                                                                                  |
| ---------- | ------------------------------ | -------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chat agent | first Chat send, on demand     | an empty app folder                                            | an empty app folder / `~/.mlx-node/chat/chat-agent` | built from nothing: no inherited variable, no key; no shell tool                                                                                                             |
| `mlx_web`  | by the Chat agent, per session | —                                                              | —                                                   | the selected search / fetch key in its MCP `env` only                                                                                                                        |
| Work agent | first Work send, on demand     | an empty app folder; each session in the folder the user chose | the real `HOME` / `~/.mlx-agent` (always set)       | the Control Panel's own, minus the app's internals and grok-build's `GROK_AUTH_PATH` / `GROK_CONFIG_PATH`, plus the selected providers' keys; the login shell's `PATH` first |

The Work agent is `mlx agent` as in a terminal: it runs shell commands in the chosen folder, and the user's own MCP
servers, hooks and shell commands can read the provider keys in its environment. The page says so.

Its `PATH` is the terminal's too. An app started from Finder or the Dock gets launchd's
`/usr/bin:/bin:/usr/sbin:/sbin`, and the agent runs plugin hooks (`sh -c`) and stdio MCP servers with its own `PATH`,
so a hook's `node` (Homebrew, fnm, Vite+) was not found (`sh: node: command not found`). The agent's shell commands
were not affected: the agent asks the login shell for them itself (see the Firecrawl item below). So at host start
the runtime runs the user's shell (the user record's, else `/bin/zsh`) once as `<shell> -ilc` with `HOME`, `USER`,
`LOGNAME`, `SHELL`, `TERM=dumb`, `TMPDIR` and `PATH` only and no stdin, and reads `PATH` between two random markers out
of whatever the rc files print (`src/chat/agent/login-path.ts`). fish prints its list with `string join :`; any other
shell that is not POSIX is not run. Work's `PATH` is the login entries, then the inherited ones they lack, each once.
Nothing else of the login env is taken, so a key an rc file exports never reaches Work. It runs only in the app
(`MLX_DESKTOP_CLI` set) or under launchd's `PATH`, never on Windows and never for Chat. A Work start waits for the one
pending answer, at most the 5 s limit. A shell that fails, prints no `PATH` or runs past 5 s (its process group is
killed) leaves the inherited `PATH`, with one `[work] login shell …` line in the log.

Work also loads the user's Claude Code setup, as `mlx agent` in a terminal does. The gear's Settings has one switch for
it, **Claude Code in Work** (default on; `ui/src/shell/settings-popover.tsx`), kept in
`~/.mlx-node/chat/work-settings.json` (`{"version":1,"claudeCode":bool}`, 0600; anything else reads as on) and read
through `GET`/`PUT /api/chat/work/settings`. Off sets `CLAUDE_CODE_OFF_ENV` (`src/chat/agent/env.ts`) over any
inherited value, each `false`: `GROK_CLAUDE_HOOKS_ENABLED`, `_SKILLS_` (skills and commands), `_AGENTS_`
(`~/.claude/CLAUDE.md`, not subagents), `_RULES_`, `_MCPS_` (`~/.claude.json`) and `_PLUGINS_` (mlx-agent v0.3.6:
every Claude plugin source, so their hooks, skills, commands, MCP servers and agents, plus `.claude/agents` and
`~/.claude/agents`). A project's own `.mcp.json` and `CLAUDE.md` have no switch and still load. The app's guard plugin
(`_meta.pluginDirs`) is a command-line plugin and loads either way. Chat sets `GROK_CLAUDE_PLUGINS_ENABLED=false` too,
always. A change marks only the Work host stale: an idle Work agent stops at once and the next message starts one with
the new value; a running answer, run, helper, command or monitor ends on the old agent first. Chat is never touched.

Work sends its rules (`WORK_RULES` of `src/chat/agent/profiles.ts`) as the session's `_meta.rules`; the agent puts them
in `<human_rules>` of the system prompt, and a workflow helper's prompt gets them too. A subagent started with
`spawn_subagent` does not (agent v0.3.5 passes rules only to workflow children), so the rules end with a
"SUBAGENT RULES" block and tell the main model to start every `spawn_subagent` prompt with it: who the subagent works
for, that a user message arriving mid-task is a correction from the user or the main agent (the agent gives both as
plain user text), the key and install rules, and a short final report. The rules also say that a subagent which ends
"Subagent was cancelled" without the model's own kill was stopped by the user and is not started again unless the
user asks (the agent gives both stops the same text), and that a finished subagent is continued with `resume_from`,
not with a message. `chat-work-workflow-real-agent.test.ts` checks that a helper's system prompt holds the block.

Work asks before every shell command. On its own the agent runs a built-in safe list (`ls`, `cat`, `echo`, `grep`,
`head`, read-only `git`, a bare `touch` or `mkdir`, …) with no permission request. So the app writes a guard plugin of
its own (`~/.mlx-node/chat/work-guard/plugin.json`, before every Work agent start) and sends it first in each Work
session's `pluginDirs`; its `PreToolUse` hook answers "ask" for every `Bash` and `monitor` call (both run a shell
command). Its name is `mlx-work-guard-<16 random hex>`, new at every start: a folder's `.grok/config.toml`
`[plugins] disabled` list is read with no trust, so a fixed name could be switched off by the folder itself. A managed
`allow_managed_hooks_only` pin (`~/.mlx-agent` or `/etc/grok`) still skips every plugin hook, and a hook whose `sh`
fails does not ask: the guard then fails open. A `[permission]` ask rule cannot carry
this: `GROK_CONFIG` drops that table, and every file that holds one lives in `~/.mlx-agent`. The hook "ask" also
outranks the user's own allow rules and saved bash grants; a deny rule still denies. File reads, listings and searches
do not ask. `GROK_CONFIG` of Work is the user's own overlay object (when there is one) with the output cap set in
`models`. Each mode's cap is its answer room plus the engine's thinking budget (`AGENT_HOST_THINKING`, 12,000): Work
16,384 + 12,000 = 28,384, Chat 8,192 + 12,000 = 20,192 (`src/chat/agent/env.ts`). A lower cap of the user's own still
holds. Work also sets `GROK_ACTIVE_AGENT_MESSAGES=1`, over any inherited value: without it the agent answers
`_x.ai/subagent/message` as not found, so the rail's Message cannot reach a subagent. It also gives the model a
`send_subagent_message` tool. Chat has no subagents and does not set it. In the app Work's `PATH` starts with the
login shell's entries (above); Chat keeps its fixed `/usr/bin:/bin:/usr/sbin:/sbin`.

The engine also runs with the family thinking levels (`AGENT_HOST_THINKING.levels`, server option `thinkingLevels`,
`packages/server/src/host/thinking-levels.ts`): its `/v1/models` rows list each chat family's levels as
`reasoningEfforts` (Qwen line Off · Low · Medium · High with High the default; none for Gemma 4), and a Messages
request's `extra_body.reasoning_effort` maps through the family row (Qwen Low = 1,024 thinking tokens, Medium = 4,096),
on `/v1/messages/count_tokens` as on `/v1/messages`, so a count is of the prompt the answer renders. A family with a
checkpoint template that reads `reasoning_effort` gets the level's word with its budget, so Low and Medium never run at
the template's default effort: `qwen3_5` and `qwen4_exp` (the Qwen3.8 templates take `low` and `medium`, and with no
word say `xhigh`) and `deepseek_v41` (its effort score; the engine renders `medium` at its High score, 75). Both
templates write the effort at the start of the prompt, so on these models the first answer after a level change
prefills the whole conversation again (Off already did). The Qwen3.5 and 3.6 templates of `qwen3_5` read no effort,
so the word changes nothing there. `qwen3`, `qwen3_5_moe` and `nemotron_h` templates read none, and their levels send the budget alone (`thinking-levels.test.ts` reads the
checkpoint templates kept in `crates/mlx-core/src/tokenizer/fixtures`). The default level sends nothing, so a request
that names it runs as one that names none. No level raises the 12,000 budget, so the caps above hold for every family
whose engine enforces that budget. Inkling2's engine does not (`ThinkingPolicy::None`):
its levels only set the effort number in its prompt (Off 0, Low 0.2, Medium 0.7, High 0.9), so at any level a long
Inkling2 thought can use the answer room. The agent (the app and `mlx agent` need mlx-agent v0.3.6 or later,
`MLX_AGENT_MIN_RELEASE`) builds its `reasoning_effort` session option from those rows and adds the level as
`extra_body.reasoning_effort` (`none` for Off, never `minimal`) only for a model whose `/v1/models` row has
`owned_by: "mlx-node"` and lists `reasoningEfforts` or `reasoning_efforts` (it reads `reasoningEfforts` first). There it goes beside `output_config.effort` and
`thinking: {type: "adaptive"}`, which the engine never reads, so the extra field carries the level; Off sends that field
alone, with no thought config. `chat-thinking-levels-real-agent.test.ts` compares its request with one for a row with
no levels: they differ in those three fields alone. It keeps the level in each session's `summary.json`, so the
Sessions detail lists `Reasoning high`. `mlx serve` by hand and `mlx launch claude` do not set the option: their
rows list only the family's `reasoning_efforts` (`reasoningEffortsFor` of `@mlx-node/lm` family-data), and the effort
a request names goes to the model as `reasoningEffort`. Every other Messages endpoint gets no `extra_body` (a strict
one answers 400 to an unknown field). Every host `/v1/models` row says `owned_by: "mlx-node"`, and only a server with
`thinkingLevels` lists `reasoningEfforts` (`__test__/server/models-list.test.ts`). A family with no levels of its own
keeps only its `reasoning_efforts` there, so the agent offers Gemma 4 Off and High.

One capability flag goes out at `initialize`, in both modes: `clientCapabilities._meta` =
`{"x.ai/incrementalBashOutput": true}` (`initializeParams`, `src/chat/agent/profiles.ts`; `fs` and `terminal` stay
false). With it each live output tick (every 100 ms with new bytes; never stored) sends `rawOutput.output: []` and the
new bytes in `output_delta` instead of the whole output as a number array, about a third of the size. The tick's text
block still holds the whole output, and that is all the app reads: it never reads `output_delta`, which can come out
empty after the agent cuts its buffer, and empty means "clear". `x.ai/bashOutputNoColor` stays off (it would drop
`CI=true` and set `TERM=dumb` for every command of the user), and so do `x.ai/statusLine`, `x.ai/userMessageEcho` and
`x.ai/folderTrust`. A running agent keeps the flags it started with.

Compaction (the agent summarizes older messages once its context passes 85 % of the window) is set for a local model
in both modes, over any inherited value:

| Setting                                            | Value          | Mode       | Why                                                                                                                                                                                                                                                                            |
| -------------------------------------------------- | -------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| env `GROK_TWO_PASS_COMPACTION`                     | `0`            | Work, Chat | the early second summary waits behind the main step on the local engine and doubles the work                                                                                                                                                                                   |
| env `GROK_COMPACTION_WALL_CLOCK_SECS`              | `0` (no clock) | Work, Chat | the agent's 300 s clock cut every local summary, so each step of a long turn waited 5 minutes and compacted nothing                                                                                                                                                            |
| `GROK_CONFIG` `models.inference_idle_timeout_secs` | `3600`         | Work       | a stall limit, not a turn cap: a summary reads the conversation for minutes with no event the agent counts, and a workflow child queued behind it ends at the agent's 600 s default; `0` would mean 10 s; a higher integer of the user's own stays (`AGENT_IDLE_TIMEOUT_SECS`) |

Chat keeps the agent's 600 s stall limit: its turn cap pauses while the agent compacts (Turns, below).

The desktop engine runs these agents, so its server sets the three thinking options of
`@mlx-node/server/host/agent-thinking` on every Messages request that is not reasoning effort `none`: a budget of
12,000 thinking tokens (only when the request sets none), the cycle guard (an exact repeat inside a thought closes it)
and the end text written before the close when a cut ends a thought. The cut thought ends on that sentence in the
Thought card. Responses on the same server gets none of them. An engine whose addon predates the
guard ignores it and the end text; the budget alone still bounds the thought.

## Control Panel liveness and recovery

A slow reply to a page request never ends a turn. MAIN's broker is the only judge of whether the Control Panel is
alive, and it judges from the child alone; a stuck database worker is restarted alone; every recovery leaves a record,
and the thread says what stopped a turn.

```
page call, no reply                   MAIN broker (dk/main/broker.ts)              Control Panel
  30 s   cancel (withdraws a           every 5 s: {type:'mlx-ping', n} ──────────▶  parent port, main thread:
         worker call not started)                                                    one synchronous function,
  35 s   THAT call fails: E_TIMEOUT    ◀── {type:'mlx-pong', n, open, convs, slow}   no worker, no fs, no route
         (504). Nothing else.          any message from the child counts as life
                                       a tick > 1 s off schedule is not counted
                                       (MAIN stalled, the Mac slept, the clock jumped)
                                       6 silent ticks (30 s), no turn in the last pong
                                       24 silent ticks (120 s), a turn runs or the
                                         child never spoke
                                         -> intent row, SIGTERM, +12 s SIGKILL, restart
worker call 60 s + 4 s with no
  reply or withdrawal ack, or the
  worker dies -> the WORKER restarts  ◀── {type:'mlx-worker', reason, route, …}      process, agents, turns stay
```

No page message can stop or restart the Control Panel: the page, the preload and MAIN have no such channel. A page call
that gets no reply of any kind in 35 s fails alone with `E_TIMEOUT` ("The app did not answer in 35 s.", and for a
mutation "The change may or may not have been applied."); streams and other calls are untouched, and the page gets no
retry timer. A turn of any length passes the liveness check as long as the Control Panel answers one ping in 120 s;
the Control Panel also pushes a pong when a turn starts and when one ends, so MAIN never holds a turn count older than
the turn. A turn the agent starts by itself (a wake) counts as running from its first update with no typed prompt open
(the typed turn's reply may come in the same stdout chunk) until its `turn_completed` line or the agent's exit; the
host keeps it (`TurnState.origin: 'wake'`), not the service. A workflow run the agent reports `active` counts as running
too, with no turn: the conversation is in the host's `active` list with `background: true`. At a
signal the Control Panel tells MAIN once which turns its close cancels; a turn the user already stopped is not listed.

A stuck or dead database worker is replaced once its exit is confirmed (a queued mutation must be dead first). Its open
calls fail with `E_UNAVAILABLE` "Dashboard database restarted; the request was not answered"; the next call goes to the
new worker. Restart delays are 0, then 5, 10, 20 and 40 s for each worker that dies, or hangs on a call posted, within
60 s of its start; after six such deaths in a row worker routes fail at once, and a call 60 s or more later starts one
new worker.

| timer                                         | value                                 | what it does                                                         |
| --------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------- |
| page call                                     | 30 s + 5 s                            | fails that call (`E_TIMEOUT`)                                        |
| database worker call                          | 60 s + 4 s                            | restarts the worker                                                  |
| MAIN silence, no turn                         | 30 s (6 ticks)                        | restarts the Control Panel                                           |
| MAIN silence, a turn runs or no first message | 120 s (24 ticks)                      | restarts the Control Panel: the one app deadline that can end a turn |
| kill grace                                    | 12 s                                  | SIGKILL after SIGTERM                                                |
| Control Panel exit cap                        | 10 s                                  | its own teardown                                                     |
| agent stop ladder                             | EOF, +1.5 s, +1 s                     | stops the agent at teardown                                          |
| `pageGoneMs`                                  | 60 s                                  | with no window, open Work requests are denied                        |
| Chat turn cap                                 | 10 min, Chat only                     | ends a Chat turn; paused while the agent compacts                    |
| restart budget                                | 6 short lives, 0 / 5 / 10 / 20 / 40 s | a liveness restart counts                                            |

**The record.** MAIN writes `~/.mlx-node/ledger/main-<startMs>-<pid>.1.jsonl` (folder 0700, file 0600, append only, one
synchronous write per row, no fsync; files older than 30 days are deleted at MAIN start). Every row has `v`, `t`
(epoch ms), `m` (monotonic ms), `p` (`<pid>.<startMs>`), `n` (row number) and `k` (kind):

| `k`           | written                                                      | fields                                                                                                               |
| ------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `proc`        | first row                                                    | `role: main`, `host: desktop`, `pid`, `ppid`, `ver`, `seq`                                                           |
| `child.spawn` | each fork                                                    | `role: panel`, `gen`, `pid` when known                                                                               |
| `slow`        | 2 silent ticks (`op: loop`), and the first message after     | `target: panel`, `state` (`open`, `done`), `ageMs`                                                                   |
| `slow`        | once per page call a pong shows 30 s old (`op: rpc`)         | `route` (a pattern such as `GET /api/sessions/:id`), `thread`, `ageMs`, `wq`, `wbusyMs`, `wroute`                    |
| `intent`      | before the SIGTERM; at SIGKILL; at a worker restart; at quit | `what`, `cause`, `stage` (`start`, `idle`, `turn`), `ageMs`, `open`, `convs`, `target`, `by`, `reason`               |
| `child.exit`  | every exit: one row per recovery                             | `child`, `pid`, `code`, `asked`, `cause`, `by`, `sinceIntentMs`, `uptimeMs`, `crashes`, `open`, `convs`, `cancelled` |
| `note`        | after a dropped field, or a failed write                     | `what`, `count`, `from`, `to`                                                                                        |

`cause` of an exit: `main.panel_no_pong` (the broker's own restart), `panel.exit_unasked` (a crash: not asked, code
other than 0), `os.signal_term` (not asked, code 0: the Control Panel's own SIGTERM/SIGINT handler ran), `user.app_quit`.
`convs` are the turns of MAIN's last pong; `cancelled` the turns the dying Control Panel said it was cancelling as
`app-closed`, posted before its teardown does anything async. One encoder writes every line: a row holds numbers,
booleans, list members, UUIDs, row ids and route patterns only; anything else is dropped and counted. A failed write
is counted, said once on MAIN's stderr, and never delays or changes a stop. MAIN hands the folder to the Control Panel
as `MLX_PANEL_LEDGER_DIR`.

**The thread.** The next Control Panel reads the ends at start (newest first, at most 32, from the last 1 MiB of the newest files) and sends them in
the host state as `appEnds`. A turn of conversation C gets the line of an end when its stored end is `cancelled`, C is
in the end's `cancelled` list and the end lies between the decision and the exit (1 s of slack each side); or when it
has no stored end, C is in either list, it started by the exit, and the conversation's next turn, if any, started after
the exit (so the line stays after Continue). The line replaces "This answer stopped before the end." or "This answer did
not finish.", with Continue:

| trigger | line                                                                                                                         |
| ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| no-ping | The app restarted its Control Panel because it did not answer for 120 s. The turn was stopped by the app, not by you.        |
| crash   | The Control Panel of the app stopped by itself (exit code 1) and was restarted. The turn was stopped by the app, not by you. |
| signal  | Something outside the app stopped its Control Panel. The turn was stopped with it, not by you.                               |
| quit    | The app was closed during this turn.                                                                                         |

A Stop the user pressed is never in the `cancelled` list, so it keeps its own words; with no `cancelled` list (the
note was lost) a stored `cancelled` end keeps "This answer stopped before the end.".

## Using it as a library

```ts
import { createDashboardRuntime } from '@mlx-node/dashboard';
import { serveRuntimeOverPort } from '@mlx-node/dashboard';

const runtime = createDashboardRuntime({ modelsDir, dbPath });
// `call({method, path})` is the whole API surface; it never rejects for an API
// failure, it returns a `{ok:false, status, code, message}` envelope.
```

## Pages

Twelve routes. Ten sit in the sidebar (`⌘1`…`⌘9` jump to the first nine; all ten sit in the `⌘K` command palette); one route table
(`ui/src/shell/routes.tsx`) feeds the router, the sidebar, the breadcrumb, the hotkeys and the palette.

| Key       | Goes to                                                                       |
| --------- | ----------------------------------------------------------------------------- |
| `⌘1`      | Overview                                                                      |
| `⌘2`      | Chat                                                                          |
| `⌘3`      | System One                                                                    |
| `⌘4`…`⌘9` | Sessions, Metrics, Engine, Cache, Models, Coding Agents                       |
| `⌘⇧O`     | A new conversation in the last used mode                                      |
| `⌘K`      | The command palette (it also lists the 20 newest conversations of both modes) |

| Route            | Page           | What it shows                                                                                                                                                                                                                                                                                                                          | Data                                                                                                         |
| ---------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `/`              | Overview       | Live engine panel and health list; "Last 24 hours" tiles (each opens a detail panel in place); turns per hour; storage tiles; latest turns                                                                                                                                                                                             | live channel, `/metrics/summary`, `/metrics/timeseries`, `/metrics/turns`, `/models`, `/cache`, `/downloads` |
| `/chat`          | Chat           | Chat and Work: the conversation list (search over both modes), the thread, the composer: the + menu (files and photos, web search, crawler), model picker, (Work) folder chip (bar once set); Work's permission and plan cards. `?c=<id>` opens a conversation, `?mode=chat\|work` the New screen. See [Chat and Work](#chat-and-work) | `/chat/*` routes, the Chat host and conversation streams, live channel                                       |
| `/system-one`    | System One     | Structured extraction (Clairic) and calibrated decisions (CLEF): visual field/rubric builders, live routing thresholds, batch evaluation studio, saved specs, run history, and copyable `/v1/systemone` API snippets                                                                                                                   | `/systemone/*` routes, `/models`, `/catalog`, live channel                                                   |
| `/sessions`      | Sessions       | `mlx agent` sessions and the app's conversations (tagged Chat or Work): search, directory filter, live filter, summary tiles, sortable table                                                                                                                                                                                           | `/sessions`                                                                                                  |
| `/sessions/:id`  | Session detail | Activity tab (session totals, per-turn charts, turn inspector with an agent timeline (one lane per agent, handoffs, replay; live while the turn runs), subagents) and Transcript tab (turn groups, find); a live session shows its open thinking block and running tool                                                                | `/sessions/:id`, `/sessions/:id/timeline`                                                                    |
| `/metrics`       | Metrics        | Six tabs: Summary · Speed · Usage · Speculation · Failures · Turns. One header with range and model filter; only the open tab queries. KPI tiles pick the main chart; `?detail=` opens a tile's panel                                                                                                                                  | the six `/metrics/*` routes                                                                                  |
| `/engine`        | Engine         | Live part: "Now" ticker, output rate, memory, turns in flight, loaded models, KV block pool. History part (1h · 6h · 24h · 7d): busy time, time by phase, output rate, memory, decode batch size, model loads, processes, supervisor                                                                                                   | live channel, `/runtime/series`, `/models`                                                                   |
| `/cache`         | Cache          | Model caches (one folder per model identity under `prefix-cache/`: size, files, last use, lock; delete an idle one); cold-tier usage against quota, objects by age with an eviction preview, hit / miss trend, bytes written and restored, cold-tier health counters; evict older, clear all                                           | `/cache`, `/cache/prefix`                                                                                    |
| `/models`        | Models         | Local models table (also the legend of the disk-use bar), usage per model over 30 days, recommended catalog with installs; `?focus=` opens a model's drawer                                                                                                                                                                            | `/models`, `/catalog`, `/catalog/updates`, `/downloads`, `/metrics/summary`, `/metrics/timeseries`           |
| `/coding-agents` | Coding Agents  | The model and command the coding agents will use, per-agent setup with the instruction block it adds                                                                                                                                                                                                                                   | `/coding-agents`                                                                                             |
| `/providers`     | Web Search     | "Now" summary of what Chat and Work get; the search provider choice (one of the configured Exa, DeepSeek, Brave, or Off) and the crawler choice (Firecrawl or Off), with the skills each installs; one key card per provider (save, check, replace, remove; the key is never shown after entry)                                        | `/providers`                                                                                                 |
| `/welcome`       | Onboarding     | First run: pick and download a model, then set up coding agents. Outside the shell; `/` sends here while the model library is empty                                                                                                                                                                                                    | `/models`, `/catalog`, `/downloads`, `/coding-agents`                                                        |

A models folder that is a link to a missing target (a folder on a disk that is not connected) is not an empty
library: `GET /api/models` answers 503 `models-dir-unavailable` with "The models folder … is a link to …, which is not
available. Connect that disk, or change the link, then try again." The Models page shows it in place of the table, and
`/` opens Models rather than Onboarding. Install answers the same 503 before a job exists, and a job whose disk goes
away fails with the same sentence; neither makes a folder in place of the link. The engine (`resolveModelsDir`,
`ModelsDirUnavailableError`) and the CLI say the same words, and the desktop sidecar exits 78 on it, so the
supervisor stops at once instead of retrying.

A System One run changes only the Output card. The four metric tiles are always there and show "—" when the mode
on screen has no result. The Output status line is always there too, with three faces in one box: rest ("No run yet"
or "Last run · model · time"), running, and failed. A failed run puts its text in that line, removes the answers and
shows each requested row with "—", so answers of an earlier input never sit under an error; a toast is added only
when the line is not in view. Nothing in the card reacts before `--duration-hold`, so a run that fails at once
changes nothing on a repeat. The Highlights tab marks character ranges in the request text only: a run with only an
image or PDF has no text to mark, and the tab says so and shows the image.

The Engine alerts for `restarting` and `failed` add "Reason: …" when the last start failed with a reason: the
message of the sidecar's `[mlx] inference host failed to start:` line, as `failureReason` of `supervisor.json`. MAIN
writes it only while the state is `restarting` or `failed`, so a good start, a stop or a new start clears it.

The Engine "In flight" card is a table with fixed columns (Model, Phase, Progress, Started, Context) under a
header that counts turns per served model ("2 turns on 1 model"). A row names the model that serves the turn
(`modelId`); when the client asked for another name, such as the app's `mlx-local` alias, the row says
"asked for `mlx-local`" under the model name, so two requests on one model never read as two models. "Started" is
the time since the turn began, on the snapshot clock.

A row of this table and of the Overview's Engine card is a lane: one client session, not one model call
(`ui/src/lib/live-lanes.ts`). An agent makes one engine call per step and runs its tool in between, so rows group by
the session the request names (`clientSession`, the `x-grok-session-id` header). When the session's call leaves the
list, the row stays as Idle; the session's next call takes the same row in the same place. An idle row leaves after
10 s (`LANE_HOLD_SECONDS`) without a new call. A request that names no session, and a second call of one session
while its first still runs, get rows of their own. Rows keep the order in which they first appeared. When a list is
full (Overview: 2 or 3 rows, Engine: 32), idle rows give up their place first. On an idle row, "Started" and the
Overview's time read the time since its last call ended.

The Phase badge and the Progress text follow the native phase of the turn (`arc-x/live-turn-row`):

| Native phase | Badge     | Progress                                                                | When                                                                     |
| ------------ | --------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `queued`     | Waiting   | model busy                                                              | the model thread works on other requests (also before it picks this one) |
| `slot`       | Queued    | waiting for a slot                                                      | the scheduler cannot start it: lanes full, memory refused, a held start  |
| `prepare`    | Preparing | restoring N cached tokens (`cachedTokens` > 0), else reading the prompt | render, prefix match, cache restore, GDN replay before its prefill       |
| `prefill`    | Prefill   | done of total tokens, with a bar                                        | as before                                                                |
| `decode`     | Decode    | tokens out                                                              | as before                                                                |
| none         | Running   | no progress for non-streamed turns                                      | a turn without a progress handle                                         |
| (lane)       | Idle      | last call · N tokens out, else last call ended                          | the session's last call ended and its next has not come yet              |

"waiting for a slot" is printed for `slot` only. From `prefill` on the native phase never moves back.

The Cache page opens with one disk cache line: the switch the running engine's host resolved (`persistPagedCache`
of its heartbeat: On, Off, or Unknown for an engine started before the field), the most one turn's capture walk may
write (`coldCapture`: up to N prefix blocks, or "the whole prompt" from 63,125 blocks, and the walk's time cap; a
walk the time cap ends early is picked up by the next turn; nothing for an older engine), and how many recorded
turns ran with the tier on (`scope.enabledTurns`) and off (`scope.disabledTurns`). The turn counts cover every
cache folder, not only the one the page shows. A Qwen3.5 dense or MoE, Flash Next or Hy4-flash turn counts by its own disk
cache, the SSD state tier its trace row carries as `stateTier`: on when `persistent`, off when `temp` or `off`. When more than one
engine runs, the line names the one it reads.
Neither the switch nor the turn counts prove a write: a turn with the tier on may write nothing. The written side is
the rest of the page (bytes and blocks on disk, restores).

Below that line, "Model caches" lists the stored-state caches of Qwen3.5, Flash Next and Hy4-flash: one folder per model
identity under `~/.mlx-node/cache/prefix-cache/`, named by 32 hex digits, plus the folder of each running engine
started with `--cache-dir` (its heartbeat names it), one panel per folder. A row is a folder the engine itself counts
(a real directory of this user, not a link) that holds a regular `lock` file: every engine open makes one, so a folder
without it is not a model cache and is neither listed nor deleted. Its size is the sum of its regular files but
`lock`, its last use the newest time of those files and of `lock` (a restore moves it too), and its state the lock:
"In use" while an engine holds it, "Idle" when free, "Unknown" where the lock cannot be checked (not macOS). The page
reads names, sizes, times and the lock state only: at most 4,096 entries of each root, and at most 20,000 entries or
250 ms of folder walks per call; a folder it has not finished shows "≥" and the page asks once more per answer until
it has. An unchanged folder is not read again. A root it cannot read says "could not be read" with the system's
answer, not "No model cache yet". A cache whose namespace a running engine
reports (`models[].stateNamespace` of its heartbeat) shows that model's name; any other shows its hash only. The tiles
add the rows; "Size on disk" names the smallest limit a running engine reports for that folder, and none when no
engine reports one (an engine with two or more Qwen3.5 models, or Flash Next or Hy4-flash alone). "Free on volume" is `statfs` of
the folder.

Delete removes one idle cache after a confirm at its row. It holds the folder open, takes the cache's existing lock
(it never makes one), and removes the files and links directly inside the folder in steps of at most 4,096 files or
1 s; the last step removes `lock` and then the folder. It never enters a subfolder (it reports one it left; the folder
is then no longer listed) and never follows a link. Node has no `unlinkat`, so before each removal the step checks
that the folder's path still names the folder it holds open and not a link; a folder moved or replaced during a step
stops it (409 `moved`). The page waits 300 ms between two steps, longer than the 200 ms between an engine's tries for
the lock, so an engine that opens the same cache during a long delete gets the lock and the next step answers `held`.
It is refused while an engine holds the lock (409 `held`), when an engine opened the cache after the page listed it
(409 `changed`: every engine open sets the lock's time, which the page sends back), where the lock cannot be checked
(409 `no-lock-check`), for a folder without a regular `lock` file (404 `not-a-cache`), for a folder the runtime does
not list now (403), and for a cache that is gone (404). A refused delete shows the reason and reloads the list.
The engine removes a cache on its own only at a later state-tier load: unused for 14 days, or the oldest idle ones
over the folder's limit.

One drawer, `TurnDrawer`, shows one turn's phase waterfall and numbers. Overview, Metrics and the Failures tab open
it (`?turn=<traceId>`, answered by `/metrics/turns?traceId=`). It is the only user of the phase waterfall
(`arc-x/span-timeline`).

Where the numbers come from:

| Number                                                                      | Source                                                                                                 |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Turn counts, tokens, latency, speed, failures                               | trace records (`traces` table). Percentiles are nearest-rank, computed in JS on the database worker    |
| Speculation (MTP) acceptance                                                | the MTP fields of trace records with `outcome = 'ok'`                                                  |
| Memory, busy time, time by phase, output rate, batch size, loads, processes | the runtime series (`runtime_*` tables)                                                                |
| Anything labelled "now" or "live"                                           | the live channel (heartbeat files + `desktop/supervisor.json`), never the index                        |
| Cold-tier usage and objects                                                 | a scan of the cold-tier directory on each `/cache` call                                                |
| Model caches                                                                | a walk of each `prefix-cache/<namespace>` folder per `/cache/prefix` call, ≤ 20,000 entries or 250 ms. |
| Sessions and transcripts                                                    | the session folders, read on each call                                                                 |

The Engine page's History part reads the work deltas of the runtime samples (see the work ledger under Runtime
sampler). **Time by phase** stacks each bucket's lane-ms ÷ the bucket's length, so one lane busy the whole bucket is
1: decode, prefill, other (preparing + running + loading) and waiting (model busy + lane wait). A range of more than
90 buckets merges neighbours into one column (24 h: 30 minutes, 7 d: 2 hours). **Output rate** is Σ `rateOutTokens` ×
1000 ÷ Σ `rateOutMs` per bucket, with a gap where the bucket has no decode. The Busy time hint adds the decode and
prefill lane-ms ÷ the observed time. A sample written before the work deltas adds 0, so a range of only such samples
shows both cards empty (`ui/src/lib/engine-history-view.ts`).

A model is keyed by `modelId ?? model` (`model_key`) and keeps one color on every page: 8 color slots in a registry
kept in `localStorage`, models 9–16 reuse the hues with a second encoding, later ones fold into "Other".

A catalog entry (`MODEL_CATALOG` in `packages/agent/src/catalog.ts`) has one or more builds, and each build installs to
its own folder under the models directory: the last part of its repo in lower case, plus `-<folderSuffix>` for a
second build of a repo another build already installs. One folder is one build: one job, one marker, one row in Local
models and one Delete. The chat builds:

| Model              | Build      | Folder                          | Size     |
| ------------------ | ---------- | ------------------------------- | -------- |
| Qwen3.8-27B        | UD-Q4_K_XL | `qwen3.8-27b-gguf`              | 17.7 GB  |
| Qwen3.8-27B        | UD-Q4_K_M  | `qwen3.8-27b-gguf-ud-q4_k_m`    | 16.6 GB  |
| Inkling-2-Mini     | UD-Q4_K_XL | `inkling-2-mini-gguf`           | 18.7 GB  |
| Inkling-2-Mini     | UD-Q4_K_M  | `inkling-2-mini-gguf-ud-q4_k_m` | 17.7 GB  |
| Inkling-2-Mini     | Original   | `inkling-2-mini`                | 20.2 GB  |
| Qwen3.8-Flash-Next | UD-Q4_K_XL | `qwen3.8-flash-next-gguf`       | 111.3 GB |
| Hy4-flash          | Q4_K_M     | `hy4-flash-gguf`                | 168.3 GB |

Repos, globs and assets repos are in [cli.md](cli.md#model-selection-and-first-run-wizard). A GGUF build takes one
variant out of its repo plus the base-model tokenizer sidecars of its `assetsRepo`; the download runner glob-filters
the manifest and stages the sidecars before publishing. Inkling-2-Mini Original is the safetensors release of
`thinkingmachines/Inkling-2-Mini` and takes the default filter. Qwen3.8-Flash-Next keeps its four shards in the repo
folder `UD-Q4_K_XL/`, and downloads keep repo paths, so they land in `<slug>/UD-Q4_K_XL/` with the sidecars at the
root. The System One section offers Clairic Flash (9B) and Clairic (27B), one build each, and CLEF Flash (9B) and
CLEF (27B), two builds each:

| Model           | Build  | Folder                   | Size    |
| --------------- | ------ | ------------------------ | ------- |
| CLEF Flash (9B) | Q8_0   | `clef-flash-gguf`        | 9.7 GB  |
| CLEF Flash (9B) | Q4_K_M | `clef-flash-gguf-q4_k_m` | 6.7 GB  |
| CLEF (27B)      | Q8_0   | `clef-gguf`              | 28.7 GB |
| CLEF (27B)      | Q4_K_M | `clef-gguf-q4_k_m`       | 19.5 GB |

A CLEF build takes one file, by its exact name, out of `ggml-org/Clef-Flash-GGUF` or `ggml-org/Clef-GGUF` (`*Q8_0*`
would also take the mmproj file, which CLEF never uses). The GGUF carries the tokenizer and the decision head, and it
has no `assetsRepo`: a base `config.json` beside it makes discovery read the folder as `qwen3_5` and drop the GGUF. Such
a build (`configlessBuildOf`: a build of a `-GGUF` repo with no `assetsRepo` on this platform, of an entry with
`ggufFamily`; today only the CLEF builds) installs the GGUF alone: when the Hub list has no file of one of its exact
names the job fails before any download with "The Hub has no file "<name>" for this build."; its file list must hold a
GGUF weight and no `config.json` (else "Repo "<repo>" is not a complete model (needs a GGUF weight file and no
config.json)"), and after the download the staged folder must hold the build by the rule below for the entry's family
(`clef`), else the job fails with "The downloaded file is not <model> <build>, so the app cannot load it. Nothing was
installed." and publishes nothing. Its record lists only the GGUF. For such a build, and only for it, a record with no
`config.json` counts as installed (`isModelInstalled` with the build): every file it lists is there, every weight file
it lists is one the build's globs select, and a GGUF it lists loads with no `config.json` and states the entry's family.
So the card shows Installed, the revision and the update badge as for any install, the skip check of a job, the backup
recovery of a crashed publish and a Locate from a folder with this app's record read the same rule, and a full-repo
`mlx download` record of a CLEF repo (every quant, BF16 and the mmproj files) is not the Q8_0 build installed. A folder
with a `config.json` (at the root, or beside the real GGUF) is not such a build at all: the card is not present, and
goes to Install for this app's folder or to Update in place / Clean up & update for another. Update hashes the GGUF and
adds only the record; Locate needs no `config.json`, and in linked mode it refuses a GGUF with a `config.json` beside
the real file ("These files are on another disk than the models folder, and a config.json is beside them, so the app
would not load them as <model>. …"). For every other build, and for every reader with no build (Local models reads a
Located folder of a GGUF that loads alone by its family, with no globs), a record counts as installed only when it
lists a `config.json`, as before. **Open** on a System One card links to `/system-one?model=<folder>&build=<build>`;
the page selects the model whose id or name is `<folder>` (links from elsewhere name a model id), else the model of
that build (discovery's `variant`) whose GGUF is in that folder, and switches to Decide for a CLEF model and to Extract
for a Clairic one. The link names the build's own folder: a card present through a sibling or a renamed folder opens
the page with no model chosen, since the catalog does not say which folder holds the build.

A build reads present (`isModelPresent`) for a folder with a root `config.json` and a loadable weight at the root, or
with a GGUF in a variant folder one level down, by discovery's folder walk for a GGUF repository folder (which
folders, which companions, split sets): only when the root holds no primary safetensors; never a folder whose name
starts with `.`, nor one that holds primary safetensors; links to folders and files followed; the GGUF a regular file,
not a companion (`mmproj`, `imatrix`, `dflash`, `dspark`, `draft`, `mtp`), and a split set counted from its first part
only with every part beside it. Nothing deeper is read. With a root `config.json`, discovery's family name, header
and sibling-asset checks are not applied.

A GGUF folder with no `config.json` reads present on the card of its own model only when the engine loads it with no
asset files and it is that card's build: the GGUF names the build, every part of its split set is a regular file, the
real folder of the file (links resolved) has no `config.json`, and its header (`general.architecture`, read by
`readGgufArchitectureSync` of `@mlx-node/lm/gguf-metadata`, at most 16384 keys and tensors and 128 MiB, as
`readGgufHeader`) names a family whose GGUF discovery needs no `config.json` or `tokenizer.json`, admits the file name
(`bareGgufFamily` of `@mlx-node/lm/model-discovery`) and is the catalog entry's `ggufFamily` (`qwen3_5` for Qwen3.8-27B,
`inkling2`, `qwen4_exp` for Qwen3.8-Flash-Next, `hy_v4_flash`, `clef` for both CLEF entries; set only on an entry with
GGUF builds, and only a build of a `-GGUF` repo uses it). Any other such folder (a header that cannot be read, a GGUF of
another model or another build, a GGUF at the Inkling-2-Mini Original folder) keeps the card it had without the rule:
**Update** or **Clean up & update**. Local models lists a folder the engine loads with no `config.json` under its family
from the header, its build from the file name and no context window, whichever catalog entry its folder name belongs to.
A folder with the download marker of the build's repo under another name counts too, by the same rules. For a repo that
two builds share (the Qwen3.8-27B, the Inkling-2-Mini and the CLEF GGUF builds), a folder counts as the build its GGUF
file name states, not the build its name suggests, and the marker alone never makes a build present. When a build's
folder holds a loadable model that is another build of the same repo, the build has `heldBy` (that build's name, or
`another model` when the weights state none): the card shows **Holds UD-Q4_K_M** with no Install, the build in the
folder reads as present on its own segment, and the server refuses an install into the folder with 409 (`"<folder>"
holds <build>. Delete it in Local models, then install <build>.`), because the owned swap would delete the other build.

A folder that reads present counts as installed on its card: the card reads **Installed** and **Details**. Without
the app's complete record (`installed` false, as for a hand-made folder or an `mlx download` of the weights only, also
through a link) the card shows no version and no update badge. To get the catalog's version, delete the folder in
Local models, then install. Delete of a row whose entry in the models folder is a link (`link` on the row) removes
only the link: its confirmation is **Remove link?** "This removes the link <name> from the models folder. Its folder
and the model files in it are not changed." with **Remove link** in the neutral tone and the toast "Link removed"; the
row's size is the size of the folder the link points to. A real folder is deleted with its files.

The recommended section is a grid of equal cards: two columns from 780 px of content, one below it, never three. Every
card in a grid has the same height in every state, because it has five fixed rows: title, description (two lines at
most), build, extras and action. A state changes what a row shows, never how many rows there are or how tall they
are. With the dev harness running, `yarn workspace @mlx-node/dashboard dev:shots --card-check` measures this in the
browser on `/models` and the gallery and exits 1 when two cards of a page differ.

- **Title:** the logo and the model name, which links to the shown build's repo; a Default or System One badge; an
  Update badge when the shown build is on disk and has an update.
- **Build:** a card with two or more builds shows a build switch ("Build of Qwen3.8-27B", one segment per build); a
  one-build card shows the build name as a tag. A segment carries a mark: a spinner (downloading), a warning (needs
  attention), an up arrow (update available) or a check (installed), each with its word for screen readers. Choosing a
  segment starts nothing; it changes only what the card shows. The card shows, first match, the build chosen on this
  page visit (page state, not stored), a build with a running job, the first build on disk, the first build. Under the
  switch, the build line reads size · source · note (`16.6 GB · Unsloth GGUF · 1.1 GB smaller`). A disk or memory
  warning replaces it, and a running job replaces it with its live numbers.
- **Extras:** the build's `included` text (`Vision and DSpark draft included`), or the DFlash2 line on Qwen3.8-27B.
- **Action:** one control for the shown build. Install (`Install UD-Q4_K_M` on a card with more builds) starts that
  build only; a download shows a spinner, a bar, the percent and Cancel (none in its last step), with the file name
  in the bar's tooltip; an installed build reads Installed, then Details (its folder in Local models) when Local
  models lists it (`build.listed` of `/api/catalog`, so it does not wait for `/api/models`) and, in System One, Open;
  an update adds an Update button. A build in a blocked state shows a status word and one button, by the first match
  of `catalogActionState`: job > update > installed > missing (Locate) > held > locked > adopt > cleanup > install.

  | state   | status word     | button               | when                                                                     |
  | ------- | --------------- | -------------------- | ------------------------------------------------------------------------ |
  | held    | Holds UD-Q4_K_M | Show in Local models | the folder is a loadable model of another build (Show why if not listed) |
  | locked  | No access       | Show why             | a real folder the app has no write permission on (`blocker.locked`)      |
  | adopt   | Needs update    | Update…              | a real folder of model files, fully listed, no kept refusal              |
  | cleanup | Needs cleanup   | Clean up & update…   | anything else in the way: a link, a file, a large or junk folder         |

  "Needs cleanup" always comes with **Clean up & update…**. The long texts are not on
  the card: the status word's tooltip holds the one-line reason (also as screen-reader text), **Update…** loads the
  plan ("Checking…") and opens its confirmation, **Clean up & update…** opens its confirmation from the catalog's
  numbers with no request, and **Show why** is a popover with the reason. A job's tooltip says what it does to the
  old entry: "Checks the files in <slug>. Nothing is deleted." (update), "Removes the link when the download is
  complete. Its folder is not changed." (cleanup of a link), "Moves the old folder to the Trash when the download is
  complete." (cleanup of a folder or file). The build line of a job reads bytes, speed and time left; the time left
  waits 10 s into each phase (checking files, downloading; the speed is measured again when the phase changes), is
  left out while checking or stalled, and reads `under a minute left`, `about N min left`, `about H h M min left`
  (minutes by 5, up to 10 h), `about H h left` or `more than a day left`.

The DFlash2 line belongs to its parent model (Qwen3.8-27B or Inkling-2-Mini), not to a single build: one copy pairs
with every build of that model by name. When auto-installed (`autoInstall: true`), it reads
`Included DFlash2 · <size>` and queues automatically alongside a fresh build install (with an **Add** button still
available if removed or skipped), never following the build switch. Then it reads the received bytes and percent with
Cancel, `added`, `added · waits for the model` while no build of that model is on disk, `update available` with Update,
or `needs update` / `needs cleanup` with Update… or Clean up & update… (`Clean up and update companion weights?`), or
`no access` with Show why. The drafts live at `qwen3.8-27b-dflash2` and `inkling-2-mini-dflash2` in the models
directory, and compatible targets find them automatically on their next load. They are not chat models and never count
as ones. Draft-only checkpoints are excluded from the local-model table and model counts; their bytes still count toward
disk use on Models and Overview. Expand **Companion weights** in the local models panel to inspect or delete them;
deleting a separate companion leaves the target model in place. Delete is not on the card: Local models has one row per
folder, and deleting a row removes that one build. The Qwen3.8-27B UD-Q4_K_XL file also carries an MTP layer inline,
which the native loader detects.

"N of M installed" counts models: a model counts when at least one of its builds is on disk, and M is the number of
chat models offered on this Mac. The System One section counts its own. The command palette has one command per build
that can act: `Install Qwen3.8-Flash-Next` for a one-build model, `Install Qwen3.8-27B (UD-Q4_K_M)` for the others,
and the same for Update, with the ids `models:install:<folder>` and `models:update:<folder>`; a command also chooses
its build on the card. The System One drawer uses the same card and grid.

Onboarding offers one choice per model, with no build control. A choice stands for the build with a running job, else
the first build on disk, else the first build, and the dock adds "Build UD-Q4_K_XL. More builds are on the Models
page." for a model with more builds. The dock uses the card's action as an alert: **Needs cleanup** with one
sentence and the same **Clean up & update…**, **No access** with the reason, or **Holds UD-Q4_K_M** with the held
sentence. The setup wizard asks for the build instead
([cli.md](cli.md#model-selection-and-first-run-wizard)).

An entry with `minMemoryGb` is offered on Models and on Onboarding only once the host reading is in and the Mac's
physical RAM reaches it (`offeredOnHost`); Qwen3.8-Flash-Next and Hy4-flash need 96 GiB, which a 96 GB Mac reports.
The reading is `GET /api/host` (`{ totalMemBytes, diskFreeBytes }`, `useHost`), a transport-thread route, so it never
waits behind the worker's walk of the model folders that `/api/models` does; the Models page asks for `/catalog`
before `/models` for the same reason. Until `/api/host` settles, the Recommended and System One sections keep their
loading shape and show no count, so "N of M installed" is never counted over a list that is still filtered. When it
fails, the entries with a floor stay hidden. Onboarding and the System One drawer read the same request. The System
One drawer and its empty card show one placeholder per System One entry once `/api/catalog` has answered, and two
before.
The setup wizard filters the same way (`visibleCatalog`). Below the floor the page hides the card, its palette
commands and its share of the "N of M installed" count. An installed copy on a smaller Mac stays in the local list,
and loads where its load admission accepts the RAM (Hy4-flash from 39 GiB).

The local-model table's **Quant** column reads the weights. For a folder whose
weights are GGUF (at the root, else in a variant folder one level down, by the
`isModelPresent` rule; companions never count, and a split set counts from
its first part with every part beside it) it is the build the GGUF file
name states (`UD-Q4_K_XL`, `Q4_K_M`, `BF16`; else the variant folder's name),
parsed by the engine's `ggufVariant` (`@mlx-node/lm/model-variant`). The
`config.json` beside a GGUF install is the base model's, so its quant block is
never read. A folder with several builds shows `GGUF`, as does a name that
states none; each build is still served under its own id. A safetensors folder
shows its `config.json` quant block, or "full precision" without one. The
**Context** column is `max_position_embeddings` (root, then `text_config`),
else `model_max_length` (`text_config`, then root) when it is a positive safe
integer, else empty.

A delete, from a row, a companion line or the drawer, takes the entry off the table, the tiles and the disk-use bar
as soon as the runtime answers ok (`useJson(...).patch`); the full `/models` refetch that follows replaces that
body, so a slow listing (a GGUF-heavy models directory can take seconds) never leaves a deleted row with a live
trash button. If that refetch fails, the page says "Could not refresh the model list" with Try again. A GET that
was in flight across any mutation never seeds the cache for the next visit (`json-cache` epoch). The Sessions list
applies a confirmed rename or delete the same way.

A catalog folder held by something the app did not create (a leftover file, a hand-made folder, a link) is described
by `blocker`: kind, file count, size, at most 3 names (counted with `lstat`, never following a link or opening a file,
at most 2,000 entries; only the root `config.json`, or without it GGUF headers, is read, to tell whether Local models
lists the entry), the link text (cut to 200 characters), whether the link is dead, whether it holds model files, and
`locked` (a real folder without write permission). An entry holds model files when it has, anywhere in it, at any depth (links to folders
followed), an entry named `config.json` or named like weights (`.safetensors`, `.gguf` including companions,
`.pdiparams`, `.pdparams`, `.npz`, `.bin`, `.pt`, `.pth`, `.pkl`), or a folder that cannot be read, or more than
20,000 entries to look through. A fully listed real folder that holds model files gets **Update…** (below); a locked
folder gets **No access** and no action, because a folder without write permission can be neither moved nor added to
("The app has no write permission on <slug>, so it cannot move it or add files to it. Change its permissions (Finder,
Get Info), then open this page again."). Every other entry gets **Clean up & update…**.

**Clean up & update…** downloads and checks the whole model into the job's own `.staging/<slug>@<sha>.<pid>.<uuid>`
tree, puts it at `<slug>`, and only then takes the old entry away. Nothing is deleted for good:

| at `<slug>`           | what happens to it                                                                                                                         | `done` event       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| a link (live or dead) | the link alone is removed (`unlink` after an `lstat` that says link); its folder is never opened, written or moved                         | `removed: 'link'`  |
| a folder or a file    | moved whole, as one item named `<slug>`, to the macOS Trash of the disk that holds the models folder (`/usr/bin/trash -s`, no shell, 60 s) | `removed: 'trash'` |
| the Trash refused     | kept at `.staging/<slug>.kept-<uuid>/<slug>`; the model is installed                                                                       | `keptDir`          |

`removed: 'trash'` needs proof: after the Trash call, `lstat` of the old entry answers ENOENT and the holder is still
the same folder (same `dev:ino`). A disk that went away also answers ENOENT, and any other answer (EACCES, EIO, a
changed holder) proves nothing, so in each such case the entry counts as kept: the holder is renamed to `.kept-`, or,
when that fails or another folder took the holder path, the `done` event names the holder path itself. The kept path is also written to the core log. An ended
job's row in `GET /api/downloads` carries `keptDir`, `removed` and, for a cleanup, `entry: { kind, linkToFile }`, so a
page that was not open when a cleanup ended shows the same toast before it dismisses the job. A toast that names
`keptDir` stays until the user closes it. The registry is in memory: after the app quits, nothing names a `.kept-`
folder any more.

The confirmation (`cleanupContent`, tone neutral) says it in three rows: **Removed** "Only the link. It points to
<target>." and **Stays** "That folder and every file in it…" for a link to a folder, **To the Trash** with the folder's numbers
(or "The file <slug> (4.0 KB).") for a folder or a file, and **Downloaded** with the model, its size and, from 20 GB,
"A download of this size can take an hour or more." The notice says the entry leaves only when the download is
complete and checked, that a cancel or a failure changes nothing, and for the Trash that the space is free only after
the Trash is emptied; an entry Local models lists (a folder, or a link to one) adds "Local models lists this folder as
a model. Make sure that you do not need it."; a short disk adds the disk line. The toasts
after the job: "The link <slug> was removed. Its folder was not changed.", "The old folder
<slug> is in the Trash.", or "The old <slug> could not go to the Trash. It is in <keptDir>. Nothing was deleted."; for
a link that could not be removed, "The link <slug> could not be removed. It is in <keptDir>." A link to a file gets no
**Stays** row and no "Its folder" sentence (the blocker's `linkToFile`). The page keeps the blocker the user confirmed
on the job, so the job line and the toast name it after the catalog has reloaded without it. The old files of a link
keep their disk space; this version always downloads the whole model and reuses no old file.

The request echoes what the user confirmed: `replaceBlocker: { kind, fileCount, sizeBytes, truncated, linkTarget }`.
The server refuses (409) a folder without write permission, and an entry that no longer matches (kind, link text, and
the numbers unless the walk was truncated), and keeps the entry's `dev:ino` in the job. The identity is taken when
the request arrives. Between the catalog read that drew the card and the click, only the shown numbers are compared,
and a folder too large to list (`truncated`) is matched by kind only; names are not compared. So the entry that goes
to the Trash is the one at `<slug>` when the request arrives, whatever the card described.

A cleanup job never takes the paths of an app install: at the job start, in place of the "already installed at this
revision" check, and at the publish, whenever `<slug>` is occupied the entry must still have the kept `dev:ino`, the
same `lstat` kind and, for a link, the same link text. Nothing is read through a link to decide this. The same entry
goes to the Trash whole even when it gained the app's record during the job (for example from `mlx download`), and a
file added to it during the download goes with it. Any other entry, with the record or without, is refused with
"changed on disk" and nothing is removed.

The swap makes a holder `.staging/<slug>.cleanup-<pid>.<ino>.<uuid>/`, where `<ino>` is the inode of the staged
folder (a rename keeps it), renames `<slug>` into it, checks the moved entry's `dev:ino` (else it is put back), then
renames the new model in. When that rename fails, the old entry goes back and the error ends "Nothing was removed." A
way back puts a file back with `link` and then `unlink` of the holder name (when that `unlink` fails, the entry is
back and its second name stays in the holder), and a link with a new link of the same text, so neither replaces what another program put at `<slug>` meanwhile (EEXIST); then, as when the path is already
taken, the holder is renamed to `.staging/<slug>.kept-<uuid>` and the error names where the old entry is. A folder
goes back by `rename`, which fails over a file or a folder with files in it; the one gap is an empty folder that
another program makes at `<slug>` in the moment of the swap (about 0.1 ms between the check and the rename): the
rename replaces it. The same holds for the rename of the new model into place. Nothing else can be replaced. From
there the model is installed, and what follows is reported, never undone. The holder is removed with `rmdir`. The app
never calls `rm` on an entry it did not create.

When the process dies with a holder in place, recovery runs when the download manager starts (a holder with the
manager's own pid counts as dead then, since no job of this process exists yet) and before every job for the slug,
for a holder whose pid is dead. It removes an empty holder; puts the entry back as above when `<slug>` is free (when
something took the path meanwhile, the holder goes to `.kept-`); finishes the job as above only when the folder at
`<slug>` is the one this cleanup renamed in (the holder's device and the inode in its name); keeps the old entry in
`.kept-`, never removed, when another complete install of the app is at `<slug>`; and leaves everything as it is in
any other case. A recovery that keeps an entry writes its path to the core log. Nothing removes a `.kept-` folder.

A refused **Update…** plan whose reason a cleanup solves answers 409 with `reason: 'needs-cleanup'` beside the
server's sentence (a file of another size, a model file that is not in the catalog set, a name that differs only in
letter case, an unreadable record, more than 2,000 entries, a link or special file inside, a file with no checksum, a
file that failed the hash in an earlier Update, and a disk that cannot link files: the plan links a zero-byte file in
`.staging` to a second name first; when the app cannot write that file the plan is refused with no reason). The card then reads **Needs cleanup** and the cleanup confirmation opens at once
at the same button, led by "Update cannot use <slug> as it is." and the sentence; it trashes the folder whole and
downloads the whole model. The page keeps the refusal for the visit, by the blocker's kind, count and size, so a
folder the user fixed by hand gets **Update…** again; after a reload the card reads Needs update until one click.
Other refusals (another job holds the folder, it changed on disk, the file list could not be read, held by another
build) carry no reason and stay a "Could not check the folder" toast. A download of the slug already in the queue
answers "A download of this model is already in the queue. Cancel it, then use Clean up & update."

**Update…** makes the app own a folder that holds the catalog model but was not made by the app, by adding what is
missing. It never moves, renames, deletes or overwrites a file that was in the folder. A click asks for a plan (`POST
/api/downloads/adopt-plan`): the server pins one 40-hex sha for the repo and one for the `assetsRepo`, lists the wanted
files the same way an install does (the wanted set must hold `config.json`, except for a build that installs the GGUF
alone), and scans the folder with `lstat` only, opening no file, at most 2,000 entries. A wanted file is **checked**
when a regular file of the manifest size is at its path, and **downloaded** when nothing is there; any other file that
is not a model file stays where it is. The plan is refused (409 with `reason: 'needs-cleanup'`; the text names the file,
and the card turns into the cleanup confirmation above) for a wanted file of another size, a wanted file the manifest
gives no checksum for (in the folder or missing from it), a file that failed the hash in an earlier Update since the app
started, a model file that is not part of the catalog model, a name that differs from a wanted name only in letter case,
links or special files, a `.mlx-download-complete.json` at the root (the app cannot read it), more than 2,000 entries,
or a disk that cannot link files; no write permission is refused with no reason (the card shows No access first). The
plan, and the commit scan, read a `.mlx-download-complete.json` at the root on the transport thread with no size limit,
as the catalog, a cleanup and every install already do: a file of hundreds of MB there stalls download progress and
Cancel for as long as the read takes (seconds on a slow disk), and then the folder is refused. The plan reads Hugging
Face with a deadline and outside any job, so it neither adds bytes to a running download nor dies with its cancel. Its
token is a sha256 over the repo, both shas, the wanted files with their checksums, the checked files with size,
`mtimeNs` and inode, the downloaded paths and the folder's `dev:ino`; other files are not in it, so a new `.DS_Store`
does not break the confirm. The confirm shows the counts, the sizes and up to 3 other names. The start request carries
the token, and a token starts one job only: a token that is not the last plan's gets 409, and so does a used one once
its job has ended. While the Update job is active, an Update request for the folder returns that job's id. The job
computes the plan again before it does anything and stops with "Nothing was changed." when the token differs.

The job hashes each checked file where it is. It opens the file with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, the `fstat`
of the descriptor must match the plan, the sha256 must equal the manifest's `lfs.oid` (or the git blob sha1 its
`oid`), and a second `fstat` must give the same `dev`, inode, size, `mtimeNs` and `ctimeNs`. Size or name alone never
proves a file. The progress frames of this step carry `phase: 'check'` and the bar reads "Checking <file>". A file
that does not match fails the job after the loop, which names every such file and changes nothing; the user moves the
file out, and the next Update downloads the catalog file in its place. The missing files are downloaded and verified
into the job's own `.staging/<slug>@<sha>.<pid>.<uuid>` tree. At the commit, the last point where Cancel works, the
folder is scanned again and must be the same folder, still not owned, with the same fingerprints and the same missing
paths. The marker is written and fsynced in staging. Then each file is added with `link()`, `config.json` last: link
fails with `EEXIST` when anything is at the path, so an add cannot overwrite a file, and no `rename` or `copyFile`
targets the folder. The marker is linked last, and only when every wanted file is in the folder as a hashed or
downloaded file. Just before the marker link every checked file is read again with `lstat` (same `dev`, inode, size,
`mtimeNs` and `ctimeNs`) and the folder must keep its `dev:ino`; a change during the adds fails the job. A change in
the microseconds after that check, after the marker, or through a descriptor or `mmap` opened before the job is not
seen, as for any installed folder. A failed add is undone newest first, and only while `lstat` shows the job's own
staged inode at the name. The name is not unlinked in place: it is renamed into `.staging/<slug>.kept-<uuid>` and
checked again there, so a file another process renamed onto the name in between is put back with `link()`, or stays
in that folder when something took its name again, and the error names the folder. Folders the job made go with a
plain `rmdir`, and the error names any added file that stays. When the added `config.json` (or the marker) cannot be
removed, the undo stops there and every added file not yet removed stays beside it. A volume that
cannot hard-link (exFAT) stops the job with nothing changed. There is no recovery step. A crash before the first add
leaves the folder as it was, and the dead staging tree is reaped by its pid. A crash during the adds leaves complete
added files and no marker, so the folder still reads as not owned, and the next Update checks those files by hash and
finishes. That holds until `config.json`, the last file, is in: with `config.json` and no marker (a crash between the
two links, or a rollback that could not remove it) the folder reads as present, the card shows Installed and no Update,
and the app does not manage it. The other added files are then all there, so the folder is never Installed without
its tokenizer. A failed rollback says so and tells the user to remove `config.json`; with it gone the card shows
Update again.

A marker that Update writes has an `adopted` record: the `lstat` size and `mtimeNs` of each wanted file as the job
proved it. It decides what the app may delete later. When an update replaces such a folder, or backup
recovery removes its old copy, only regular files that the marker lists and whose size and `mtimeNs` still equal the
record are unlinked, and the folders left empty are removed with `rmdir`. Each such file is first renamed to a private
folder inside the old copy and checked there (the same inode, size and `mtimeNs`), so a file renamed onto its name in
between is put back and kept; a writer that already holds the file open can still change it after that check. Everything else, the marker included, is
moved to `.staging/<slug>.kept-<uuid>`, which nothing removes; the `done` event carries that path as `keptDir` and the
toast names it. When a sidecar refresh would delete an old sidecar from such a folder, it downloads the model again
and swaps instead. A record in any other shape reads as empty and proves nothing, so nothing is deleted. An old copy
whose marker cannot be read is kept whole in `.staging/<slug>.kept-<uuid>`. The `mlx` CLI keeps the record when it
rewrites the marker; a file it downloaded again no longer matches the record, so it is kept, not deleted.

Delete in Local models removes a loadable folder for good (a link row loses only its link); a catalog card never does:
Clean up & update only unlinks a link or moves an entry to the Trash, and a held folder (another build, a working
model) gets only Show in Local models. Update and Delete can reach the same folder; a Delete during an Update makes the Update stop.

### Locate

**Locate…** sits beside Install on the card of a build that has nothing in its folder (and in the Onboarding dock under
**Download model**). It uses model files that are already on this Mac, for example in `~/.lmstudio/models` or a
Hugging Face cache, instead of downloading them. It is not on the DFlash2 line, on hidden entries, on a build that is on
disk, has a job or has something in its way, and not in the command palette.

The user picks a folder in the macOS dialog ("Locate model files", button **Locate**, hidden folders shown): the folder
that holds the model, or the folder one level above it. The page sends only a request id and the word `model`; MAIN
holds the dialog's options, one dialog at a time for Work and Locate together. A cancelled dialog does nothing. While
the server checks the folder (`POST /api/downloads/locate-plan` with `{ slug, path }`, which writes nothing) the button
reads **Checking…**. A refusal is one line in the card's build row, in the server's words; the line opens the whole
sentence. Otherwise a confirmation, **Use these files?**, says how many files of which size the app found and where,
what it checks, how the files are used and what it downloads, and that nothing in the user's folder is changed, moved
or deleted. **Use files** starts the job (`POST /api/downloads` with `{ slug, locate: token }`); it waits in the same
queue as downloads, shows the usual progress ("Checking <file>" while it reads a file) and Cancel, and ends with the
toast "Model located". The token follows the Update rules, except that while the Locate job is active only its own
token returns its id; any other token gets "A download of this model is already in the queue. Cancel it, then use
Locate."

What the app checks: the folder (or one of its direct child folders) must hold every weight file of the chosen build
under its catalog name and size. A folder that holds another build of the same model is named ("This folder holds
UD-Q4_K_M. Choose UD-Q4_K_M on the card, then use Locate again."). The job then reads each weight file once and
compares its sha256 with the catalog's; a file of the right size with other bytes stops the job with nothing changed.
A folder that has this app's own install record of the build is trusted: the job checks that the listed files are
there and reads none of them, and needs no network. The record is trusted for content too: a file replaced under a
listed name after the record was written is used as it is, as the recorded build at the recorded revision, and is not
read. Use Install, or remove the record from that folder, to have the files checked. The models folder on a missing
disk is refused with the same sentence as Install. A snapshot of the app's own download cache is refused when a file
must be downloaded, because the download would create that file in the snapshot ("This folder is part of the app's
download cache, and "<name>" is missing or different there. Use Install."). The check compares real paths: the cache
paths a download writes (the file in its snapshot and the repo's `blobs` folder), with every link in their existing
part followed, must be outside the folder that holds the files, so a cache repo folder that is a link into that
folder is refused too. When that folder holds the whole download cache, the sentence names the cache.

The app makes its own folder `<models>/<slug>`, built in `.staging` and renamed into place, in one of two modes:

- **Same disk** (`shared`): each weight file gets a second name in the app's folder. The file is stored once and
  takes no extra space; its content, name and modified time stay as they are.
- **Another disk** (`linked`): the app's folder links to each weight file. The model works only while that disk is
  connected. The engine reads the small files beside the real weights, so the user's folder must hold them under the
  catalog names, with the same bytes; a Hugging Face cache, whose files have other names, can be used only from the
  disk of the models folder.

A link uses the file by its path: a file replaced or rewritten there later is used as it is and is not read again, and
the card keeps Installed at the recorded revision. In shared mode the same holds for a file rewritten in place; a file
that another program replaces under its name keeps its old bytes in the app's folder.

The small files (configs, tokenizer) are always the app's own copies, made with an exclusive create; missing files are
downloaded into the app's folder. The marker carries `located: { from, mode }`. The app never creates, renames,
deletes or writes a file in the user's folder. Inside the located folder it writes only in the models folder, when the
located folder contains it (model files at the root of the disk that holds the models folder): `<slug>` and `.staging` are made
inside the models folder, and nothing else in the located folder is created. The search never takes anything from the
models folder: it skips the models folder as a child folder, and a file that a link (or a catalog path into a child
folder) leads to inside it refuses the plan, with the text that "<name>" here is a file in the models folder and
Locate uses only files outside it.

After Locate the card reads **Installed** with "Uses files in <folder>" on its build row. A linked model whose disk is
away reads **Files not found** (the line opens "<folder> is not available. Connect the disk, or locate the files
again.") with **Check again**, **Locate…** and **Download instead**; nothing is removed until the user acts, and the
engine does not list a folder whose weight links are dead (nor a split set or safetensors checkpoint with one part
whose link is dead). Local models shows a **Linked** badge (or **Not connected**)
and the drawer a **Shares files with** or **Linked to** row.

Delete of a located model removes only the app's folder (its names and links). When every weight file in it is a link
or still has another name, the confirmation is **Remove model?**: "This removes <name> from the app. Its
model files stay on disk; you located them in <folder>." When the app's folder holds the last name of a weight file
(the user deleted the originals, or `mlx download` put its own copy in place of a link), the normal **Delete model?**
text is shown, because the bytes do go. **Remove** sends `{ keepsFiles: true }` with the delete, and the server checks
again just before it removes: when the folder now holds the last name of a weight file it answers 409 and removes
nothing, and the page reads the list again and shows the **Delete model?** text, which the user confirms anew. **Update
available** on a located model asks first ("Download the new version?"): the update downloads the whole build into the
models folder, the old app folder is removed by the usual swap, and the files in the user's folder stay and are no
longer used. **Download instead** does the same without a question. `mlx download` into a located folder replaces a
file by removing its name first, so it never writes through to the user's file ([cli.md](cli.md)).

Not supported: copying or moving weights into the models folder, an older version of a file or a renamed file (only
the current catalog file passes, unless the folder has this app's install record), picking a single file, a search
deeper than the direct child folders, models that are not in the catalog, and updating a located model in place.

## Chat and Work

`/chat` is one page with two modes, switched by the segmented control in its bar (`⌘⇧M`). Chat runs the bundled
`mlx-agent` with the app's `web_search` / `web_fetch` tools and no shell; Work runs it in a folder the user chooses.
The page is `ui/src/pages/chat.tsx` plus `ui/src/pages/chat/`.

- Workflows: Work only. The Chat agent starts with `GROK_WORKFLOWS=0` (the environment beats any config file), so a
  Chat prompt lists no workflow and the model has no `workflow` tool to call.
- Places: `/chat?c=<id>` is a conversation, `/chat?mode=chat|work` the New screen of a mode. A send from the New screen
  creates the conversation and pushes `?c=<id>`. The last mode and the model picked per mode are kept in
  `localStorage`; drafts stay in memory only, one per place, and are never written to storage.
- Streams: the Chat host stream (`use-chat-host.ts`, one per window, shared with the sidebar badge and the palette)
  stays open while the window is hidden; a conversation stream (`use-conversation.ts`) closes while it is hidden and
  opens again when it shows. Both open again after a reconnect, and a `seq` gap asks for a fresh `sync`. Nothing
  polls.
- Send is off, first match wins: reconnecting, agent missing / installing / install failed, agent too old, the mode
  blocked (engine not reachable, agent build does not match, damaged skill files), the agent keeps stopping, engine
  stopped, no models, Work without a folder, text that starts with `/`. On the New screen such a state replaces the
  hero with a panel and its action; inside a conversation it is a strip over the composer. A send the server refuses
  puts the text back and shows the strip for its `reason` (`blockers.ts`).
- Keys: the tooltip of the round button names its key (`Send ↵`, `Stop Esc`). The empty text field shows `⇧↵ New line`
  at the end of its first line (icons with hidden key names; not drawn under 520 px of composer).
- Tools row: a round `+` first, then the model picker and the chips, then Send / Stop in a column of its own that never
  shrinks. The `+` never shrinks. As the row gets short, first the folder label shortens, then (Work) the quantization
  badge goes, then the model name, each label down to a floor (a label of 8 characters or fewer never shortens). A
  badge that does not fit whole is not drawn and its room goes back to the folder label (`data-fit-whole`, composer);
  under 520 px of composer the chips are icons, under 320 px they wrap. The Bypass word and the Send button never give
  way. The reason Send is off keeps up to two lines beside the chips.
- Tools menu (`+`; `arc-x/menu`, `pages/chat/composer-tools.tsx`): opens upward, `role="menu"`. It holds only what
  exists: three rows. The first is Add files or photos (paperclip, `⌘U`, below). The second is Web search, whose right
  side says the provider, `Off`, `Key rejected` or `No credit`. The third is Crawler (below). The Web search row
  opens a panel beside it (pointer rest, click, Enter or the right arrow; the left arrow or Esc closes it) that picks
  the search provider:
  a list (`role="menu"`, the menu's keys) of Exa, DeepSeek, Brave and Off as `menuitemradio` rows with a check mark on
  the one the host summary names, then the row Web Search settings, whose note names the providers with no key ("Add
  a key for DeepSeek or Brave.", else "Keys and page reading"); it opens the Web Search page and closes both. A row
  with no key is `aria-disabled`, says `No key`, and its name says where to add it ("DeepSeek, no key. Add it in Web
  Search settings."). A rejected or no-credit key keeps its dot and words and can still be picked. The keys come from
  `GET /api/providers`, read in Chat and Work each time the menu opens and again when the host summary's `search`,
  `searchCheck`, `firecrawl` or `fetch` changes while it is open; the Web search and Crawler panels
  share that one read. Until the first read answers, or when it fails, only the current row and Off can be picked;
  while the Keychain does not answer, a provider whose key is not known cannot be picked. A pick sends
  `PUT /api/providers/selection` with `{search}` only (never `firecrawl`); the row says
  `Switching…` and every other row waits until the first host state after the reply, or until the reply fails. The
  menu stays open. One caption above the list says, in this order: why a pick failed (red; it clears on the next pick
  and when the menu closes, and the keys are read again), a key gone at app start (`vanished`, amber; a toast when the
  menu closes before the panel opened), the Keychain (amber), `Reading keys…` / `Could not read the keys.`, a change
  that waits for running answers. A pick that fails after the menu closed is a toast. In Work a grey line under the
  list counts the search skills Work gets. The composer shows nothing about web search while it works or is off.
  A rejected search key or a search account with no credit puts a 6 px dot on the `+` (danger / warning) and the
  button's name and tooltip say it (`Tools, web search key rejected`); the dot stays search only, crawler problems
  show on the Crawler row. Before the first host state the `+` is disabled.
- Crawler row (FileText icon): picks who reads pages, Firecrawl or Off. Its right side says `Off`, `Firecrawl`, or a
  problem with its dot: `Key rejected` (danger), `No credit` (warning). On or off come from the host summary
  (`firecrawl`); the problem words need the `/api/providers` body, so they show from the menu's first read on. The
  panel is built like the Web search one: a caption, the list "Reads pages" with the rows Firecrawl and Off (always
  these two, in this order), a separator, then Web Search settings. Firecrawl with no key is `aria-disabled`, says
  `No key`, and is named "Firecrawl, no key. Add it in Web Search settings."; while its key is not known (no body yet,
  or a Keychain that does not answer) it is disabled unless it is the current one; a rejected key or no credit keeps
  its dot and words and can still be picked. The Off row's second line says who reads pages then: "Exa reads pages
  for Chat." (or DeepSeek), "Brave gives Chat page excerpts only." or "Chat cannot read pages." (no search provider).
  The settings row's note is "Add a Firecrawl key." with no key, else "Keys and page reading". A pick sends
  `PUT /api/providers/selection` with `{firecrawl: true}` or `{firecrawl: false}`, never `search`; the row says
  `Switching…` and the other row waits until the host summary shows the pick (the HTTP reply may still wait for the
  Work `PATH`), or until the reply fails. The check mark follows the host summary only. The caption says, in this
  order: why a pick failed (red; "Add a Firecrawl key in Web Search settings.", the damaged-skills sentence, or "Could
  not switch the crawler. Try again."), the Keychain, `Reading keys…` / `Could not read the keys.` (before the first
  body), a change that waits for running answers: "Conversations that are open keep the old crawler until every
  running answer ends." (a loaded conversation keeps the `mlx_web` it was given, and a stale Work host its
  environment, until the host restarts). In Work a grey line under the list says what Work gets: the Firecrawl skills
  with the key (the agent's commands, MCP servers and hooks can read it), no skills until the `firecrawl` CLI is
  installed, skills not installed, or no skills (Off).
- Files and photos: the row Add files or photos, `⌘U`, paste and drop on the composer all take files the same way
  (`ui/src/lib/chat-attachments.ts`). Small UTF-8 text files work in Chat and Work; PNG, JPEG and WebP pictures work
  in Chat when the model row of `ChatHostState.models` has `images: true`. The page reads the bytes of what was picked,
  pasted or dropped; it never learns a path. The text file goes to the agent as one ACP `resource` block
  (`file:///attached/<name>`, `text/plain`), a picture as one `image` block (no `uri`), both after the one text block,
  and on the first `session/prompt` of a turn only (a Work continue sends text only). No block carries `_meta`. The
  agent shows a file to the model as `<file_contents path="/attached/<name>">` with numbered lines.
  - `⌘U` works from the text field, with the `+` menu open (the menu closes) and while a tooltip shows; not while a
    dialog is open (Rename, Permission, Confirm, Settings, the palette), not while the `+` is disabled, and not with 4
    files held. The picker is one hidden `<input type="file" multiple>` with no `accept`; after it closes or is
    cancelled the focus goes back to the text field. Paste takes files only when the clipboard has no plain text.
  - Row note: Chat and a model that does not see images: "Text files only. This model cannot see images."; Work: "Text
    files only in Work"; 4 files held: the row is disabled with "Up to 4 files per message".
  - The content decides, not the type the browser gives: only `image/png`, `image/jpeg` (also `image/jpg`) and
    `image/webp` take the picture path and `application/pdf` is refused; every other file (`.ts` labelled
    `video/mp2t`, `.svg`, HEIC, a zip) gets the text path. A text file is first looked at in its first 8,192 bytes, so a
    40 MB zip says "not a text file", not "too long". A picture is always decoded and drawn again in the page
    (`createImageBitmap` with the photo's EXIF turn applied), scaled to fit 2000 px a side and 2,400,000 px (the 8 px
    and 512 px floor is checked on that drawn size, as the agent checks it: a 4000 x 15 strip is refused), and leaves
    as PNG, or as JPEG (quality 0.9, 0.8, 0.7, then 0.5) for a JPEG source or a PNG over 1 MB. Its own bytes, EXIF, GPS
    and camera data are never sent. The route checks a picture's type, base64, size and first bytes only: the redraw
    (upright pixels, no EXIF or GPS, the pixel limits) happens in the page, the route's only caller. The agent decodes
    every picture in full: one it cannot decode is dropped with a note to the model and the turn answers from the text;
    one over its pixel limits is made smaller before the engine sees it.
  - Strip (`arc-x/attachment-strip`): between the notice strip and the text field, one chip per file (thumbnail for a
    picture, name, size, remove), one status line with the first refusal of the last add ("{n} more not added" for the
    rest), and the caption "About N tokens". The caption turns to a warning, "The context window is nearly full.", when
    the conversation's context plus the estimate passes 80 % of the model's window. A read that ends after the user
    moved to another place lands in the place it started in, and its refusal shows as a toast; one that ends after its
    conversation was deleted adds nothing. One queue serves the whole window (`attachDrafts`, `pages/chat/drafts.ts`),
    so a read started before Chat was closed ends before the next Chat page's read starts. A read adds only what still
    fits under 4 when it ends and refuses the rest ("Up to 4 files per message"), and the composer on screen follows
    the store, whichever page wrote it. With files and no text
    the field says "Add a message for these files": a message still needs text.
  - Send is off, first match, after the reasons above: "Reading files…"; a picture in Work; a picture and a model that
    does not see images (the picture chips say why; nothing is removed); a picture when the conversation's stored
    pictures and the new ones pass 40 MiB of base64 (below); text plus files over 90,000 bytes; the estimate over 80 %
    of a known window. A window the host state does not know yet blocks nothing. `send()` runs the
    same checks, so Retry and Send again cannot pass them.
  - Retry and Send again resend the files while this window holds them. A failed send keeps one `failed` record per
    place (conversation or the New screen; text, files and the failure's strip, in `pages/chat/drafts.ts`). The message
    is put back into the composer only when it has no text, no files and no file still being read; otherwise the newer
    draft stays as it is and the strip keeps Retry. The record outlives a move to another place and closing Chat, so
    the strip and Retry show again on return. A failure shown as a toast (a busy conversation) keeps one too when the
    message could not be put back. Acting on the strip (Pick another, Choose folder…) takes it away: a message the
    composer holds needs no record, one it does not keeps a Retry strip. The record goes on Retry, when a send from
    its place goes through, when the user types into a composer the message was put back into, and when the place is
    forgotten (a deleted conversation).
    `lastSent` is one record per window. After a reload, or once another message was sent, the thread's button reads
    "Send again without files" and sends the text only.
  - Thread: the user message shows read-only chips, live and after a reload, in Chat and in the Sessions transcript,
    whose find also matches file names. A file chip has its name and size, a picture chip "Image", its type and size
    (the agent stores no picture name). No thumbnail in the thread: it is rebuilt from the agent's file and a picture
    would send up to 1.3 MB on every read. A Chat conversation that holds pictures on a model that does not see
    images shows one warning line: "This model cannot see images. The images in this conversation are left out of its
    answers."

  | Limit                                       | Value                                                                                     | Checked by                                                            |
  | ------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
  | attachments per message                     | 4                                                                                         | page (one queue per window, held again when a read ends); route       |
  | first look at a file that is not a picture  | 8,192 bytes; a NUL or bad UTF-8 = not a text file                                         | page                                                                  |
  | text file                                   | 20,000 bytes with line numbers (`N→line`, as the agent renders it; about 5,000 tokens)    | page; route                                                           |
  | message text plus all text files            | 90,000 bytes UTF-8 (the agent cuts a prompt over 100,000 bytes)                           | page; route                                                           |
  | file name                                   | 255 characters in; stored as the base name, controls and `#"<>/\` as `_`, cut to 120      | route                                                                 |
  | picture file picked                         | 20 MiB, PNG / JPEG / WebP                                                                 | page, before the decode                                               |
  | picture sent                                | PNG or JPEG, ≤ 1,000,000 bytes, ≤ 2000 px a side, ≤ 2,400,000 px, ≥ 8 px a side, ≥ 512 px | page; route checks length, base64, bytes and the first bytes          |
  | pictures                                    | Chat only, model row `images: true`                                                       | page; service (`attachment-refused`); engine (400, below)             |
  | `/v1/messages`, `/v1/messages/count_tokens` | 16 MiB body (other engine routes 10 MiB), listed as `max_request_bytes` in `/v1/models`   | engine router; the agent drops old pictures from 13 MiB down to 8 MiB |
  | pictures one chat keeps                     | 40 MiB of stored base64 (about 31 full-size pictures), so text and tools keep 24 MiB      | page; service (`pictures-full`): "Start a new chat to send more."     |
  | the session file the thread is read from    | 64 MiB; past it the thread shows only "Some messages could not be read"                   | reader                                                                |

  Token estimate (page only): bytes of the message and the numbered files ÷ 3, plus 2,800 per picture.

  Model gate: `images` on a host state model row is the loaded model's own `supportsImages()` while it is in memory
  (the engine's live file, per model `images`), else discovery's answer (the endpoint file). Discovery gives a
  `qwen3_5` GGUF image support only with a sibling `mmproj-*.gguf` (`config.json` does not count), a `gemma4` GGUF only
  with both the vision marker of its `config.json` and a sibling `mmproj-*.gguf` (its load looks for a projector only
  then), and a `qwen3_5_moe` GGUF none (its load takes no mmproj); MLX checkpoints and Clairic keep their rule.
  Discovery reads only the file name of the projector: one of the wrong kind or width (Qwen `projection_dim`, `gemma4v`
  against `gemma4uv`) or a Prism-Hadamard Qwen file still reads as images until the first load. That cold turn
  answers without the picture, then the loaded model's own answer takes over; after each unload the guess comes back.
  A request with an image for a model whose `supportsImages()` is not true gets 400 `Could not process image: the
  model "<id>" does not take images.` before any inference; the agent then retries the request without its images, so
  the turn still answers. It leaves the images out of that retry only and keeps them in its history (measured with
  `mlx-agent-v0.3.3`, its log says `payload_heuristic`), so every later turn of such a chat costs one refused request
  and one retry, with one picture or more. A picture sent while a model change is still in flight can reach a model
  that cannot see: the engine refuses it, the turn answers without it, and the warning line shows.

  Stored: drafts (text and files) live in this window's memory only, one per place; the New screen's place is shared
  by Chat and Work, so its files are too. A queued message (its text, file text and picture bytes) is kept in
  `composer/<id>.json` until it is sent or the conversation is deleted. The agent writes each sent block to its session folder: the
  file text and the picture's base64 in `updates.jsonl`, the picture again in `chat_history.jsonl` and as
  `assets/image-<uuid>.<ext>`. Only the redrawn picture is stored. `updates.jsonl` keeps every sent block, so the chip
  stays when the agent's retry left the picture out. The reader turns stored `resource` and `image` user blocks into
  `attachments` on the user entry (name from the `uri`, sizes from the stored text or base64); no file text and no
  pixels cross to the page. Titles and the conversation list use the message text only. The Chat address list for
  `web_fetch` is built from the typed text only, so an address inside a file is not opened.

- Context hint: an 18 px ring just before Send / Stop, outside the box the chips shrink in, with no text. The arc
  (`--info` blue; warning from 80 %, danger from 95 %; a share above 0 shows at least a short arc) is the conversation's
  context over its model's window. The numbers are only in its tooltip and name (`Context: 11,971 of 262,144 tokens
  (5%)`); it takes keyboard focus so the tooltip opens without a mouse. Used: the agent's `params._meta.totalTokens`
  on each `session/update` (its count of the current context, not a usage sum; a line without it keeps the last
  value). The transcript reducer keeps it, a `context` op carries it live, `sync` carries it (`contextUsed`), and a
  stored conversation gets it back from the same `updates.jsonl` lines its thread is rebuilt from (the session
  detail's `contextUsed`), so it survives a reload and an app restart. Window: the model's `contextTokens` from the
  host state: the window an agent process last listed for it, kept in `model-windows.json`, so it is known while no
  agent runs (after the 10 min idle stop, after a restart). Not drawn while either is unknown, beside a reason or the
  counter, or under 320 px of composer. The ring is the button of the Context panel (next bullet).
- Context panel (Chat and Work): the ring (a `button`, `aria-haspopup="dialog"`, `aria-expanded`; tooltip kept), the
  palette's "Show context and usage" and `historyActions.showContext` open it as a popover (`role="dialog"`, name
  "Context") upward from the ring, right edge on the ring; while the ring is not drawn it hangs from the end of the
  tools row. Esc closes it and gives focus back to the ring (to the message box when the ring is hidden). Top: "158K of
  256K tokens · 62%" (the ring's own numbers), a bar and a list of the agent's parts ("System prompt", "Tools (42)",
  "Messages", "Free"), the line "Compacts by itself at 85% · compacted once" with a dashed mark on the bar (under the
  list instead when the bar cannot mark it: no window, or 0 or 100%), and a "Compact now" button (the `⋯` row's action
  and off reasons; off, it stays focusable and reads its reason). The count of compactions is the agent's own: it grows
  as a run starts, so a stopped or failed Compact now counts too. The parts are the agent's estimates (`messageTokens`
  is bytes / 4), so they are scaled to the context's use (the three add up to it exactly) and the bar and the list show
  the same numbers; the bar is `role="img"` whose name holds every part, and the list is a `<dl>`, so nothing depends
  on color. They come from `GET /api/chat/conversations/:id/context`: `ChatService.context` asks
  `AgentHost.contextInfo`, which sends `_x.ai/session/info` (mlx-agent v0.3.4 `agent/handlers/session.rs`,
  `session/acp_types.rs` `ContextInfo`) only for a session loaded and guarded in the running process, under a soft 5 s
  deadline (a miss shows no parts and keeps the agent). A cold conversation never starts or loads the agent for the
  panel: the route answers `{used, parts: null}` (the stored use) and the panel shows a plain meter of the use and
  "Send a message to see the parts."; a failed read looks the same, with no error text. Three placeholder rows show
  while the first read runs; the panel reads again after each answer's end and each Compact now end while it stays
  open. No window known: no bar, no "Free", "158K tokens". Bottom, "This conversation": Turns, Model calls, Read
  ("2.1M tokens (1.8M from cache)"), Written, Turn time, summed by `ui/src/lib/session-usage.ts` from the stored turn
  metrics (`turn_completed.usage` on disk), never from `x.ai/session/usage`, whose totals reset with the agent process.
  A turn with no usage counts as a turn and adds no call and no tokens; rewound turns are not counted. No cost row:
  "Runs on this Mac. No charge.", then "Inspect" to the Sessions page.
- Model: before each message goes to the agent, the app selects the conversation's model on the engine
  (`POST /v1/messages/count_tokens` over `node:http` to 127.0.0.1 only, the token in the `x-api-key` header, a 30 min
  ceiling; `src/chat/agent/select-model.ts`). Only that model loads. While the engine loads it, the thread shows the
  load card; Stop then ends the turn with "The message was not sent." and the engine continues the load. While the
  engine prepares GPU kernels for that model (load progress `phase: 'kernels'` with `steps` known), the card's title
  is "Preparing GPU kernels", its head counts "n of m" steps, a bar fills by steps and the note says "The first load
  after an update prepares the GPU. It can take a couple of minutes."; the weight read, or an engine that reports no
  steps, keeps the plain card. The first load of a model that moves its stored chats to 32-token pages
  ([paged-cache.md](paged-cache.md#32-token-pages-with-token-major-keys-qwen3_5-draft-loads)) also does that move
  inside the load; the card shows no step for it, so it reads as a longer plain load. Loads can overlap: the card shows the load of the conversation's own model, else the
  oldest one (`loadFor`, `ChatHostState.loading` is a list, oldest first). A model change
  in an open conversation loads nothing until the next message. A model in memory is selected at once, also while
  answers run on it (the engine skips its load gate for a resident name). If the select still waits, the card says the
  model is in memory and does not say it loads. When the turn has no output yet and the live file shows that the
  model runs as many answers as its limit while a request waits, and every running answer started before the turn (so
  none is its own request; `ChatHostState.answers`), the thread says "Waiting for a running answer".
- Wait line: while a turn runs and nothing moves on screen, the end of the thread shows one status line beside a 20 px
  dotted orb (`arc-x/wait-line`, `arc-x/wait-orb`; the orb is `aria-hidden`), with a clock (`m:ss`) that counts from
  the start of this wait: the turn's start before its first entry, else the newest time of the turn's entries (the end
  of the last step). It shows before the first entry and again between any two steps of a long turn. It hides while a
  drawn tool row runs (`in_progress` or `pending`), while a permission or plan card is open, and while text or a
  thought got a stream event in the last 3 s. While it shows, that last thought or text is drawn finished ("Thought",
  no live mark), so the turn has one live indicator. Its shape says which wait: a slow ring while the turn waits (starting,
  stopping, the engine, a queue, the busy model, no event for 3 s), a waving band while the model works for this
  conversation. The load card has the orb in its head: a pulsing sphere while the engine loads this model, the slow
  ring otherwise. What it says, first match (`pendingRow`, `engineWait` in `ui/src/lib/chat-turns.ts`):

  | Engine state                                                       | Line                                                                      |
  | ------------------------------------------------------------------ | ------------------------------------------------------------------------- |
  | the turn is created / stopping                                     | "Starting…" / "Stopping…"                                                 |
  | this conversation's request is `queued` or `slot`                  | "Waiting for the model. Another answer runs first." (full: below)         |
  | this conversation's request is in `prepare` or `prefill`           | "Reading the conversation · 69.3K of 121.3K tokens" and a bar             |
  | this conversation's request is in `decode`                         | "Thinking…"                                                               |
  | the model loads, the engine starts, the turn starts                | the load card, "Starting the engine…", "Starting…"                        |
  | the model is full and every running answer started before the turn | "Waiting for a running answer. … runs one answer at a time."              |
  | the request may be this one's but the app cannot tell (see below)  | "The model is busy · reading 5,000 of 20K tokens" / "…busy…"              |
  | nothing known; after 3 s with no event                             | "Working…" ("Thinking…" before the first entry); "Waiting for the model…" |

  The link from a conversation to its engine request is exact: the agent sends `x-grok-session-id: <ACP session id>`
  on every model request; `/v1/messages` and `/v1/responses` keep it on the traced config (`clientSession`,
  `clientSessionOf` in `packages/server/src/turn-trace.ts`), the engine's live file lists it per turn
  (`turns[].clientSession`), and the host state lists the app engine's requests (`ChatHostState.requests`: session,
  model, phase, start, prefill counts, oldest first) on the host stream, pushed on each heartbeat. A request names this
  conversation when its session equals the conversation id. A request that named no session (an older engine, a
  terminal client) on the conversation's model that started in this wait, or requests in the server's own queue (the
  model's `queued`, which the live file does not list) while every listed request on the model is another's, give the
  model's words, never "the conversation". Requests in that server queue have no phase until they reach the model.

- Compaction: when the agent's context passes 85 % of the window it summarizes older messages; on a local model one
  summary reads the whole conversation and then writes for 10 to 25 minutes. The agent sends four xAI updates on
  `_x.ai/session_notification` (mlx-agent v0.3.3 `extensions/notification.rs`): `auto_compact_started`
  (`tokens_used`, `context_window`, `percentage`, `reason`), then `auto_compact_completed` (`tokens_before`,
  `tokens_after`, `elapsed_ms`), `auto_compact_failed` (`error`; text that starts "it'll retry on the next turn" when
  it tries again at its next step) or `auto_compact_cancelled` (Stop). A manual run (Compact now, below) sends
  `completed` alone; a typed `/compact` runs Compact now in the app and is refused as prompt text, so it never reaches
  the agent. For an
  automatic run the host keeps `{ phase: 'running', startedAt, tokensBefore }` per conversation
  (`AgentHost.compaction`) from `started` until an end update, the agent's `turn_completed`, the end of the turn or the
  agent's exit, and sends it on the conversation
  stream (`compaction` event, `sync.compaction`); it is not a second channel and not part of the turn, so a compaction
  with no typed turn (a model-family switch, a woken turn) shows too. While it runs the thread end shows ONE line in
  place of the wait line (never both): the `fold` orb (five dotted lines that squeeze into one bar, then the next page
  fades in; `arc-x/wait-orb/fold.ts`, a still frame under reduced motion), "Compacting the conversation", a quieter
  note from the conversation's newest engine request since the start ("Reading 113.2K of 172.2K tokens" with the bar,
  then "Writing the summary · 3,200 tokens" from the request's `generated`, nothing while it waits) and the clock from
  the start. Under the first compaction of a conversation one line says "Older messages are summarized to free space.
  This can take several minutes on a local model." Stop works as in any turn: `session/cancel` cancels the
  compaction. The agent writes the end updates to `updates.jsonl` (`_x.ai/session/update`), so the transcript reducer
  makes a `compaction` entry from them, live and from disk alike, and the marker survives a reload and an app restart:
  "Conversation compacted · 222.8K → 31.2K tokens · 12:04" (counts only when the agent sent both), "Compaction
  failed. The agent tries again." (or "Compaction failed." when it does not), "Compaction stopped." The Sessions
  transcript draws the same marker. The context ring drops through its usual path: the next update after the summary
  carries the smaller `totalTokens`.

- Compact now (Chat and Work): the thread bar's `⋯` ("Conversation actions") has "Fork" (below), then "Compact now" with the note
  "Summarize older messages" and the context share at its right; the palette has "Compact conversation". When the row
  is off it stays focusable and its note says why: the engine or agent blocker, "Wait for the answer to finish." while
  an answer, a wake or a compaction runs, "A workflow run is live. Stop it or wait.", and "Nothing to compact yet."
  before the first finished answer. Both call `historyActions.compact` (`ui/src/pages/chat/history-actions.ts`), the
  one way in for every history op, which posts `POST /api/chat/conversations/:id/compact`. The route answers 202 `{}`
  once the run started, since a page call fails after 30 s + 5 s and a local summary runs for minutes; the end comes on
  the conversation stream as the `compaction` event with `compaction: null` and `end: 'compacted' | 'stopped' |
  'failed' | 'load-slow'` (live only, never in a `sync`; `load-slow`: the cold load passed its soft deadline). With the engine stopped the route refuses with `inference-stopped` and the
  toast says the engine's blocker sentence. The host (`AgentHost.compact`) refuses with `conversation-busy` while
  `engineBusy` holds (a turn, a wake, a live workflow run or another history op), loads a cold conversation first
  (`ensureLoaded`, the load a turn uses, with a soft deadline: see Soft calls), then sends `_x.ai/compact_conversation {sessionId}` with no deadline
  (mlx-agent v0.3.4 `extensions/memory.rs`: `{}` after the `auto_compact_completed` update; an error carries
  `data.kind` `compact_failed` or `compact_cancelled`, which `AcpRpcError.kind` keeps). While it runs the session holds
  an `op`: `isBusy` and `isWorking` count it, so a prompt is refused and the idle stop waits. The conversation's
  `compaction` is `{phase: 'running'}` from the start and `'stopping'` after Stop, so the thread end shows the
  compaction line and the composer face reads Stop. Stop or Esc posts `cancel`: the host sends `session/cancel` (the
  agent cancels a compaction with no turn), or ends the run with nothing sent when the request has not gone out yet. A
  stop is a toast "Compaction stopped.", a failure an error toast "Compaction failed."; a done run leaves the marker and
  the live region says "Conversation compacted, 41K to 9,000 tokens." The row's press closes the menu and leaves focus
  in the composer: a menu row whose action moves focus elsewhere keeps it there (`Menu` does not pull it back to its
  trigger when the panel unmounts after its exit).
- Rewind to here (Chat: the conversation; Work: the conversation, the files, or both): a sent message has Copy and `⋯` ("Message actions") with one
  row, "Rewind to here…" with the note "Remove this message and what came after"; a steer, a note, the automatic
  `continue` and a wake have no `⋯`. When the row is off it stays focusable and its note says why (the same reasons as
  Compact now, with no "nothing yet" one). `/rewind` (`/undo`) and the palette's "Rewind to the last message…" open it
  on the last sent message ("Nothing to rewind yet." when none). The row posts `POST …/rewind/plan {entry, ts}` (the
  entry's index and its time as the page holds them; another entry there now is 409 `rewind-changed`, and a page sync
  that puts another entry where the dialog's message was turns the open dialog into "The conversation changed. Open
  Rewind again."): the service reads the conversation from disk and finds the agent's prompt index
  (`src/chat/rewind-target.ts`): a stored message by its own `promptIndex` when the disk holds that prompt with the
  same text, a message sent in this app run by order and text after the last stored one; a message whose turn ended
  before its prompt went to the agent (`unsent`, marked by `Conversation.promptUnsent`) takes no part; a Send now that
  ran as its own turn (its live `message` wake row, stored as a typed prompt) takes its stored prompt in that walk, so
  the message after it pairs with its own. The walk runs from the start and from the end; a message the two walks pair
  differently (the same text twice, one never stored and not marked) is refused like a message the agent never ran (409
  `rewind-not-point`, "This message is not a rewind point."). Fork maps an entry the same way. The plan answers what goes (`removes.messages`, `removes.steps`), the live commands
  and monitors and the schedules the removed part started (`background`), helpers and attachments, and a token
  (`baseVersion:entries:entry`; one plan per conversation is held, so the commit of a plan that a later one, from
  another window, replaced is refused and never runs on the later plan's message). The dialog quotes the message's
  first line, says what it removes, that the text goes back to the box, and each command and schedule it ends; first
  focus is Cancel. Rewind posts `POST …/rewind {token, mode}` (`all`, `conversation` or `files`; Chat takes only
  `conversation`, any other is 400); a token the thread moved past is refused (409 `rewind-changed`,
  "The conversation changed. Open Rewind again."). The route answers 202 `{}` once it started. The host
  (`AgentHost.rewind`) refuses with `conversation-busy` while `engineBusy` holds and with `engine-lost`, loads a cold
  conversation first, stops each removed command and monitor (`stopTask(…, 'teardown')`, no wake) and deletes each
  removed schedule (`_x.ai/scheduler/delete`, soft 10 s) at once; one that fails ends the rewind with nothing rewound
  ("A background command could not be stopped. Stop it in Activity, then rewind."). A rewind that puts files back
  (`all`, `files`) first sends the dry run again (`force: false`, its own `mode`): the agent's commit overwrites every
  file it finds changed outside it, so a file the plan showed clean (or did not show) that changed since ends the
  rewind `changed` with nothing rewound, nothing stopped and nothing sent with `force: true`. An `all` rewind that
  stops anything sends the dry run once more after the stops, as the stopped work ran until its stop: a file that
  changed during them ends it `changed` there, with that work stopped and nothing rewound. Either way the page asks
  the plan again and the dialog says "A file changed after the last check. Look at the files again." over the new list (it
  opens on Conversation only, as for any conflict). Then it sends
  `_x.ai/rewind/execute {sessionId, targetPromptIndex, force: true, mode}` (`all`, `conversation_only`, `files_only`; both always sent, as an absent `mode` is the agent's `all`) with no deadline and
  never kills the agent for it (mlx-agent v0.3.4 `extensions/rewind.rs`: a bare `RewindResponse`; the agent writes a
  `rewind_marker` line to `updates.jsonl` and sends nothing live). The session holds an `op` while it runs; Stop does
  not cancel it. A done rewind cuts the host's thread in memory (`Conversation.cut`: the entries before the message
  and the metrics of the kept turns, one `sync`), which is what a disk read gives (`filterRewindLines`); Chat builds
  its address ledger again from the kept entries, so an address only the cut part held is refused again; Work stores a
  `rewind` line in the permission log (`{kind, at, mode, files, from}`, no path or text) and sends its mark. The end
  comes on the stream after the `sync` as the `rewind` event (`end: {outcome: 'done', mode, restored, entry}`,
  `{outcome: 'partial', mode, restored, entry, failed, failedMore}`, `changed`, or `refused | tasks | stopped |
  failed`) with its number `n` (1, 2, … per conversation in this app run), and every later
  `sync` carries the last one as `rewindEnd: {n, end}`, so a page that synced again between the cut and the end still
  gets it. The dialog takes an end whose `n` is past the one it saw when Rewind was pressed. The page closes the dialog, puts the message text back in an empty
  box (a box with a draft keeps it, and a toast with no timeout offers "Put message back"; the toast is the rewound
  conversation's: pressed while the page shows another, it opens that conversation with the message in its box and
  leaves the other's draft as it is), focuses the box and says
  "Rewound. The message is in the message box."; a rewind to the first message leaves one line, "Rewound to the
  start. Your message is in the box.". Another conversation opened while the rewind runs closes the dialog, but the
  rewind's conversation keeps its stream (`watchConversation`) until its end: a done end puts the message in that
  conversation's own box (or, when that box has a draft, a toast with no timeout offers "Put message back", which
  opens it with the message), and a toast names it (its title) with Open; any other end is a toast with Open too.
  In Work the thread keeps a "Rewound to here" line at the time of the rewind,
  and an allowed or denied mark with no row inside the removed span (`[from, at)`) is not drawn. After 20 s the
  button reads "Still restoring…"; after 120 s the dialog offers Keep waiting or Restart the agent (it says how many
  background commands a restart stops), and again 120 s after each Keep waiting, so the dialog always has a way out.
  Esc and a press outside do nothing while it runs. The dialog's last line says "This cannot be undone. Fork first to
  keep a copy."
- Rewind files (Work): the plan also asks the agent's dry run, `_x.ai/rewind/execute {force: false, mode: 'all'}`
  (`AgentHost.rewindPreview`: a cold conversation is loaded first, "Opening the conversation…", else "Checking
  files…"; soft 20 s, never kills the agent; it sets no `op`, so a message sent meanwhile makes the plan's token stale;
  Work's plan also needs the folder and the engine: `inference-stopped`). The agent changes nothing and answers
  `clean_files` and `conflicts` (`modified_externally | deleted_externally | created_externally`); a dry run's
  `error` with no conflict is a target past its prompts (409 `rewind-not-point`). The plan carries `files: [{path,
  state: 'clean' | 'modified' | 'deleted' | 'created'}]` (paths relative to the folder, redacted, hidden characters as
  `\u{…}`, the changed ones first, at most 200) and `filesMore`; Chat's plan has none and sends no dry run. A miss is
  503 `agent-slow`: the dialog says "The agent did not answer." with Try again. With files the dialog offers "What to
  rewind": Conversation and files ("Puts 3 files back as they were"), Conversation only ("Files stay as they are
  now"), Files only ("The conversation stays as it is"), an `OptionList` whose checked row has first focus (Enter on a
  row picks it, never confirms). It opens on Conversation and files, or on Conversation only when a file changed
  outside the agent: then the file choices carry a dot, their note says "· 2 changed outside", each such file reads
  "changed since the agent wrote it" (or deleted, or "made again since the agent removed it"), a line says "2 files
  changed outside the agent. Restoring overwrites that." and the button reads "Restore and overwrite" (the agent's
  `force` overwrites them). A file choice shows "Files to restore (n)", 8 rows and the rest behind "n more", and the
  limits: "Only files the agent edited are restored. Changes from shell commands and helper agents stay. Files the
  agent made are deleted." (pinned on v0.3.4 by `chat-rewind-real-agent.test.ts`: the agent's file points hold only
  what its own edit tool wrote in this session; a command's change and a helper's edit, in the background or not, stay;
  a hand edit is `modified_externally`). Conversation only folds the list away ("Files stay as they are now. The agent
  will not remember changing them."). Files only reads "Puts 3 files back as they were before this message. The
  conversation stays.", "Restore files", and stops no command or schedule: the thread stays, nothing goes back in the
  box, the toast and the live region say "3 files restored", and the rule reads "Files restored · 3 files" after the
  last entry (its line has no `from`). Conversation and files toasts "Rewound · 3 files restored" and its rule reads
  "Rewound to here · 3 files restored". With no file to put back the dialog has no choices and says "The agent changed
  no files after this message."; so does a fork's copy (the agent copies no file points into it). An agent that stops
  while it puts files back ends with "The agent stopped while rewinding. Check the files in app."; the restore runs
  through the agent's own writes, with no permission request, in Ask, Auto and Bypass alike. The agent skips a file
  it cannot write, answers success and drops its copy of it (v0.3.4 `acp_session_impl/rewind.rs`): a file of the dry
  run missing from `reverted_files` ends the rewind `partial`. The thread is cut and the rule stored as for `done`
  (with the files put back), the queue holds, and the dialog stays with "Rewound, but 2 files could not be put back."
  ("2 files could not be put back." for Files only) over the list of them ("Files not put back, 2"); the message goes
  back to the box as for `done` (a box with a draft keeps it, and a toast "Your draft stayed. The removed message is
  here." offers "Put message back").
- Fork (Chat and Work): the thread bar's `⋯` has "Fork" first ("Copy this conversation"), a sent message's `⋯` has
  "Fork with this answer" ("New conversation up to this answer") under Rewind, `/fork` runs the first and the palette
  has "Fork conversation". Both rows are off with the reasons of Rewind (an answer, a wake, a compaction, a workflow
  run or a subagent; the engine restarted under commands; the engine or agent blocker); a background command or
  monitor alone does not stop them. Both call `historyActions.fork`, which posts `POST …/fork` with `{}` or
  `{entry, ts}` (the message's index and its time, as the plan of a rewind) and shows a "Forking…" toast. The service
  (`ChatService.fork`) refuses an entry that is not the page's message any more with 409 `fork-changed` ("The
  conversation changed. Pick the message again.", `CHAT.history.forkChanged`), maps it to the agent's prompt index as
  Rewind does (`targetOf`; a message the agent never ran is 409 `fork-not-point`, "This message is not a fork point."), then
  takes the automatic continues of that answer too (`answerEnd`). The host (`AgentHost.fork`) refuses with
  `conversation-busy` while `engineBusy` holds or a recap is written and with `engine-lost`, starts a stopped agent but
  loads nothing, holds the session's `op` (a prompt sent meanwhile with `whenBusy: "queue"` queues and goes once the op
  ends, however it ended), and sends `_x.ai/session/fork {sourceSessionId, sourceCwd,
  newCwd[, targetPromptIndex]}` with both folders the conversation's own (Chat: the fixed Chat workspace) and no model,
  soft 60 s (mlx-agent v0.3.4 `session/fork.rs`: a bare camelCase reply `{newSessionId, …}`; `targetPromptIndex` keeps
  prompts `0..N` and their answers, inclusive, unlike a rewind's; the copy's `summary.json` keeps the title and gets
  `session_kind: "fork"`, `parent_session_id`, `forked_at`; rewind points are not copied). A miss is `agent-slow`: the
  agent lives and the lists are read again; when the late reply comes (`AcpSoftTimeoutError.late`) and names a copy,
  the host tells the service (`onForkLanded`), which lands it as an answered fork: Work writes its index line (a failed
  write deletes it) and `conversations-changed` goes out, so the copy shows in the list. An error reply
  is `agent-protocol-error`. Work then adds the copy to the Work index; a failed write deletes the copy through the
  agent. The route answers `{id}` and `conversations-changed` goes out. While a fork runs, a second Fork, Compact now
  and Rewind are refused with 409 `fork-running` ("A fork is already running.", `CHAT.history.forkRunning`), and the
  page's rows read the same; while a recap is written they are refused with 409 `recap-running` ("Wait for the
  recap.", `CHAT.history.offRecap`). `ChatHostState.active[].op` (`'rewind' | 'compact' | 'fork' | 'recap'`) names the
  op or recap that runs; the page reads it, with its own Fork press until the host lists it and the stream's `recap`.
  With an op (not a compaction) and no answer running the composer is in phase `op`: Enter and the round button queue
  the text (Queue, tooltip "Runs once the conversation is free (Enter)"), there is no Stop and Esc stops nothing; with
  an empty box the button is a still ring named "Busy". The page opens the copy (cold: its first message
  loads it), focus goes to the box, and the live region says "Opened a fork of Fix the login test."; the toast turns
  into "Forked. This is the copy." with "Back to the original" (Work adds "Both conversations work in app/."). A page
  that shows another conversation by then stays there: the toast reads "Forked. The copy is in the conversation list."
  with Open. The palette's Delete asks about the conversation it was opened in and closes when the page shows another.
  A
  refusal turns it into an error ("Could not fork the conversation.", the busy or blocker sentence, or the fork-point
  one) and the page stays. The copy starts in Ask with an empty queue and an empty permission log; the original gets
  no mark. `ConversationInfo.forkOf` (`{id, title}`, from the session detail's `parent` when `kind` is `fork`; `title:
  null` once the original is gone, also for a copy this run holds when the original is deleted, and a rename of the
  original, by the user or the agent's own title line (`onTitle`), renames it there) draws the thread's first line, "Fork of Fix the login test · Open" (`ForkOrigin`,
  `ui/src/components/history-mark.tsx`; "Fork of a deleted conversation" with no link). List rows carry `forkOf` too
  (the worker reads the title from the same list, `null` when the original is not listed) and draw an outline "Fork"
  tag after the title, read in the row's name; the dot column keeps only the state dot.
- Recap (Chat and Work): the thread bar's `⋯` has "Recap" third ("Where things stand, in a few lines"), `/recap` runs
  it and the palette has "Recap conversation". The row is off, focusable, with its reason: the engine or agent
  blocker, "Wait for the answer to finish." while an answer, a wake or a history op runs, the engine-lost sentence,
  "Nothing to recap yet." before the first finished answer, and "Writing a recap…" while one is written. A live
  workflow run, helper or command does not stop it. It calls `historyActions.recap`, which posts `POST …/recap`; the
  route answers 202 `{}` once the recap started, since the page call fails after 30 s and the agent's model call may
  take 120 s. The service (`ChatService.recap`) checks as Compact now does (`inference-stopped` with the engine
  stopped). The host (`AgentHost.recap`) refuses with `conversation-busy` while `isBusy` holds or a recap is written
  and with `engine-lost`, loads a cold conversation first (`ensureLoaded` with no caller), then sends `_x.ai/recap
  {sessionId, auto: false}` soft 10 s (mlx-agent v0.3.4 `extensions/recap.rs`: `{result: {ok: true}}` at once, or
  `{result: {ok: true, disabled: true}}` when recaps are off). The recap comes after as one xAI line, sent and stored:
  `session_recap` (`summary`, `auto`) or `session_recap_unavailable` (a manual one only: no finished answer, another
  recap running, the model call failed or gave nothing, or a typed prompt came). The host reads that line to end the
  wait; the session holds no `op`, so a plain prompt is not refused, and the agent drops the recap when one comes. A
  prompt with `whenBusy: "queue"` (the page's Enter) queues instead and goes once the wait ends, and a queued row does
  not go while it runs (Send next included). The wait counts as work (`hasWork`, `holdsIdle`): neither the idle stop
  nor the stale restart stops the agent while a recap is written. The wait
  is `{startedAt}` on the conversation stream (`recap` event, `sync.recap`) and ends with `recap: null` and `end:
  'shown' | 'none' | 'off' | 'cancelled' | 'failed' | 'load-slow'` (live only): `none` also when no line came in 120 s,
  `cancelled` when a typed message started a turn or the conversation was deleted, `failed` when the load, the ack or
  the agent failed, `load-slow` when the cold load passed its soft deadline. While it runs the thread
  end shows "Writing a recap…" with a clock (no running answer), said once. `none` toasts "No recap this time.",
  `off` "Recaps are turned off in the agent settings.", `failed` the error "Could not write a recap.". The shared
  reducer makes a `recap` entry from `session_recap` (`text`: the summary trimmed, cut at 2,000 characters with "…"),
  live and from disk alike, added at once (not held like a notice), and nothing from `session_recap_unavailable`. It
  is a side entry (`isSideEntry`, `ui/src/lib/chat-turns.ts`): it never starts a turn, never counts as work and never
  moves the wait clock, and a recap after an answer forms its own group, so the answer's end line stays on the
  answer. The row (`RecapRow`, `history-mark.tsx`, named "Recap, 12:04") reads "Recap" with the agent's text under
  it as plain text; live, the live region says "Recap. …" once. The Sessions transcript draws the same row and its
  search finds the text. `__test__/chat-recap-real-agent.test.ts` pins this on the real agent: the unavailable and
  the shown line as the fake agent plays them, the same row live and stored, and a typed message ending a wait
  `cancelled`.
- **Goal** (Work only, #17; `ui/src/lib/goal.ts`, `ui/src/components/arc-x/goal-strip/`). `/goal <objective>
  [--budget <tokens>]` is an agent command (allowed built-in, sent as typed). mlx-agent v0.3.4 stores the goal's state
  as `_x.ai/session_notification` `goal_updated` lines (`goal_id`, `objective`, `status` active · user_paused ·
  back_off_paused · no_progress_paused · infra_paused · blocked · budget_limited · complete · cleared, `phase`,
  `tokens_used`, `elapsed_ms`, `token_budget?`, `pause_message?`, `verifying_completion?`, `planning?`), sends a
  live-only tick of the same line when a helper's progress moves the goal's tokens (no helper, no tick), and a
  cleared goal as a line with an empty id. It runs its
  rounds inside the typed `session/prompt` (one `turn_completed` at the end; the reply waits) and, when a turn ends
  with the goal still on, starts the next round by itself as a wake `goal-summary-<uuid>` ("Started by the goal
  loop"). It stores the typed prompt as its goal rules (`<system-reminder>\nA goal has been set: <objective>\n…`),
  which the shared reducer reads back as `/goal <objective>` (with ` --budget n`), or `/goal resume` for the rules
  of a resume, so disk and live show the typed text. The reducer keeps one `goal` entry per goal id, held like a
  notice so it sits right after the message that set it (the agent stores the first line before the prompt); later
  lines change it in place, with a `set` op only when a shown field moved (state, phase, objective, budget, tokens in
  steps of 1,000, reason, checking, planning), so ticks do not flood the page, and a cleared line closes the last goal
  that is not done. `pause_message` is cut at 300 characters (the redactor drops a cut key's start). The strip sits
  in the dock above the composer (`GoalStrip`, a group named "Goal"): the last goal unless cleared, a done goal until a
  message is typed after it. Running it reads Planning, Working, Checking or "Waiting for you" (a card is open) with a
  clock counted on the page from the line's `elapsed_ms` and time, and the tokens ("31k tokens", "31k of 200k"), with
  Pause; stopped it reads "Paused", "Paused: no progress" (the agent's reason under the line), "Blocked: …",
  "Budget used · 200k" or "Done · 41:10 · 96k tokens", with Resume and Clear (Clear only when out of budget or done);
  an active goal of a cold conversation reads "Paused: the agent is not running". Narrow, the tokens go first, then
  the word Goal and a running goal's state words. The buttons post `POST …/goal {action}`: Pause while an answer runs
  is Stop (the agent pauses the goal; `{stopped: true}`), else each sends the agent's own `/goal pause|resume|clear`
  through the checks of a typed message (a host turn; Resume runs the goal again); Resume and Clear while an answer
  runs are 409 `conversation-busy`, Chat 400; on the page they are off then ("Wait for the answer to finish."), and
  Pause stays on. A failed press is a toast ("Could not pause the goal."), the strip keeps
  its last true state; after Clear the composer takes focus, after Pause the focus stays on the button, now Resume.
  The live region says each state change once ("Goal paused.", "Goal done.", "Goal cleared.", "Goal blocked: …"),
  never the clock. The thread keeps one row where the goal began ("Goal · <objective> · Done in 41:10"), the Sessions
  transcript the same; the rail has a Goal section after Needs you (its row "Show goal in the thread" jumps there),
  which never opens the docked rail by itself (`hasActivity` leaves it out). While the goal runs (`active` in a live
  agent; `AgentHost.goalRuns`) the queue sends no row: its header reads "Queued · n · Runs when the goal pauses." and
  the rows go once a `goal_updated` leaves it paused, blocked, out of budget, done or cleared with no answer running.
  Send now reaches the running round, also between two rounds (`steerable`). Rewind to a message before the open
  goal's row is off ("A goal started after this message. Clear it first.", 409 `rewind-goal`): the message menu, the
  palette row and a typed `/rewind` (its reason slot says so and Enter sends nothing). The `/goal` row of the
  command list carries the tip "Asks as your permission mode says. Auto fits best.".
  `__test__/chat-goal-real-agent.test.ts` pins the wire on the real agent.
- Soft calls: a missed `ACP_TIMEOUTS` deadline is a protocol error that stops the agent with every run and command it
  holds. `AcpClient.requestSoft(method, params, ms)` rejects with `AcpSoftTimeoutError` past `ms` instead (its `late`
  settles with the reply when it comes) and leaves the agent running; every call added that is not a turn uses it.
  The cold load of a history op, a recap or a schedule's Delete is soft too (`ensureLoaded` for any caller but a turn):
  `session/load` goes with no deadline and the guard's `_x.ai/session/info` soft 5 s, and the op waits at most
  `LOAD_SOFT_MS` (30 s, `HostTimings.loadSoftMs`). A miss ends that op `load-slow` ("The agent is slow to open this
  conversation. Try again.", `CHAT.history.loadSlow`): a plan or a Delete answers 503 `load-slow`, a rewind ends
  `{outcome: 'load-slow'}` on the stream (its dialog offers Try again), Compact now and a recap end `load-slow`. The
  agent and every command, monitor and run it holds go on, and the load does too: its late reply passes the guard and
  the conversation is loaded (Work: it reads hot, from `onSessionReady`), with nothing of the op run after it. A turn
  keeps its hard load (a miss stops the agent, as it always has); a turn that waits on a soft load gives that
  `session/load` the turn's deadline from then (`AcpClient.harden`), and the guard's read after it is hard too. The
  fake agent's `slowLoad` step answers the next load of its session that late, once (the harness word "lateopen").

- Retries (Chat and Work): when the model stops answering (no token for the agent's idle time, 300 s by default), the
  connection fails, the engine errors, or too many requests run at once, the agent sends the request again and says so
  in one-shot xAI updates it also stores (mlx-agent v0.3.4 `extensions/notification.rs`): `retry_state` with `type`
  `retrying` (`attempt`, `max_retries`, `reason`, `error_type`), `exhausted` (`attempts`, `reason`) or `failed`
  (`error_type`, `message`), and `auto_recovery_started` / `auto_recovery_exhausted`. There is no "retry ended" update.
  The shared reducer makes one `notice` entry (`code: 'retry'`) per episode, live and from disk alike: tries that follow
  each other are one entry (`attempts`, `max` and the cause of the last try); text, a thought or a tool call ends it as
  `recovered`; `failed` / `exhausted` end it as `gave-up` with the agent's words (cut at 300 characters); the turn end
  ends it with no outcome; a `failed` with no try before it makes no entry (the turn's footer has the error). Because
  the agent sends one-shots past text it made earlier, a late chunk may end an episode early: the next try reopens the
  same entry unless a tool call came between. The cause comes from `error_type` (`idle_timeout` stalled, `http`
  connection, `api` / `serialization` / `empty_response` server, `rate_limited` or a "Too many requests" reason busy,
  `doom_loop_detected` loop), auto-recovery is its own cause, anything else keeps the agent's reason as its words. A
  notice never starts a turn: one made while no turn is open shows in the next turn. Its row is a one-shot, placed as a
  hook row is (below): before the first call stamped after it, else at the turn's end, the same live and from disk
  (`__test__/chat-retry-real-agent.test.ts` pins the store's order on the real agent: a 503 on the first call and on
  the call after a denied call). The open episode the line shows is the reducer's live copy (`retryLive`), which the
  stream carries as the `retry` event and `sync.retry`, not an entry. While an episode is
  open the thread end shows one line in place of the wait line (a running tool row does not hide it; an open request
  card and the compaction line do): "The model stopped answering. Trying again · try 2 of 3" with the bar of the
  conversation's new engine request while it reads its prompt and the clock from the first try; it is announced once.
  After it, one quiet row in the turn: "Retried · The model stopped answering · 2 tries · waited 1m 04s", "Gave up"
  in the danger tone with the agent's words under it, and "Retry not finished" for an episode with no end once no turn
  runs (the agent stopped mid-retry). The Sessions transcript draws the same row (not under "Messages"); the timeline's
  retry marks read the same lines as before. A helper's retries stay on its own Session page.
- Pictures left out or made smaller (Chat and Work): the agent says so in two one-shot xAI updates it also stores
  (`extensions/notification.rs`): `image_dropped` (`notes`: one sentence per picture, e.g. "Image 2 was dropped before
  send: …" for a prompt or tool-result picture that fails its checks, or "This request failed over its images …" when
  the engine refused a request and the agent sent it again without them) and `image_compressed` (`images`: index, bytes
  and pixel sides before and after; `message`). The shared reducer makes one `notice` entry (`code: 'picture'`, `what`
  `dropped` or `shrunk`, `count`: the pictures made smaller, or a dropped line's notes (one note can speak for several
  pictures), up to 6 `notes` cut at 300 characters, `sizes`), a one-shot placed as a hook row is, live and
  from disk alike; an `image_compressed` with `images: []` (a re-encode that failed, original kept) makes none. The
  row: "Picture · Left out of the request" in the warning tone, with no count (the agent's own note says how many, as
  "1 image(s) were left out of the retry" for N pictures in one note), and the agent's notes under it, one per line, or "Picture
  · A picture was made smaller · 2.1 MB → 921.6 KB · 4032×3024 → 2000×1500" (several: one line per picture). The app
  shows its own headline and never parses the note. Live, the headline is announced once; a row the thread opened with
  is not. The Sessions transcript draws the same row. The agent makes a prompt's picture lines after the prompt's
  chunks, but its store writes them before the chunk it still holds; the placement gives the same list.
  `__test__/chat-pictures-real-agent.test.ts` pins this on the real agent (a 2,100 px wide picture is made smaller, a
  4×4 one left out): the fields of the fake agent's lines, and the same two rows live and stored, right after the prompt.
- No model (Chat and Work): when the agent has no model it can use it sends `model_auto_switched`
  (`previous_model_id`, `new_model_id`, `reason`; `agent_ops.rs` `send_model_auto_switched`), live only and never
  stored. The host takes the line (the reducer never sees it) and puts it on the running turn: `''` reads as null and
  the reason is cut at 300 characters and redacted. A blocked prompt (both ids `''`, the agent answers `end_turn` at
  once and stores nothing) and a load whose stored model is gone with no fallback (`new_model_id` `''`) end the turn
  `no-model`: "No model answered. Your message got no answer." with Send again; a load that fell back to another
  model is still refused by the guard, as before. The turn's end (`TurnEnd.model`, `sync.lastEnd`, memory only) carries
  `{previous, next, reason}`, and the footer adds one muted line: "The agent has no model it can use: <reason>",
  "The agent could not use <previous>: <reason>" or "The agent could not use <previous>. It wanted <next>. <reason>".
  The end is announced with that sentence. One for a session the host does not know yet (inside `session/new`), or
  with no turn running, is dropped: the create or the guard already says "That model is not available.". A queue
  holds after a `no-model` end. After a reload the line is gone with the unsent message.
- Hooks (Chat and Work): the agent says what its hooks did in three xAI updates (`extensions/notification.rs`,
  `session/acp_session_impl/hook_dispatch.rs`). `hook_execution` (`event_name`, `tool_name?`, `prompt_id?`, `runs`:
  `name` and `status` `{status: success, elapsed_ms}` / `{status: skipped}` / `{status: failed, error, elapsed_ms,
  blocked?}`) is sent and stored after each batch; `blocked: true` is a hook's decision, not a failure. The shared
  reducer makes a `notice` entry (`code: 'hook'`: `event`, `tool?`, `hook` without its `:{event}[i].hooks[j]` tail,
  `detail` cut at 600 characters, `times`, `lastTs`) for the first failed run of a hook, event, tool and error in the
  session; each repeat counts on that entry (`set` in place, no new row), so a stop hook that fails on every turn is one
  row. Passed, skipped and blocked runs make nothing. `hook_annotation` (`message`, `kind?`: `note` or `tool_outcome`)
  is sent and stored: a note is a `notice` entry (`code: 'hook-note'`, the leading ⚠ or ↩ dropped, cut at 600
  characters); a `tool_outcome` ("`tool` blocked by hook: reason") pairs, oldest with oldest, with the open turn's calls
  that failed with "Hook denied: " and lands on that call as `blocked` (the agent sends its words past the call's lines,
  so either can come first); one left over at the turn's end is a note. A hook row (failed run or note) is placed
  before the first `tool_call` stamped after it, else at the turn's end (its `turn_completed`, the next prompt, or the
  end of the file of a turn that never ended): the agent's store holds its last ACP line and merges later text into it,
  and writes a one-shot straight away (`session/persistence.rs` `maybe_merge_notification`), so on disk a stop hook's
  failure sits before the answer it came after. Placed that way, live and stored give the same list, and the answer's
  text stays one entry; a row made while no turn is open shows in the next turn. Every one-shot row (failed hook,
  hook note, retry episode, picture) is placed this way (`oneShots`, `placeOneShots`); mode lines are ACP lines, which
  the store keeps in order, so they are placed when they come. A `session_start` batch runs while `session/new` is
  still open (`DispatchSessionStartHook`), so its line, and the `hook_annotation` of each hook's own `systemMessage`
  sent right after it (`send_hook_execution`), can come before the reply names the session: the host hands such lines
  on while a `session/new` is in flight, the service keeps them by session id (at most 64 lines) until it holds the
  conversation, and their rows then stand in the first answer as on disk. A prompt a hook blocked
  stores no prompt chunk (`turn.rs`): its note and the end it came with stay together, and its stored `turn_completed`
  carries `_meta.cancellationCategory` `HookDenied`, which the reducer keeps as the end's `hookStopped`; live the prompt
  reply says the same and the turn ends `hook-stopped`: "Stopped by a hook." with no Send again, the same after a
  reload. Live the service drops the prompt's bubble (`Conversation.dropTypedPrompt`, a fresh `sync`), so the thread
  holds the hook's row and the end, as after a reload, and the end carries the typed text (`TurnEnd.blocked`; an
  automatic `continue` carries none). The page puts it back in an empty message box ("A hook blocked your message. It
  is back in the message box."), with the files it was sent with (from the send still in flight, else the page's
  last sent message to that conversation, `lastSent`), added after the box's own files up to the cap; a box with text
  keeps it, and a toast "A hook blocked your message." offers "Put message back", which puts back the text and the
  files the same way, in that conversation's own box: pressed while the page shows another, it opens that one. It does so only for an end it saw come, or for the first message it sent from the New screen (a
  fast hook ends that answer before the page holds the conversation). The rows: "Hook failed · global/lint · before the answer ends" in the warning tone with the error under it,
  then "12 times · last 11:20" in place of its time; "Hook · Prompt blocked by global/policy: no deploys on Friday";
  the app's own Work guard reads "the app's command check". A blocked call's own row reads "Blocked" with "by a hook"
  and the agent's words in place of its output ("Blocked · by a hook" on the Sessions page). A new row is announced
  once; a count change is not. `hook_run_started` (`event_name`, `tool_name?`, `prompt_id?`, `count`) is live only:
  the host takes it (the reducer never sees it) and holds `hookRun` (`{event, tool?, count, startedAt}`, event
  `hook-run`, `sync.hookRun`) until the batch's own `hook_execution` (same event, tool and prompt id), the model's text
  or thought stamped after the batch started (the agent buffers text it made before), the turn's end line or reply, a
  compaction start, or the agent's exit; any other line keeps it, so a stop hook that holds the turn for minutes beside
  a background command's output still shows. Once the batch is 1 s old the thread end shows "Running a hook · before
  the command" ("Running 2 hooks · before the answer ends") with its clock, after an open request, the compaction line
  and the retry line and before the wait line, even while a tool row runs; it is announced once ("A hook runs before
  the command."). A helper's hook lines stay in its own session. With `MLX_AGENT_E2E_BIN`,
  `__test__/chat-hooks-real-agent.test.ts` pins this on the real agent with a hooks file in the agent home (a
  `PreToolUse` deny, a failing `Stop` hook, a `UserPromptSubmit` block): the lines have the fake agent's fields, the
  host takes `hook_run_started`, and live and stored give the same rows.

- Picker: the pill shows the base model and a badge with the thinking level of that model (`Qwen3.8-27B` `High`: the
  picked level, else the model's default; no badge when the model has no levels or the agent has not listed it; the
  badge hides under 520 px of composer, and in Work it is drawn whenever the row has room for it (while no folder is
  chosen, with the folder label at its floor), else not at all). The quantization is in the popover's middle list, and
  the pill's title and its name always say it (`Qwen3.8-27B-UD-Q4_K_XL, Thinking: High`), the title the full id. The
  popover is a cascade: one row per base on the left, the quantizations of the open base in the middle, and Thinking
  effort on the right (see **Thinking**): Model → Quantization → Thinking effort for a base with several
  quantizations, Model → Thinking effort for a base with one (its levels take the middle column), and no Thinking
  effort column for a model with no levels. A base row with levels shows the level a pick keeps, muted after its name
  (the conversation's level on the conversation's base, else the model's default); on a narrow row that level goes
  before the name shortens. A row with a next column (a `›` at its end) never picks: a click, `Enter`, `Space` or `→`
  opens that column and moves focus there. Only leaf rows pick: a level, a base with one quantization and no levels, a
  quantization with no levels (Qwen3.8-27B → UD-Q4_K_XL → High is three clicks, Hy4-flash → Max two). A base opens
  never on hover; in the middle list the active row (pointer or keys) shows its levels on the right without moving
  focus. A pick of a level sets the model, the quantization and the level together and closes the menu. Keys: `↑` `↓` `Home` `End` and letters move, `←` goes
  back a column, `Esc` closes; each list is one tab stop with `aria-activedescendant`. The panel keeps 16 px from the
  window edges. When the chip's left edge plus three columns (713 px) and that gutter do not fit in the window, the
  levels replace the quantizations instead of sitting to their right, under a back row (`‹ UD-Q4_K_XL`, or `←`), and a
  quantization opens its levels only on a click or a key, not on hover. Both lists put the model in memory first, then the model in load, then the
  rest (bases by name, quantizations by size). The quantization that is active when a base opens: the conversation's
  model, the one in memory, the one in load, the last one picked for that base (`localStorage`
  `mlx-node:chat-variant:v1`), the smallest. `base`, `variant`, `format` and `sizeBytes` come with each entry of
  `ChatHostState.models` (discovery names them, `@mlx-node/lm` `model-variant.ts`); every other page shows the id. The
  size of a split GGUF counts the parts the loader opens beside the listed path (`ggufSplitPartNames`); a set it cannot
  open has no size, so it is never the smallest.
- Memory: the picker footer names the chat model in memory, with Unload; after a stop before the prompt was sent, the
  thread's end line says the same. Unload calls `POST /api/chat/engine/unload`, which calls the engine's
  `POST /v1/models/unload`. It is off while an answer runs in Chat or Work or the engine loads a model (409
  `engine-busy`, also when the engine itself answers 409). After it, no place says the model is in memory, and the
  next message loads it again. The Engine page has no Unload.
- Turns: the Stop button, the palette's "Stop answer" and `Esc` in the composer cancel the running turn. `Esc` stops
  nothing on a key repeat, while an input method composes, or with the focus anywhere else. A Chat turn stops after 10
  minutes (`turn-timeout`); the time the agent compacts the conversation is not counted (the cap pauses at
  `auto_compact_started` and runs on with the time that was left after the end update). A Work turn has no time limit: it runs until the agent ends it or you press Stop, and the
  app ends it only when the engine or the agent is gone, or after two length stops in a row (below). The end of a turn says why it stopped; `output-limit` offers
  Continue, `tool-refused`, `arguments-unreadable` and `tool-loop`
  offer Send again. A `refused` turn says "Chat could not be started safely." (Work alike) with Update the agent,
  except one whose failure is `model-unavailable`: "That model is not available." with "Pick another model", which
  opens the composer's model picker.
- Length stop in Work: when the model's reply hits the output cap, the app sends the Continue button's own message
  (`continue`) once, inside the same turn: after the agent's own end line of the reply (no timer) and a fresh read of
  the engine's endpoint. The thread shows a "Continued" divider where the prompt bubble would be, in every permission
  mode; the pending row reads "Continuing after the length limit…" until the answer comes. A tool call since the
  prompt was written resets the count. Two length stops in a row with no tool call between end the turn:
  "The answer hit the length limit twice in a row. Continuing did not help." and Continue. Stop, a
  lost endpoint, the agent's exit or a stale host between the two prompts end the turn `output-limit` and nothing is
  sent to the agent. Chat never continues by itself. The turn's result carries `lengthStops`, `continues`,
  `noContinue` (`second-length`, `stop`, `endpoint`, `agent-exit`, `host-stale`) and `late` for the stop record.

  | What ends a Work turn                               | Who               |
  | --------------------------------------------------- | ----------------- |
  | The agent ends it                                   | the agent         |
  | Stop                                                | you               |
  | The engine or the agent is gone                     | the app           |
  | Two length stops in a row with no tool call between | the app, by count |

- Thought time: a thought made from live lines shows the time from its first to its last chunk. A thought read from
  disk shows "Thought" alone: the agent stores one merged record with its last stamp, so no real time is known. When a
  conversation goes cold and is read again, its earlier thoughts lose their time.
- Titles, one rule on the server (`src/chat/titles.ts`) for Chat and Sessions: the agent's title, else the first typed
  prompt (first non-empty line, at most 60 characters with the `…`, cut at a word), else no title. A page prints its
  own word for no title: "New chat", "New Work conversation", or "Session" and the first 8 characters of the id.
- Untrusted text: model output and web results render as text. A link to a host that no search result or user message
  gave is plain text, never a link.
- Markdown: every answer, Thinking body, Work report, plan and Sessions transcript message goes through `ChatMarkdown`
  (`arc-x/chat-markdown/`), which runs `markstream-react` 2.0.32 with `stream-markdown-parser` 1.2.31 (both pinned
  exactly; read the diff and rerun `__test__/chat-markdown-lib.test.ts` on a bump) on our own node components
  (`md-nodes.tsx`). The parser setup is one function, `ui/src/lib/markstream-config.ts`:
  - Raw HTML is never parsed: `<b>`, `<script>`, `<img>` and `<a href>` show as the text that was written.
  - Links: only http(s) opens, in a new window. In Chat only a host of the conversation is a link; any other link is
    its words and ` (host)`. Link words that name another host show the host the link opens. In Work every http(s)
    link opens and shows its host. File names, e-mail addresses and `www.` words are text (`ui/src/lib/link-rule.ts`).
  - An image is never loaded: it is "Image: alt" under the same link rule, and its alt text alone in a Thinking body.
  - `[n]` (1-99) that names a source of the answer is a source chip; any other `[n]` stays text.
  - Code fences, Mermaid, D2 and diff fences included, are code panels. Math, footnotes, `:::` boxes, task boxes,
    `==mark==`, `^sup^` and smart quotes are off. A `[1]: https://…` line stays visible. Bold next to CJK
    punctuation works (`**说明**。`). `&amp;` shows as `&`.
  - markstream's optional renderers (KaTeX, Mermaid, D2, infographics, stream-diffs) are aliased to a stub that fails
    to load (`ui/vite.config.ts`, `ui/src/lib/markstream-off.ts`), so nothing is fetched, no worker starts and no
    markstream stylesheet is loaded.
  - While an answer streams, new words fade in (under reduced motion they appear at once), a settled block keeps its
    object and is not drawn again, a half link shows its words, and a fence shows nothing until its first line.
    `scripts/dev/markdown-replay.ts` replays recorded answers chunk by chunk and prints the parse time per chunk.
  - A streaming parse and render that takes more than a frame (16 ms) holds the next text for 4× its cost, at most
    10 s (`paceHoldMs`), so a looping answer cannot freeze the window; the finished text is drawn at once.
    `markdown-replay.ts --hostile` prints the main thread's busy and free time on two such answers.
- Rename (`F2`) and delete work from the bar, the list and the palette; both are off while an answer runs.
- The list (`arc-x/conversation-list`, `pages/chat/list-pane.tsx`): Chat in day groups (Today, Yesterday, Previous 7
  days, Previous 30 days, then months). Work in one group per folder (`groupByFolder` in `lib/conversations.ts`, the
  exact `cwd`), named by the folder's own name; two folders of one name each show the fewest parent parts that tell
  them apart, and the header's title is the `~` path. Groups order by the local day of their newest row, then by name,
  then by path; rows are newest first and one line (a Bypass row keeps the word on a second line). A header click
  folds its group; the folded keys stay in `localStorage` (`mlx-node:work-folded:v1`), a folded header shows a dot
  (waiting, else running) and the count, and going to a conversation in a folded group opens it (not on the window's
  first look). An open group shows its 5 newest rows plus every running, waiting or open row, then "Show n more" (this
  window only). The header's "+" opens New Work in that folder. Conversations started without a folder are one group,
  "No folder" (last among the groups of its day; title "Each conversation here has its own folder in ~/Portato.";
  its "+" starts another one). Search is flat over both modes and matches the title or the folder's own name, not the
  folders above it; a No folder row matches "no folder". One tab stop: ↑ ↓ Home End move over headers, rows and "Show n
  more"; → on a header opens it or reaches "+", ← folds it, and ← on a row goes to its header.

### Work

- **Folder.** Work runs in one folder per conversation, fixed for its life. On the New screen (both modes, until the
  first send) a strip sits under the composer card (`arc-x/composer-tray`): `--surface-muted`, tucked under the card's
  rounded bottom corners, chips on the left only, opening and closing on the `smooth` spring (at once under reduced
  motion). In Work it holds the folder chip, the only folder control on New (the tool row, the bar and the hero have
  none): the folder's name (`~` path as its title), or "No folder" (dashed, no branch chip). Its popover ("Folder")
  lists the newest folders of the Work list (at most 6, never a folder the app made for No folder), then "No folder"
  ("A new empty folder in ~/Portato"), then "Choose another folder…" ("Choose folder…" with no recent folder), which
  opens the native folder dialog (`ui/src/lib/pick-folder.ts`). A choice made in this window wins; else, once the Work
  list is in, the place of the newest Work conversation (its folder, or No folder) is preselected, and No folder when
  the list is empty (`newPlace` in `ui/src/pages/chat/drafts.ts`). While the list loads the chip reads "Loading…"
  ("Loading your folders…") and Send waits with that reason; when the list failed there is no default: the chip reads
  "Choose a folder" and Send waits ("Choose a folder, or No folder, to start.") until a pick. Moves up of the mode
  wait for the place too, and a mode draft made before the list came in keeps its id when it does.
  A folder the server refuses (`cwd-invalid` from the branch read) gives the chip a danger dot and the
  sentence as its title. In Chat the strip holds "Work in a folder": the popover with the recent folders only
  ("Recent folders"; with none a click opens the dialog), and a pick chooses the folder,
  then switches to Work (text and files carry over, as both modes share the New place; a picture gets Work's refused
  mark); Cancel stays in Chat. After the first send the strip goes and the folder is fixed (the conversation's real
  path); an open conversation's chip sits at the top right of the bar and only shows the folder: its popover has the
  whole path and Copy path. The server
  checks the folder (`src/chat/work-folder.ts`): not absolute, missing, not a folder, unreadable, `/` or the top of the
  data volume, the home folder or a folder that holds it (as written or where its links lead), and a folder inside,
  holding or picked through the app data folders are refused (compared by disk identity, so another path to the same
  folder is refused too) with a fixed sentence and `reason: 'cwd-invalid'`,
  shown in a strip with
  Choose folder…. The same rule runs again right before the agent gets the folder (`session/new`, `session/load`) and
  before each prompt. An open conversation whose folder is gone or no longer usable shows its sentence and Send is off.
  Paths show `~` for the home folder; a long one is cut in the middle and shown whole on hover and in the open
  conversation's folder popover (with Copy path).
- **No folder.** A send with No folder posts `noFolder: true` (never with `cwd`; Work only; `false` is a 400). After the
  command, engine, model and picture checks the Control Panel makes `~/Portato` when it is missing (mode 0700; an
  existing one is left as it is), then a new folder in it named by the local minute, `YYYY-MM-DD-HHmm` (0700, never one
  that exists: `-2` to `-9`, then the create fails), records it in `~/.mlx-node/chat/scratch-folders.json` (its real
  path with its device and inode numbers; 0600, written whole; a record that cannot be written removes the folder and
  fails the send with `folder-not-made`), runs the folder rule on it as on a picked folder, and starts the agent there
  (`src/chat/work-scratch.ts`). The record is written before `session/new`. From then on it is a normal Work
  conversation with that folder for life, so Auto covers that folder only (an edit in another No folder conversation's
  folder asks). A folder whose encoded path is over 255 bytes is not made. A folder that cannot be made fails the send
  with `reason: 'folder-not-made'` ("The app could not make a folder in ~/Portato. Choose a folder instead." and Choose
  folder…). A create that fails before the conversation is stored removes the new folder (`rmdir`, never recursive) and
  its record; a create that gave up on a slow `session/new` does so only once that call ended (a late session is deleted
  first), so a retry in the same minute gets `-2`. Once the conversation is stored the folder stays. A conversation is
  `noFolder: true` (on `ConversationInfo` and its list row) only when its folder has a record and the folder at that
  path still has the recorded device and inode: a fork keeps the mark with its folder, and a folder picked by hand
  (`~/Portato` and the folders in it included) is never marked or removed, nor is a new folder made at a recorded path.
  The hero reads "Work without a folder" with its own starters, the composer "Describe a task", the bar chip "No folder"
  (popover: the path, "This conversation has its own folder.", Copy path), the palette "Work · No folder · …". Delete
  says the folder is removed when empty: after the agent deleted the session the Control Panel calls `rmdir` on a
  recorded folder when no other Work conversation of the index (on disk or held) has that folder and no create or fork
  is still opening a conversation in it (a fork past its soft deadline holds it until its late copy lands), so a folder
  with any file stays and a fork keeps it. A removed folder's record goes with it. The app never deletes a file there.
  The Sessions page shows such a session by its real folder.
- **Branch.** In a git work tree the Work strip also holds a branch chip (`GitBranch` and the branch; a detached HEAD:
  `GitCommitHorizontal` and the short hash; a branch with no commit yet says so in its name), cut at the end with the
  full name as its tip. In a narrow strip nothing wraps: the branch chip shrinks first (to 5rem), then the folder chip
  (to 6rem). The chip is off (greyed, `aria-disabled`, still focusable) while the work tree has uncommitted changes:
  its tip reads "{folder} has {n} uncommitted changes. Commit or stash them to change the branch.", or, when git could
  not count them in time, "Could not check {folder} for uncommitted changes, so the branch cannot change now." A clean
  tree's chip opens the branch menu (`arc-x/branch-menu`): "Search branches" (focused; a case-folded substring search
  on the page), the work tree's top folder (path as its title), the local branches (current first with a check, then
  the newest commit, ties by name; at most 1,000, "Showing the 1,000 newest of N branches."), then one muted line or
  an alert. Remote-only branches and "work in a new worktree" are not offered. The field is a `combobox` with
  `aria-activedescendant`: ↑ ↓ Home End move, Enter picks, Esc closes to the chip, Tab leaves. A branch another
  worktree has checked out is off and says "Checked out in {folder}". While a Work conversation works in the same
  work tree (a turn, wake, run, helper, history op, command or monitor, a running goal, or a create; the session's
  folder and the top are the same or one holds the other; a sibling worktree does not count) every row is off and the
  menu says "A Work conversation is running here." A pick shows the loader on its row, the other rows and Send go off
  ("Switching branch…"); on success the menu closes, focus goes to the composer and "On branch {name}." is announced;
  git's refusal ("Git did not switch:" and its text) stays in the menu as an alert. A switch the app refused before
  git ran (`dirty`, `status-unknown`, `folder-busy`) reads the branch again, and the menu shows the chip's tip or the
  busy sentence as its note. The page reads the branch again
  when Work New shows, the folder changes, the menu opens, the window regains focus or shows (at most once a second),
  and when the set of working Work conversations changes; an answer for an older folder is dropped
  (`pages/chat/work/use-work-git.ts`).
- **Picking a branch** (`src/chat/work-git.ts`, the service's `switchWorkBranch`). In the work tree's chain: the folder
  rule, then the running guard (409 `folder-busy`, no git run), then the read again: the name must be a listed
  `refs/heads/<name>` (404 `branch-unknown`, no git run; a name starting with `-` is never listed), the current branch
  answers `switched` with no run, and the uncommitted change count must be 0 (409 `dirty`; 409 `status-unknown` when
  it is unknown; no git run). Then `git switch --no-guess --no-progress <name>`, never `--force`,
  `--discard-changes`, `-c` or `--detach`, and nothing is ever stashed. Exit 0 is `switched`; any other exit is
  `refused` with git's text (`error: ` / `fatal: ` cut, a last "Aborting" dropped, at most 12 lines and 2 KB, hidden
  characters shown as `\u{…}`): a branch held by another worktree, an `index.lock`, a hook that failed. A switch past
  2 minutes gets SIGTERM, then SIGKILL 2 s later: `timeout`, "Git did not finish in 2 minutes and was stopped. Check
  the folder in a terminal." Every reply carries the read after it. While `git switch` runs, a Work create, a message
  or a queued row in that work tree is refused with 409 `folder-busy` (the page says "Git is still changing the branch
  of this folder. Try again when it is done."; a queued row holds the list), and a create that holds a folder there
  refuses the switch the same way. On New, Send and Try again wait while the page's own switch runs.
- **Rows.** Shell commands (the whole command, then its output), file reads, listings, searches and edits (the change
  as `-` / `+` lines) are rows in the thread (`ui/src/lib/work-tools.ts`); an input of another shape is a plain row.
  Hidden and bidi characters in a command or change show as `\u{…}`. A row whose input or output the transcript cut at
  16 KB says so. A command's output is the text it printed (the agent's `rawOutput` of `type: "Bash"` is never shown as
  JSON; the Sessions transcript reads it the same way); a finished command that printed nothing says "No output.", and
  one that ended with a non-zero exit code, a stop the agent named or a timeout says so on its row ("Exit code 2").
  Commands keep printing color (the agent runs them with `TERM=xterm-256color`, `FORCE_COLOR=1`, `CI=true`). The
  reducer drops the escape codes from a tool's text (`stripAnsi`: `ESC [`, parameter bytes, one final byte; `ESC ]` up
  to BEL or `ESC \`; where the agent's head-and-tail cut or a tick's last read split a code, a CSI loses only `ESC [`
  and its parameters, and an ESC left alone as the last character or right before the cut marker goes) before it
  decides the text is binary, for live and stored lines alike, so colored output reads as text, not `[binary · 4 KB]`.
  Any other ESC still marks the text binary; a `\r` redraw stays as text.
- **Edit diffs** (`src/agent-sessions/line-diff.ts`, `arc-x/diff-view`). The agent sends each edit as `diff` content
  blocks: the asked-for change on the call's start (search_replace: old and new string, with `_meta.old_line` when it
  found the match; write: the whole new text, no old text yet) and the made one on its result (search_replace: one
  `_meta.details[]` per replaced match with its new-file line, up to 3 lines around it and the start of its first
  line; write: the whole old and new file). The shared reducer turns them into `edits` on the tool entry (`readEdits`,
  then `diffEdits` in `finishTool`): per file `added`, `removed`, hunks of ` ` / `-` / `+` rows with 3 lines of context
  (changes 6 or fewer lines apart share a hunk; replace_all matches whose context lines meet share one, so no line
  shows twice; a deleted block shows no after context, since the agent's starts one line past the line that took its
  place), `create` for a result with no old text, `endsOnly` when only line endings or the last newline differ. Myers line diff after the common start and end; past an edit distance of 2,000
  the middle is all removed, then all added. Caps per tool entry: 400 rows, 24 KB, 400 characters a row (a cut row
  ends in `…`); `cut` counts the changed lines left out. The whole old or new file never reaches the page, and the
  result's `rawOutput.EditsApplied` (the files again) is dropped from the reducer's memory. A diff-only edit has no
  `output`; a failed one keeps its asked-for change and shows the agent's error as Output. The edit row reads
  "+4 −2" ("New file · +40"; "Writes the whole file · +40" for a write before its result), after "Allowed by Auto" when
  a mode allowed it, and opens to the diff: two number columns (one for a new file or a column under 520 px), red and
  green rows with hidden "Removed:" / "Added:", "n unchanged lines" between hunks, 14 rows then "Show n more lines"
  (open: 480 px, scrolls, focusable), "Copy change" (unified diff), "Not applied yet" before the result and "Not
  applied" on a failed or denied one; a call's 4th file on sits behind "n more files". While the call's request waits,
  its row opens by itself (no focus moves), so the change shows before Allow. An input cut at 16 KB (a big write) still
  gets its edit row from the first diff block's path. The Sessions transcript draws the same diff; a key in a hunk is
  redacted, and a cut row's end that could start a key is dropped. With `MLX_AGENT_E2E_BIN`,
  `__test__/chat-edit-diffs-real-agent.test.ts` runs the harness "diff" edits on the real agent and on the fake one:
  the same `diff` blocks, and the same `edits` from live lines and from `updates.jsonl`.
- **Pictures in tool results** (`arc-x/picture-strip`, `arc-x/picture-viewer`). A `read_file` of a picture or a PDF
  answers with one ACP `image` block per picture (a PDF: one `image/jpeg` per page) and the same bytes again in
  `rawOutput` (`ReadFile` `ImageContent`, or `PdfPageImages` with `page_number`, `total_pages`, `file_size`). The
  shared reducer turns the blocks into `pictures` on the tool entry (`readPictures`: `n` from 0, `mediaType`, decoded
  `bytes`, a PDF's `page`; `pdfPages` the file's page count), drops every `data` from the `rawOutput` it keeps
  (`slimRawOutput`) and prints no `[image]`: the bytes never reach an entry, the host's memory or the page's store. The
  Work read row reads "1 picture · 412 KB" or "3 of 12 pages · 1.4 MB" (the pictures' size) and opens to 96 px tiles
  ("Pictures the model read"; a PDF tile names its page). Only an opened row asks for bytes:
  `GET /api/sessions/:id/pictures/:call/:n` (worker; either agent home) has the session reader (`picture()`) read
  `updates.jsonl` with rewinds applied, take the call's newest line with image blocks (the call id compared, never a
  path) and answer `{mediaType, data}`; 404 for an unknown session, call (also a rewound one), index or a file past
  64 MiB, 415 for a type other than png / jpeg / webp / gif, 413 past 4 MiB. The page makes blob URLs, reads the
  tiles one at a time, asks a 404 once more after 1 s ("This picture is no longer on disk." with Try again; "This
  picture is too large to show."; "This kind of picture is not shown."), keeps the URLs per conversation
  (`PictureScope`) and revokes them when it closes or changes. A tile opens the viewer: a full-window dialog with
  "2 of 3" (polite), ‹ › and ← →, zoom − + and "Actual fit" (keys `-` `+` `0`), a thumbnail row, focus on Close, Esc
  closes and focus goes back to the tile; the picture's `alt` names its source ("Page 5 of docs/spec.pdf"); with
  reduced motion it only fades. It is built on the Radix dialog until the Arc Pro `lightbox-gallery` is pulled. The
  Sessions transcript draws the same tiles. With `MLX_AGENT_E2E_BIN`, `__test__/chat-tool-pictures-real-agent.test.ts`
  reads a PNG and pages 4 to 6 of a 12-page PDF on the real agent (past 10 pages the agent needs the model's `pages`,
  which both `rawInput`s carry): the fake agent's lines have its shape, live and stored give the same `pictures`, and
  the reader serves the wire's bytes from disk.
- **Permission card** (`ui/src/pages/chat/work/permission.tsx`, `arc-x/interaction-card`). One card at a time, under
  the turn, in arrival order ("1 of 3"). It shows what the agent asks for exactly: the whole command, path or tool
  input in a scrolling well, never cut, as plain text with hidden, blank-looking and bidi characters made visible (the
  server does this, the card shows its text). A request longer than 100,000 characters is denied without a card. Two
  or more empty lines in a row, or a long run of spaces, show as a marked count; a well that does not fit says how
  many lines it has and fades at its end until scrolled there. An edit is judged where the agent writes it (trimmed,
  unquoted, `..` followed on disk); one whose path leads elsewhere through a link on disk also shows where it lands, and
  one whose landing place cannot be known says "The app cannot tell which file this changes" (a link it cannot
  follow; or a missing file whose folder holds a name that differs only by a no-break space, U+00A0 or U+202F,
  against a space: the agent then edits that file). Warning lines (runs as administrator, deletes
  files, reaches outside the folder, installs software, may sign in or put a key on the command line, prints secrets,
  starts provider jobs, too long to check) block nothing; the scan also reads `sh -c` text, `eval`, heredocs fed to a
  shell and `find -exec` commands, and past a fixed step budget or nesting depth it says the command is too long to
  check. Buttons: Deny, Allow once, and the "always"
  option worded by what the agent saves ("Always allow this command", "Always allow this tool", "Allow all edits in
  this conversation"), matched by option id and kind; `reject_always` and `enable-always-approve` are never drawn.
  The agent's own wording of the "always" option is shown under the buttons; the edits option also says it allows
  edits outside the work folder. Focus goes to Deny when the card arrives, unless the user is typing in the composer.
  A button that allows ignores clicks in the card's first 600 ms, so a double click on one card cannot answer the
  next. Deny ends the turn: "You denied a request. The answer stopped." Deny is a split button: its arrow opens
  "Deny and say why…" ("Deny and tell <helper> why…" on a helper's card), a step with a field ("What should it do
  instead?", 4,000 characters, Escape goes back to the buttons, Cmd+Enter sends) and "Deny and send", off while the
  field is blank. The reply is `reject-once` with `_meta.followup_message`; the agent ends the call "not executed",
  reads the note as a user message and goes on, so the turn is not marked denied and its later requests still ask. A
  note that is only white space is a plain Deny. The host draws the note itself as a "Your note" row inside the answer
  (the agent sends no user chunk live); the call's row says "Denied · note sent". On reopen the row comes from the
  agent's log, which holds each note twice as a user chunk with no `promptIndex`: in a session that has shown a
  `promptIndex`, not a helper's, a bare chunk while a turn is open is a note and two equal ones are one row. A note
  never starts a turn and is not the turn's prompt. A helper's call denied with a note counts on its subagent row ("1
  call denied"); its note lives in the helper's own session. With `MLX_AGENT_E2E_BIN`,
  `__test__/chat-work-workflow-real-agent.test.ts` pins this on the real agent: the turn goes on, no user chunk comes
  live, and `updates.jsonl` holds the note as exactly 2 bare chunks (v0.3.4); a build that stores one fails it, and
  the pair rule goes.
  Never allow: on a shell card Deny's arrow also offers "Never allow `git push`…". Its step (Back or Escape returns)
  picks the scope: the short words (the program, plus its subcommand for git, package managers and known tools) or
  every word of the first real command (past `cd`, `export`, assignments and wrappers such as `env`, `timeout`,
  `nohup`; at most 16 words and 200 characters). A part the app hides or cannot show is not offered. The app keeps
  the rule itself in `~/.mlx-node/chat/never-allow.json` (mode 0600, written whole), per Work folder (its real path,
  compared as the disk does), at most 50 per folder. It never sends the agent's `reject-always`: that ends the turn
  on every match, and ACP has no way to list or remove it. The call is answered `reject-once` with
  `_meta.followup_message` ("I never allow commands that start with `git push` in this folder. Do not run it, …"), so
  the answer goes on; a toast "Never allow git push in this folder" offers Undo, which removes the rule of the
  conversation it was saved in, also after the page moved to another. From then on the host checks the
  rules before the mode, in Ask, Auto and Bypass and for a helper's request, for every tool that runs a shell command
  (`run_terminal_command` and `monitor`, which runs its `command` in the background terminal; `shellCommand` in
  `permissions.ts`). A command with any simple command that starts with a rule's words is denied with no card and the
  same note; a monitor's mark and history line say `access: "other"`, as its card does. A shell call whose `command`
  is not text counts as a command the app cannot read. The matcher (`never-rules.ts` `match`, a
  scanner, no regex) reads each simple command as written past its assignments, then past each wrapper it starts with
  (`sudo`, `env` with its options and `NAME=value` words, `command -p`, `exec`, `nohup`, `time`, `timeout`, `nice`,
  `builtin`; `command -v` runs nothing), and reads the line of a `sh -c`, `bash -c` or `zsh -c` with the same scanner
  (`sudo bash -c "env A=1 git push"` matches `git push`); a rule on a wrapper itself (`sudo`) still matches. A program
  written as a path matches by its name, and on macOS the program word is compared without case (`GIT push`,
  `/usr/bin/Git push`), as the disk finds it; the words after it keep their case. A `-c` line the app cannot read
  makes the command unread (a rule found elsewhere in it wins), and so do a wrapper option it does not know
  (`env -S "git push"`, a `timeout` length it cannot read) and more than 8 wrappers and `-c` lines in one simple
  command. In Bypass, a command the app cannot read, in a
  folder with rules, opens a card ("Bypass is on. This still asks: …"). An answer gets at most 3 such notes; the 4th
  match is a plain deny and the answer ends "A never-allowed command was tried again and again. The answer stopped." (`never-loop`). The call's row
  reads "Never allowed · git push" ("Never allowed · rule removed" once the rule is gone). The note row reads "Your
  rule · never allow git push" with Remove rule; Sessions draws it with no button. The mode log stores `by:"rule"`
  with the `ruleId`, never the words. A call a rule denies while the page is open is said in the status line. While
  the folder has rules the permission chip's menu has "Never allowed · N": its list shows each rule's words, when it
  was made and a remove button. `GET /api/chat/conversations/:id/never` lists the rules; `DELETE …/never/:ruleId`
  removes one (404 when there is no such rule). Every loaded conversation of the folder then gets the new count, and
  a removed rule's marks a `rule-gone` event. It is not a sandbox: a script or another program can still do the same.
  With `MLX_AGENT_E2E_BIN`, `__test__/chat-work-workflow-real-agent.test.ts` pins this on the real agent: in Bypass
  a rule denies the agent's own request with no card, its note keeps the answer going and is stored as the same
  "Your rule" row, and the 4th hit in one answer ends `never-loop`.
  A grant the agent saves lives in
  `~/.mlx-agent` and also applies to `mlx agent` in a terminal; in Work a saved shell grant does not skip the card (the
  guard asks anyway). The guard's own line at the end of the agent's title is not shown. Under the card's title, the
  agent's own title is one line, and only when it says something the well does not: on a command or file card it is
  hidden when it only names that command or file (whole, or cut by the server at 200 characters), and only the words
  after it are shown when more follow (a hook's reason). The command or file counts as named only where it ends at a
  word edge (`Run gitk` on the command `git` is shown whole). A one-word title is hidden only when it is the target or
  the agent's own verb or tool name. MCP and other cards always show it (the tool's name). The server says when it cut
  a title (`titleCut`); such a title ends with "…", at a whole word when the last word is short; the whole title is
  its hover text. A request a helper agent made (the agent's `x.ai/requester`) names it in the card's header: "Asked by
  researcher-0 · deep-research" for a helper of a workflow run, "Asked by helper explore-1" for one of no run, and
  "Works in <folder>" in place of the conversation's folder when the helper works in another one. The card reads in three groups:
  what is asked (title, description, wells), what to know (Auto's reason line, then the warnings), the answer
  (buttons, then the notes): 8 px between groups, 12 px before the buttons, 6 px before the first note.
- **Workflows** (`ui/src/components/run-card.tsx`). A run the agent starts with its `workflow` tool is one card in
  the turn that started it, kept in place as the run moves (one `workflow` transcript entry per run, replaced by each
  higher `revision`; the Sessions transcript draws the same card, read only). The card: "Workflow · deep-research",
  the status word with a dot (Running, Waiting for you while a helper's request is open, Stopping…, Paused, Waiting,
  Out of helpers, Done, Failed, Stopped, Ended early), the steps with "Step 2 of 4: Research", "5 of 128 helpers used"
  with a bar, a pause message, Helpers (one row per helper: label, state, step, tokens, time; open while the run
  works; the only place a run's helpers are drawn, see below) and Report (the run's `result_summary` as Markdown, cut at 16 KB, open when the run is Done). While a run is
  `active` and no turn runs, the card's moving dot is the only sign of work: the thread shows no wait line. A helper's
  engine requests carry the helper's own session, so they never show on the conversation's wait line, also while a
  typed turn runs: the run card is the sign. A card
  the agent still says is `active` but the host no longer holds (the agent stopped, the app restarted) reads "Ended
  early". Stop workflow is in "Stop workflow" below; no wake turn follows a stopped run. The composer's Stop stops the
  turn only, a typed one or a wake, never a run. During a run: a typed prompt is allowed, the model picker and Delete
  are off ("A workflow runs here. Stop it first."), Rename stays on, Unload waits, the agent's idle stop and a config
  restart wait for the run's end (a config restart also waits for the wake that follows a run's end, or for the idle
  stop when none comes), and an engine address change stops the run. The list and the mode switch count the
  conversation as running. Requests of the run's helpers come one at a time, as cards under the thread like any other;
  such a card does not count as the typed turn waiting, so that turn's wait line stays, and its Deny refuses only
  that helper's call ("Deny refuses this call of the helper only."), never the typed turn. When a run ends the agent may start a turn with no prompt (the wake): it shows
  the wait line and Stop like any turn, and its row reads "Woken by workflow <name>" from its first line (the host
  names the run that ended; the wake's `turn_completed` corrects it when the agent woke for something else).
- **Background work holds the agent process.** The Work host reads what each conversation's process runs in the
  background (`ActivityTracker`, `src/chat/agent/activity-tracker.ts`) from its live lines, after the live filter and
  before the page: helpers (`subagent_spawned` to `subagent_finished`; a workflow run's helpers count as the run's),
  commands and monitors (`task_backgrounded` or the tool result `BackgroundTaskStarted` / `Monitor`, to
  `task_completed` or a `background_tasks` list that ends them; a later list that says running brings one back, since
  the agent reads a list before it sends it) and schedules (`scheduled_task_created` or `_fired`, to
  `scheduled_task_deleted` with any reason but `shutdown`). It lives in host memory and is cleared when the process
  exits. Two kinds of work count: engine work (a turn, a wake, a run, a helper, a history op) needs the model, and
  process work (a command, a monitor) needs only the agent process. Engine work holds the model picker, Unload and a
  history op ("A subagent runs here. Wait for it to end."); process work does not. Both hold Delete ("A command runs
  here. Stop it first.", or the subagent sentence), the quit guard, the list's running dot, a config
  restart and the 10 min idle stop. A schedule holds the idle stop only, and not while a config restart waits. A
  config restart that waits sets
  `applyPending` ("A change applies when the running work ends."). An end line with `will_wake: true` holds the
  restart until that wake ends, and names the wake: the host's wake line carries the prompt id the agent gives it
  (`task-completed-<task>`, `subagent-completed-<subagent>`), so the row reads "A command ended: npm test", "A monitor
  ended: Watch the build log" or "Woken by subagent: Find the flaky test" from its first line. One due wake is kept, the
  latest: the agent drops many due wakes (one helper's wake reports every helper end not yet reported, and a helper
  whose end was already reported wakes nothing), so a list would shift every later name. A wake with nothing due is stamped with nothing. When the idle stop finds only helpers holding it, it asks the agent
  `_x.ai/subagent/list_running {sessionId}` (5 s, a soft call that never stops the process) and drops each helper the
  reply does not name, by child session id or subagent id, since a lost `subagent_finished` would hold the process
  for ever. The reply judges only the helpers tracked when the request went: the agent read its list before a helper
  that started since, so that helper stays.
- **Helpers tick while they run (#10).** The agent sends each running helper's numbers every ~2 s
  (`subagent_progress` on the parent session: turns, tool calls, context tokens and percent, errors, run time; at least
  one every 8 s) and stores none of them. The host takes the line before the live filter, so the reducer never sees
  it, keeps the last tick per helper in the tracker (a tick for a helper the conversation did not start, such as a
  helper's own helper, is dropped), and sends the page the whole live state as the conversation event `activity` at
  most once a second (the first change starts a 1 s timer; ten ticks in it make one event); `sync.live` carries it to
  a page that attaches later. A helper's end, the idle guard dropping it, or the process exit takes it out. The rail's
  helper row gets a third line: "Starting…" before the first tick, then "12 calls · 41% full · 18.2K tokens"
  (errors first, in danger ink; a context of 80% or more in warning ink; "Quiet for 0:35", muted, after 20 s with no
  tick), with "turn 2" on its second line; the line is the row button's description, not a live region. The thread's
  subagent row holds three facts in its head: "Running · 2:14 · 12 tool calls" while it runs (Quiet for … in place of
  Running), "completed · 2m 14s · 21.5K tokens" after its end, "not finished" when no end is stored and no agent
  process holds the conversation; the rest is a second muted line ("turn 2 · 41% of context · 18.2K tokens", after the
  end "3 turns · 14 tool calls" from `subagent_finished`, then "3 commands allowed" or "1 call denied"). Its words link
  to the subagent's own session (`/sessions/<childId>`) when the stored transcript found its folder (`exists`), and
  while it runs once its first tick came: its session is made at its start, and the Sessions page refetches it at most
  every 5 s. A row made from live lines keeps its link after it ends when `subagent_finished` counts a turn or a tool
  call; one that ended with neither has no link until the conversation is read from disk again. A workflow
  run card's running helper shows its step, its run time and tool calls, with context and tokens under it, while the
  host holds the run; a done helper keeps the numbers of `workflow_updated`. A conversation not loaded in a running
  agent lists no helper on the rail.
- **A workflow run's helpers are drawn once (`ui/src/lib/run-helpers.ts`).** A helper's `subagent_spawned` names its
  run (`workflow_run_id`); the reducer keeps it as the entry's `run` when the run's `workflow` entry came before it,
  and the retried attempt as `resumedFrom` (`resumed_from`). Such a helper has no Subagent row in the thread or on the
  Sessions transcript: its run card lists it. The card's rows are the run's roster (`agents` of `workflow_updated`, in
  its order, each the latest attempt by child id) joined to the helpers' entries, then the run's helpers the roster
  does not list, by spawn: the agent stores the roster only at the run's start, at each step and at its end, so a
  transcript read from disk can hold helpers its roster does not name. A helper the run retried is one row, its latest
  attempt; a roster row that still says running for a helper whose `subagent_finished` came after it takes that line's
  facts. A script call that resumes a helper (`agent({resume_from})`) has a roster row of its own, so both rows stay,
  each with its own call's attempts; a roster row folds into the spawn that resumed it only when that spawn is on no
  roster row's line (a roster stored before the retry). A row built from the entries sums its attempts' run times. A
  row with no end reads Running only while its run is live (not ended, and in the thread held by the host), else
  "not finished". The thread keeps a run's list of rows while they stay the same, so streamed text does not redraw
  the run cards. A card row links to the helper's session like a Subagent row, and in the thread carries the helper's allowed
  and denied calls (all its attempts'). On the rail, a run the host holds is one row with "2 agents running" in its
  detail, and its helpers are not rows of their own; a helper whose run the host no longer holds keeps its row.
- **Stop a command or monitor (T1).** Each command and monitor row of the rail has a Stop button beside its jump
  button (two tab stops; it shows on hover and focus beside the thread, always in the drawer and on touch), while the
  host's tracker of the running agent still holds its task (`live.tasks[]` names it: the row's `live`); a row the
  transcript still lists as running that the host no longer tracks keeps its jump button only. It asks
  "Stop this command?" (or monitor) "It stops now. The agent is told and may answer." with Keep running (focused) /
  Stop, then sends `POST /api/chat/conversations/:id/tasks/:taskId/stop`. The host marks the task stopping and sends
  the page its live facts at once (`live.tasks[]`: id, kind, `stopping`; not held to the one-second window, since the
  press is the user's), so the row reads "Command · Stopping…" with no Stop, then calls `_x.ai/task/kill {sessionId,
  taskId, source: "clientUi"}` as a soft call of 10 s: a reply that does not come never stops the agent and the other
  commands it holds. `killed` and `already_exited` answer `{}` and the page says "Stopped."; the row leaves on the
  agent's `task_completed`, whose `explicitly_killed` the reducer keeps as the task's `killed` (live and on disk), so
  the thread row reads "Stopped in the background" (else "In the background" while it runs, "Ended in the background"
  after its end or when no agent process holds the conversation). A UI stop is not delivered to the model, so the
  agent wakes and tells it: the wake row reads "A command ended: …". `not_found` drops the task from the host's live
  work (it no longer holds the process) and answers 404, as does a conversation not loaded in the running agent, a
  task the host does not hold, or a Chat conversation; the page shows the toast "It already ended.". An error reply
  or no reply in time gives the row its Stop back and answers 503; the page shows "Could not stop npm run dev." with
  Try again, which sends the Stop once more. When the row leaves while it holds focus, focus goes to the next row,
  else to the section heading, else to the rail. The host's `stopTask(sessionId, taskId, source)` also takes
  `teardown`, the rewind's stop: the agent counts it delivered and wakes nothing (`will_wake: false`).
- **Agents: Stop, Message and the Agent view.** Every agent the host's tracker holds as running gets the same row on
  every surface (`AgentRow`, `ui/src/components/arc-x/agent-row/`): a Task subagent in the thread and on the rail, a
  workflow agent in its run's block and under its run on the rail (`live.agents[]`: `childId` of the newest attempt,
  `run`, `attempts`, `stopping`, `messages`). Its buttons (`AgentControls`, `ui/src/lib/agents.ts` `agentButtons`) are
  always drawn while it runs and no stop is on its way: a Task subagent has Message and Stop, a workflow agent Stop
  alone (this agent build takes no message for one), and an agent that waits on a request has Review (it brings that
  card to the front) and Stop; a stopping or ended row has none. Message stays drawn but off, with the reason as its
  tooltip and description, when `messages` is `"off"` (the process answered the method as not found: "This agent
  version cannot take messages.") or `"no-address"` (the tracker keeps each live `subagent_spawned`'s `agentAddress` in
  host memory only; the agent drops it before it stores the line, and the page never gets it). Buttons are 24 px beside
  a pointer and 32 px in the rail's drawer or on touch. On a focused row X opens its Stop and M its Message; ↑ ↓ Home
  End move between the rows of a list (one tab stop per list: a run block's list, a run's agents on the rail, and the
  rail's Running subagents; rows under a closed Done fold are skipped), and Tab still reaches each button. A row's click,
  Enter or Space opens its Agent view. Stop asks "Stop this subagent?" "It stops now. If the main agent waits for it,
  it goes on without the result." for a Task subagent, or "Stop this agent?" "It stops now. The workflow gets a
  cancelled result for it, and its script decides what happens next. It may skip it, or fail." for a workflow agent,
  with focus on Keep running, then sends `POST /api/chat/conversations/:id/subagents/:childId/stop` (any attempt's id).
  The host marks it stopping and sends the live facts at once, then calls `_x.ai/subagent/cancel {subagentId}` with the
  newest attempt's id as a soft call of 10 s. `cancelled` answers `{}` (the page says "Stopped."), the row leaves on
  `subagent_finished` `cancelled` and reads "Stopped by you" (`live.stoppedByYou`; a run's script gets a cancelled
  result and goes on). `already_finished` and `not_found` drop the agent from the live facts only while the attempt
  the cancel named is still its newest (a contract retry that took the row since keeps running, and its Stop can be
  pressed again); they answer 404, as does an agent the host does not hold or a Chat conversation, and the page shows
  "It already ended.". An error reply or no reply in time gives the buttons back and answers 503: "Could not stop
  verify:claims." with Try again. When a row's buttons go while they hold focus, its label takes it; a row that moves
  into a closed Done gives focus to its list's tab stop.
- **Stop workflow.** A run the host holds that is live, or that waits on the person and is not `budget_limited`, has
  Stop on its block's head (X while the head has focus), on its rail row, in the dock strip while the rail is a drawer,
  and in the ⌘K "Agents" group ("Stop workflow {name}", a dialog). Each asks "Stop this workflow?" with "Its 2 running
  agents stop now. If you resume later, the group they ran in starts again whole." (one agent: "Its 1 running agent
  stops now…"; none: "It stops now. Finished work is kept.") and Stop workflow / Keep running (focused), then sends
  `POST /api/chat/conversations/:id/runs/:runId/stop`. The block reads "Stopping…" while its agents drain, then
  "Stopped by you in step 2, Verify". 404 "No such run." shows in the confirm alone; 409 "The run could not be
  stopped." and 503 "The agent could not be reached." add the toast "Could not stop the run" with the reason.
- **The Agent view** (`ui/src/components/agent-view.tsx`, `ui/src/pages/chat/agent-pane.tsx`). Opening an agent sets
  `?agent=<any attempt's id>` (replace), so a reload opens it again. On a chat page 1,448 px or wider it takes the
  rail's slot at 440 px; from 1,296 px it is a drawer over the docked rail, `min(560px, 100%)`; under that, a
  full-height drawer. The head has "‹ Activity" (back to the rail, with "· 1 needs you" in warning ink while requests
  wait), Show in thread, Open in Sessions and, in the column, Close; then the state word with its facts and Review or
  Stop, the run and step, model and "41K of 128K context", the turn and tool counts with "2 allowed, 1 denied" (a link
  that opens the permission lists under the row in the thread), and "Attempt 1 | Attempt 2" when a retry took the row.
  The body is one stream: "Copied from attempt 1" (folded), the prompt (or the retry's reason), then the agent's lines.
  The page watches one attempt per window with `POST /api/chat/conversations/:id/subagents/:childId/watch {window}`:
  the answer holds the agent's file plus its kept live lines, redacted, and while it runs the window's `agent-entries`
  events on the conversation stream carry its later ops, then the stored list at its end. A third live view in one
  window is 409; an agent that is not the conversation's own is 404. Closing sends `DELETE …/watch?window=<id>`
  (404, with the view left open, for an agent another conversation started), and a `sync` watches again. The foot of
  a running Task subagent the host holds, with messages on, is its message field ("Message verify:claims"; "It reads
  this at its next step. ⌘↩ sends."), kept per agent as a draft; with an open request of it the field says "It waits
  for your answer on a request. It reads this after.". Send is off for blank text and past 32 KB (UTF-8 bytes: from
  28 KB the field reads "30.4 of 32 KB", past it "Too long: 33.0 of 32 KB." in danger ink) and goes as
  `POST /api/chat/conversations/:id/subagents/:childId/message {text}`; the host sends `_x.ai/subagent/message
  {sessionId, agentAddress, content: [{type: "text", text}]}` (no `queue`: a steer read at the agent's next step) as
  a soft call of 10 s, only while it holds the agent as running and not stopping. `sent` clears the field, the toast
  "Sent to verify:claims." has Open, and "Message sent." is said; `unsure` keeps the text with the warning toast "Not
  sure verify:claims got it. Check its transcript before you send again."; `not-ready`, `busy`, `late`, `too-long` and
  `error` keep the text with the reason under the field; `off` is a toast; 404 drops the text and says "It ended before
  the message arrived.". The agent stores the message in the agent's own session only, so the parent thread gets no
  row, and the view shows it on its next watch or at the end. Elsewhere the foot is one line: "It ended. A message
  cannot reach it.", "It is stopping. A message cannot reach it.", "This agent version cannot take messages for
  workflow agents." or why messages are off. Escape closes the view and focus goes back to the row (to Message when it
  opened from there); `[` and `]` step to the previous and next agent of the same list. The agent's Sessions page shows
  the same controls and field while the host holds its conversation, else "Read from disk. Controls work while its
  conversation is open in Work." with Open in Work.
- **Monitor events (T2).** A monitor sends each output line (or a batch of lines, cut by the agent at 500 and 3,000
  characters) as `_x.ai/monitor_event {sessionId, update: {sessionUpdate: "monitor_event", task_id, description,
  event_text}}`, live only: the agent stores none (the model gets the XML copy as a reminder mid-turn, or, idle, as a
  turn of its own, `notifications-<uuid>`). The host takes the line before the live filter, so the reducer never sees
  it, and the tracker counts it for a monitor the running agent holds (any other task id, or another session's line, is
  dropped), keeping the last 5 with their host time, color codes removed and cut at 500 characters with `…` and
  `cut: true` (`live.tasks[].events`, `recent`; the page gets them in the one-second `activity` window and in
  `sync.live`; the conversation redacts the text on the way out, and for a cut event only drops an end that could
  start a key: a line the monitor itself ends with `…` keeps its end). They go when the monitor
  ends or the process exits. The rail row reads "Monitor · 12 events" (the button's description) with the newest line
  under it in the code font (`aria-hidden`); the thread row reads "In the background · 12 events" with a closed
  "Events (12)" under it: the last five, newest first, each with its clock to the second, as plain text with hidden
  characters made visible, and "Only the last 5 are kept." past five. A row read from disk has no Events. The monitors
  that reported since the last `turn_completed` become the report of the next wake (`live.reports[]`: the wake row's
  `ts` and each monitor's task id with the description its last event named, redacted, cut at 200 characters and
  with hidden characters as codes, last 20): the host files it as the wake starts, sends it at once (not held to the
  window), stamps the wake line with its own time (`agentTimestampMs`, so the row's `ts` is the report's) and, when
  nothing else is due, with `notifications-`. A `notifications` row with no ended task whose `ts` has a report reads
  "A monitor reported: Watch the build log" (two: "2 monitors reported: …"; the words are `WAKE_REPORTED` in
  `ui/src/lib/sessions.ts`) from its first line, each monitor named as its tool row names it, or by its event's
  description when the page holds no such row (or one with no words); from disk it reads "Woken by background
  notifications". The agent
  also sends a running command's or monitor's output as ticks of the call that started it (`in_progress`
  `tool_call_update`, no `_meta`, never stored), between turns too. The tracker keeps those call ids, and the host
  takes such a tick before the live filter (it still checks it for a printed provider key): it starts no wake, and
  the row stays as its stored result ended it, live as on disk.
- **Schedules: Delete, paused, running now (T3).** The tracker keeps each schedule the running agent holds with the
  run its last fire started (`scheduled_task_fired` names it by `subagent_id`, the run's child session id, live only)
  until that helper's `subagent_finished` (or the idle guard dropping it), and the next fire time that fire named
  (`next_fire_at`), and the page gets them in `live.schedules[]` (id, `deleting`, `firing`, `nextTs` once a fire
  came, until the next `scheduled_task_created`, which an update of the schedule also sends). The fire is never
  stored, so the shared reducer does not read it: the schedule's entry keeps the time its last
  `scheduled_task_created` names (made, updated or reloaded), live as on disk, and the row's "Next" reads the live
  `nextTs` first. A schedule row of the rail reads its state at its end: "Next 14:05" (or "Due") while it
  waits, "Running now" while its fire's run runs (the run is a Subagent row), "Paused" (tooltip "Runs again after your
  next message.") while no running agent holds the conversation, since the agent fires a schedule only in a loaded
  session. Each row has a Delete beside its jump button (two tab stops; shown like Stop). It asks "Delete this
  schedule?" "It will not run again." with Keep (focused) / Delete; for a paused schedule whose time passed it adds
  "Its time has passed: it may run once while it opens." (the agent fires an overdue schedule as its session loads).
  Then `POST /api/chat/conversations/:id/schedules/:scheduleId/delete`: 404 "No such schedule." for an id the
  transcript does not list as live (no load, no call) and for a Chat conversation. The host marks the Delete and sends
  the live facts at once: a conversation the running agent holds reads "Removing…"; any other reads "Opening…" while
  the host loads it first (`ensureLoaded`, the head of a turn: its folder and the engine must be there, and on load
  the agent announces its stored schedules again with `scheduled_task_created`), then "Removing…". The call is
  `_x.ai/scheduler/delete {sessionId, taskId}`, a soft call of 10 s (a reply that does not come never stops the
  agent). `deleted: true` answers `{}`, the page says "Schedule deleted.", the schedule leaves the live facts at the
  reply (it no longer holds the idle stop) and the row leaves on the stored `scheduled_task_deleted` (reason
  `deleted`); `deleted: false` answers 404 and the page shows "It was already removed."; an error reply or no reply
  in time answers 503, the row gets its Delete back, and the page shows "Could not delete the schedule" with Try
  again. A Delete in flight holds the agent process (the idle stop and a config restart wait for it). While the engine
  restarted under live commands the Delete is refused (409 `engine-lost`) and the button is off with "The engine
  restarted. Your next message restarts the agent." as its tooltip and description: the load would restart the agent
  and stop those commands.
- **The engine restarted under live commands.** An engine address change stops the agent unless commands or monitors
  run; then the process stays, the host is stale with `engineLost: true`, and a history op is refused (`engine-lost`,
  409: "The engine restarted. Your next message restarts the agent."). Only a message (a turn or a new conversation)
  restarts it, which stops those commands. The composer says so before the send, as a warning line under the turn
  line: "Sending stops 2 background commands." (one: "Sending stops 1 background command."), with "The engine
  restarted. Your next message restarts the agent and stops them." and the count in other conversations in its
  tooltip. A turn that finds the address moved by itself does the same.
- **Plan card.** The plan as Markdown in a scrolling well; Approve plan, or Send feedback… with a text field of at
  most 4,000 characters (Escape goes back). A request with no plan text (`planContent` null, absent or blank) shows
  "The agent sent no plan text." and the buttons Keep planning and Approve plan; no feedback is sent then (with no
  plan file the agent drops it). The agent's `exit_plan_mode` row reads "Plan review".
- **Question card** (`ui/src/pages/chat/work/question.tsx`, `arc-x/answer-list`, `src/chat/agent/questions.ts`).
  Work sends `askUserQuestion: true` (Chat keeps it off and answers any request `cancelled`), and one Work rule asks
  the model to ask only when a wrong guess would waste real work, at most 3 questions. The agent's
  `_x.ai/ask_user_question` opens a card: one question per step ("Question 2 of 3"), the options as a radio group or,
  for `multiSelect`, checkboxes (a radio group is one tab stop and arrows pick; each checkbox is a tab stop and Space
  toggles), numbered (keys 1–9 pick while focus is in the card and not in a text field), then an
  "Other" row with its own field (picking it moves focus into the field, typing picks it, Escape goes back to the row).
  A label ending in "(Recommended)" shows a tag, never a pick: nothing is picked on arrival, Enter on an option that is
  not picked only picks it, and Send (and Enter to send) counts only after the card has been on screen 600 ms. Send is
  off until one question has an answer; unanswered questions are left out of the reply; Skip ("Skip all") answers
  `cancelled`. The page sends indexes (`{answers: [{picks, other?} | null]}` or `{reply: 'skip'}`); the host builds the
  reply from the agent's own texts (the card shows them redacted, hidden characters as codes): the labels, own text
  beside picks as `annotations[q].notes`, own text alone as `["Other"]` plus the note. Over 10 questions, 12 options,
  1,000 characters of question, 200 of label, 1,000 of description or 16 KB of preview no card opens: the agent gets
  `cancelled` and the conversation shows "The agent's question was too long to show. It goes on without an answer."
  until the next message. A turn or wake gets 3 question calls; the 4th is `cancelled` with no card, and a new one
  settles an older open question of the session as expired. Stop, the agent's own time limit (`interaction_resolved`)
  and a closed window end it with no answer, and the status region says "The question closed with no answer." A
  question does not block a mode change. An option's preview (single select only) shows under the list as plain text
  in a scrolling well, "Preview: Sidebar layout", for the focused option, else the picked one; a pick of an option
  with a preview sends that preview back word for word in `annotations[q].preview` (the agent's own text, not the
  redacted one). In plan mode (`mode: "plan"`) the card adds "Talk it over" and "Plan now" beside Skip, each with its
  note, not armed (like Skip): the page sends `{reply: 'discuss' | 'plan-now', answers}` with the answers so far, and
  the host replies `chat_about_this` or `skip_interview` with `partial_answers`: answered questions only, one label
  string each (a multi-select question's picks joined by ", ", own text alone "Other"), no notes. A plan reply on a
  card not in plan mode is refused. The agent's terminal never sends these two; `__test__/chat-question-real-agent.test.ts`
  pins the preview echo and both plan replies on agent v0.3.4 with a scripted engine (a run on a real model in plan
  mode is still to do). The step and the half-filled answers stay in page memory by request id until the card
  closes, so the card comes back as it was after a conversation switch or when another card was brought to the front.
- **Question row** (`chat-thread.tsx` `QuestionRow`, `src/agent-sessions/ask-result.ts`). The agent stores only its
  result sentence; the reducer reads it into `entry.ask` with the call's own question texts and labels as exact
  anchors (a hand-written scanner; a pick is never guessed), the same live and from `updates.jsonl`. "Asked 2 questions
  · answered" opens to each question and its answer (picks joined by ", ", then " · " and the own text); an answer that
  reads more than one way shows as written. Other ends: "waiting for you" while its card is open, "not answered" (Skip
  or the time limit, which disk cannot tell apart), "you chose to talk it over", "you chose to plan now", "failed", and
  "not finished" (Stop, where the agent stores no result, or the app closed meanwhile); a sentence it does not know
  shows as plain text. Its texts are cut with "…": a question at 1,000 characters, an answer at 4,000, an unknown
  sentence at 16 KB. The row is never folded into "Used N tools".
- **MCP card** (`ui/src/pages/chat/work/elicit.tsx`, `arc-x/schema-field`, `src/chat/agent/elicit.ts`). In Work an MCP
  server's `elicitation/create` reaches the app as the agent's `_x.ai/mcp/elicit` (`{sessionId, toolCallId:
  "mcp-elicit-<uuid>", serverName, message, mode, requestedSchema}`, after a live-only `pending_interaction` with kind
  `mcp_elicitation`; nothing of it is stored). Chat answers `cancel`. A link request (`mode: "url"`, with `url` and
  `elicitationId`) opens the link card below. A form opens a card titled "{server} asks for input" for the typed
  turn, else the wake or a running workflow (none: `cancel`): the server's message in a well, then one field per
  property in the schema's order, as the agent reads it (`schema.rs`: at most 32 fields, a title of 128 characters, a description of
  512, 32 options of 128; a field of a kind it does not know is left out and counted under the form, "1 field the app
  cannot show was left out."). Text is an input (email, url, date or date-time box by its `format`; a multi-line box
  over 200 characters of `maxLength`); numbers a text box with a number keypad; yes/no a switch; one pick of up to 4
  options radio rows, of more a menu that starts on "Pick one"; many picks check boxes. Every box stops at 4,096
  characters, as the agent's own form does (the host refuses a longer value). Each field has its label (a
  needed one with "*", and "needed" in its name), the server's description and the host's error, tied to the control
  by `aria-describedby`. The texts are drawn as plain text, cut, redacted and with hidden characters as codes. Notes:
  "This goes to the MCP server "{server}", not to the model." and "Never type a password or a key here." Decline
  answers `decline`; Send (and Enter in a one-line field) counts only after 600 ms. The page sends one value per
  field by index (`{mcp: 'accept', values}`: text as typed, a pick as its option index, a date and time with the
  local offset); the host checks them as the agent does (`validate.rs`: text not trimmed, numbers trimmed, whole
  numbers in i64 and within ±2^53, the formats, the bounds; an empty optional field is left out) and builds the
  content from the server's own names and values. A refused Send keeps the card: the reply is `{fields}`, one error
  or null per field; each wrong field shows its words and `aria-invalid`, "2 fields need a change." is said at once
  and focus goes to the first wrong field. Typed values stay in page memory while the card is open (shown again, it
  keeps them) and are dropped when it closes, however it closes; never in a log, event or draft store. A form the app cannot read (bad schema, over the limits) or with a needed field of a kind it
  does not know offers Decline only, with "The app cannot show this request (…)." Stop, the agent's own end of the
  wait (`interaction_resolved`) and a closed window answer `cancel`; the status region says "The MCP request closed
  with no answer." The card does not block a mode change. Its rail row reads "MCP server" with the plug icon; its
  name is "{server} asks for input, MCP server, waiting {time}" (the server asks, not the agent). A request that came
  while no typed answer or wake ran is the running workflow's (`McpInteraction.run`, the run's name; the request names
  no helper): its rail row reads "MCP server · Workflow {run}" (named "…, MCP server, Workflow {run}, waiting …"),
  and like a run helper's ask it leaves the composer alone (`askRun`, `ui/src/lib/activity.ts`). Every card's notes
  (these two, the link card's, a question's, a command card's) are read with the card: `InteractionCard` adds them
  to its `aria-describedby` after the asked-about text, the warnings and the folder.
- **MCP link card** (`McpLinkCard` in `elicit.tsx`, `linkCheck` in `src/chat/agent/elicit.ts`). A link request opens a
  card titled "{server} asks you to open a page", owned as a form is: the server's message, the whole address in an
  "Address" well and "Site: {host}" under it. The host checks the address as the agent's own client does
  (`check_elicit_url`): it must parse, be `http:` or `https:`, name a host and hold no user name or password, at most
  2,048 characters; the card shows the parser's form of it (spaces trimmed, the site in lower case, an international
  site as `xn--…`, other characters percent-encoded), which is exactly what opens. Plain `http:` adds "This address
  is not encrypted (http)." A site with any part in its `xn--` form (`link.lookalike`) adds "This address uses
  letters that can look like other letters. Check the site name." An address that fails, or that holds a provider
  key, shows no address and offers Decline only ("The app cannot show this request (its address is not a safe web
  address)."); so does a request with an empty `elicitationId`, which the agent passes on (before, the host answered
  it `cancel` unseen). Focus arrives on Decline; the
  status region says "MCP server {server} asks you to open a page. Answer in the conversation." **Open page** is a
  real link (`ButtonLink`, `target="_blank"`, named "Open page in your browser, {host}") that the window's external
  link policy opens in the browser; a click counts only 600 ms after the card shows and only once, and it sends
  `{mcp: 'accept'}` with no values, which the host answers `{outcome: 'accept'}` with no `content`: the page was
  opened, not that sign-in worked. A middle click never follows it. Note: "The page opens in your browser. The app
  does not see what you do there. Check the site name first." The host keeps the server and `elicitationId` of each
  page opened in the agent process's life (two servers may use one id; at most 32 per conversation, forgotten when
  the process exits); the agent's `_x.ai/mcp/elicit_complete` (`{sessionId, elicitationId, serverName?}`, the
  server's `notifications/elicitation/complete`) gives one "finished in the browser" line only for such a page, once,
  and only when its `serverName` (when sent) is the server that asked. A declined, stopped or unknown id gives none. The
  address and the id are never stored.
- **MCP line** (`chat-permission-marks.tsx` `McpLine`). Each closed MCP card writes an `mcp` line to the permission log
  (`{kind, at, server, mode, outcome}`: the server name with hidden characters as codes, at most 64 characters; never a
  field or a value) and sends its mark: "github asked for input · sent" (or "declined", or "closed with no answer"),
  "github asked you to open a page · opened" for a link, and "github · finished in the browser" (outcome
  `completed`, a link only) when its server says the page is done; in time order, live and after a reload. A line
  inside a rewound span is not drawn.
- **Plan first** (`PlanChip` in `ui/src/pages/chat/composer-tools.tsx`, Work only, after the permission chip). It is
  turned on by `/plan` (app command, below) and the chip is drawn only while it is on or the agent is in plan mode;
  off and not planning, it is not in the toolbar. Pressing it off while the agent is not in plan mode removes it; while
  the agent still plans it stays, unpressed. One
  toggle (`aria-pressed`) holds what the next message carries: pressed sends `plan: true`, else `plan: false`. It
  follows the agent's mode: each new mode line sets it (on for plan), and a press holds until the next mode line. The
  label reads "Planning" while the agent is in plan mode and it is pressed. It stays live while an answer runs (tooltip
  "Applies to your next message."); the press lives in the page's draft store. Every Work prompt names its mode in
  the request's `_meta.mode` (`plan` or `agent`, never on a block); the automatic `continue` names none, so the agent
  keeps the mode it has; Chat sends none and a Chat body with `plan` is 400. The agent's
  `current_mode_update` (sent and stored) becomes a `plan-mode` entry (`plan`, `default`, `ask`; an unknown id reads
  `default`; a repeat of the mode makes none), drawn as a rule line "Plan mode on" / "Plan mode off" / "Ask mode on"
  (Sessions too) and announced when it arrives. Like a notice it never starts a turn: the agent stores the mode line
  before the prompt's chunk, and the reducer holds it until right after that prompt, so live and disk give one list.
  A `plan: true` message that ends with no mode line, sent while the agent was not in plan mode, makes the notice
  `no-plan-mode` "This agent version has no plan mode." and turns the chip off; the next message clears it. Until the
  stream brings the conversation the page cannot know the agent's mode, so Send, Try again and Queue wait ("Loading the
  conversation…") rather than send `plan: false`.
- **Thinking** (the Thinking effort column of `ui/src/components/arc-x/model-picker/model-picker.tsx`, Chat and
  Work). The model menu's last column: the levels of the quantization in view, one row each ("Off · Low · Medium ·
  High" for the Qwen line, two rows for a model with two), with a check on the level a pick of that quantization keeps
  (the conversation's level on the conversation's base, stepped as below, else the model's default); a model with no
  levels has no column. The ⓘ in its heading holds "Less thinking is faster. Off answers at once." ("Less thinking is
  faster." for a model with no Off; "Applies to your next message." while an answer runs). The levels are the ones the agent lists for each model at its start (`_meta.reasoningEfforts` of
  `modelState.availableModels`, its copy of the engine's `/v1/models` rows), as `ChatHostState.models[].thinking`
  (`src/chat/thinking.ts`), kept in memory. Until an agent of this app run lists a model, the host gives it the levels
  of its family (`levelsFor` of `@mlx-node/server/host/thinking-levels`, keyed by the endpoint row's `modelType`, which
  is the engine row's `model_family`): the same levels the agent will list, so the row shows on the New screen before
  the first message. An agent's list replaces them; a model whose endpoint row names no family has no row until then.
  The chip's badge shows the level of the model (the default until one is picked; while an answer runs it shows the
  picked one), set apart from the quantization, which stays in the title, the name and the menu. A level of the
  conversation's model says "Thinking: Low." and PATCHes `{thinking}` (the New screen keeps it per mode beside the
  model and sends it with the first message); it stays live while an answer runs, when only the model rows are off and
  only the conversation's model has the column. A pick of another model carries the level picked in its column. The page PATCHes `{modelId}` first (the host steps the stored level to the new model,
  below), then `{thinking}` when that step lands elsewhere, so the picked level wins; a refused model change sends no
  level. The New screen stores both. The level lives in `composer/<id>.json` (`thinking`; absent: the default). Before each
  prompt, after the load and its guard, the host sends `session/set_config_option` `{configId: "reasoning_effort",
  value}` (soft 10 s) when the session holds another level: the session reply's `configOptions` say which it holds,
  a live `config_option_update` (taken by the host, never a transcript line) says it again after a set or a model
  change, and a conversation with no pick wants the model's default (the agent's process default may be another
  conversation's level). A stored level the model does not list (a model change made before an agent listed the
  levels) goes stepped as the row shows it: the nearest level below that the model has, else its default. A refusal,
  a miss or a level the session does not offer makes the notice `thinking-not-set` "Thinking level not set. This
  answer runs at the agent's level." (a warning dot on the chip); the answer goes on, a late OK records the level,
  and a later set that took, a next answer whose session already holds the level, a level change or a model change
  clears it. A model change on the host steps the level down to the nearest the model has, else to its default; the
  status says where a pick lands. A
  fork's copy keeps the level and starts with no queue (rows held in the original stay there).
- **Queue** (`src/chat/queue.ts`, `src/chat/composer-store.ts`, `ui/src/components/arc-x/queue-list/`, Chat and Work).
  While an answer, a wake or Compact now runs, Enter queues the message (⌘Enter too in Chat; in Work it is Send now,
  below): the round button reads Queue (tooltip "Runs when this answer ends (Enter)") and Stop moves to a small square
  before it; with no text the round button is Stop, as before, and Esc still stops. The page posts the prompt with
  `whenBusy: 'queue'`; the host keeps the
  row (an idle conversation runs such a message at once). A row carries its files and, in Work, the Plan first value it
  was queued with. The list sits in the composer above the files: "Queued · n", up to five rows (three in view), each
  with its text (a click opens the edit: Enter saves, Shift+Enter is a new line, Esc leaves; while it is open, Esc in
  the message box also closes it, and only the next Esc stops the answer), ✎ and ✕; Clear empties it. The host ends
  every edit when no page watches the conversation (a hidden window, a resync); a later view that shows the row still
  open in this window without the host's hold holds it again (`PATCH … {editing: true}`), except while its save is out.
  A save whose row had already left the queue (sent or removed) is no success: its text goes to an empty message box
  ("… Your edit is in the message box.") or the toast says it was not saved. A queue or Send now the host refuses
  puts the message back into an empty box; with a newer draft there it stays as the place's failed message with Try
  again, which queues it while the answer runs and sends it once the conversation is idle. A sixth row is
  refused: the reason slot says "5 messages wait. Remove one to queue more." and Queue is off. The page counts as the
  host does: `QueueState.count` holds the listed rows and the row whose turn starts (out of the list, still in the
  cap), so four listed while one starts read as five; in Work ⌘Enter text
  that can go now still goes (a Send now takes no place in the list). When an answer (or a
  wake, or Compact now) completes, the host sends the first row through the same checks as a typed message, as a new
  turn with its own model select and folder check; one at a time, oldest first. The row leaves the list as its turn
  starts but stays in the host and in `composer/<id>.json` until its `session/prompt` went to the agent (the turn's
  `running` phase): a Stop or a failure before that (the model select, the session load, an unsent end) gives it back
  in its place, with its files, under the hold that end makes. A drain asked for while another checks its row (a
  Remove of that row, Send next) runs once that check ended, so the next row never waits for nothing. It waits while a
  row is open for edit (a
  page that leaves gives the edit back), while a wake is due (a command, helper or run ended with `will_wake`; the row
  goes once that wake ended, or once the agent process exited with that wake still due: `onWakeDropped`) and while the engine restarted under a live command (held "Held: the engine restarted.
  Sending stops n background commands.", counting every Work conversation's and naming how many run in others, since the
  send would restart the agent). The user's Stop holds the list ("Held: you stopped the answer"), so does any end but
  `completed` ("Held: the last answer failed"), a failed check ("Held: {its sentence}", the row stays) and an app
  restart ("Held since the app restarted": the rows come back from `composer/<id>.json`, never sent by themselves). Send
  next ends the hold and is the user's yes. A held list never blocks the box: with the agent idle, Enter sends the typed
  message at once and the held rows stay under it. A message typed while a history op (Compact now, Fork, Rewind) or a
  recap runs would have gone at once without it, so it queues unheld (`QueueRow.unheld`, memory only, never stored):
  when the op ends it goes past the held rows, which stay held; a hold made after it (the op's Stop or failure, a later
  Stop) holds it with the rest. The rows live in the host (the agent's own queue would start the next
  turn with no host step between); `_x.ai/queue/*` stays unused. The page's view is redacted; files show by name and
  size.
- **Send now** (Work only; `AgentHost.steer`, `ChatService.queueSend`). While an answer or a wake of the conversation
  runs past its start, each queued row has Send now (an icon under 520 px of composer; tooltip "The agent reads it at
  its next step (⌘Enter)"), and ⌘Enter (Ctrl+Enter) in the box sends the typed text the same way (`whenBusy: 'now'`).
  The host sends `_x.ai/interject {sessionId, text, interjectionId}` as a soft call of 5 s (`STEER_MS`); the agent
  answers `{result: {status: "queued"}}` at once and reads the text after its next tool batch or before the answer
  ends. It goes only while a prompt is open: not while the turn starts (the row's Send now is off, "Wait until the
  answer has started."), not between the prompts of an automatic continue, not during a history op, never with no
  answer (the agent would run it as a turn the host did not start). A row with files ("A message with files runs as its
  own turn.") or starting with `/` ("A command runs as its own turn.") cannot go now. Taken: the row leaves and the
  host pushes the row itself (`Conversation.steerSent`, a `user_message_chunk` with `_meta.interjection`; the agent sends
  no live chunk, and its `_x.ai/session/interjection` echo stays unread), a bubble inside the running answer with the
  caption "Sent while working · 14:02" always shown under it
  (`Message` `caption`, not the hover footer). The agent stores it only when the model read it (`_meta.interjection`,
  the wrapped text, the typed text in `content._meta.displayText`), which the reducer reads as the same entry, maybe a
  few rows lower. A Stop drops a steer the model did not read: the live bubble's caption then reads "The answer
  stopped. This may not have been read.", and a reopen from disk has no such row. A steer that missed the answer's last
  read runs as its own turn (`interject-fallback-<uuid>`): live a wake row "Your message ran as its own turn"; stored
  as a typed prompt. Refused while the answer still runs (its start, the gap of an automatic continue, a history op,
  the agent's error): 409 `steer-busy`, the row stays, a warning toast "The agent cannot take a message at this step.
  Try Send now again in a moment."; ⌘Enter text then joins the queue (`{queued, steer: 'refused'}`, the same toast),
  or with five rows waiting goes back to the box with that sentence. Refused with no answer running: 409
  `steer-not-now` and no toast, since the row's Send now is gone by then (it shows only while an answer runs) and the
  list sends the row by itself, or its held header says why it waits; ⌘Enter text then starts a new answer. A second
  press while the row's first Send now waits for its reply: 409 `steer-late`. No reply in 5 s: 503 `steer-late`, the
  row stays and the list does not send it until the reply comes ("The agent did not answer yet. The message stays
  queued."); a late yes still moves it into the answer, but a row edited meanwhile stays queued with its new text.
  ⌘Enter text with five rows waiting has no row to wait in: the call waits for the late reply. Chat has no Send now (409 `steer-work-only`): a steer that missed its answer would be a turn with no
  typed message, whose first web call Chat refuses; there ⌘Enter queues. A steer is not an allow and resets nothing in
  Auto.
- **Slash commands** (`src/chat/commands.ts`, `ui/src/components/arc-x/command-menu/`, `ui/src/pages/chat/slash.ts`).
  Text whose first word starts with `/` is a command of one of two classes, decided by name before anything is sent.
  _App commands_ run in the app and are never prompt text: `/compact` calls `historyActions.compact` (Compact now),
  in Chat and Work; it is never queued, and while its op is off the reason slot says why ("Nothing to compact yet.")
  and Send stays off. `/context` (also typed `/usage`, `/cost`) opens the Context panel (`historyActions.showContext`),
  also while an answer runs; a new conversation has no panel ("Send a message to see the parts."). `/rewind` (also
  typed `/undo`) opens Rewind on the last sent message. `/fork` forks the whole conversation ("Nothing to fork yet." on
  a new one). `/recap` asks for a recap (`historyActions.recap`; "Nothing to recap yet." before the first finished
  answer). `/plan` (Work only; not in Chat's list, and typed there it is an unknown command like `/goal`) toggles the
  "Plan first" value (off to on, on to off; "Plan first: the agent plans before it edits."), also on a new conversation
  and while an answer runs; it needs no conversation. An app command takes no words: when it runs, the `/` word leaves the box and the words after it stay
  (`slashRest`), so the "+" menu's Commands row, which puts `/` before the draft, never costs the draft. The host
  refuses every app command as prompt text. _Agent
  commands_ (Work only) go through `POST …/prompt` as typed, a queued row too, and the agent expands them: `/goal`,
  `/loop`, `/deep-research`, `/workflow`, and the user's skills and workflows from the agent's list. A built-in goes
  only when the session's own list offers it (the agent lists one only while its gate is on: `/loop` needs the
  scheduler); else it is refused as an unknown name, and the page reads it as unknown once the list came. `parseSlash` is
  the agent's own parse (`parse_slash_prefix`: trimmed, the name up to the first white space, no `/` or `\` in it,
  lower-cased). The allow rule (`checkCommand`) is static and checked first, aliases included: the
  app commands (also typed `/usage`, `/cost`, `/undo`), the blocked built-ins (`/always-approve` and `/yolo`, which turn
  every permission request off; `/hooks-*`, `/plugins`, `/reload-plugins`, `/memory`, `/flush`, `/dream`, `/feedback`)
  and the hidden `/session-info` (`/status`, `/info`) are refused in both modes by the route itself; Chat sends no agent
  command; in Work a name no listed allowed built-in, listed skill or listed workflow has is refused (`slash-text`, "No
  such command here."), never passed to the model as text. `AGENT_BUILTINS` lists the agent's 19 built-ins and 5 aliases;
  `__test__/chat-commands.test.ts` pins it to `__test__/fixtures/agent-builtins.json`, taken from the agent source, so
  a built-in the agent adds fails until it is classed. The list: a Work session's `available_commands_update` push is
  read before the live filter and is never a transcript line (the agent never stores it). After each answer the
  agent sends it again to carry the answer's token count (`_meta.totalTokens`, its `UsageMeta` refresh); such a line
  goes on to the reducer for the context count alone (no entry). After each load the host
  pulls `_x.ai/commands/list {sessionId}` as a soft call of 10 s (`COMMANDS_LIST_MS`; a miss keeps the static list);
  the page gets the rows it may send as the `commands` event and `sync.commands` (each word one line, cut at 80
  characters, redacted, hidden characters made visible). The first push comes before `session/new` answers and is
  dropped, as any push for a session the host does not hold. When the agent process exits the host sends `commands:
  null` for each session that held a list, so the page's `/` list goes back to the static one until the next load
  pulls it again. A cold Work conversation lists `/compact`, `/context` and
  the four built-ins with "… More after the first message"; a skill or built-in typed in full may still go, and the host
  checks it once the session loaded (the pull is awaited): a name not listed ends the answer unsent with `slash-text` before
  `session/prompt`. Chat takes the agent's push and lists none of it. The page: typing `/` opens the list over the
  composer (Commands, Skills, Workflows; it never takes focus, the text field carries `aria-expanded`,
  `aria-controls`, `aria-activedescendant` and `aria-autocomplete="list"`); ↑ ↓ move and wrap, Enter or Tab fills
  `/name ` and sends nothing, a second Enter runs it, a name typed in full runs on the first Enter, Esc closes the list
  only; another name a command is typed as (`/usage`) is typed in full too. A filled command shows its words in the
  reason slot (`<goal> [--budget <tokens>]`, or "Enter to run"), which leaves Send on. "No such command here." is
  said (polite) once the list has nothing for the typed name, not while a name that matches is still being typed. The "+" menu has a Commands row ("Type / in the message box", no key: `/` outside a text field is
  the page search). A built-in such as `/goal pause` is the agent's host turn: the reply is a `hostTurn` agent chunk and
  the typed text is stored as a `hostTurn` user chunk, so live and reopened threads show the same bubble and answer.
  `__test__/chat-commands-real-agent.test.ts` pins this on the real agent: the pull after the load and the push after
  an answer give the same rows (the four built-ins and a skill from the Work folder's `.grok/skills`), the fake agent
  plays the same rows, `/yolo` and `/compact` never reach it, a skill goes as typed and the agent adds its body, `/goal`
  and `/workflow runs` are host turns with no model call, and the live rows equal the rows read back from disk.
- **Permission mode** (`ui/src/pages/chat/work/permission-mode.tsx`, `src/chat/agent/auto.ts`,
  `src/chat/agent/auto-scan.ts`, `src/chat/permission-log.ts`). Each Work conversation has one of three modes, the
  chip before Plan first in the tools row (and, while not Ask, a chip in the thread header): **Ask** (every request is
  a card), **Auto** (the app answers plain commands and edits inside the folder itself; the rest is a card) and
  **Bypass** (the app answers every request itself). These are the app's modes: the agent's own modes
  (`--permission-mode` and its always-approve switch) stay off, and the agent is told nothing in any mode. It
  asks as always, the app's guard hook still makes every shell call ask, and the app answers allow-once (never an
  "always" answer). A new conversation starts in Ask. A move up (Ask → Auto, Ask → Bypass, Auto → Bypass) needs a
  confirm text in the chip's menu or the palette's "Permission mode: Auto…" / "Permission mode: Bypass…" dialog, and
  counts only a trusted click made 600 ms or more after the text it confirms was shown. A move down is one click (chip,
  header chip, palette), also during a turn, and wins over a move up sent before it (that one is refused with
  `mode-changed`). A move up is refused while the agent cannot run, while the folder has a problem (on the New screen:
  none chosen, or too long), while the first message is on its way and while the port is down; Ask → Auto also while a
  request of the conversation is open (409 `card-open`). A move up
  writes its mode line first; when it cannot, it is 503 and the mode stays. `POST
  /api/chat/conversations/:id/permission-mode {mode, up}` is the only way up, and only with `up: true`, which the page
  sends from a confirm's action alone. Every other pick is sent with `up: false`: below the mode it is a move down; above
  it (the page still showed an older, higher mode, for up to the host state's coalesce or until a reconnect after a
  Control Panel restart) it moves nothing, calls back a move up that waits, and is 409 `mode-changed`, which the page
  shows as no toast while the stream brings the mode. A request sent with `up: true` to a mode below the current one is
  a move down. The mode lives in the Control Panel's memory, is never read from a file, and is Ask for every
  conversation after the Control Panel restarts (quitting the app does that; the thread says so). A command that
  already runs is not stopped by a move down.

  _Before the first message._ On the New Work screen the chip works the same way, for the conversation the first
  message makes. The page names a draft id (made in the window, never stored); the confirm's action sends `POST
  /api/chat/drafts/:draft/permission-mode {mode, up: true, cwd | noFolder: true, temp?}` and pins the place it named.
  The Control Panel checks the folder as a create does (No folder: the temp root of a folder not made yet in
  `~/Portato`) and keeps one pick in memory (`ChatHostState.permissionDraft`), bound to
  that draft id and place; a second pick, from any window, replaces it. A No folder pick is taken only by a `noFolder`
  create, a folder pick only by a create in that folder. The first message names the draft
  (`draft` on create), never a mode: after the session exists the Control Panel takes the pick when the id and the
  folder match, writes the mode line (`from: ask`) and only then starts the turn, so the first turn shows the mode
  line before its first call. A pick it cannot apply leaves the conversation in Ask, and the page shows a toast when the
  reply's mode is below the one the chip showed. The pick goes back to Ask when the folder changes, the page goes to
  Chat or leaves the New screen (the page posts `{mode: 'ask', up: false}` for the old id), the last window closes or
  the Control Panel restarts. A send that fails before the conversation exists keeps it. While the first message is on
  its way the chip keeps the mode it showed and only a move down is sent, always to the draft id the first message
  named (a folder picked meanwhile does not change it); one that arrives after the pick was taken is applied to the new
  conversation. The Control Panel keeps that draft-to-conversation link apart from the one pick, so a later pick or
  the last window closing does not drop it; it goes when the conversation is deleted. The confirm reads the hooks line from `GET /api/chat/work/hooks?cwd=` (No folder: `?noFolder=1`, read for a
  folder of `~/Portato` that is not made yet) and starts its
  arm clock when that answers; its last line says the pick is for this new conversation only.

  _On the page._ The chip shows the mode the stream says, never its own click: Ask is a plain chip, Auto has a quiet
  fill, Bypass the danger border, tint and ink, and keeps its word where a narrow row drops the words of the other
  chips (the header chip drops Auto's word below 45rem). Its menu lists the three modes with one line each; a lower
  mode is sent at once (also while a move up is still on its way), a higher one turns the menu into its confirm text
  (Back, then the action; Escape closes it and sends nothing). A higher row is disabled with its reason as tooltip.
  The Bypass text says what it allows, that your hooks are answered Allow when the app found hooks of yours, and how
  many open requests it allows at once. The arm clock starts again when the text changes (the folder, the key
  sentence, the hooks line, the count of open requests). In Bypass the composer's placeholder reads "Reply. Bypass is
  on.", and a card that is open shows Deny and Allow once only (with no "Allow once" offered: Deny and the line
  "Bypass is on. This still asks: …"); back in Ask it shows every option the agent offered. The status region says
  the new mode when it changes; single allowed calls are not announced.

  _Auto._ For a file edit that lands inside the folder and for a shell command written plainly, the gate
  (`src/chat/agent/auto-scan.ts`) allows only what it models in full (a small subset of shell text, a known command,
  only the options on that command's table, and every write placed inside the folder on disk; a disk read that fails
  with anything but "not there" asks). Some of its rules a user can meet: `printf` passes only with `%s`, `%%`, `\n`,
  `\t` and `\\`; `mv` with more than one source asks; a pattern with a non-ASCII character asks; a project tool's
  operand must name an existing file or real folder with no link on its path, so a test selector (`a.py::test`,
  `a.test.ts:3`) or a filter that names no file asks, and `python -m unittest` passes only with no operand; `test` and
  `[` place the files they look at; every git call asks unless `.git` is a real folder (no `commondir`, nothing under
  `modules`, no alternates, no link among its entries, no submodule in its index) whose config holds only keys on the
  app's list (core basics, user, remotes and branches, pull/push/fetch/merge/rebase basics), and no other repository
  lies between the Work folder and the folder git starts in (`cd sub && git status`); fetch, pull and push also need
  every remote to be a network URL. The global and system git config and git variables of the login environment are
  not read: a credential helper, signing, or filter and diff drivers set there still run. While a call that may change
  the folder runs, in any conversation on that folder or on a folder that holds it or that it holds, and until the
  agent reports its end (a Stop does not end it), every other call there is a card, and a call that may change the
  folder (every edit and write) is a card while any call runs there; a call the user or Bypass allowed counts for
  every Work folder. A Bypass call that can go on after it reports its end (a background command, a lone `&`, a helper,
  a workflow, a monitor, a schedule) counts until the agent process exits (its idle stop after 10 minutes, a restart or
  a crash). Facts read while such a call started or ended are read again once, then the call asks. A project tool
  still walks links that are already in the folder (`pytest`, `vitest run` with no operand, `npm test`). Every other
  request is the card with one line that says why it still asks (a command that names a key variable and has no
  warning line says so in that line); while the mode is Auto it answers only with Deny and Allow once. The guard hook
  must have asked for each shell command (this start's name). At most 100 calls pass this way in one answer: a call
  counts when the app decides, so calls that arrive together (parallel tool calls, helper agents, whose requests carry
  the conversation's session) cannot pass the cap; the cap counts per owner (the typed turn, the wake, or each run);
  with no window open, every call is a card. Starting a workflow or a helper is a card once (`helper-start`: "this
  starts helpers that run on their own. Each of their calls is checked like yours."); after that each helper request
  is judged like the conversation's own, by the folder the request names: a helper whose folder is not the Work folder
  asks (`helper`: "this helper works in another folder."), per request. A helper's edit inside the Work folder is
  allowed with no card, as for a typed turn (also from a workflow whose prompt says not to edit). A check that fails with
  an error is a card (`unread`).

  _Auto and the temp folder._ The places Auto answers for are a closed list of roots, given as data by the host
  (`src/chat/agent/roots.ts`): the Work folder first, then the temp roots of `ChatServiceOptions.tempRoots` (tests
  pass `[]` or a throwaway root). The default is `resolveTempRoot('/tmp', home, appTemps)`
  (`src/chat/work-folder.ts`): `/tmp` with its real path (`/private/tmp` on macOS) as a second spelling, only when it
  is a root-owned folder
  with the sticky bit, the home folder neither holds it nor lies in it, and no temp folder of the app or the agent
  (`os.tmpdir()`, the Work environment's `TMPDIR`) is it or lies in it. A temp root the Work folder holds or equals is
  dropped for that conversation; a Work folder under a temp root is still the Work folder (its rules, by either
  spelling). With a temp root, a shell read, a write and a file edit there run like one in the Work folder, and also
  an absolute `cd` or reader pattern into it (into the Work folder those still ask). Every place in a temp root is
  walked as written, link by link: each part must be yours, in folders no one else can write (mode bits; an ACL is
  not seen), a real folder or a followed link with a regular file or folder at the end, on the root's disk; a
  first-level name, also one the call would make, may not be a dot name, `com.apple.*` or carry your uid or login as a
  piece (names and the app-data folders are compared case-folded); the top itself, app or
  agent data, and every project tool or script that starts there or is named there (it looks for config in the folders
  above, up to the shared top) ask with the reason `temp`. git under a temp root uses the nearest folder below the top
  that holds `.git`: a folder on the way with `HEAD` asks, `.git` must hold `HEAD`, `objects` and `refs` in the form
  git accepts, and `.git` and its entries must be yours and not writable by others, beside the Work folder's checks.
  After an absolute `cd` into a temp root, the parts joined by `&&` in the same list (no `||` in it) are checked in
  that folder only. All of the temp root is one place: while a call that may change it runs (every project tool run
  counts), another Auto call that reaches it asks. Limits: a link or file one of your own programs changes between the
  check and the run is not seen, a program's folder with a plain name reads as yours, and a first-level name that does
  not exist yet can be taken by another user first. Three list rules hold in every root: `ps` passes with the forms of
  its table (`aux`-style letters `a u x w`, dash clusters of `A a x f w`, `-p` ids, `-o` keys; any `e` form asks), an
  `echo` word is text unless it is `-` and letters all of `n e E` (then only `-n` passes), and `sed` passes only as
  `sed -n` with one print script (`Np`, `N,Mp`, `$p`, `N,$p`). The host state carries `tempRoot` (`/tmp`, or null
  with no temp root). The Auto confirm then says "Edits in <folder> or /tmp …" and adds one line: commands read and
  change your own files in `/tmp` with no card, while project tools, scripts, deletes, the top of `/tmp`, and folders
  that look private or are not yours still ask; the mode's tooltip and menu note say the same in one sentence. The arm
  clock starts again when `tempRoot` arrives. The confirm sends the temp folder it named (`{mode, up, temp}`); a move
  up to Auto whose text did not name the temp folder the conversation gets is refused (409 `temp-unnamed`), so no
  conversation is widened without the text. That mode line and its mark carry `temp` (also a move down from Bypass to
  Auto), and the mark reads "Edits in the folder or /tmp and known commands run without asking." A card in Auto names
  the part that asks (`autoAskDetail`): `Auto is on. This still asks: "ps" is not on the app's list.`, `"…" and 2 more
  parts are not on the app's list.`, `the app cannot place "…" inside napi-rs or /tmp.`, `other users can write to a
  folder on the way to "/tmp/x".`, and one line per cause of the temp rules (`top` and a global Git config name no
  part); a card with no detail keeps the general line. An allowed call whose real place is in a temp root carries
  `temp` on its mark and its history line, and its row reads "Allowed by auto · /tmp"; a call in a Work folder that
  lies under `/tmp` reads "Allowed by auto". The dev harness spells a folder of its fixture home `/tmp`
  (`harnessTempRoot`): Work prompt `longcmd` is allowed in Auto, `tmpshared` shows an allowed read and then a card.

  _Bypass._ Every request of the conversation (shell commands of any form, file edits anywhere, MCP tools, helper
  agents, workflows, requests a hook of the user asked for, tool names the app does not know) is answered allow-once
  with no card, no gate, no protected names, no folder limit and no per-answer cap, also with no window open, also for
  a command too long to show (Ask and Auto deny that one). A plan still waits for you; a request that does not offer
  "Allow once" is a card (only Deny answers it while the mode is Bypass); one that offers neither "Allow once" nor
  "Deny" is cancelled in Auto and in Bypass. A card that is open when you switch to Bypass is allowed at once.

  _The record._ Before each allow of Auto or Bypass, a line is written to `~/.mlx-node/chat/permission/<id>.jsonl`
  (time, mode, tool call id, kind, and for a helper agent's call its session id; no command, path or title); a failed
  write shows the card and moves the conversation to Ask. The mode is read again after that write and after every
  other wait: a request is decided under the mode it has when the app acts, and a line whose call got no allow (Stop,
  a mode change in that instant) is taken back by a `withdrawn` line, so a reload does not list it. A deny with a note
  writes a `denied` line (time, tool call id, kind, `by: "note"`, the helper's session id; never the note) after the
  reply is sent. A line written in
  the instant the Control Panel stopped can still be listed for a call that never ran. In the thread an allowed row
  says "Allowed by auto" or "Allowed by bypass" and opens to the text its card would have shown. A helper agent's
  calls fold into its subagent row: the row counts them ("14 commands allowed") and a closed "Allowed calls"
  disclosure under it lists their lines. A call with no row of its own and no subagent row (a line written before the
  session id was kept, a helper's own helper) is a line. A mode change is a mark in time order: a hairline across the
  thread with the icon, "Mode: Ask|Auto|Bypass", one short
  sentence and the time centered in it, on one line (`arc-x/thread-mark`, `role="note"`, no box, no tab stop; Bypass
  in the danger ink with red-tinted hairlines). A second, centered line holds only what is shown nowhere else: Bypass →
  Auto "Everything else asks again. Running commands are not stopped.", the "Allow all edits" note and the two hooks
  sentences. The settings mark is the same rule in the warning tone: "Agent settings changed", "It can make the agent
  stop asking, also in Ask.", and under it "Changed: `<path>`" (two paths at most, then "and N more"; each path in code
  font, kept on one line with the punctuation after it; a path over 56 characters shows its first and last two
  segments with `…` between, the whole path as its hover text). The app compares the agent's settings files (`~/.mlx-agent/config.toml`,
  `~/.claude/settings*.json`, `~/.claude.json`, `~/.cursor/hooks.json`, the agent's managed config and hook paths,
  and the same names in the folder) at each mode change, at each session start while not in Ask and after a restart;
  a change adds an "Agent settings changed" line and a toast, both naming the files that changed (home as `~`). The
  record keeps one hash per file under fixed names, never the folder's path. Of `~/.claude.json` only what the agent
  reads counts: the top-level `mcpServers` and, for the Work folder and its git root, the project entry's
  `mcpServers`, `enabledMcpjsonServers` and `disabledMcpjsonServers`; its usage counters do not. A file over 16 MB
  counts by size and modification time. A record line from before this form is skipped. A Bypass command runs as you, so it can change those files and make the agent stop asking
  in every Work conversation, also in Ask: the line warns, it cannot prevent it. The list and the drawer mark a
  conversation in Auto or Bypass; the Sessions pages show no marks. Reading files never asks in any mode, and calls the
  agent does not ask about (an earlier "always" grant, your own allow rules) run with no mark.

- **Key printed.** When a command's output held a provider key, the output is redacted and the conversation shows a
  notice to replace the key and delete the conversation.
- **Needs you** (`ui/src/lib/activity.ts` `asksOf`, `arc-x/activity-rail/ask-row.tsx`). Every open card (permission,
  plan and question; a question's row reads its first question and "{n} questions") carries `openedAt` (when the host opened it) and, for a helper's request, `from.child` (the helper's own
  session id). The Activity rail's first section, "Needs you", lists them oldest first: an icon by kind, a one-line
  title (a command's first line, an edit's path cut from the start, else the card's title, at most 80 characters),
  the kind, who asks ("Subagent: <label>", "Workflow <run>: <label>"; nothing for the agent itself) and "Auto asks"
  for a request Auto did not cover, and how long it has waited (the clock is hidden; the row's name says the wait in
  words). The card shown under the thread is the front one: the one picked, else the oldest; "1 of 3" counts in that
  order. A row brings its card to the front, scrolls it into view and focuses its first button; in the drawer it
  closes the drawer first. The palette's "Go to the next request" does the same for the oldest request that is not in
  front. A request alone opens the docked rail (with the rail preference "auto"); the drawer never opens by itself.
  The Activity button shows the count with a warning dot in place of the running count ("Show activity, 2 need you");
  the conversation list shows the count in a warning badge in place of the time, with the words in the row link. The
  host state's `active[].asks` is the count of open cards; the service sends a new host state when a card opens or
  closes. When the turn's own request is open (not a run helper's) the composer reads "Answer on the card above…",
  and while the thread is scrolled away from its end the strip says "1 waiting for you" (or "{n} waiting for you")
  with Go to request. The status region says "{n} requests wait for you." when a second or later request arrives,
  once per new count.
- **A request the agent gives up on.** The agent sends `_x.ai/session_notification` lines `pending_interaction` and
  `interaction_resolved` (`tool_call_id`) from the asking session (a helper's own session for a helper's request);
  the host takes both before the transcript reducer. On `interaction_resolved` it closes the open card of that session
  and raw tool call id as `expired`, answering the request the way a closed card always does (`cancelled`); a resolve
  with no open card changes nothing. The page says "A request was withdrawn." A resolve that arrives while the host
  still decides a permission request (Auto still checking, an edit's landing still reading) keeps its card from
  opening; the request gets `cancelled`.
- **Notices.** A conversation notice has a `tone` (`warning` or `danger`). One shows at a time above the composer
  (danger first, then the newest); the others wait behind "+n", which opens them in a popover. A notice is one line
  with an icon, its sentence cut (the whole sentence is its hover text) and a close button that hides it on this page;
  "Key printed" stays an alert with no close button; `no-plan-mode` (Plan first) and `thinking-not-set` (Thinking)
  are warnings with a close button.
  `Conversation.clearNotice(code)` removes every notice of a code and sends `notice-cleared` to every page.
- An answer to a request that is no longer open closes the card with a toast; an answer the server refuses for its
  content keeps the card and shows the server's sentence.

## Data sources

All state lives on disk; SQLite is only an index.

| Source                                         | Location                                                                                            | Written by                      | Kept                       |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------- | -------------------------- |
| Trace records (v2)                             | `~/.mlx-node/metrics/traces/<date>-<pid>.jsonl`                                                     | inference process, one per turn | 30 days                    |
| Runtime series                                 | `~/.mlx-node/metrics/runtime/<date>-<pid>.jsonl`                                                    | inference process sampler       | 14 days                    |
| Heartbeat                                      | `~/.mlx-node/metrics/runtime/live-<pid>.json`                                                       | inference process sampler       | while the process runs     |
| Supervisor state                               | `~/.mlx-node/desktop/supervisor.json`                                                               | desktop MAIN process            | latest only                |
| Disposable SQLite index                        | `~/.mlx-node/dashboard.db`                                                                          | Control Panel database worker   | rebuilt when needed        |
| Local models                                   | `--models-dir` → `MLX_MODELS_DIR` → `modelsDir` in `~/.mlx-node/config.json` → `~/.mlx-node/models` | downloads, `mlx convert`        | —                          |
| Cold tier (paged prefix blocks)                | `~/.mlx-node/cache/paged/v1/` (`MLX_COLD_CACHE_DIR` override)                                       | inference process               | quota, LRU                 |
| Stored states (Qwen3.5, Flash Next, Hy4-flash) | `~/.mlx-node/cache/prefix-cache/<namespace>/` (`--cache-dir` override)                              | inference process               | 14 days unused, root limit |
| `mlx agent` sessions                           | `<agentHome>/sessions/<encoded cwd>/<id>/` (`agentHome` = `MLX_AGENT_HOME` or `~/.mlx-agent`)       | `mlx agent`                     | —                          |

Chat and Work add, under `~/.mlx-node/chat` (0700):

| File                                                   | Holds                                                                                                          |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `conversations.json`                                   | the Work index: session id and creation time only, no title, no path                                           |
| `model-windows.json`                                   | the context window an agent last listed per model: model id and token count only                               |
| `providers.json`                                       | provider selection and check results; never a key                                                              |
| `plugins/<id>/`                                        | provider skill folders copied from the bundle                                                                  |
| `work-guard/plugin.json`                               | the Work guard plugin: one `PreToolUse` hook that asks before every shell command                              |
| `work-process/`                                        | the Work agent's process cwd: empty, so `$HOME` (and grok-build's `~/.grok`) is never read as a project        |
| `chat-agent/`, `chat-home/`, `chat-workspace/`, `tmp/` | the Chat agent's own home, cwd and temp folders; the agent writes its sessions there                           |
| `composer/<id>.json`                                   | the messages queued while an answer runs (text and files) until they are sent, and the thinking level; 0600    |
| `scratch-folders.json`                                 | the folders Work without a folder made: real path, device and inode numbers; 0600                              |
| `web-ledger/<uuid>.json`                               | Chat: the normal forms of the addresses `web_fetch` may read; 0600 in a 0700 folder (see Web search providers) |
| `never-allow.json`                                     | Work: the never-allow rules, the command words and the Work folder's real path; 0600                           |
| `permission/<id>.jsonl`                                | Work: one conversation's permission record, metadata only; removed with the conversation                       |
| `work-settings.json`                                   | Work: the app-wide Work settings, `{"version":1,"claudeCode":bool}`; 0600                                      |

Work sessions are the agent's own, in `~/.mlx-agent/sessions/`, as from a terminal. Keys live in the Keychain only.

The Work strip's branch comes from git, never from a file the app reads itself. The binary is the first `git` on the
login shell's `PATH` merged before the app's, so a Homebrew git wins as in a terminal. On macOS `/usr/bin/git` is the
Command Line Tools shim, which can raise the install prompt: it is never run; `/usr/bin/xcode-select -p` (2 s) names
the developer folder and its `usr/bin/git` runs if it is executable, else the strip shows no branch. Below git 2.23
(`switch`, `%(worktreepath)`) the strip shows no branch either. A binary found is kept for the app run; a miss is
looked for again at the next read.

Every record on disk is metadata only: a number, a closed enum or an identifier. Prompt text, tool arguments, model
output, error text, stderr and URLs are never written by the app, with four exceptions. `composer/<id>.json` holds
the messages the user queued (see Queue), with their file text and picture bytes, until they are sent, and goes with
the last row and with the conversation. `web-ledger/<uuid>.json` holds the normal forms of the addresses of a Chat
session's user messages, parsed search results and pages read, query and user info kept; it goes with the session
and at host close. `never-allow.json` holds the command words of each never-allow rule and its Work folder's real
path, until the user removes the rule. `scratch-folders.json` holds the real paths of the folders Work without a
folder made. The agents write their own session files.

```
 inference process ──writes──▶ ~/.mlx-node files ──read──▶ Control Panel service ──MessagePort──▶ SPA
 packages/server               traces, runtime series,     packages/dashboard/src                 packages/dashboard/ui
                               live-<pid>.json             SQLite index, routes, live watcher
 desktop MAIN ──writes──▶ desktop/supervisor.json ─────────▲
```

### Trace record v2

`packages/server/src/metrics-trace.ts` (`MetricsTraceRecord`, `v: 2`). One line per dispatched turn — `ok`, `error`
or `cancelled` — written by the turn wrapper in `turn-trace.ts`, so every client of `/v1/responses` and `/v1/messages`
lands in the same stream.

| Group           | Fields                                                                                                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Identity        | `traceId`, `ts`, `pid`, `sessionId`, `model`, `modelId`, `modelType`                                                                                                                                         |
| Request         | `outcome`, `errorKind`, `failPhase`, `endpoint`, `stream`, `dispatch`, `finishReason`                                                                                                                        |
| Timing          | `durationMs`, `queueMs`, `loadWaitMs`, `resolveMs`, `preDispatchMs`, `waitMs`, `slotMs`, `prepareMs`, `ttftMs`, `prefillMs`, `decodeTps`                                                                     |
| Tokens          | `promptTokens`, `cachedTokens`, `outputTokens`, `reasoningTokens`, `prefillDoneTokens`, `prefillTokens`, `ctxWindow`                                                                                         |
| Thinking        | `thinkingBudget`, `thinkingBudgetFrom`, `thinkingGuard`, `outputCap` (what the server sent); native `reasoningClose`, `reasoningCutAt`, `reasoningCyclePeriod`, `reasoningStrayCloses`                       |
| Speculation     | `mtpCycles`, `mtpMeanAccepted`, `mtpMeanDrafts`, `mtpMeanDepth`, `mtpAcceptByPos` (at most 16 positions)                                                                                                     |
| Counts, overlap | `toolCount`, `toolCallCount`, `imageCount`, `concurrency`, `concurrencyPeak`                                                                                                                                 |
| Memory          | `memActiveBytes`, `memPeakBytes`                                                                                                                                                                             |
| GPU kernels     | `pipelinesCreated`, `pipelineCreateMs`: pipelines the process created while serving between the turn's begin and end (process-wide, so overlapping turns share them); absent while the native counter is off |
| Cold tier       | `cold*` deltas and totals of the turn (hits, misses, bytes, queue drops, evictions, corruptions, write errors, sidecar counts)                                                                               |

A rate or time that was not measured is absent, never 0. A media-restart error is not recorded as a failure.
`thinkingBudget` and `outputCap` are the budget and `maxNewTokens` of the native config, after the window scaled them;
`thinkingBudgetFrom` is `request` or `server` (Messages only); `thinkingGuard` is `true` when the cycle guard was sent. The native four
are written for every finish reason: `reasoningClose` is `model`, `budget`, `cycle` or `open` (absent when the turn
did not start in thinking), `reasoningCutAt` the thinking tokens when a cut began, `reasoningCyclePeriod` the period of
the cycle the guard saw, `reasoningStrayCloses` the close tokens after the thought ended. An addon that predates them
sends none, and the record carries none.
`failPhase` is one of `queued`, `slot`, `prepare`, `prefill`, `decode`; a line from an older engine wrote `queued` for
any stop before the prefill. Native reports `done` before it sends an error, so the server reads where the turn
stopped from its timers: decode, else prefill, else `slot` when `slotMs` > 0, else `prepare` when `prepareMs` > 0,
else `queued` (`turn-trace.ts` `failPhaseOf`).

#### Native waits before the first token

Streamed turns of an engine that reports them carry three timers from the native progress handle (on `ok`,
`error` and `cancelled` lines alike; an older engine writes none):

| Name        | Meaning                                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| `waitMs`    | Native request created → the model thread picked it up: the model was busy. Outside `ttftMs`                     |
| `slotMs`    | Sum of the spans the scheduler could not start the turn (lanes or memory). Inside `ttftMs`                       |
| `prepareMs` | Sum of the spans spent on render, prefix match, cache restore and GDN replay before the prefill. Inside `ttftMs` |

```
created ──waitMs──► pickup ──[ queued | slot | prepare ]*──► begin_prefill ──prefillMs──► first token
                    └──────────────────────────── ttftMs ────────────────────────────────────┘
```

The phase bars and the turn drawer place them as `wait` (bounded by duration − time to first token), then `slot` and
`prepare` inside the time to first token before the prefill; Setup is what is left of it. A turn that never had a
first token keeps its slot and prepare time in the rest of its duration. The parts never pass the turn's total
(`api/sql.ts` `PHASE_SQL` and `turnPhases` agree row by row). The drawer lists Model busy, Waiting for a slot and
Preparing under Speed when the record has them and they are above 0, and "GPU kernels created: n (time)" under the
details when the turn created any.

#### Time to first token, setup and prefill

```
 dispatch   generation_start    begin_prefill()      first_token()     first token emitted        dispatch returns
    │ native wait, │ setup, part a     │ PREFILL WINDOW    │ setup, part b │        decode            │
    │ template,    │ prefix lookup,    │ forward over the  │ one decode    │ (output − 1) tokens      │
    │ tokenize     │ cache restore,    │ new tokens, then  │ forward, then │                          │
    │              │ allocation,       │ the first-token   │ the token is  │                          │
    │              │ memory wiring     │ sample            │ emitted       │                          │
                   ├───────────────────────── ttftMs ───────────────────────┤
                                       ├──── prefillMs ────┤
             setupMs = ttftMs − prefillMs − slotMs − prepareMs = part a + part b
```

| Name            | Meaning                                                                                                                                                                                                                                |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ttftMs`        | `generation_start` to the first token emitted. Shown as First token                                                                                                                                                                    |
| `prefillTokens` | Tokens the prefill window processed (native `ChatStreamProgress.prefillTotal`, the cached prefix excluded)                                                                                                                             |
| `prefillMs`     | `begin_prefill()` to the first token sampled (native `prefillElapsedMs`, frozen at the first token)                                                                                                                                    |
| Setup           | `ttftMs − prefillMs − slotMs − prepareMs`, bounded as above (`SETUP_MS_SQL`): prefix lookup, cache restore, allocation before the prefill, or only the wait behind other requests once `prepareMs` > 0; the first decode step after it |
| Prefill speed   | `prefillTokens × 1000 / prefillMs`, only from `PREFILL_MIN_TOKENS` = 256 tokens up (`src/prefill-rule.ts`)                                                                                                                             |
| To first token  | `ttftMs` of a turn without a prefill window: setup and prefill cannot be told apart                                                                                                                                                    |

The server reads the window from the stream handle's progress snapshot at the terminal event and writes both fields
for an `ok` streamed turn only. Non-streamed turns (no progress handle), paths that never call `begin_prefill`, failed
and cancelled turns, and lines written before this change carry no prefill window, and get no prefill speed and no
setup time. Below 256 tokens the fixed part of the window (graph build, the sample, one decode step on some cores)
dominates, so a small prefill shows its time and size but no rate. Under continuous batching the window is wall time
and also holds other turns' decode steps.

### Runtime sampler

`packages/server/src/host/runtime-sampler.ts`, started by the server host; one per process.

| Output                               | Content                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Cadence                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `live-<pid>.json` (`v: 3`)           | status, health, memory, loaded models with their KV pool, turns in flight with their phase and `waitMs` / `slotMs` / `prepareMs` / `prefillElapsedMs` / `decodeElapsedMs`, the loads in flight (`loading`: model, start, `phase` `weights` or `kernels`, `stepsDone` of `steps`, read per load path from native `loadProgress`), counters, `work.ms` and `rates` (see below), `stateCache: { dir, limitBytes }` (the `--cache-dir`, null for the default folder, and one open persistent state tier's limit, null when the engine's figure is a sum over two or more Qwen3.5 models, the tier is not persistent, or there is none), and per model `stateNamespace` (its open persistent state cache's folder name, from `stateTierNamespace()`; null without one) and `images` (`supportsImages()`: false for a model with no such method, null when the call throws); about 0.6 KB plus about 380 B per turn in flight (at most 32 listed) plus each resident model, not capped | 1 s while busy, 5 s while idle (tmp + rename), plus a write 250 ms after each turn or load start and end |
| `<date>-<pid>.jsonl` `proc`          | host kind, version, port, wired limit, total memory, chip                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | first line of each file                                                                                  |
| `<date>-<pid>.jsonl` `sample`        | busy time, memory, in flight, queued, resident models, turn and token deltas, `pipelinesCreatedWhileServing` (cumulative; absent while the native counter is off), and the work deltas since the previous sample (see below)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 5 s while busy, 60 s while idle, plus one on each busy flip, burst or resident-set change                |
| `<date>-<pid>.jsonl` `sched`         | `schedulerStats()` snapshot of one model, plus `turns`: the turns sent to that model since its load, all lanes, counted when the call was sent                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | once after each load, then once after each busy burst, only when that model is idle (see below)          |
| `<date>-<pid>.jsonl` `load` / `exit` | one model load (with `warmupMs`, `warmupSteps`, `warmupSkipped`, `warmupReset`, `warmupPipelines`, `warmupCompileMs`, `warmupPeakBytes` when the load ran a warm-up; not indexed); process exit                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | on the event                                                                                             |

**Work ledger** (`packages/server/src/work-ledger.ts`). Every traced turn is a lane. Each sampler tick reads the live
turns' native progress and credits the change of each native clock to a state: `waitMs` + `slotMs` to `waiting`,
`prepareMs` to `prepare`, `setupMs` + `prefillElapsedMs` to `prefill` (a whole turn reads `prefill` from its start, so
its setup before the prefill window counts there, whatever reads land around `begin_prefill`), `decodeElapsedMs` to
`decode`; a lane with no progress (non-streamed turns, a stream before its reader exists) goes to `running`, and so does
each System One decision, from its start edge to its settle edge. `loading` is wall time with at least one load in
flight. These are lane-ms: two lanes decoding for one second add two seconds. A turn's final snapshot is credited when
it settles, so a phase shorter than one tick still counts in full; native stops the clocks when the turn ends, so a
terminal chunk that waits for a slow reader adds nothing. Two token/time pairs ride along: output (each streamed lane's
kept tokens after its first, over the ms in which at least one lane decoded, so the union and not the sum) and prefill
(the computed prompt tokens of prefills of 256+ tokens, over their prefill window). A pair moves only when its tokens
move: a one-shot prefill, or a core that reports its tokens at the end, adds its time together with its tokens. A
lane's decode time waits for that lane's own tokens; a lane that ends with no kept token after its first gives its
decode time to no rate. Counters only grow.

| Field                                                                       | Where     | Meaning                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `work.ms: { loading, waiting, prepare, prefill, decode, running }`          | live file | lane-ms per state since the process started, rounded                                                                                                                                                                                                                                                              |
| `rates: { windowMs, outputTps, prefillTps }`                                | live file | over the last `windowMs` (10,000) of work: output pair tokens × 1000 ÷ its ms, prefill pair likewise. Each credit keeps the interval its work ran in and the window cuts it at its start, its tokens spread evenly over that interval; null when the window holds under 250 ms of that time or no tokens, never 0 |
| `turns[].prefillElapsedMs`, `turns[].decodeElapsedMs`                       | live file | native prefill clock (prefill start to first token, or to now) and time from the first token, both stopped at the turn's end, rounded; null before they start                                                                                                                                                     |
| `loadingMs`, `waitingMs`, `prepareMs`, `prefillMs`, `decodeMs`, `runningMs` | `sample`  | `work.ms` delta since the previous sample                                                                                                                                                                                                                                                                         |
| `rateOutTokens`, `rateOutMs`, `ratePrefillTokens`, `ratePrefillMs`          | `sample`  | the two pairs' deltas since the previous sample. Deltas are taken on the rounded totals, so the samples of a process sum to its totals                                                                                                                                                                            |

`schedulerStats()` is a scheduler barrier, so the sampler never calls it on a model with work: it goes through
`SessionRegistry.tryIdleProbe`, which holds any dispatch that arrives until the reply settles. Scheduler counters are
cumulative, so the difference of two idle snapshots is exact for the burst between them. A turn in the one-turn-at-a-time
lane moves no scheduler counter (any family under `MLX_SERVE_FORCE_SERIAL=1`, MoE DFlash2, media and other barrier
turns), so `turns` against `admitted` tells such a model from an idle scheduler; a record without `turns` reads as
unknown. Live concurrency comes from TS-side state. At 16 MiB a series file takes only `load` and `exit` records (each still led by this process's
`proc` header when the file has none from it yet); `sample`, `sched` and `proc` records are dropped until the next
day's file.

### Supervisor state

The desktop MAIN process writes `~/.mlx-node/desktop/supervisor.json`
(`packages/desktop/src/main/supervisor-state-file.ts`): supervisor state (`stopped`, `starting`, `running`,
`restarting`, `failed`, `lying`), sidecar pid, generation, the restart policy's `streak` and `shortLivedFailures`,
last exit verdict, health and the native error count. Exit reasons, stderr and native error text are never written;
the one text is `failureReason`, the message of the sidecar's start-failure line (written before any request runs),
one line of at most 300 characters, set only while the state is `restarting` or `failed`. A reader ignores the file
when MAIN's pid is dead.

### The index

`~/.mlx-node/dashboard.db` is **disposable**: deleting it loses nothing; it is rebuilt from the JSONL on next start.
`SCHEMA_VERSION` is 14 (`src/db/open.ts`): `traces`, `trace_files`, and five runtime tables (`runtime_files`,
`runtime_procs`, `runtime_samples`, `runtime_sched`, `runtime_loads`). Any DDL change bumps the version; a file with
another version, or a damaged one, is moved aside and rebuilt. Version 10 replaced `traces.prefill_tps` with
`prefill_ms` and `prefill_tokens`; version 11 added `runtime_sched.turns`; version 12 added `traces.wait_ms`,
`slot_ms`, `prepare_ms`, `pipelines_created` and `pipeline_create_ms`; version 14 added the ten work deltas of a
sample to `runtime_samples` (`loading_ms`, `waiting_ms`, `prepare_ms`, `prefill_ms`, `decode_ms`, `running_ms`,
`rate_out_tokens`, `rate_out_ms`, `rate_prefill_tokens`, `rate_prefill_ms`; NULL for a record that predates them).

Ingest runs on boot, then every 30 s, after a finished turn (see the live channel) and on `POST /api/ingest`. It reads
only the tail appended since the last read of each file. Files past retention are pruned after ingest.

## Live channel

Live state is pushed over the same port; no page polls for it. The one timer that reads data is the Coding Agents
page, which re-reads `GET /api/coding-agents` every 30 s while the window is visible (and on window focus). The Chat
load card and the thread's wait line (`components/wait-clock.tsx`) have a 1 s clock while they are shown, and so do a
running helper's rows (the rail, the thread row and the run card: `useNow` in the same file); it changes only their
elapsed-time and quiet-time text (the kernel step count, the engine's request phases and the helper's numbers come
with the live snapshot and the host state).

```
 SPA  ── { kind: 'watch', id, topic: 'live' } ──▶  service
 SPA  ◀── { kind: 'live', id, event }  ──────────  event = { type: 'runtime', snapshot: RuntimeLive }
                                                         | { type: 'data', version, traces, runtime }
```

- The service (`src/live.ts`) stats the `live-*.json` files and `supervisor.json` once a second, and only while a
  window is subscribed. `fs.watch` is not used: the heartbeat is replaced by rename, which drops a file watch.
- A finished turn moves the writer's counters. That starts one ingest (at least 2 s apart) and a `data` event.
- In the SPA, `subscribeLive` (`ui/src/lib/api.ts`) feeds `useRuntimeLive()` (`ui/src/lib/live.ts`). A query marked
  live refetches on a `data` event at most once per 1 s or 5 s (`LIVE_REFETCH_FAST_MS`, `LIVE_REFETCH_MIN_MS`).
- `GET /api/runtime/live` answers the same snapshot once, for the first paint before the first push.

A process is `running` while its heartbeat is at most three intervals plus 1.5 s old, `unresponsive` after that, and
`dead` when its pid is gone or the file is 5 minutes old. The engine status in the sidebar footer:

| Status                             | When                                                |
| ---------------------------------- | --------------------------------------------------- |
| `untrusted`                        | the supervisor state is `lying`                     |
| `loading`                          | a running process reports health `loading`          |
| `running`                          | a process is running                                |
| `starting`, `restarting`, `failed` | no running process; the supervisor is in that state |
| `unresponsive`                     | no running process; one is unresponsive             |
| `stopped`                          | none of the above                                   |

## Design system

One design system: Arc (`uiarc.dev`) in a desktop density.

| Folder / file                       | Holds                                                                                                                                                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ui/src/components/arc/`            | Vendored Arc **free** components (MIT), copied by hand. `PROVENANCE.md` lists each file, its source and every local change; `LICENSE` beside it. Excluded from our format and lint                                                   |
| `ui/src/components/arc-x/`          | Own components, written from scratch on Arc tokens: surfaces, tables, the chart kit, number and motion parts, transcript parts. A folder holds `<name>.tsx` and, when it has styles, `<name>.module.css`                             |
| `ui/src/components/arc-x/wait-orb/` | The one exception in `arc-x/`: `engine.ts` holds orb geometry derived from the MIT package `thinking-orbs` 0.3.2, with its `LICENSE` and a `PROVENANCE.md` that lists every local change. The component and its painter are own code |
| `ui/src/styles/tokens.css`          | Density and own tokens on top of Arc `foundation.css`: controls 28 / 32 / 40 px, radii 10 / 14 / 20 px, type 11–32 px, 8 model colors, data and phase colors, durations, staggers                                                    |
| `ui/src/styles/tailwind-theme.css`  | A locked Tailwind theme: layout and text utilities only, no palette. Pages use Tailwind for layout; `arc-x/` and `shell/` use CSS modules only                                                                                       |
| `ui/src/styles/fonts.css`           | Inter, Geist and Geist Mono (variable, latin + latin-ext): six bundled `woff2` files, no network                                                                                                                                     |
| `ui/src/theme/theme.ts`             | `data-theme` on `<html>`: preference `system` / `light` / `dark` in `localStorage`                                                                                                                                                   |

Rules the tests hold (`__test__/style-rules.test.ts`, `no-regex.test.ts`, `tokens.test.ts`):

- colors, durations and blur radii in own stylesheets come from tokens;
- springs are named once in `arc-x/motion/motion.ts` (`motionX`: `expand`, `glide`, `morph`, `travel`, `reshape`,
  plus `swap`, `reveal`, `page`, `layer`, and `orb`: draws per second, the still frame, the hold pace); no component
  writes a duration, an easing or a spring;
- every own stylesheet that animates has a `prefers-reduced-motion` block, and every infinite loop pauses while the
  window is hidden (`:root[data-hidden]`, set from `visibilitychange`);
- a canvas loop (`WaitOrb`) is not paused by that rule, so it gates itself in JS: it asks for animation frames only
  while the page is visible, the canvas is in view and motion is allowed, draws at most `motionX.orb.rate` times per
  second, and under `useStaticMotion()` draws one still frame;
- `useStaticMotion()` is true under the OS reduced-motion setting and in tests (`MotionGlobalConfig.skipAnimations`):
  every own animation then renders its end state, so tests read final text;
- no regex anywhere in the SPA, the dev scripts or the dashboard tests; hand-written scanners live in
  `ui/src/lib/text-scan.ts`.

Motion follows measured data only: numbers count up once and roll on change (`NumberValue`), live labels shimmer,
charts draw in once and morph between questions. `null` stays `—` and never moves. A number counts up only when it is
on screen as its first value shows (or when motion turns on and its wheels are first drawn); one that is off screen
then (below the fold, in a closed part) appears at its value, so no zeros wait for a scroll.

## Dev harness

A dev tool only. It runs the real runtime on a generated fixture home plus a Vite dev server in one process; the SPA
talks to the runtime over Vite's HMR socket instead of a MessagePort.

```bash
yarn build:ts                                        # once: packages the runtime imports
yarn workspace @mlx-node/dashboard dev               # fixture if missing, then http://127.0.0.1:5179/
yarn workspace @mlx-node/dashboard dev --fresh       # regenerate the fixture first
yarn workspace @mlx-node/dashboard dev:fixture       # write the fixture home only (--home <dir>, --fresh)
yarn workspace @mlx-node/dashboard dev:shots         # PNGs, light/dark x 1180x800/720x480, every route, into .dev-shots/
```

- `http://127.0.0.1:5179/__gallery` shows every own component in its states (loading, ready, working, error).
- The fixture home defaults to `packages/dashboard/.dev-home` (gitignored). `assertFixtureHome` refuses the user's
  home, the real `~/.mlx-node`, anything inside it and every ancestor of them, so the harness never reaches real data.
- Routes that would reach the network (a real download, the update check, the Update and Locate plans) answer
  `E_UNAVAILABLE` unless `MLX_DEV_NETWORK=1`; a Locate click there ends in the "Disabled in the dev harness" line.
- `MLX_DEV_ENGINE_STATES=1` loads five more models, one per state of the Engine "Block pool and batching" section: turns
  that ran alone with 0 or 1,200 blocks held, turns only before the last hour, a flat-cache model and a paged model
  before its first snapshot. `MLX_DEV_TICKER_TPS` and `MLX_DEV_TICKER_MEM_GB` fix the Output rate and MLX memory of the
  heartbeat.
- `MLX_DEV_FLIGHT=one-model` puts both turns in flight on one model, one asked through the alias `mlx-local` (the
  default is one turn on each of two models). `MLX_DEV_DISK_CACHE=native|off|older` sets the disk cache the heartbeat
  reports: on with the native 128-block limit, off, or no field at all (default: on, the whole prompt).
  `MLX_DEV_FLIGHT=phases` adds three turns before their prefill: model busy, waiting for a slot, preparing.
  `MLX_DEV_FLIGHT=chat` runs a 40 s cycle on the harness's chat model: 10 s busy (one request of another session
  reads its prompt for 6 s, then generates, while the model's `queued` counts the Work conversations' requests,
  unlisted), then one request per Work conversation named with its id (queued 10 s, prefill 15 s, decode 5 s): send a
  Work prompt with "slow" and the wait line shows each state, the model-busy ones included, during the 45 s pause
  after the Read row. The request's decode writes 640 tokens a second. A prompt with "compact" (Chat or Work) runs a
  45 s compaction in the fake agent (`compact` step: started, then completed with 222,822 → 31,204 tokens): the
  compaction line shows, in Work with the reading and writing notes of the cycle, then the marker. Compact now (the
  thread bar `⋯`) runs 6 s in the fake agent (`compactNow`), then ends 41,000 → 9,000 tokens; Stop ends it sooner.
  Rewind to here (a sent message's `⋯`, `/rewind`, the palette) runs 3 s in the fake agent (`rewind.ms`); in Work
  after "background" the dialog names the command and schedule it ends, and the thread keeps a "Rewound to here" line.
  In Work, Rewind on the "diff" message lists the four files it wrote with src/log.ts changed outside (`rewind.changed`
  in the script, path → conflict type; the fake agent keeps the files its own edit steps wrote per prompt beside the
  session's `updates.jsonl`, so they come back after a restart, and a fork's copy has none), opens on Conversation
  only and offers Restore and overwrite.
  Fork (the thread bar `⋯`, a message's "Fork with this answer", `/fork`, the palette) copies in 0.8 s in the fake
  agent (`fork.ms`): the copy opens with "Fork of …" on top and a Fork tag in the list.
  Recap (the thread bar `⋯`, `/recap`, the palette) shows "Writing a recap…" for 4 s in the fake agent (`recap.ms`),
  then a Recap row; a conversation with no finished answer gets none ("No recap this time."). The script's `recap`
  also turns recaps off (`disabled`), gives none (`unavailable`), holds the line (`holdOn`), sends no line (`silent`)
  or no ack (`noAck`), or answers an error (`rpcError`). The live fixture session stores one recap after its 3rd
  answer, which the Sessions transcript draws too.
  The context ring opens the Context panel: the fake agent's `_x.ai/session/info` gives its parts from the last
  `totalTokens` it sent (after Compact now the use drops and "compacted once" shows); a cold conversation reads "Send a
  message to see the parts.". The live fixture session reads 61K (its turns store a growing `totalTokens`).
  The queue: send "slow" (Chat 30 s, Work 45 s), then type and press Enter to queue rows under the composer; they go
  one by one when it ends, Stop holds them (Send next sends them), and they come back held after a harness restart.
  In Work a row's Send now (or ⌘Enter) during "slow" goes into the answer: the agent reads it after the 45 s wait
  (`readSteers` step); Stop before that drops it. The live fixture session holds one stored steer ("also update the
  changelog").
  A goal: in Work send `/goal Fix the flaky cache tests` (the Work script `/goal `, first in its list, plays every
  `/goal <objective>`; `/goal pause|resume|clear|status` stay the agent's host turns). The strip reads "Working · …"
  with the clock and tokens while the typed answer runs two rounds (16 s, ticks every 0.9 s), then the agent starts two
  more by itself 1.5 s apart (wake rows "Started by the goal loop", 9 s each) and the strip ends at "Done · 0:34 · 33k
  tokens". A row queued meanwhile reads "Runs when the goal pauses." and goes once it is done; Pause or Stop pauses
  it, Resume and Clear send the agent's own commands. The fake agent's `goal` script (`wakeRounds`, `wakeRoundMs`,
  `gapMs`, `end`, `endMessage`, `after`, `budget`, `planning`) and its `goalRound` and `goalStatus` steps play the
  stored lines and the live-only ticks.
  Type `/` in the composer: Compact, Context, Fork, Recap and Rewind in both modes; in Work also the four built-ins,
  and once the conversation is loaded the harness skills (commit, review-pr) and workflow (release) of
  `work.session.commands`.
  `/goal` (alone or with status, pause, resume, clear), `/workflow` and `/deep-research` answer as the agent's host
  turns (`hostTurnText`); `/goal <objective>` is a model turn, as on the agent; `/yolo` or an unknown name reads "No
  such command here.".
  A Work prompt with "colors" runs `yarn test` with colored output in three ticks 600 ms apart (`shell` step: the
  flagged tick shape once the app sent `x.ai/incrementalBashOutput`); the row shows plain text. The fixture sessions'
  `cargo test` output is colored too, so stored rows show the same cleaning.
  A Work prompt with "needs" opens three requests (`giveUp` step: a subagent's `cargo test -p app`; the turn's own
  `rm -rf build/`; a workflow run "release" whose helper "packer" asks for `cargo publish`): the rail lists all three
  under Needs you, and the composer follows only the turn's own; after 30 s the subagent gives up and its card closes
  as withdrawn.
  A Work prompt with "diff" runs five edits as the agent plays them (`edit` step: start diff, request, result diff
  with `_meta.details`): a change in `src/app.ts`, a replace-all with two matches in `src/log.ts`, a new
  `docs/notes.md`, a whole-file write of `README.md` and a search_replace that finds no match; each row opens to its
  diff while its card waits. The fixture sessions' plan edits are search_replace results with a diff, so stored rows
  show one too.
  A Work prompt with "leftpad" asks for `npm install left-pad`: "Deny and say why…" with a note gives the "Denied ·
  note sent" row and the "Your note" row, and the agent goes on with `pnpm add left-pad` (`permission` step: a
  `reject-once` with `_meta.followup_message` plays the failed call and stores the note twice); a plain Deny ends
  the answer. The fixture's live session ("Add retries to the upload client") has a stored note in its 4th turn: a
  failed `npm install left-pad` and the note twice in `updates.jsonl`, which the Sessions transcript draws as one
  "Your note" row. The harness's Work folder starts with two never-allow rules (`git push`, `npm publish`), so the
  chip's menu reads "Never allowed · 2". "Never allow npm install…" on the "leftpad" card keeps a third, and the next
  "leftpad" is denied by the app with no card.
  A Work prompt with "plan" has the model enter plan mode itself (`planMode` step), then asks for the plan; Approve
  leaves plan mode. The fake agent moves its mode on each prompt's `_meta.mode` as the agent does (scenario
  `noPlanMode` turns that off, for the notice). The fixture's live session stores "plan" before its 1st prompt and
  "default" before its 2nd, so its Sessions transcript shows both lines.
  A Work prompt with "question" opens a card with three questions (`question` step: a single pick with a Recommended
  option, checks to pick any of, a yes or no; `giveUpOn` plays the agent's time limit in tests); the answer becomes the
  row. The fixture's live session has an answered question call in its 5th turn (two questions, picks and a note).
  A Work prompt with "interview" has the model enter plan mode itself and ask two questions: layouts whose previews
  show under the list, and pages to pick any of; the card has "Talk it over" and "Plan now". The fixture's live session
  has a plan-mode question in its 1st turn answered with "Talk it over" (the row "you chose to talk it over").
  A Work prompt with "mcp" has the server "github" ask for input (`elicit` step: the live `pending_interaction`, the
  `_x.ai/mcp/elicit` request with `schema` or, in link mode, `url`, then `interaction_resolved`; `giveUpOn` plays the
  agent's end of the wait in tests): eight fields, a menu of six repositories among them; the answer becomes the line.
  `__test__/chat-elicit-real-agent.test.ts` pins this on the real agent with a stdio MCP server in the agent home
  (`__test__/helpers/elicit-mcp-server.mjs`): the request has the fake agent's keys, a refused Send keeps the card,
  and the server gets the exact content, then a decline, then a cancel on Stop.
  A Work prompt with "signin" has "github" ask to open its sign-in page (`elicit` step in link mode with `complete`:
  3 s after the answer the fake agent sends `_x.ai/mcp/elicit_complete`); Open page leaves "github asked you to open a
  page · opened", then "github · finished in the browser". The real-agent test pins link mode too: the request's
  keys, the bare accept the server reads as `{action: 'accept'}`, and the agent's `_x.ai/mcp/elicit_complete` with
  `serverName`, which counts for the opened page and not for a declined one.
  A prompt with "retry" (Chat or Work) plays two stalls 6 s and 5 s apart, then the answer (`retry` step: a
  `retry_state` line, sent and stored): the retry line counts from the first try, then the "Retried" row stays. A
  prompt with "retryfail" plays three server errors, then `failed`, a `turn_completed` with `stop_reason` "error" and
  the error reply, in the agent's order (`turnCompleted` step with `agentResult` and `errorKind`): the "Gave up" row
  and the turn's error, the same after a reload. The fixture's long session has a stored stall in its 9th turn, which
  the Sessions transcript draws as a "Retried" row.
  A Chat prompt with "picture" plays the agent's `image_compressed` for the message's picture, stored after the
  prompt's chunk (`promptPictures` of the script), then an `image_dropped` for the retry the engine refused (`picture`
  step): a "made smaller" row and a "left out" row. A Work prompt with "picture" reads `docs/shot.png` (`pictureRead`
  step: the call, its title update and the result with the picture's bytes): its read row reads "1 picture · …" and
  opens to the tile, then the agent leaves the picture out. A Work prompt with "readpdf" reads pages 4 to 6 of a
  12-page `docs/spec.pdf`: "3 of 12 pages". The fixture's live session reads `docs/screenshot.png` and three pages of
  `docs/api-spec.pdf` in its 2nd answer, after that answer's picture rows. A prompt with "nomodel" (Chat or Work) plays a blocked prompt (`noModel` of the script: the
  `model_auto_switched` line and `end_turn`, nothing stored): "No model answered." with the agent's words under it;
  the scenario's `session.modelGoneOnLoad` plays the load cases. The fixture's live session stores both picture
  one-shots after its 2nd prompt's chunk, which the Sessions transcript draws before that answer's first call.
  A prompt with "hookstop" (Chat or Work) answers, then two stop hooks run 4 s (`hookStarted` step: the live-only
  `hook_run_started`; `hook` step: the stored `hook_execution`; `hookNote` step: a `hook_annotation`): the wait line
  from 1 s, then "Hook failed" for the lint hook and the other hook's note; sent again, the same row counts "2 times". A
  prompt with "hookblock" (Chat or Work) plays a prompt hook that blocks the message after 1.5 s (`promptBlocked` of the
  script: no chunk stored, the note, a cancelled end with `HookDenied`): the note and "Stopped by a hook.", and the
  text back in the box. The scenario's `session.sessionStart` (`{runs, notes?, afterReply?}`) plays a `session_start`
  batch and its notes before (or just after) the `session/new` reply; the fixture's live Work session stores one that failed before its
  first prompt ("Hook failed · global/env"). A Work
  prompt with "hookdeny" plays a `pre_tool_use` hook that denies `rm -rf build` (`hookDenied` step, the verdict before
  the failed update): the row reads "Blocked · by a hook" with the hook's words. The fixture's live session has that
  blocked command in its 1st turn, a stop hook failure on every turn (one row, "5 times"), and a hook's note in its 3rd.
  `MLX_DEV_LOAD_KERNELS=3/50` reports the harness engine's load as "Preparing GPU kernels", 3 of 50.
- The fixture home holds three model caches under `prefix-cache/` (sparse files, last used 2 minutes, 3 days and 12
  days ago); the harness holds the lock of the newest one, so the Cache page shows one "In use", and the fixture
  heartbeat names its model. The runtime's roots all come from `fixturePaths(home)`, so it never lists the real
  `~/.mlx-node/cache/prefix-cache`. A fixture home made before this shows "not created yet" until `--fresh`.
- `MLX_DEV_APP_END=no-ping|crash|signal|user` hands the runtime one earlier Control Panel end over every Work
  conversation of the fixture home, so the last unfinished turn of each shows its line.
- Chat and Work run only on the scripted fake agent (`__test__/helpers/fake-agent.mjs`, script
  `__test__/fixtures/chat-scenarios/chat-harness.json`, picked by a word of the prompt), with keys in memory and the
  providers on loopback. The folder dialog answers a throwaway folder in the fixture home: `work/` for Work,
  `lmstudio/models` for Locate.
- The fixture home holds one catalog model that Locate made (Clairic (27B), `CATALOG_LOCATED`): its folder links to
  the weights in `lmstudio/models/Brooooooklyn/Clairic-GGUF`, so the card shows "Uses files in" and Local models the
  Linked badge. The other Locate states (checking, a refusal, files not found) are in the gallery.
- Nothing of the harness ships: its files live in `packages/dashboard/scripts/`, no file under `ui/` imports them,
  and `__test__/pack.test.ts` checks that `web/` holds no harness string.
- The harness runs without the app's CSP, drag region and traffic lights. Check those in the real window:
  `yarn workspace @mlx-node/dashboard build:ui && yarn workspace @mlx-node/desktop start`.

## Sessions

Read straight from the grok-build session folders on the database worker. No
SQLite table: each call re-lists the folders and reuses parsed `summary.json`,
`meta.json` and `updates.jsonl` while their (mtime, size) is unchanged.

```
<agentHome>/
  active_sessions.json            TUI registry {session_id, pid}; read only, no lock
  sessions/<encoded cwd>/<id>/
    summary.json                  title, cwd, model, kind; rename rewrites it
    updates.jsonl                 transcript + turn_completed metrics (rewinds applied)
    subagents/<child>/meta.json   the ONLY parent -> child link
```

A subagent's own steps are in its own session folder; the timeline route reads them, the detail route does not.

A turn that a background workflow, subagent, task or notification started has no typed prompt: the agent wrote a hidden
one (`hideFromScrollback`). The reader keeps it as a `wake` entry, never as a user message, with the cause read from the
prompt id the agent recorded (`turn_completed.prompt_id`, else the `user_prompt_submit` hook line); live, the wake's
own `turn_completed` id replaces whatever the host's line guessed, so the row ends as the stored one. The prefixes
(`WAKE_PREFIXES`, first match wins): `workflow-completed-`, `subagent-completed-`, `task-completed-`, `bash-completed-`
and `monitor-completed-` (a task; the last two are the agent's queued names, never a turn id in v0.3.4),
`monitor-` and `notifications-` (notifications), `goal-summary-` and `goal-classifier-nudge-` (goal),
`interject-fallback-` (message: a Send now that missed its answer, titled "Your message ran as its own turn"); anything
else is unknown. A task wake links the command or monitor (`task: {id, kind, label}`) from the tool row that started it, else
from its `task_completed` line. A notifications wake lists what ended with `will_wake: false` since the last turn end,
at most 8 (`tasks`), leaving out an end the agent never reports (`block_waited`, or killed with the result delivered).
The page titles the turn by its cause (`Woken by workflow deep-research`, `A command ended: npm test`,
`2 commands ended: npm test, cargo build`; the words are `WAKE_ENDED` in `ui/src/lib/sessions.ts`, labels drawn with
hidden characters made visible; the Chat page alone names a monitor report, "A monitor reported: …", from the host's
live facts), and its timeline starts at the previous turn's end when background
subagents ran in the gap: the main lane is idle there, and the status counts the turn alone. A typed prompt that
starts with the workflow host's retry words ("Your final message did not satisfy the output contract: ") is titled
"Output contract retry: {reason}".

- List: top-level sessions only. Subagent children, `headless` runs and empty
  husks are left out; a child is reachable from its parent's detail page.
- Rename does what grok's `/rename` does to a dormant session: `generated_title`,
  `session_summary` only while it is empty, `title_is_manual: true`,
  `updated_at`, plus `title_refresh_idx = 2`. Title: 1..100 characters after
  removing control and bidi characters and trimming. Written as temp file +
  fsync + rename, pretty JSON like serde.
- Delete removes the session folder and its subagent children (followed through
  `meta.json` whose `parent_session_id` matches, looked up by id in every cwd
  folder). Like grok it leaves forks, `prompt_history.jsonl`,
  `session_search.sqlite` and the cwd folder alone.
- Delete first renames each folder to `<cwd>/.deleting-<uuid>`, deepest child
  first and the requested session last, then removes those folders. Dot names are
  skipped by our scan and by grok, so the rename is the delete:

  ```
  rename fails part way   -> requested session still listed; retry deletes the rest
  crash between renames   -> same; the renamed children are just gone
  removal fails (EACCES)  -> 200; .deleting-* stays, invisible, retried on the next delete
  ```

Write guards (all answer 409 and change nothing; checked again right before the write):

| Guard        | Rule                                                                                                                                                                         |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| registered   | the id, or any subagent ancestor, is in `active_sessions.json` with a live pid (`kill(pid, 0)`, EPERM = alive)                                                               |
| recent write | any regular file in the session folder changed in the last 30 s (backup for mlx-agent < v0.3.0, where ACP and `-p` runs never register); our own rename does not count       |
| open turn    | the transcript ends inside a turn and was written in the last 10 min                                                                                                         |
| unreadable   | `active_sessions.json` cannot be parsed                                                                                                                                      |
| containment  | the folder must be a real `<sessions>/<cwd>/<id>` directory (no symlink at either level, realpath inside the sessions root); ids with `/`, `\`, NUL or a leading `.` are 400 |

Turn metrics come from each `turn_completed`. Its `perf` block holds the turn's totals over every model call
(mlx-agent v0.3.3+; a turn has many calls). Only `ttft_ms` is the **first** call's: it is shown as First token (the
inspector adds `first of n model calls`, the chat footer tooltip says `First token (first call)`).
`decode_tokens_per_second` is token-weighted over all calls, Σ(nᵢ − 1) ÷ Σ decode time, the same formula as Decode
speed on the Metrics page; it is the chat footer's tok/s. Session Activity combines turns the same way, never a mean
of turn rates: a turn's rated tokens are its output tokens − `usage.modelCalls` (each call's first token is not
rated), its decode time is rated tokens ÷ turn rate, and the speed is Σ rated tokens ÷ Σ decode time. A turn without
a rate, a `modelCalls` count or a rated token is left out. `usage` also holds the calls of a subagent whose usage
landed inside the turn, while `perf` covers only the turn's own calls, so such a turn still counts the subagent's
tokens as rated. `prefill_ms` is the sum of the calls' time to
first token, not the native prefill window, and is read only together with `prefill_tokens`;
`prefill_tokens_per_second` and `prefill_input_tokens` are not read. Prefill speed reads only `prefill_ms` +
`prefill_tokens` (both, or neither), with `prefill_calls` for the per-call size floor and `usage.modelCalls` for
`k of n model calls`. mlx-agent v0.3.4 writes neither `prefill_tokens` nor `prefill_calls`, so the page says
`not measured in this session`.

grok's `*.lock` files are never opened: grok blocks on `LOCK_EX`, so a lock held
from Node would stall the agent. From mlx-agent v0.3.0 every resident session
(TUI, `-p`, ACP, subagents) has a row in `active_sessions.json`, so an idle holder
is detected. With an older binary, an idle ACP/headless holder with no recent
writes cannot be detected.

## Persist paged cache (`persistPagedCache`)

The `ColdCacheManager` SSD cold tier of the restore-allowlisted families: when
enabled, full paged prefix blocks are captured to the cold tier on publish and
restored on a hot-cache miss before falling back to a normal prefill. Qwen3.5
dense and MoE are not among them: their SSD tier holds DFlash2 states
(`mlx serve --max-cache-disk`, persistent under the same switch; see
[paged-cache.md](paged-cache.md#dflash2-states-on-ssd-qwen3_5--qwen3_5_moe)), and
the Cache page lists it under Model caches (above).

- One switch for every family: `mlx serve`, `mlx agent`, `mlx launch claude`
  and the desktop engine turn it **on** by default (`--no-persist-cache`, or
  `MLX_PERSIST_PAGED_CACHE=0` without a flag, turns it off) and write the
  resolved value to `MLX_PERSIST_PAGED_CACHE` before any load. A load with no
  host reads the model's `config.json` `persist_paged_cache`, else
  `MLX_PERSIST_PAGED_CACHE`, else off.
- The engine's `live-<pid>.json` heartbeat carries `persistPagedCache` (the
  resolved switch) and `coldCapture: { blocks, ms }` (the capture limits below;
  `null` when the cache is off). The dashboard live snapshot reads both per
  process; both are `null` from an engine that predates them. Whether a turn of a
  restore-allowlisted family used the cold tier is the trace row's `coldEnabled`;
  the Cache page shows both. A Qwen3.5, Flash Next or Hy4-flash turn's state tier is not in
  `coldEnabled`: its row carries `stateTier` (Qwen3.5: `persistent`, `temp` or
  `off`, the most lasting mode over the engine's loaded models, with its numbers
  in `/health` `memory_governor.dflash2_states`; Flash Next and Hy4-flash: the model's own
  `stateTierMode()`, `persistent`, `temp` or `off`, with its numbers in
  `residencyInfo()`'s `prefixDisk*` counters; `prefixDiskAdopted` is what a
  restarted load took back from the SSD before any turn).
- **One turn writes the whole prompt in both agent paths** (the restore-allowlisted
  families; not the Qwen3.5 state tier). By default a turn's capture walk writes at
  most 128 blocks (2,048 tokens) within 250 ms. `mlx agent` (through
  `mlx serve --persist-whole-prompt`) and the desktop engine ask for
  `MLX_COLD_CAPTURE_BLOCKS_PER_TURN=63125` and `MLX_COLD_CAPTURE_BUDGET_MS=10000`
  instead.
  - 63,125 = ⌈1,010,000 / 16⌉: the largest window a load serves
    (`MAX_YARN_CONTEXT_TOKENS`) in 16-token blocks, the block size every family
    uses unless its config sets `paged_block_size`. The block cap never ends a walk
    before the prompt does; the time cap does.
  - Precedence, per variable: a value in the env that native parses
    (`trim` + unsigned integer) wins, then the entry's whole-prompt value, then the
    native default (128 / 250, not written). The host writes only when the disk
    cache resolves on. `mlx serve` alone and `mlx launch claude` keep 128 / 250.
    `mlx serve` prints `[mlx] paged KV disk capture per turn: <blocks> blocks, <ms> ms`
    when the cache is on, and `InferenceHost.coldCapture` holds the same numbers.
  - Native reads both once per process, at the first paged load
    (`cold_tier::cold_capture_budget`, a `OnceLock`). Changing them needs an engine
    restart.
  - **Cost at turn end.** The walk runs on the model thread before the turn's
    final event, so the client sees the turn end after the write, and other
    sessions on that model wait too. Only blocks not already on disk are written,
    so the cost is paid once per new prefix; later turns write only their new
    tail. Worst case on a stalled disk: 10 s for the walk plus up to 10 s more for
    a hybrid family's state sidecar to get a writer slot.
  - **When the 10 s cap ends a walk,** the blocks written so far stay; the next
    turn's walk skips them and goes on, so the rest lands one turn later.
- **Restore is gated to an allowlist: `qwen3`, `gemma4`, `muse_glimmer`, `lfm2`, and `lfm2_moe`.**
  Dense qwen3 sizes its pool over every layer, so paged blocks fully determine its
  layer state. The others are hybrid — their pools cover the full-attention
  layers only — but each persists its out-of-pool state as a cold-tier
  **sidecar**: grouped sliding-window K/V for gemma4 and muse_glimmer, and
  ShortConv recurrent state for LFM2/2.5. A
  `ColdSidecarPolicy` makes the restore
  reconcile DOWN to a boundary a validated sidecar actually backs (vLLM's per-group
  rule), and the `aux_prefix_unbacked` latch fails closed if an in-process hot hit
  would resume a K/V prefix whose out-of-pool half is missing. Gemma4 instead
  installs every paged group at one validated boundary or resets them all to zero.
  A family joins the allowlist only after its restart-parity
  gate passes on real weights with `hits > 0` and `corruptions == 0` — see
  `docs/paged-cache.md`. The allowlist is enforced natively in
  `cold_tier::resolve_persist_cold`, so a family that is off it never persists or
  restores even under an explicit `persist_paged_cache` or `MLX_PERSIST_PAGED_CACHE=1`.
- The tier is fail-open: any validation or I/O failure falls through silently to a
  fresh prefill. Quota is 10 % of FS capacity capped at 100 GiB, LRU by rebuilt
  mtime recency. A write never takes the disk's free space below a reserve of
  5 % of capacity or 5 GiB, whichever is larger: it evicts the tier's own oldest
  entries first, then refuses the write (`crates/mlx-paged-attn/src/cold_cache.rs`
  `evict_for_write`). On a nearly full disk the tier therefore stays small rather
  than filling it. All destructive I/O is descriptor-relative and no-follow, contained
  to a managed `mlx-paged-v1` child.
- **Cold-tier eviction is per-block global-LRU and NOT chain-aware.** Prefix chains
  are written head-to-tail, so under sustained quota pressure the oldest blocks —
  the root/head of a chain — are evicted first. Because restore stops at the first
  missing block, an evicted head makes the rest of that chain unrestorable while its
  suffix blocks still occupy quota; the tier then falls open to a normal prefill
  (correct, just no speedup). The common case — a working set within quota — is
  unaffected. Chain-aware eviction (evict leaves/tails first, or cascade a head's
  descendants) is a future improvement.
- The quota is a **per-process best-effort cap**, not a strict cross-process limit:
  each `mlx agent` process enforces it against its own startup scan, so several
  processes sharing one root before either writes can transiently exceed it (up to
  ~N×quota for N concurrent writers). It self-corrects — the next process to scan the
  root evicts LRU back down to quota on its first write — and the free-space floor is
  enforced against a live `statvfs` re-sampled per eviction. Strict cross-process
  quota enforcement (interprocess locking) is out of scope for v1.
- `MLX_COLD_CACHE_DIR=<parent>` relocates the tier; the cache operates in a
  `mlx-paged-v1` child of that parent (never the parent verbatim). Default root is
  `~/.mlx-node/cache/paged/v1`. `coldCacheStats()` (NAPI) exposes cumulative
  counters, root path, and quota.

## Environment variables

| Var                                | Effect                                                                                                                                                                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_AGENT_METRICS=0`              | Turns off both telemetry sinks of the inference process: trace records and the runtime sampler (series + heartbeat). `0` / `false` / `off`, case-insensitive. Default on. With it off, Metrics and Engine have nothing to show.                                           |
| `MLX_PERSIST_PAGED_CACHE`          | Turns the paged KV disk cache on (`1`/`true`/`on`/`yes`) or off (`0`/`false`/`off`/`no`) when no flag decides; beats the entry default (on for `mlx serve`, `mlx agent`, `mlx launch claude` and the desktop engine).                                                     |
| `MLX_COLD_CACHE_DIR`               | Parent dir for the cold tier (operates in its `mlx-paged-v1` child). Default `~/.mlx-node/cache/paged/v1`.                                                                                                                                                                |
| `MLX_COLD_CAPTURE_BLOCKS_PER_TURN` | Blocks one turn's cold-tier capture walk may persist. Native default `128` (2048 tokens at block size 16); `mlx agent` and the desktop engine write `63125` when it holds no number native parses and the disk cache is on. A number you set wins. Read once per process. |
| `MLX_COLD_CAPTURE_BUDGET_MS`       | Wall-clock ceiling on one capture walk. Native default `250`; `mlx agent` and the desktop engine write `10000` when it holds no number native parses and the disk cache is on. A number you set wins. The turn ends after the walk. Read once per process.                |
| `MLX_MODELS_DIR`                   | Local models directory when `--models-dir` is omitted.                                                                                                                                                                                                                    |
| `MLX_AGENT_HOME`                   | `mlx agent` config home read by the Sessions page and set as `GROK_HOME` of the Work agent (`agentHome` runtime option overrides). Default `~/.mlx-agent`. `GROK_HOME` is grok-build's and is never read.                                                                 |
| `MLX_DEV_NETWORK=1`                | Dev harness only: let the download and update-check routes reach the network.                                                                                                                                                                                             |

## Security model

The HTTP attack surface no longer exists. There is no listening socket, so there is
nothing to bind, nothing to firewall, and no host/origin guard to get wrong — the
class of bug is deleted rather than mitigated. What remains:

- **The renderer is sandboxed** (`contextIsolation: true`, `nodeIntegration: false`,
  `sandbox: true`) and reaches the runtime only through a transferred MessagePort.
- **A strict CSP ships as a response header** on `app://`. `script-src` is `'self'`
  with no `unsafe-inline` and no `unsafe-eval`; `connect-src` is `'self'` only,
  because the SPA speaks over a port and has no reason to reach the network at all.
  The scheme is registered with `bypassCSP: false`, or the header would be decorative.
- **`app://` serves only inside the bundle.** A miss under `/assets/` returns 404
  rather than falling back to `index.html`, so a stale or mis-copied bundle fails
  loudly instead of returning HTML labelled `text/javascript`.
- **The folder dialog is the only main-process surface the page adds.** The renderer posts a request id; MAIN opens
  the native folder dialog for the Control Panel window only (one at a time) and answers that id with the chosen path
  or null. Nothing else crosses: no path goes in, no file is read. An attached file is read by the page from what the
  user picked, pasted or dropped; its path is never known (see [Chat and Work](#chat-and-work)).
- **Git runs for the Work strip only, and only as written** (`src/chat/work-git.ts`). `spawn` with no shell, stdin
  closed, its own process group (killed whole at a deadline), stdout at most 4 MiB, stderr 64 KiB. Every call starts
  `git -C <dir> -c core.fsmonitor=false -c color.ui=false -c core.quotePath=false`, so the app starts no fsmonitor
  daemon. The environment is built from nothing: `HOME`, `USER`, `LOGNAME`, `TMPDIR`, the merged `PATH`, `LC_ALL=C`,
  `GIT_TERMINAL_PROMPT=0`, and for reads `GIT_OPTIONAL_LOCKS=0`; no key and no inherited `GIT_*`. Reads (`rev-parse`,
  `symbolic-ref`, `for-each-ref`, `status --porcelain=v2 -z`) share one 5 s deadline. The only write is `git switch` to
  a local branch the host itself just listed, on a clean work tree, with no running Work conversation and no create in
  it, and no Work turn starts there until it ends; it runs
  the folder's hooks and filters as a terminal `git switch` would, since the user asked for it. One chain per work
  tree: reads and switches never overlap.
- **Work without a folder writes in one place.** The app makes folders only under `~/Portato` (the scratch root,
  `join(home, 'Portato')`), each new and 0700, and checks each with the Work folder rule before the agent gets it.
  It removes one only with `rmdir` (never recursive), so it never deletes a file there; the agent's rules for the
  conversation are those of that one folder, never of the scratch root.
- **Mutations are still the dangerous part** — model delete, cache clear, model cache delete, session rename/delete —
  and they are reachable by anything that gets the port. The port is minted
  by MAIN, transferred once to the Control Panel window it created, and consumed on receipt.

## API

The same route table, addressed the same way, over a MessagePort instead of HTTP.
`runtime.call({ method, path })` is the whole surface; it **never rejects for an API
failure**, it returns `{ok: true, status, body}` or `{ok: false, status, code, message}`.
That envelope is structured-clonable by design — it is what makes the port transport
cheap, and it gives code written against the in-process runtime an identical signature
over a port.

`GET /health` (and `GET /api/health`) returns `{ status, modelsDir, tracesDir, runtimeDir, homeDir }`.
`homeDir` is the user's home folder: the window shows a path under it as `~/…`, and its hover title and
copy button keep the full path.

Common params of the metrics routes, parsed by hand: `from`, `to` (epoch ms), `model` (exact `model_key`), `bucket`
(`5m`, `hour`, `day`; hour and day follow local time), `groupBy` (`model`). A request spans at most 2000 buckets.

| Route                                 | Method       | Thread    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------- | ------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/models`                         | GET          | worker    | Local models: name, path, family, quant, size, ctx window, `servedIds`, `companions`, `warnings`, `dir` (no host reading: see `/api/host`)                                                                                                                                                                                                                                                                                                                      |
| `/api/host`                           | GET          | transport | `{ totalMemBytes, diskFreeBytes }`: this Mac's memory and the free space of the models disk (`null` when the folder cannot be read); never waits for the worker                                                                                                                                                                                                                                                                                                 |
| `/api/models/:name`                   | DELETE       | worker    | Delete a model dir (path-checked)                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `/api/catalog`                        | GET          | worker    | Recommended catalog: entries with `builds[]`, each with its folder (`slug`) and state (`installed`, `present`, `heldBy`, `blocker` with `locked`, `listed`: Local models lists the folder), and the `draft` companion                                                                                                                                                                                                                                           |
| `/api/catalog/updates`                | GET          | transport | Upstream sha per catalog repo; the page joins it against `/api/catalog`                                                                                                                                                                                                                                                                                                                                                                                         |
| `/api/downloads`                      | GET / POST   | transport | List active jobs (`{id, slug, repo, kind, state, receivedBytes, totalBytes}`) / start a catalog download: body `{slug}` (a build or draft folder) → 202 `{id, slug, repo, kind}`, kind `install` \| `update` \| `cleanup` \| `locate`; at most one of `replaceBlocker: { kind, fileCount, sizeBytes, truncated, linkTarget }` (Clean up & update), `adopt: <token>` (Update), `locate: <token>` (Locate); 409 on change, or when the folder holds another build |
| `/api/downloads/adopt-plan`           | POST         | transport | `{slug}` → Update plan: `token`, `revision`, `assetsRevision`, `check` / `download` / `other` counts and bytes, up to 3 other names; 409 names the file it refuses, with `reason: 'needs-cleanup'` when Clean up & update solves it                                                                                                                                                                                                                             |
| `/api/downloads/:id`                  | DELETE       | transport | Cancel a download                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `/api/downloads/:id/events`           | GET          | transport | Not served over the port — use `runtime.subscribe(jobId, fn)`; answers 503. `progress.phase: 'check'` while Update hashes a file; `done.keptDir` names kept files                                                                                                                                                                                                                                                                                               |
| `/api/coding-agents`                  | GET / POST   | transport | Coding agent state with the managed instruction `block` / body `{action: 'refresh'}` or `{action: 'install', agent}`                                                                                                                                                                                                                                                                                                                                            |
| `/api/coding-agents/models`           | GET          | worker    | Local model ids the coding agents can be pointed at                                                                                                                                                                                                                                                                                                                                                                                                             |
| `/api/metrics/models`                 | GET          | worker    | Model keys in first-seen order (color registry seed, model filter)                                                                                                                                                                                                                                                                                                                                                                                              |
| `/api/metrics/summary`                | GET          | worker    | Totals, latency quantiles, per model, finish reasons, failures, endpoints, dispatches; `previous` = the window of equal length before `from`; totals carry `decodeRatedTokens/Ms` (Decode speed = Σ(n−1) ÷ Σ decode ms) and `prefillRatedTokens/Ms` (rows of 256+ computed tokens)                                                                                                                                                                              |
| `/api/metrics/timeseries`             | GET          | worker    | Per-bucket series, optionally per model                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `/api/metrics/distributions`          | GET          | worker    | Histograms with bins chosen by the API                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `/api/metrics/speculative`            | GET          | worker    | MTP per model (acceptance per draft position; `decodeWith` / `decodeWithout` = `{tokens, ms}` sums for decode speed with and without MTP) and a trend                                                                                                                                                                                                                                                                                                           |
| `/api/metrics/turns`                  | GET          | worker    | Turns, newest first; cursor paging (`beforeTs` + `beforeTraceId`), `limit` ≤ 2000, `traceId`, `outcome`                                                                                                                                                                                                                                                                                                                                                         |
| `/api/runtime/series`                 | GET          | worker    | `from`, `to`, `bucketMs` (5 s … 1 h) → points (per bucket: samples, memory, in flight, `busyMs`, turn and token deltas, `phaseMs: { loading, waiting, prepare, prefill, decode, running }`, `rateOutTokens`, `rateOutMs`, `ratePrefillTokens`, `ratePrefillMs`; the work fields are sums, 0 for samples that predate them), scheduler deltas, processes, loads                                                                                                  |
| `/api/runtime/live`                   | GET          | transport | `RuntimeLive` snapshot: status, supervisor, processes                                                                                                                                                                                                                                                                                                                                                                                                           |
| `/api/cache`                          | GET / DELETE | worker    | Cold-tier scan and daily trend (`trend[].t`, local day start) / clear all or evict older-than-N-days                                                                                                                                                                                                                                                                                                                                                            |
| `/api/cache/prefix`                   | GET          | worker    | Model caches per folder: `roots[]` (`namespaces`, `volume`, `readError`, `engines` with their limit), `counting`, `canLock`                                                                                                                                                                                                                                                                                                                                     |
| `/api/cache/prefix/:namespace`        | DELETE       | worker    | Body `{root, openedMs}` → one step `{removed, freedBytes, done, left, directoryRemoved}`; repeat until `done`. 400, 403, 404, 409 `held`/`changed`/`moved`/`no-lock-check`                                                                                                                                                                                                                                                                                      |
| `/api/ingest`                         | POST         | worker    | Trigger an incremental rescan                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `/api/sessions`                       | GET          | worker    | `?q=&cwd=&limit=&offset=` → `{sessions, total, cwds}`; limit 1..500, default 100                                                                                                                                                                                                                                                                                                                                                                                |
| `/api/sessions/:id`                   | GET          | worker    | `{session, transcript, turns, transcriptError?, resumeCommand}`; subagent ids work too; tool, text and subagent entries carry `endTs`                                                                                                                                                                                                                                                                                                                           |
| `/api/sessions/:id/timeline`          | GET          | worker    | `?turn=<n>` → `SessionTimelineResponse`: one turn as lanes, spans and handoffs; metadata only (below)                                                                                                                                                                                                                                                                                                                                                           |
| `/api/sessions/:id/pictures/:call/:n` | GET          | worker    | `{mediaType, data}`: the `n`-th picture (from 0) a tool call read, from `updates.jsonl` with rewinds applied; 404 unknown session, call or index, 415 not png / jpeg / webp / gif, 413 past 4 MiB                                                                                                                                                                                                                                                               |
| `/api/sessions/:id`                   | PATCH        | worker    | Body `{title}` → `{id, title}`; 400 bad title, 404, 409 live/recent/changed                                                                                                                                                                                                                                                                                                                                                                                     |
| `/api/sessions/:id`                   | DELETE       | worker    | `{deleted: [id, ...children]}`; 404, 409 when it or any child is live                                                                                                                                                                                                                                                                                                                                                                                           |

`/api/sessions`, `/api/sessions/:id` and its timeline read both session homes. `origin` is `agent`, `chat` (the Chat
home) or `work` (an agent-home session the Work index `~/.mlx-node/chat/conversations.json` names). `resumeCommand` is
null for a `chat` session: it cannot be resumed in a terminal. The pages tag `chat` and `work` rows, link them to
`/chat?c=<id>` (Open in Chat), and rename and delete them through the Chat routes (refused only while an answer
runs), not through the session routes.

### Chat routes

| Route                                                      | Method | Thread    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------- | ------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/chat/state`                                          | GET    | transport | `ChatHostState`: inference, the load, the models in memory, models, agent binary, both modes, providers, active turns, mode picks                                                                                                                                                                                                                                                                                |
| `/api/chat/conversations`                                  | GET    | worker    | `?mode=chat\|work` → `{conversations: ConversationRow[]}`                                                                                                                                                                                                                                                                                                                                                        |
| `/api/chat/conversations`                                  | POST   | transport | `{mode, text, modelId?, cwd?, noFolder?, draft?, attachments?, plan?, thinking?}` → 201 `{conversation, turnId}`; starts the first turn; `noFolder: true` (Work, never with `cwd`) runs in a new folder of `~/Portato`, 409 `folder-not-made`; `draft` (Work) applies its picked mode; `plan` Work only; `thinking` one of `none`, `low`, `medium`, `high`, `max`                                                |
| `/api/chat/conversations/:id/prompt`                       | POST   | transport | `{text, attachments?, plan?, whenBusy?}` → 202 `{turnId}`; busy: `{queued}`, `{steered}` (`'now'`); `plan` Work only, 400 in Chat; 409 `queue-full`, `steer-busy`                                                                                                                                                                                                                                                |
| `/api/chat/conversations/:id/queue/:qid`                   | PATCH  | transport | `{text}` (the row's new text, its edit ends) or `{editing}` → `{}`; 404 for a row that is gone                                                                                                                                                                                                                                                                                                                   |
| `/api/chat/conversations/:id/queue/:qid`                   | DELETE | transport | no body → `{}`; Remove                                                                                                                                                                                                                                                                                                                                                                                           |
| `/api/chat/conversations/:id/queue`                        | DELETE | transport | no body → `{}`; Clear                                                                                                                                                                                                                                                                                                                                                                                            |
| `/api/chat/conversations/:id/queue/release`                | POST   | transport | no body → `{}`; Send next: the hold ends, the first row goes when the conversation is free                                                                                                                                                                                                                                                                                                                       |
| `/api/chat/conversations/:id/queue/:qid/send`              | POST   | transport | `{how: "now"}` → `{steered: true}`; Send now, Work only (prompt's `whenBusy: 'now'` alike); 409 `steer-*`; 503 `steer-late`, the row stays; 404 a row that is gone                                                                                                                                                                                                                                               |
| `/api/chat/conversations/:id/cancel`                       | POST   | transport | → `{cancelled}`                                                                                                                                                                                                                                                                                                                                                                                                  |
| `/api/chat/conversations/:id/compact`                      | POST   | transport | no body → 202 `{}` once the run started; the end is the stream's `compaction` `end`; 409 `conversation-busy` (turn, wake, run, op); 503 `inference-stopped`                                                                                                                                                                                                                                                      |
| `/api/chat/conversations/:id/rewind/plan`                  | POST   | transport | `{entry, ts}` → `{token, removes, background, helpers, attachments, files, filesMore}` (Work: the agent's dry run); 409 `rewind-not-point`, `rewind-changed` (the entry at that place is another message), `rewind-goal` (the open goal began after it), `conversation-busy`, `engine-lost`; Work 503 `inference-stopped`, `agent-slow` (soft 20 s), `load-slow`                                                 |
| `/api/chat/conversations/:id/rewind`                       | POST   | transport | `{token, mode: "all" \| "conversation" \| "files"}` (Chat: `conversation` only) → 202 `{}` once started; its end is the `rewind` event; 409 `rewind-changed`, `conversation-busy`, `engine-lost`; 503 `inference-stopped`                                                                                                                                                                                        |
| `/api/chat/conversations/:id/fork`                         | POST   | transport | `{}` or `{entry, ts}` → `{id}` of the copy once it is on disk (Work: in the index); 409 `fork-changed` (another entry at `entry`), `fork-not-point`, `conversation-busy`, `engine-lost`; 503 `agent-slow` (soft 60 s)                                                                                                                                                                                            |
| `/api/chat/conversations/:id/recap`                        | POST   | transport | no body → 202 `{}` once the wait started; its state and end ride the stream's `recap` events; 409 `conversation-busy` (turn, wake, history op, another recap); 503 `inference-stopped`                                                                                                                                                                                                                           |
| `/api/chat/conversations/:id/goal`                         | POST   | transport | `{action}` (`pause`, `resume`, `clear`) → 202 `{turnId}` (the agent's `/goal <action>`) or `{stopped: true}` (Pause while an answer runs is Stop); 400 in Chat; 409 `conversation-busy` (Resume, Clear while busy)                                                                                                                                                                                               |
| `/api/chat/conversations/:id/context`                      | GET    | transport | → `{used, parts}`: the agent's parts (`_x.ai/session/info`, soft 5 s) while loaded, else `parts: null` and the stored use; never starts the agent; answers while it works                                                                                                                                                                                                                                        |
| `/api/chat/conversations/:id/runs/:runId/stop`             | POST   | transport | no body → `{}`; Stop run of one live workflow run; 404 "No such run." when the host does not hold it; 409 when the agent rejects the stop, 503 when its reply does not come                                                                                                                                                                                                                                      |
| `/api/chat/conversations/:id/tasks/:taskId/stop`           | POST   | transport | no body → `{}`; Stop of one live command or monitor (`_x.ai/task/kill`, soft 10 s); 404 "No such task." when not live; 503 when the agent errs or does not answer                                                                                                                                                                                                                                                |
| `/api/chat/conversations/:id/subagents/:childId/stop`      | POST   | transport | no body → `{}`; Stop of one of the conversation's own subagents; 404 "No such subagent." when it is not running here; 503 when the agent's reply does not come                                                                                                                                                                                                                                                   |
| `/api/chat/conversations/:id/subagents/:childId/message`   | POST   | transport | `{text}` (at most 100,000 characters) → `{result}`, how the send ended; 404 "No such subagent." when it is not running here or the agent refused it as gone                                                                                                                                                                                                                                                      |
| `/api/chat/conversations/:id/subagents/:childId/watch`     | POST   | transport | `{window}` → `{childId, attempts, entries, live}`: the Agent view; while `live`, the stream's `agent-entries` events for `window` carry its later ops; 404 "No such subagent." (not the conversation's own); 409 a third live view in one window                                                                                                                                                                 |
| `/api/chat/conversations/:id/subagents/:childId/watch`     | DELETE | transport | `?window=<id>`, no body → `{}`; the view closed; 404 "No such subagent." (not the conversation's own)                                                                                                                                                                                                                                                                                                            |
| `/api/chat/conversations/:id/schedules/:scheduleId/delete` | POST   | transport | no body → `{}`; Delete of one schedule the transcript lists (`_x.ai/scheduler/delete`, soft 10 s), the conversation loaded first when no running agent holds it; 404 "No such schedule." when not listed or the agent no longer holds it; 409 `engine-lost`; 503 `load-slow`, or when the agent errs or does not answer                                                                                          |
| `/api/chat/conversations/:id/interactions/:iid`            | POST   | transport | `{optionId}` (`{optionId: 'reject-once', note}`: Deny with a note of at most 4,000 characters), `{never: 'short' \| 'exact'}` (Never allow: Deny, then the rule is saved; → `{never: {saved, ruleId?, words?}}`), `{decision, feedback?}`, `{answers}`, `{reply, answers?}` (a question), `{mcp: 'accept', values}` (`{fields}` when refused; a link: no `values`) or `{mcp: 'decline'}`; Work only; 404 in Chat |
| `/api/chat/conversations/:id/never`                        | GET    | transport | → `{folder, rules: [{id, words, at}]}`: the never-allow rules of the Work folder, oldest first; 404 in Chat                                                                                                                                                                                                                                                                                                      |
| `/api/chat/conversations/:id/never/:ruleId`                | DELETE | transport | no body → `{}`; removes the rule; 404 "No such rule."; 500 when the rule file was not written (the rule stays)                                                                                                                                                                                                                                                                                                   |
| `/api/chat/conversations/:id/permission-mode`              | POST   | transport | `{mode, up}` → `{mode}`; Work only; 409 `card-open` or `mode-changed`; 503 if a line write fails: see [Chat and Work](#chat-and-work)                                                                                                                                                                                                                                                                            |
| `/api/chat/drafts/:draft/permission-mode`                  | POST   | transport | `{mode, up, cwd?, noFolder?, temp?}` → `{mode}`; the New Work screen's pick; `up: true` needs `cwd` or `noFolder: true`, never both; 409 `mode-changed` or `temp-unnamed`                                                                                                                                                                                                                                        |
| `/api/chat/work/hooks`                                     | GET    | transport | `?cwd=` or `?noFolder=1` → `{userHooks}`; the folder rule of create first (none for `noFolder`)                                                                                                                                                                                                                                                                                                                  |
| `/api/chat/work/git`                                       | GET    | transport | `?cwd=` → `{kind: 'not-git' \| 'no-git' \| 'slow'}` or `{kind: 'repo', top, head, branches, total, busy, changes}`; `head` `{branch}`, `{detached}` or `{unborn}`; a branch `{name, at, current, elsewhere}`; `changes` the uncommitted change count, null when unknown; 400 `cwd-invalid` (the folder rule first)                                                                                               |
| `/api/chat/work/git/switch`                                | POST   | transport | exactly `{cwd, branch}` (branch 1–1,024 characters) → `{outcome: 'switched', git}` or `{outcome: 'refused' \| 'timeout', text, git}`; 400 `cwd-invalid`; 404 `branch-unknown`; 409 `folder-busy`, `dirty`, `status-unknown`, `git-unavailable`                                                                                                                                                                   |
| `/api/chat/work/settings`                                  | GET    | transport | `{claudeCode}`: the app-wide Work settings                                                                                                                                                                                                                                                                                                                                                                       |
| `/api/chat/work/settings`                                  | PUT    | transport | exactly `{claudeCode: boolean}` → `{claudeCode}`; saved first, then the Work host goes stale                                                                                                                                                                                                                                                                                                                     |
| `/api/chat/conversations/:id`                              | PATCH  | transport | `{title}`, `{modelId}` or `{thinking}` → `ConversationInfo`; `{thinking}` also while an answer runs                                                                                                                                                                                                                                                                                                              |
| `/api/chat/conversations/:id`                              | DELETE | transport | → `{}`; open streams end with `gone`                                                                                                                                                                                                                                                                                                                                                                             |
| `/api/chat/agent`                                          | POST   | transport | `{action: 'restart', mode?}` or `{action: 'install'}` → `ChatHostState`                                                                                                                                                                                                                                                                                                                                          |
| `/api/chat/engine/unload`                                  | POST   | transport | no body → `{unloaded: string[]}`; 409 `engine-busy` while an answer, a workflow or a load runs                                                                                                                                                                                                                                                                                                                   |

Bodies hold exactly the listed fields: any other field (`_meta`, `mcpServers`, `pluginDirs`, `env`, …) is 400. Text is
1–100,000 characters (413 above); text whose first non-space character is `/` is a slash command: a name the agent
would not read, an app command, a blocked or a hidden built-in is 400 with `reason: 'slash-text'` ("No such command
here.") at the route; the service then refuses what the conversation's mode and the agent's list do not allow
(Slash commands, above), so such text never reaches the agent. `attachments` is a list of `{kind: 'file', name, text}` and
`{kind: 'image', mediaType: 'image/png' | 'image/jpeg', data}` (base64, no `data:` prefix); an empty list is no list,
and an item that breaks a limit of [Chat and Work](#chat-and-work) is 400 with `reason: 'attachment-refused'` ("The
attached files were refused."). The service refuses a picture in Work, or for a model whose row is not `images: true`,
the same way. A Chat route failure carries a fixed sentence and, where it has one, a `reason` (a
`ChatFailureCode`); any other error becomes `E_UNAVAILABLE` "The agent could not be reached." and its text is dropped.

Streams are not routes: `runtime.watchChat({scope: 'host'}, fn)` sends `{type: 'state'}` (coalesced to 100 ms) and
`{type: 'conversations-changed'}`; `runtime.watchChat({scope: 'conversation', conversationId}, fn)` sends a full `sync`
first, then `ops`, `turn`, `interaction`, `interaction-closed`, `notice`, `notice-cleared`, `conversation`, `permission-mark`, `compaction`, `recap`,
`runs` and `gone`, each with `seq` one above the last. `runs` (and `sync.runs`) is the whole list of the conversation's live
workflow runs, `{runId, stopping}`; a `turn` whose `origin` is `'wake'` is a turn the agent started with no prompt. A page that sees a gap re-opens the stream for a fresh `sync`. Over the port the request kind is
`chat`; streams close on `unsubscribe` and when the port closes.

`GET /api/sessions/:id/timeline?turn=<n>` → `{turn, turns, open, live, stopReason, asOfMs, range, lanes, spans,
handoffs, notes}`: kinds, times, statuses, tool and model names and counts, never prompt, text, tool input or output or
error text. It reads at most 8 subagent event files of that turn, each ≤ 8 MB. 400 for a bad `turn`, 404 for an
unknown session or turn; an unreadable transcript is 200 with `transcriptError`. In the detail route a hidden wake
prompt is a `wake` entry (cause, workflow, subagent, task or tasks, text); the timeline route carries none of it. `range.startMs` is the axis origin and
`range.turnStartMs` where the turn itself starts; they differ only for a woken turn with background subagents in the gap
before it, which the main lane draws as one `idle` span.

The worker holds no keys, so the main thread redacts the answers of `GET /api/sessions`, `/api/sessions/:id`,
`/api/sessions/:id/timeline` and `/api/chat/conversations` for every session with the provider keys and the inference
token (a Work command may print one), waiting at most 10 s for the Keychain.

## Web search providers

Chat's `web_search` / `web_fetch` tools (the `mlx_web` MCP server) and the Work agent's provider skills use one
search provider (Exa, DeepSeek or Brave: exactly one or none) and, independently, Firecrawl. When Firecrawl is on it
does every `web_fetch`; otherwise the search provider does.

- **Keys** live in the macOS Keychain, service `mlx-node`, accounts `search-exa-api-key`, `search-deepseek-api-key`,
  `search-brave-api-key`, `crawl-firecrawl-api-key` (`@napi-rs/keyring`). The providers service reads each once into
  memory on first use; every Keychain call gives up after 8 s, and then the page shows `keychain: 'unavailable'` and
  the hosts start with no key. While it is unavailable, each `GET /api/providers` reads every provider again (8 s
  each) and takes only the keys the cache never had; a late answer to a timed-out call is taken too. No create or load
  path calls the Keychain. A replaced or removed key stays in memory as a redaction needle until the process exits.
- **Settings** `<chat root>/providers.json` (0600): `{version: 1, search, firecrawl, checks: {id: {status, at}}}`. No
  key. A selection whose key is gone is cleared on start and on a Check; one cleared on start is reported once as
  `vanished` (the first `GET` carries it). A Check that ends after a key PUT or DELETE of the same provider is
  dropped, and a web result's `rejected` / `no-credit` is recorded only while the key that got it is still current.
- **A new key is checked first** (Exa `GET /monitors?limit=1`, where 403 `FEATURE_DISABLED` also means the key is
  good, DeepSeek `GET /user/balance`, Brave one `test` search, Firecrawl `GET /v2/team/credit-usage`, 10 s):
  `rejected` (401/403) is not saved and any earlier key stays; `no-credit` (402, or the account says so) and
  `unverified` (any other answer) are saved. The first search key becomes the selection when none is set, and a
  Firecrawl key turns Firecrawl on, only when its skill folder installs; otherwise the key is saved and the provider
  stays off.
- **Skills** are bundled in `packages/dashboard/skills/`: `vendor/{exa,brave,firecrawl}` from pinned upstream commits
  (each with `LICENSE` and `PROVENANCE.md` listing every local edit), `own/deepseek`, and `manifest.json` (sha256 and
  size per file). Selecting a provider installs its folder into `<chat root>/plugins/<id>` (temp folder, then rename;
  links are never followed); deselecting removes it, but a sync in which any wanted folder fails removes nothing. "Installed" means the folder holds exactly the manifest's files
  with matching hashes; it is checked again on every `GET /api/providers`. A damaged bundle changes nothing and
  answers `reason: 'skill-bundle-corrupt'`; a copy that fails on disk changes nothing and answers `E_UNAVAILABLE`
  "Skills could not be installed." with no reason.
- **Work** gets the selected providers' key variables (`EXA_API_KEY`, `DEEPSEEK_API_KEY`,
  `BRAVE_SEARCH_API_KEY`/`BRAVE_API_KEY`, `FIRECRAWL_API_KEY`) and their skill folders. The key is readable by
  everything the Work agent runs: shell commands, MCP servers and hooks. The Firecrawl folder and `FIRECRAWL_API_KEY`
  are passed only when a `firecrawl` executable is on the `PATH` a Work command gets (`work.missing: 'firecrawl-cli'`
  otherwise); the app never installs the CLI. With a Firecrawl key, Work also gets
  `FIRECRAWL_API_URL=https://api.firecrawl.dev`: the CLI takes that variable before an address it saved itself
  (`firecrawl config --api-url`), so the key goes to Firecrawl unless a command passes `--api-url`. `~/.zshenv` can
  still set it again (each command runs with `$SHELL -c`, below). That `PATH` is not the app's own (an app started from Finder has
  `/usr/bin:/bin:/usr/sbin:/sbin`): `mlx-agent` asks the user's login shell (`$SHELL -lc`, with `~/.zshrc` or
  `~/.bashrc` sourced, 5 s limit), puts its own entries after the answer, and runs each command with `$SHELL -c`, which
  reads `~/.zshenv` or `$BASH_ENV` once more. `chat/agent/work-path.ts` asks the same two questions and reads back only
  `$PATH`. In the app it starts from the agent's own `PATH`: the one `-ilc` answer of the host start (Work
  environment, above) first, then the app's, so a CLI only the login shell puts on `PATH` is found. It asks only while Firecrawl is on: once when the providers service starts, and again on
  `GET /api/providers` (at most once in 15 s) and on a route that turns Firecrawl on while the CLI is not found. No
  host start and no turn waits for it; until the first answer the CLI counts as not found, and the answer restarts idle hosts as any change
  does. A login shell that fails or is too slow gives the app's own `PATH`, as in the agent. A login shell close to
  the 5 s limit can still read as found while the agent's capture times out (the agent runs a second copy of the rc
  files beside it for its alias snapshot). The file follows `capture_login_env` and `spawn_static_command` of
  mlx-agent v0.3.3 (`xai-grok-tools/src/computer/local/terminal.rs`); read them again when the agent release moves,
  and run row U16 of the live smoke.
- **Chat** never sees a key in a prompt or argv: `mlx_web` gets it in its per-session MCP `env`
  (`MLX_WEB_SEARCH_PROVIDER`, `MLX_WEB_SEARCH_KEY`, `MLX_WEB_FETCH_PROVIDER`, `MLX_WEB_FETCH_KEY`). The server talks
  only to the four fixed provider hosts, with no redirects (a 3xx is `provider-error`), a 15 s (search) / 30 s
  (fetch) timeout and a 2 MiB reply cap; `web_fetch` refuses local names, private and special IPv4 ranges, IP literals
  in brackets, user info, empty host labels, white space, control characters, backslashes and non-ASCII before the
  path, before any call, and forwards the address without its fragment. It also reads only an address its session's
  ledger file lists (`MLX_WEB_LEDGER`: `chat/web-ledger/<uuid>.json`, 0600 in a 0700 folder, a JSON array of normal
  forms): the addresses of user messages, of parsed search results and of the text of pages read. The host rewrites
  the file (temp and rename, synchronous) at every change, so it is current before the next call's permission
  request is read; it is removed with the session and at host close, and a start removes the files of sessions the
  host does not know. Any other address, or no readable file, is `WEB_ERROR blocked`: a tool error, so the answer
  goes on.
- **The Web Search page**: the Use panel's second list, "Crawler · One at a time", shows Firecrawl only with a key,
  and Off (its note says who reads pages then); with no key "Add a Firecrawl key below." stands in place of the list.
  Both lists are locked while the Keychain is down.
- **A change** (key, selection, a late Keychain answer) marks both agent hosts stale: each restarts when no turn is in
  flight (`applyPending` while one is). The selection is also changed from the composer's `+` menu: the Web search
  panel sends `search` only, the Crawler panel `firecrawl` only.

| Route                         | Method | Body                                      | Reply                                       |
| ----------------------------- | ------ | ----------------------------------------- | ------------------------------------------- |
| `/api/providers`              | GET    |                                           | `ProvidersResponse`                         |
| `/api/providers/:id/key`      | PUT    | `{key}`                                   | `{saved, check, providers}`                 |
| `/api/providers/:id/key`      | DELETE |                                           | `ProvidersResponse`                         |
| `/api/providers/:id/validate` | POST   |                                           | `{check, providers}` (`check` null: no key) |
| `/api/providers/selection`    | PUT    | `{search?: id \| null, firecrawl?: bool}` | `ProvidersResponse`                         |

All run on the transport thread. Errors: `E_BAD_REQUEST` "That does not look like an API key." (the key is never
quoted) or an unknown field; `E_NOT_FOUND` for an unknown provider id; `E_CONFLICT` "Add a key for X first.";
`E_UNAVAILABLE` with `reason: 'keychain-unavailable'` or `'skill-bundle-corrupt'`; `E_INTERNAL` "The key changed, but
the settings could not be saved." when a key PUT or DELETE changed the Keychain but not `providers.json` (the hosts
are still told). The page's types are
`ui/src/lib/providers-types.ts`, a copy of `src/chat/providers/types.ts`. The dev harness keeps keys in memory and
points every provider at fake servers on 127.0.0.1 (a key starting `bad` is rejected, `nocredit` has no credit,
`slow` times out the check).
