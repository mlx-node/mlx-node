# Dashboard (Control Panel window)

Browse local models, `mlx agent` sessions, inference metrics, and the
paged-attention cold cache. It ships inside `@mlx-node/dashboard` and is the Control Panel
window of the **mlx-node desktop app** — open it from the tray.

**There is no `mlx dashboard` command and no HTTP server.** Both were removed once
the app replaced them: the SPA is served from a custom `app://` scheme and speaks to
the runtime over a MessagePort. That deletes a whole class of exposure rather than
guarding it — there is no port to bind, nothing to firewall, and no `--host` that can
put an unauthenticated control panel API on a LAN.

The dashboard still never links the native addon (no Metal init, instant start), and
all data still comes from disk under `~/.mlx-node`.

The [macOS desktop app](../packages/desktop/README.md) downloads app updates in the
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
cannot silently land on the wrong thread. The transport thread keeps the four download
routes, the catalog update check, the two coding-agent action routes and
`/api/runtime/live`: network I/O, progress events and the live snapshot must never
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

| Process    | Started                        | cwd                                    | `HOME` / `GROK_HOME`                                | Environment                                                                           |
| ---------- | ------------------------------ | -------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Chat agent | first Chat send, on demand     | an empty app folder                    | an empty app folder / `~/.mlx-node/chat/chat-agent` | built from nothing: no inherited variable, no key; no shell tool                      |
| `mlx_web`  | by the Chat agent, per session | —                                      | —                                                   | the selected search / fetch key in its MCP `env` only                                 |
| Work agent | first Work send, on demand     | the folder the user chose, per session | the real `HOME` / `~/.mlx-agent`                    | the Control Panel's own, minus the app's internals, plus the selected providers' keys |

The Work agent is `mlx agent` as in a terminal: it runs shell commands in the chosen folder, and the user's own MCP
servers, hooks and shell commands can read the provider keys in its environment. The page says so.

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
`models`.

## Using it as a library

```ts
import { createDashboardRuntime } from '@mlx-node/dashboard';
import { serveRuntimeOverPort } from '@mlx-node/dashboard';

const runtime = createDashboardRuntime({ modelsDir, dbPath });
// `call({method, path})` is the whole API surface; it never rejects for an API
// failure, it returns a `{ok:false, status, code, message}` envelope.
```

## Pages

Eleven routes. Nine sit in the sidebar (`⌘1`…`⌘9`, also in the `⌘K` command palette); one route table
(`ui/src/shell/routes.tsx`) feeds the router, the sidebar, the breadcrumb, the hotkeys and the palette.

| Key       | Goes to                                                                       |
| --------- | ----------------------------------------------------------------------------- |
| `⌘1`      | Overview                                                                      |
| `⌘2`      | Chat                                                                          |
| `⌘3`…`⌘8` | Sessions, Metrics, Engine, Cache, Models, Coding Agents                       |
| `⌘9`      | Web Search                                                                    |
| `⌘⇧O`     | A new conversation in the last used mode                                      |
| `⌘K`      | The command palette (it also lists the 20 newest conversations of both modes) |

| Route            | Page           | What it shows                                                                                                                                                                                                                                                                        | Data                                                                                                         |
| ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `/`              | Overview       | Live engine panel and health list; "Last 24 hours" tiles (each opens a detail panel in place); turns per hour; storage tiles; latest turns                                                                                                                                           | live channel, `/metrics/summary`, `/metrics/timeseries`, `/metrics/turns`, `/models`, `/cache`, `/downloads` |
| `/chat`          | Chat           | Chat and Work: the conversation list (search over both modes), the thread, the composer with model picker, web chip and (Work) folder chip; Work's permission and plan cards. `?c=<id>` opens a conversation, `?mode=chat\|work` the New screen. See [Chat and Work](#chat-and-work) | `/chat/*` routes, the Chat host and conversation streams, live channel                                       |
| `/sessions`      | Sessions       | `mlx agent` sessions and the app's conversations (tagged Chat or Work): search, directory filter, live filter, summary tiles, sortable table                                                                                                                                         | `/sessions`                                                                                                  |
| `/sessions/:id`  | Session detail | Activity tab (session totals, per-turn charts, turn inspector with an agent timeline (one lane per agent, handoffs, replay; live while the turn runs), subagents) and Transcript tab (turn groups, find); a live session shows its open thinking block and running tool              | `/sessions/:id`, `/sessions/:id/timeline`                                                                    |
| `/metrics`       | Metrics        | Six tabs: Summary · Speed · Usage · Speculation · Failures · Turns. One header with range and model filter; only the open tab queries. KPI tiles pick the main chart; `?detail=` opens a tile's panel                                                                                | the six `/metrics/*` routes                                                                                  |
| `/engine`        | Engine         | Live part: "Now" ticker, output rate, memory, turns in flight, loaded models, KV block pool. History part (1h · 6h · 24h · 7d): memory, busy time, decode batch size, model loads, processes, supervisor                                                                             | live channel, `/runtime/series`, `/models`                                                                   |
| `/cache`         | Cache          | Cold-tier usage against quota, objects by age with an eviction preview, hit / miss trend, bytes written and restored, cold-tier health counters; evict older, clear all                                                                                                              | `/cache`                                                                                                     |
| `/models`        | Models         | Local models table (also the legend of the disk-use bar), usage per model over 30 days, recommended catalog with installs; `?focus=` opens a model's drawer                                                                                                                          | `/models`, `/catalog`, `/catalog/updates`, `/downloads`, `/metrics/summary`, `/metrics/timeseries`           |
| `/coding-agents` | Coding Agents  | The model and command the coding agents will use, per-agent setup with the instruction block it adds                                                                                                                                                                                 | `/coding-agents`                                                                                             |
| `/providers`     | Web Search     | "Now" summary of what Chat and Work get; the search provider choice (one of the configured Exa, DeepSeek, Brave, or Off) and the Firecrawl switch, with the skills each installs; one key card per provider (save, check, replace, remove; the key is never shown after entry)       | `/providers`                                                                                                 |
| `/welcome`       | Onboarding     | First run: pick and download a model, then set up coding agents. Outside the shell; `/` sends here while the model library is empty                                                                                                                                                  | `/models`, `/catalog`, `/downloads`, `/coding-agents`                                                        |

The Engine "In flight" card is a table with fixed columns (Model, Phase, Progress, Started, Context) under a
header that counts turns per served model ("2 turns on 1 model"). A row names the model that serves the turn
(`modelId`); when the client asked for another name, such as the app's `mlx-local` alias, the row says
"asked for `mlx-local`" under the model name, so two requests on one model never read as two models. "Started" is
the time since the turn began, on the snapshot clock.

The Phase badge and the Progress text follow the native phase of the turn (`arc-x/live-turn-row`):

| Native phase | Badge     | Progress                                                                | When                                                                     |
| ------------ | --------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `queued`     | Waiting   | model busy                                                              | the model thread works on other requests (also before it picks this one) |
| `slot`       | Queued    | waiting for a slot                                                      | the scheduler cannot start it: lanes full, memory refused, a held start  |
| `prepare`    | Preparing | restoring N cached tokens (`cachedTokens` > 0), else reading the prompt | render, prefix match, cache restore, GDN replay before its prefill       |
| `prefill`    | Prefill   | done of total tokens, with a bar                                        | as before                                                                |
| `decode`     | Decode    | tokens out                                                              | as before                                                                |
| none         | Running   | no progress for non-streamed turns                                      | a turn without a progress handle                                         |

"waiting for a slot" is printed for `slot` only. From `prefill` on the native phase never moves back.

The Cache page opens with one disk cache line: the switch the running engine's host resolved (`persistPagedCache`
of its heartbeat: On, Off, or Unknown for an engine started before the field), the most one turn's capture walk may
write (`coldCapture`: up to N prefix blocks, or "the whole prompt" from 63,125 blocks, and the walk's time cap; a
walk the time cap ends early is picked up by the next turn; nothing for an older engine), and how many recorded
turns ran with the tier on (`scope.enabledTurns`) and off (`scope.disabledTurns`). The turn counts cover every
cache folder, not only the one the page shows. When more than one engine runs, the line names the one it reads.
Neither the switch nor the turn counts prove a write: a turn with the tier on may write nothing. The written side is
the rest of the page (bytes and blocks on disk, restores).

One drawer, `TurnDrawer`, shows one turn's phase waterfall and numbers. Overview, Metrics and the Failures tab open
it (`?turn=<traceId>`, answered by `/metrics/turns?traceId=`). It is the only user of the phase waterfall
(`arc-x/span-timeline`).

Where the numbers come from:

| Number                                          | Source                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Turn counts, tokens, latency, speed, failures   | trace records (`traces` table). Percentiles are nearest-rank, computed in JS on the database worker |
| Speculation (MTP) acceptance                    | the MTP fields of trace records with `outcome = 'ok'`                                               |
| Memory, busy time, batch size, loads, processes | the runtime series (`runtime_*` tables)                                                             |
| Anything labelled "now" or "live"               | the live channel (heartbeat files + `desktop/supervisor.json`), never the index                     |
| Cold-tier usage and objects                     | a scan of the cold-tier directory on each `/cache` call                                             |
| Sessions and transcripts                        | the session folders, read on each call                                                              |

A model is keyed by `modelId ?? model` (`model_key`) and keeps one color on every page: 8 color slots in a registry
kept in `localStorage`, models 9–16 reuse the hues with a second encoding, later ones fold into "Other".

Every catalog entry installs its Unsloth UD-Q4_K_XL GGUF (one variant out of
the repo's dozens) plus the base-model tokenizer sidecars fetched from the
entry's `assetsRepo`; the download runner glob-filters the manifest and stages
the sidecars before publishing. No current entry carries a `draft` companion —
Qwen3.8-27B ships its MTP speculative weights inside the same repo and the
native loader auto-detects them. The catalog still renders a draft companion
(with its own install button, progress, cancel, and update controls)
whenever a catalog entry carries one, for example a manually installed
**DFlash2** draft: the roughly 3.85 GB checkpoint lives at
`qwen3.8-27b-dflash2` in the configured models directory, and compatible
Qwen3.8-27B targets find it automatically on their next load. It is not another
chat model. Draft-only checkpoints are excluded from the local-model table and
model counts. Their bytes still count toward disk use on Models and Overview.
Expand **Companion weights** in the local models panel to inspect or delete them;
deleting a separate companion leaves the target model in place.

A delete, from a row, a companion line or the drawer, takes the entry off the table, the tiles and the disk-use bar
as soon as the runtime answers ok (`useJson(...).patch`); the full `/models` refetch that follows replaces that
body, so a slow listing (a GGUF-heavy models directory can take seconds) never leaves a deleted row with a live
trash button. If that refetch fails, the page says "Could not refresh the model list" with Try again. A GET that
was in flight across any mutation never seeds the cache for the next visit (`json-cache` epoch). The Sessions list
applies a confirmed rename or delete the same way.

A catalog slug held by something the app did not create (a leftover file, a hand-made folder, a link) shows
**Needs cleanup** with what is in the way (`blocker`: kind, file count, size, at most 3 names; counted with
`lstat`, never following a link or opening a file, at most 2,000 entries; only the root `config.json` is read, to
tell whether Local models lists the entry) and a **Reinstall** button. Reinstall
downloads and verifies the model first and only then moves the old entry aside through the downloader's swap and
deletes it; a failed or cancelled download removes nothing. The request echoes the numbers the user confirmed, and
the server refuses (409) when the entry no longer matches them, at the request, at job start and right before the
swap. An entry that holds model files is never replaced and its card shows no Reinstall: anywhere in it, at any depth
(links to folders followed), an entry named `config.json` or named like weights (`.safetensors`, `.gguf` including
companions, `.pdiparams`, `.pdparams`, `.npz`, `.bin`, `.pt`, `.pth`, `.pkl`), or a folder that cannot be read, or more
than 20,000 entries to look through. A directory the app has no write permission on is refused at the request, before
any download. A link is removed as a link; its target is not touched.

The old entry is moved to `.staging/<slug>.reinstall-<pid>.<uuid>`. Before it is deleted, the moved entry is checked
again: the same `dev:ino` as at the last check and, unless it is a link, still no model files and the same numbers. A
writer outside the app (the `mlx` CLI, Finder) that changed it in between makes the job fail and the entry go back to
its path; if something else took the path meanwhile, the entry is kept at `.staging/<slug>.kept-<uuid>`, which nothing
removes, and the error names it. When the process dies between moving the entry aside and deleting it, recovery (when
the download manager starts, and before every job for the slug) puts the entry back when the path is empty, deletes it
when a complete model installed by the app is in place and the backup holds no model files, and otherwise keeps it.
Delete in Local models and Reinstall never reach the same entry: Delete needs a `config.json`, which Reinstall refuses.

## Chat and Work

`/chat` is one page with two modes, switched by the segmented control in its bar (`⌘⇧M`). Chat runs the bundled
`mlx-agent` with the app's `web_search` / `web_fetch` tools and no shell; Work runs it in a folder the user chooses.
The page is `ui/src/pages/chat.tsx` plus `ui/src/pages/chat/`.

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
- Tools row: the chips, then the key legend (`↵ Send`, `⇧↵ New line`, or `Esc to stop` while an answer runs and the
  text field has the focus; icons with hidden key names), then Send / Stop in a column of its own that never shrinks.
  A legend part is drawn whole or not at all. As the row gets short, first the legend parts go (New line, then Send),
  then the folder label shortens, then the web label, then (Work) the quantization badge goes, then the model name,
  each label down to a floor (a label of 8 characters or fewer never shortens). A badge that does not fit whole is not
  drawn and its room goes back to the folder and web labels (`data-fit-whole`, composer); under 520 px of composer
  the chips are icons, under 320 px they wrap. The Bypass word and the Send button never give way. The reason Send is
  off keeps up to two lines beside the chips.
- Model: before each message goes to the agent, the app selects the conversation's model on the engine
  (`POST /v1/messages/count_tokens` over `node:http` to 127.0.0.1 only, the token in the `x-api-key` header, a 30 min
  ceiling; `src/chat/agent/select-model.ts`). Only that model loads. While the engine loads it, the thread shows the
  load card; Stop then ends the turn with "The message was not sent." and the engine continues the load. While the
  engine prepares GPU kernels for that model (load progress `phase: 'kernels'` with `steps` known), the card's title
  is "Preparing GPU kernels", its head counts "n of m" steps, a bar fills by steps and the note says "The first load
  after an update prepares the GPU. It can take a couple of minutes."; the weight read, or an engine that reports no
  steps, keeps the plain card. Loads can overlap: the card shows the load of the conversation's own model, else the
  oldest one (`loadFor`, `ChatHostState.loading` is a list, oldest first). A model change
  in an open conversation loads nothing until the next message. A model in memory is selected at once, also while
  answers run on it (the engine skips its load gate for a resident name). If the select still waits, the card says the
  model is in memory and does not say it loads. When the turn has no output yet and the live file shows that the
  model runs as many answers as its limit while a request waits, and every running answer started before the turn (so
  none is its own request; `ChatHostState.answers`), the thread says "Waiting for a running answer".
- Wait line: until the turn's first entry the thread shows the status as text beside a 20 px dotted orb
  (`arc-x/wait-line`, `arc-x/wait-orb`; the orb is `aria-hidden`). Its shape says which wait: a slow ring while the
  turn waits (starting, stopping, the engine, a full model, no event for 3 s), a waving band while events arrive. The
  load card has the orb in its head: a pulsing sphere while the engine loads this model, the slow ring otherwise.
- Picker: the pill shows the base model and a badge with its quantization (`Qwen3.8-27B` `UD-Q4_K_XL`; the badge hides
  under 520 px of composer, and in Work it is drawn whenever the row has room for it with the folder and web labels
  at their floors, else not at all; the picker, its title and its name always say it),
  the title the full id. The popover has two lists: one row per base on the left, the
  quantizations of the open base on the right. A base opens on a click, `Enter`, `Space` or `→`, never on hover; a base
  with one quantization picks it at once. Both lists put the model in memory first, then the model in load, then the
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
  minutes (`turn-timeout`). A Work turn has no time limit: it runs until the agent ends it or you press Stop, and the
  app ends it only when the engine or the agent is gone. The end of a turn says why it stopped; `output-limit` offers
  Continue, `tool-refused`, `address-refused`, `arguments-unreadable` and `tool-loop`
  offer Send again.
- Titles, one rule on the server (`src/chat/titles.ts`) for Chat and Sessions: the agent's title, else the first typed
  prompt (first non-empty line, at most 60 characters with the `…`, cut at a word), else no title. A page prints its
  own word for no title: "New chat", "New Work conversation", or "Session" and the first 8 characters of the id.
- Untrusted text: model output and web results render as text. A link to a host that no search result or user message
  gave is plain text, never a link.
- Rename (`F2`) and delete work from the bar, the list and the palette; both are off while an answer runs.

### Work

- **Folder.** Work runs in one folder per conversation, fixed for its life. On the New screen the folder chip offers
  the newest folders of the Work list and "Choose another folder…", which opens the native folder dialog
  (`ui/src/lib/pick-folder.ts`); the newest folder is preselected, a choice made in this window wins. The server
  checks the folder (`src/chat/work-folder.ts`): not absolute, missing, not a folder, unreadable, `/` or the top of the
  data volume, the home folder or a folder that holds it (as written or where its links lead), and a folder inside,
  holding or picked through the app data folders are refused (compared by disk identity, so another path to the same
  folder is refused too) with a fixed sentence and `reason: 'cwd-invalid'`,
  shown in a strip with
  Choose folder…. The same rule runs again right before the agent gets the folder (`session/new`, `session/load`) and
  before each prompt. An open conversation whose folder is gone or no longer usable shows its sentence and Send is off.
  Paths show `~` for the home folder; a long one is cut in the middle and shown whole on hover and in the open
  conversation's folder popover (with Copy path).
- **Rows.** Shell commands (the whole command, then its output), file reads, listings, searches and edits (the change
  as `-` / `+` lines) are rows in the thread (`ui/src/lib/work-tools.ts`); an input of another shape is a plain row.
  Hidden and bidi characters in a command or change show as `\u{…}`. A row whose input or output the transcript cut at
  16 KB says so. A command's output is the text it printed (the agent's `rawOutput` of `type: "Bash"` is never shown as
  JSON; the Sessions transcript reads it the same way); a finished command that printed nothing says "No output.", and
  one that ended with a non-zero exit code, a stop the agent named or a timeout says so on its row ("Exit code 2").
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
  next. Deny ends the turn: "You denied a request. The answer stopped." A grant the agent saves lives in
  `~/.mlx-agent` and also applies to `mlx agent` in a terminal; in Work a saved shell grant does not skip the card (the
  guard asks anyway). The guard's own line at the end of the agent's title is not shown. Under the card's title, the
  agent's own title is one line, and only when it says something the well does not: on a command or file card it is
  hidden when it only names that command or file (whole, or cut by the server at 200 characters), and only the words
  after it are shown when more follow (a hook's reason). The command or file counts as named only where it ends at a
  word edge (`Run gitk` on the command `git` is shown whole). A one-word title is hidden only when it is the target or
  the agent's own verb or tool name. MCP and other cards always show it (the tool's name). The server says when it cut
  a title (`titleCut`); such a title ends with "…", at a whole word when the last word is short; the whole title is
  its hover text. The card reads in three groups:
  what is asked (title, description, wells), what to know (Auto's reason line, then the warnings), the answer
  (buttons, then the notes): 8 px between groups, 12 px before the buttons, 6 px before the first note.
- **Plan card.** The plan as Markdown in a scrolling well; Approve plan, or Send feedback… with a text field of at
  most 4,000 characters (Escape goes back). A request with no plan text (`planContent` null, absent or blank) shows
  "The agent sent no plan text." and the buttons Keep planning and Approve plan; no feedback is sent then (with no
  plan file the agent drops it). The agent's `exit_plan_mode` row reads "Plan review".
- **Permission mode** (`ui/src/pages/chat/work/permission-mode.tsx`, `src/chat/agent/auto.ts`,
  `src/chat/agent/auto-scan.ts`, `src/chat/permission-log.ts`). Each Work conversation has one of three modes, the
  last chip of the composer's tools row (and, while not Ask, a chip in the thread header): **Ask** (every request is
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
  /api/chat/drafts/:draft/permission-mode {mode, up: true, cwd, temp?}` and pins the folder it named. The Control
  Panel checks the folder as a create does and keeps one pick in memory (`ChatHostState.permissionDraft`), bound to
  that draft id and folder; a second pick, from any window, replaces it. The first message names the draft
  (`draft` on create), never a mode: after the session exists the Control Panel takes the pick when the id and the
  folder match, writes the mode line (`from: ask`) and only then starts the turn, so the first turn shows the mode
  line before its first call. A pick it cannot apply leaves the conversation in Ask, and the page shows a toast when the
  reply's mode is below the one the chip showed. The pick goes back to Ask when the folder changes, the page goes to
  Chat or leaves the New screen (the page posts `{mode: 'ask', up: false}` for the old id), the last window closes or
  the Control Panel restarts. A send that fails before the conversation exists keeps it. While the first message is on
  its way the chip keeps the mode it showed and only a move down is sent, always to the draft id the first message
  named (a folder picked meanwhile does not change it); one that arrives after the pick was taken is applied to the new
  conversation. The Control Panel keeps that draft-to-conversation link apart from the one pick, so a later pick or
  the last window closing does not drop it; it goes when the conversation is deleted. The confirm reads the hooks line from `GET /api/chat/work/hooks?cwd=` and starts its
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
  the conversation's session) cannot pass the cap; with no window open, every call is a card. A check that fails with
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
  (time, mode, tool call id, kind; no command, path or title); a failed write shows the card and moves the conversation
  to Ask. The mode is read again after that write and after every other wait: a request is decided under the mode it
  has when the app acts, and a line whose call got no allow (Stop, a mode change in that instant) is taken back by a
  `withdrawn` line, so a reload does not list it. A line written in the instant the Control Panel stopped can still be
  listed for a call that never ran. In the thread an allowed row says "Allowed by auto" or "Allowed by bypass" and
  opens to the text its card would have shown; a call with no row of its own (a helper agent's) is a line. A mode
  change is a mark in time order: a hairline across the thread with the icon, "Mode: Ask|Auto|Bypass", one short
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
- An answer to a request that is no longer open closes the card with a toast; an answer the server refuses for its
  content keeps the card and shows the server's sentence.

## Data sources

All state lives on disk; SQLite is only an index.

| Source                          | Location                                                                                            | Written by                      | Kept                   |
| ------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------- |
| Trace records (v2)              | `~/.mlx-node/metrics/traces/<date>-<pid>.jsonl`                                                     | inference process, one per turn | 30 days                |
| Runtime series                  | `~/.mlx-node/metrics/runtime/<date>-<pid>.jsonl`                                                    | inference process sampler       | 14 days                |
| Heartbeat                       | `~/.mlx-node/metrics/runtime/live-<pid>.json`                                                       | inference process sampler       | while the process runs |
| Supervisor state                | `~/.mlx-node/desktop/supervisor.json`                                                               | desktop MAIN process            | latest only            |
| Disposable SQLite index         | `~/.mlx-node/dashboard.db`                                                                          | Control Panel database worker   | rebuilt when needed    |
| Local models                    | `--models-dir` → `MLX_MODELS_DIR` → `modelsDir` in `~/.mlx-node/config.json` → `~/.mlx-node/models` | downloads, `mlx convert`        | —                      |
| Cold tier (paged prefix blocks) | `~/.mlx-node/cache/paged/v1/` (`MLX_COLD_CACHE_DIR` override)                                       | inference process               | quota, LRU             |
| `mlx agent` sessions            | `<agentHome>/sessions/<encoded cwd>/<id>/` (`agentHome` = `GROK_HOME` or `~/.mlx-agent`)            | `mlx agent`                     | —                      |

Chat and Work add, under `~/.mlx-node/chat` (0700):

| File                                                   | Holds                                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `conversations.json`                                   | the Work index: session id and creation time only, no title, no path                 |
| `providers.json`                                       | provider selection and check results; never a key                                    |
| `plugins/<id>/`                                        | provider skill folders copied from the bundle                                        |
| `work-guard/plugin.json`                               | the Work guard plugin: one `PreToolUse` hook that asks before every shell command    |
| `chat-agent/`, `chat-home/`, `chat-workspace/`, `tmp/` | the Chat agent's own home, cwd and temp folders; the agent writes its sessions there |

Work sessions are the agent's own, in `~/.mlx-agent/sessions/`, as from a terminal. Keys live in the Keychain only.

Every record on disk is metadata only: a number, a closed enum or an identifier. Prompt text, tool arguments, model
output, error text, stderr and URLs are never written by the app; the agents write their own session files.

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
| Speculation     | `mtpCycles`, `mtpMeanAccepted`, `mtpMeanDrafts`, `mtpMeanDepth`, `mtpAcceptByPos` (at most 16 positions)                                                                                                     |
| Counts, overlap | `toolCount`, `toolCallCount`, `imageCount`, `concurrency`, `concurrencyPeak`                                                                                                                                 |
| Memory          | `memActiveBytes`, `memPeakBytes`                                                                                                                                                                             |
| GPU kernels     | `pipelinesCreated`, `pipelineCreateMs`: pipelines the process created while serving between the turn's begin and end (process-wide, so overlapping turns share them); absent while the native counter is off |
| Cold tier       | `cold*` deltas and totals of the turn (hits, misses, bytes, queue drops, evictions, corruptions, write errors, sidecar counts)                                                                               |

A rate or time that was not measured is absent, never 0. A media-restart error is not recorded as a failure.
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

| Output                               | Content                                                                                                                                                                                                                                                                                                                                                                                                                  | Cadence                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `live-<pid>.json` (`v: 2`)           | status, health, memory, loaded models with their KV pool, turns in flight with their phase and `waitMs` / `slotMs` / `prepareMs`, the loads in flight (`loading`: model, start, `phase` `weights` or `kernels`, `stepsDone` of `steps`, read per load path from native `loadProgress`), counters, decode rate; about 0.6 KB plus about 380 B per turn in flight (at most 32 listed) plus each resident model, not capped | 1 s while busy, 5 s while idle (tmp + rename), plus a write 250 ms after each turn or load start and end |
| `<date>-<pid>.jsonl` `proc`          | host kind, version, port, wired limit, total memory, chip                                                                                                                                                                                                                                                                                                                                                                | first line of each file                                                                                  |
| `<date>-<pid>.jsonl` `sample`        | busy time, memory, in flight, queued, resident models, turn and token deltas, `pipelinesCreatedWhileServing` (cumulative; absent while the native counter is off)                                                                                                                                                                                                                                                        | 5 s while busy, 60 s while idle, plus one on each busy flip or resident-set change                       |
| `<date>-<pid>.jsonl` `sched`         | `schedulerStats()` snapshot of one model, plus `turns`: the turns sent to that model since its load, all lanes, counted when the call was sent                                                                                                                                                                                                                                                                           | once after each load, then once after each busy burst, only when that model is idle (see below)          |
| `<date>-<pid>.jsonl` `load` / `exit` | one model load (with `warmupMs`, `warmupSteps`, `warmupSkipped`, `warmupReset`, `warmupPipelines`, `warmupCompileMs`, `warmupPeakBytes` when the load ran a warm-up; not indexed); process exit                                                                                                                                                                                                                          | on the event                                                                                             |

`schedulerStats()` is a scheduler barrier, so the sampler never calls it on a model with work: it goes through
`SessionRegistry.tryIdleProbe`, which holds any dispatch that arrives until the reply settles. Scheduler counters are
cumulative, so the difference of two idle snapshots is exact for the burst between them. A turn in the one-turn-at-a-time
lane moves no scheduler counter (any family under `MLX_SERVE_FORCE_SERIAL=1`, DFlash2 and other barrier
turns), so `turns` against `admitted` tells such a model from an idle scheduler; a record without `turns` reads as
unknown. Live concurrency comes from TS-side state. At 16 MiB a series file takes only `load` and `exit` records (each still led by this process's
`proc` header when the file has none from it yet); `sample`, `sched` and `proc` records are dropped until the next
day's file.

### Supervisor state

The desktop MAIN process writes `~/.mlx-node/desktop/supervisor.json`
(`packages/desktop/src/main/supervisor-state-file.ts`): supervisor state (`stopped`, `starting`, `running`,
`restarting`, `failed`, `lying`), sidecar pid, generation, the restart policy's `streak` and `shortLivedFailures`,
last exit verdict, health and the native error count. Exit reasons, stderr and native error text are never written. A
reader ignores the file when MAIN's pid is dead.

### The index

`~/.mlx-node/dashboard.db` is **disposable**: deleting it loses nothing; it is rebuilt from the JSONL on next start.
`SCHEMA_VERSION` is 12 (`src/db/open.ts`): `traces`, `trace_files`, and five runtime tables (`runtime_files`,
`runtime_procs`, `runtime_samples`, `runtime_sched`, `runtime_loads`). Any DDL change bumps the version; a file with
another version, or a damaged one, is moved aside and rebuilt. Version 10 replaced `traces.prefill_tps` with
`prefill_ms` and `prefill_tokens`; version 11 added `runtime_sched.turns`; version 12 added `traces.wait_ms`,
`slot_ms`, `prepare_ms`, `pipelines_created` and `pipeline_create_ms`.

Ingest runs on boot, then every 30 s, after a finished turn (see the live channel) and on `POST /api/ingest`. It reads
only the tail appended since the last read of each file. Files past retention are pruned after ingest.

## Live channel

Live state is pushed over the same port; no page polls for it. The one timer that reads data is the Coding Agents
page, which re-reads `GET /api/coding-agents` every 30 s while the window is visible (and on window focus). The Chat
load card has a 1 s clock while it is shown; it changes only its elapsed-time text (the kernel step count comes with
the live snapshot).

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
- Routes that would reach the network (a real download, the update check) answer `E_UNAVAILABLE` unless
  `MLX_DEV_NETWORK=1`.
- `MLX_DEV_ENGINE_STATES=1` loads five more models, one per state of the Engine "Block pool and batching" section: turns
  that ran alone with 0 or 1,200 blocks held, turns only before the last hour, a flat-cache model and a paged model
  before its first snapshot. `MLX_DEV_TICKER_TPS` and `MLX_DEV_TICKER_MEM_GB` fix the Output rate and MLX memory of the
  heartbeat.
- `MLX_DEV_FLIGHT=one-model` puts both turns in flight on one model, one asked through the alias `mlx-local` (the
  default is one turn on each of two models). `MLX_DEV_DISK_CACHE=native|off|older` sets the disk cache the heartbeat
  reports: on with the native 128-block limit, off, or no field at all (default: on, the whole prompt).
  `MLX_DEV_FLIGHT=phases` adds three turns before their prefill: model busy, waiting for a slot, preparing.
  `MLX_DEV_LOAD_KERNELS=3/50` reports the harness engine's load as "Preparing GPU kernels", 3 of 50.
- Chat and Work run only on the scripted fake agent (`__test__/helpers/fake-agent.mjs`, script
  `__test__/fixtures/chat-scenarios/chat-harness.json`, picked by a word of the prompt), with keys in memory and the
  providers on loopback. The folder dialog answers a throwaway folder in the fixture home.
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
prompt id the agent recorded (`turn_completed.prompt_id`, else the `user_prompt_submit` hook line). The page titles the
turn by its cause (`Woken by workflow deep-research`), and its timeline starts at the previous turn's end when background
subagents ran in the gap: the main lane is idle there, and the status counts the turn alone.

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

Turn metrics come from each `turn_completed`. Its `perf` block describes the turn's **first** model call (a turn
has many): `ttft_ms` is shown as First token (the inspector adds `first of n model calls`); the first-call
`prefill_tokens_per_second` (prompt tokens ÷ time to first token) and `prefill_input_tokens` are not read. Prefill
speed reads only `prefill_ms` + `prefill_tokens` (both, or neither), with `prefill_calls` for the per-call size floor
and `usage.modelCalls` for `k of n model calls`. Until the agent writes those sums, the page says `not measured in
this session`.

grok's `*.lock` files are never opened: grok blocks on `LOCK_EX`, so a lock held
from Node would stall the agent. From mlx-agent v0.3.0 every resident session
(TUI, `-p`, ACP, subagents) has a row in `active_sessions.json`, so an idle holder
is detected. With an older binary, an idle ACP/headless holder with no recent
writes cannot be detected.

## Persist paged cache (`persistPagedCache`)

The `ColdCacheManager` SSD cold tier is wired into paged inference.
When enabled, full paged prefix blocks are captured to the cold tier on publish and
restored on a hot-cache miss before falling back to a normal prefill.

- **On by default in both agent paths.** `mlx agent` (its spawned `mlx serve`)
  and the desktop app engine (Chat, Work, and every client of the app's engine)
  run with the disk cache on. `mlx serve` and `mlx launch claude` keep the library
  default, **off**. One switch per engine, applied to every allowlisted family —
  `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4`, `muse_glimmer`, `lfm2`, and
  `lfm2_moe` — in GGUF and safetensors checkpoints alike.
- Precedence, highest first: an explicit flag (`mlx agent --no-persist-cache`,
  `mlx serve --persist-cache` / `--no-persist-cache`), then
  `MLX_PERSIST_PAGED_CACHE`, then the entry's default. The entry hands the switch
  to `createInferenceHost` as the typed `persistPagedCache` option; the host
  resolves it once (`packages/server/src/host/persist-cache.ts`) and writes the
  result to `MLX_PERSIST_PAGED_CACHE` before any load, which is what the native
  loads read. A checkpoint whose own `config.json` sets `persist_paged_cache`
  still wins over that env for its model (native precedence: explicit config,
  then env).
- `mlx serve` prints `[mlx] paged KV disk cache: on|off` at start, and the
  engine's `live-<pid>.json` heartbeat carries `persistPagedCache` and
  `coldCapture: { blocks, ms }` (the capture limits below; `null` when the cache
  is off). The dashboard live snapshot reads both per process; both are `null`
  from an engine that predates them. Whether a turn actually used the tier is the
  trace row's `coldEnabled`; the Cache page shows both.
- **One turn writes the whole prompt in both agent paths.** By default a turn's
  capture walk writes at most 128 blocks (2,048 tokens) within 250 ms, so a
  restart restored only 1.5–6 % of a long prompt (measured on a 0.8B qwen3_5:
  1,136 of 18,442 tokens, 928 of 60,264). `mlx agent` (through
  `mlx serve --persist-whole-prompt`) and the desktop engine ask for
  `MLX_COLD_CAPTURE_BLOCKS_PER_TURN=63125` and `MLX_COLD_CAPTURE_BUDGET_MS=10000`
  instead; measured on the same model this restores 99.7 % (18k) and 99.9 % (60k).
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
    sessions on that model wait too. Measured on the 0.8B: +414 ms at 18k tokens,
    +1,210 ms at 60k (136 / 401 MB). Only blocks not already on disk are written,
    so the cost is paid once per new prefix (the first turn of a session, or after
    a restart that restored less); later turns write only their new tail. Worst
    case on a stalled disk: 10 s for the walk plus up to 10 s more for a hybrid
    family's state sidecar to get a writer slot.
  - **When the 10 s cap ends a walk,** the blocks written so far stay; the next
    turn's walk skips them and goes on, so the rest lands one turn later. For
    qwen3_5 the state sidecar of that turn anchors at the deepest prefill
    checkpoint the chain reaches (the ladder holds ¼-spaced rungs, e.g.
    1,728 / 6,960 / 27,888 / 111,600 for a 111,611-token prompt), so a restart
    before the next turn restores that rung, not zero.
- Writes are queued and land on disk in the background. `InferenceHost.close()`
  waits for the queue (`coldCacheDrain`, at most `COLD_CACHE_DRAIN_TIMEOUT_MS`,
  2 s) once no request is left, so a one-shot `mlx agent -p`, an app quit or a
  serve stopped right after a turn still keeps that turn's blocks and state
  sidecar. The writer queue holds at most 8 jobs plus the one the writer is
  committing, and the capture walk waits for a free slot, so a finished turn
  leaves at most 9 jobs behind (one of them the sidecar, 146.8 MiB on a 27B
  qwen3_5) however long the prompt was. The queue is
  FIFO and the sidecar is enqueued last; if the drain times out, what is lost is
  the tail, usually the sidecar, and a qwen3_5 restore then falls back to the
  deepest sidecar an earlier turn wrote. An app quit during a capture walk loses
  the rest of that walk the same way. The quit itself stays bounded by the
  supervisor's 8 s SIGKILL window.
- **Restore is gated to an allowlist: `qwen3`, `qwen3_5`, `qwen3_5_moe`, `gemma4`, `lfm2`, and `lfm2_moe`.**
  Dense qwen3 sizes its pool over every layer, so paged blocks fully determine its
  layer state. The other three are hybrid — their pools cover the full-attention
  layers only — but each now persists its out-of-pool state as a cold-tier
  **sidecar**: grouped sliding-window K/V for gemma4, GDN recurrent state for
  qwen3.5/3.6 dense and MoE, and ShortConv recurrent state for LFM2/2.5. A
  `ColdSidecarPolicy` makes the restore
  reconcile DOWN to a boundary a validated sidecar actually backs (vLLM's per-group
  rule), and the `aux_prefix_unbacked` latch fails closed if an in-process hot hit
  would resume a K/V prefix whose out-of-pool half is missing. Gemma4 instead
  installs every paged group at one validated boundary or resets them all to zero.
  A family joins the allowlist only after its restart-parity
  gate passes on real weights with `hits > 0` and `corruptions == 0` — see
  `docs/paged-cache.md`. The allowlist is enforced natively in
  `cold_tier::resolve_persist_cold`, so a family that is off it never persists or
  restores even under an explicit `persistPagedCache` or `MLX_PERSIST_PAGED_CACHE=1`.
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
| `MLX_PERSIST_PAGED_CACHE`          | Turns the paged KV disk cache on (`1`/`true`/`on`/`yes`) or off (`0`/`false`/`off`/`no`) when no flag decides; beats the entry default (on for `mlx agent` and the desktop engine, off for `mlx serve` / `mlx launch claude`).                                            |
| `MLX_COLD_CACHE_DIR`               | Parent dir for the cold tier (operates in its `mlx-paged-v1` child). Default `~/.mlx-node/cache/paged/v1`.                                                                                                                                                                |
| `MLX_COLD_CAPTURE_BLOCKS_PER_TURN` | Blocks one turn's cold-tier capture walk may persist. Native default `128` (2048 tokens at block size 16); `mlx agent` and the desktop engine write `63125` when it holds no number native parses and the disk cache is on. A number you set wins. Read once per process. |
| `MLX_COLD_CAPTURE_BUDGET_MS`       | Wall-clock ceiling on one capture walk. Native default `250`; `mlx agent` and the desktop engine write `10000` when it holds no number native parses and the disk cache is on. A number you set wins. The turn ends after the walk. Read once per process.                |
| `MLX_MODELS_DIR`                   | Local models directory when `--models-dir` is omitted.                                                                                                                                                                                                                    |
| `GROK_HOME`                        | `mlx agent` config home read by the Sessions page (`agentHome` runtime option overrides). Default `~/.mlx-agent`.                                                                                                                                                         |
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
  or null. Nothing else crosses: no path goes in, no file is read.
- **Mutations are still the dangerous part** — model delete, cache clear, session rename/delete —
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

| Route                        | Method       | Thread    | Purpose                                                                                                                                      |
| ---------------------------- | ------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/models`                | GET          | worker    | Local models: name, path, family, quant, size, ctx window, `servedIds`; `host` memory and free disk                                          |
| `/api/models/:name`          | DELETE       | worker    | Delete a model dir (path-checked)                                                                                                            |
| `/api/catalog`               | GET          | worker    | Recommended catalog + installed/installable state; `blocker` names what holds a blocked slug                                                 |
| `/api/catalog/updates`       | GET          | transport | Upstream sha per catalog repo; the page joins it against `/api/catalog`                                                                      |
| `/api/downloads`             | GET / POST   | transport | List active jobs / start a catalog download; `replaceBlocker: { kind, fileCount, sizeBytes, truncated }` is a Reinstall (409 on change)      |
| `/api/downloads/:id`         | DELETE       | transport | Cancel a download                                                                                                                            |
| `/api/downloads/:id/events`  | GET          | transport | Not served over the port — use `runtime.subscribe(jobId, fn)`; answers 503                                                                   |
| `/api/coding-agents`         | GET / POST   | transport | Coding agent state with the managed instruction `block` / body `{action: 'refresh'}` or `{action: 'install', agent}`                         |
| `/api/coding-agents/models`  | GET          | worker    | Local model ids the coding agents can be pointed at                                                                                          |
| `/api/metrics/models`        | GET          | worker    | Model keys in first-seen order (color registry seed, model filter)                                                                           |
| `/api/metrics/summary`       | GET          | worker    | Totals, latency quantiles, per model, finish reasons, failures, endpoints, dispatches; `previous` = the window of equal length before `from` |
| `/api/metrics/timeseries`    | GET          | worker    | Per-bucket series, optionally per model                                                                                                      |
| `/api/metrics/distributions` | GET          | worker    | Histograms with bins chosen by the API                                                                                                       |
| `/api/metrics/speculative`   | GET          | worker    | MTP per model (acceptance per draft position, decode speed with and without MTP) and a trend                                                 |
| `/api/metrics/turns`         | GET          | worker    | Turns, newest first; cursor paging (`beforeTs` + `beforeTraceId`), `limit` ≤ 2000, `traceId`, `outcome`                                      |
| `/api/runtime/series`        | GET          | worker    | `from`, `to`, `bucketMs` (5 s … 1 h) → samples, scheduler deltas, processes, loads                                                           |
| `/api/runtime/live`          | GET          | transport | `RuntimeLive` snapshot: status, supervisor, processes                                                                                        |
| `/api/cache`                 | GET / DELETE | worker    | Cold-tier scan and daily trend (`trend[].t`, local day start) / clear all or evict older-than-N-days                                         |
| `/api/ingest`                | POST         | worker    | Trigger an incremental rescan                                                                                                                |
| `/api/sessions`              | GET          | worker    | `?q=&cwd=&limit=&offset=` → `{sessions, total, cwds}`; limit 1..500, default 100                                                             |
| `/api/sessions/:id`          | GET          | worker    | `{session, transcript, turns, transcriptError?, resumeCommand}`; subagent ids work too; tool, text and subagent entries carry `endTs`        |
| `/api/sessions/:id/timeline` | GET          | worker    | `?turn=<n>` → `SessionTimelineResponse`: one turn as lanes, spans and handoffs; metadata only (below)                                        |
| `/api/sessions/:id`          | PATCH        | worker    | Body `{title}` → `{id, title}`; 400 bad title, 404, 409 live/recent/changed                                                                  |
| `/api/sessions/:id`          | DELETE       | worker    | `{deleted: [id, ...children]}`; 404, 409 when it or any child is live                                                                        |

`/api/sessions`, `/api/sessions/:id` and its timeline read both session homes. `origin` is `agent`, `chat` (the Chat
home) or `work` (an agent-home session the Work index `~/.mlx-node/chat/conversations.json` names). `resumeCommand` is
null for a `chat` session: it cannot be resumed in a terminal. The pages tag `chat` and `work` rows, link them to
`/chat?c=<id>` (Open in Chat), and rename and delete them through the Chat routes (refused only while an answer
runs), not through the session routes.

### Chat routes

| Route                                           | Method | Thread    | Purpose                                                                                                                               |
| ----------------------------------------------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/chat/state`                               | GET    | transport | `ChatHostState`: inference, the load, the models in memory, models, agent binary, both modes, providers, active turns, mode picks     |
| `/api/chat/conversations`                       | GET    | worker    | `?mode=chat\|work` → `{conversations: ConversationRow[]}`                                                                             |
| `/api/chat/conversations`                       | POST   | transport | `{mode, text, modelId?, cwd?, draft?}` → 201 `{conversation, turnId}`; starts the first turn; `draft` (Work) applies its picked mode  |
| `/api/chat/conversations/:id/prompt`            | POST   | transport | `{text}` → 202 `{turnId}`                                                                                                             |
| `/api/chat/conversations/:id/cancel`            | POST   | transport | → `{cancelled}`                                                                                                                       |
| `/api/chat/conversations/:id/interactions/:iid` | POST   | transport | `{optionId}` or `{decision, feedback?}`; Work only, 404 for a Chat conversation                                                       |
| `/api/chat/conversations/:id/permission-mode`   | POST   | transport | `{mode, up}` → `{mode}`; Work only; 409 `card-open` or `mode-changed`; 503 if a line write fails: see [Chat and Work](#chat-and-work) |
| `/api/chat/drafts/:draft/permission-mode`       | POST   | transport | `{mode, up, cwd?, temp?}` → `{mode}`; the New Work screen's pick; `up: true` needs `cwd`; 409 `mode-changed` or `temp-unnamed`        |
| `/api/chat/work/hooks`                          | GET    | transport | `?cwd=` → `{userHooks}`; the folder rule of create first                                                                              |
| `/api/chat/conversations/:id`                   | PATCH  | transport | `{title}` or `{modelId}` → `ConversationInfo`                                                                                         |
| `/api/chat/conversations/:id`                   | DELETE | transport | → `{}`; open streams end with `gone`                                                                                                  |
| `/api/chat/agent`                               | POST   | transport | `{action: 'restart', mode?}` or `{action: 'install'}` → `ChatHostState`                                                               |
| `/api/chat/engine/unload`                       | POST   | transport | no body → `{unloaded: string[]}`; 409 `engine-busy` while an answer or a load runs                                                    |

Bodies hold exactly the listed fields: any other field (`_meta`, `mcpServers`, `pluginDirs`, `env`, …) is 400. Text is
1–100,000 characters (413 above); text whose first non-space character is `/` is 400 with `reason: 'slash-text'` and
never reaches the agent. A Chat route failure carries a fixed sentence and, where it has one, a `reason` (a
`ChatFailureCode`); any other error becomes `E_UNAVAILABLE` "The agent could not be reached." and its text is dropped.

Streams are not routes: `runtime.watchChat({scope: 'host'}, fn)` sends `{type: 'state'}` (coalesced to 100 ms) and
`{type: 'conversations-changed'}`; `runtime.watchChat({scope: 'conversation', conversationId}, fn)` sends a full `sync`
first, then `ops`, `turn`, `interaction`, `interaction-closed`, `notice`, `conversation`, `permission-mark` and `gone`, each with `seq` one
above the last. A page that sees a gap re-opens the stream for a fresh `sync`. Over the port the request kind is
`chat`; streams close on `unsubscribe` and when the port closes.

`GET /api/sessions/:id/timeline?turn=<n>` → `{turn, turns, open, live, stopReason, asOfMs, range, lanes, spans,
handoffs, notes}`: kinds, times, statuses, tool and model names and counts, never prompt, text, tool input or output or
error text. It reads at most 8 subagent event files of that turn, each ≤ 8 MB. 400 for a bad `turn`, 404 for an
unknown session or turn; an unreadable transcript is 200 with `transcriptError`. In the detail route a hidden wake
prompt is a `wake` entry (cause, workflow or subagent, text); the timeline route carries none of it. `range.startMs` is the axis origin and
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
- **A new key is checked first** (Exa `GET /v0/teams/me`, DeepSeek `GET /user/balance`, Brave one `test` search,
  Firecrawl `GET /v2/team/credit-usage`, 10 s): `rejected` (401/403) is not saved and any earlier key stays;
  `no-credit` (402, or the account says so) and `unverified` (any other answer) are saved. The first search key
  becomes the selection when none is set, and a Firecrawl key turns Firecrawl on, only when its skill folder
  installs; otherwise the key is saved and the provider stays off.
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
  are passed only when a `firecrawl` executable is on the Work `PATH` (`work.missing: 'firecrawl-cli'` otherwise); the
  app never installs the CLI.
- **Chat** never sees a key in a prompt or argv: `mlx_web` gets it in its per-session MCP `env`
  (`MLX_WEB_SEARCH_PROVIDER`, `MLX_WEB_SEARCH_KEY`, `MLX_WEB_FETCH_PROVIDER`, `MLX_WEB_FETCH_KEY`). The server talks
  only to the four fixed provider hosts, with no redirects (a 3xx is `provider-error`), a 15 s (search) / 30 s
  (fetch) timeout and a 2 MiB reply cap; `web_fetch` refuses local names, private and special IPv4 ranges, IP literals
  in brackets, user info, empty host labels, white space, control characters, backslashes and non-ASCII before the
  path, before any call, and forwards the address without its fragment.
- **A change** (key, selection, a late Keychain answer) marks both agent hosts stale: each restarts when no turn is in
  flight (`applyPending` while one is).

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
