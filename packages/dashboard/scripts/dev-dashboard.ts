/**
 * Dev harness: the REAL dashboard runtime on a generated fixture home, plus a Vite dev server for the SPA, in one
 * process. The tab talks to the runtime over Vite's HMR socket (`dev/harness-plugin.ts`). Dev only.
 *
 *   oxnode scripts/dev-dashboard.ts [--fresh] [--home <dir>] [--port <n>]
 *
 * `MLX_DEV_LOAD_MS`: how long the fake engine "loads" a model (default 4000). `MLX_DEV_LOAD_HIDDEN=1`: the heartbeat
 * does not report the load, as when a running answer holds the engine. `MLX_DEV_UNLOAD_BUSY=1`: the engine answers
 * every unload with 409. `MLX_DEV_ENGINE_STATES=1`: five more loaded models, one per state of the Engine pool section.
 * `MLX_DEV_TICKER_TPS` / `MLX_DEV_TICKER_MEM_GB`: fixed Output rate and MLX memory values in the heartbeat.
 * `MLX_DEV_FLIGHT=one-model`: both turns in flight on one model, one asked through the alias `mlx-local`.
 * `MLX_DEV_FLIGHT=phases`: three more turns before their prefill: model busy, waiting for a slot, preparing.
 * `MLX_DEV_FLIGHT=chat`: a 40 s cycle on the chat model for the thread's wait line: 10 s model busy (another
 * session's request reads, then generates, while the conversations' requests wait unlisted), then each Work
 * conversation's own request queued, in prefill, in decode; start a "slow" prompt to see it between steps.
 * `MLX_DEV_LOAD_KERNELS=<done>/<steps>`: the heartbeat reports a load as "Preparing GPU kernels", done of steps.
 * `MLX_DEV_DISK_CACHE=native|off|older`: the disk cache the heartbeat reports (default: on, whole prompt).
 * `MLX_DEV_APP_END=no-ping|crash|signal|user`: an earlier Control Panel end over every Work conversation of the fixture.
 * Compact now (the thread bar `⋯`) runs 6 s in the fake agent, then ends 41,000 to 9,000 tokens; Stop it within that.
 * Rewind to here: a sent message's `⋯` (or /rewind, /undo, "Rewind to the last message…" in the palette) shows what goes,
 * then rewinds in 3 s in the fake agent: the thread ends before the message and its text is back in the box. In Work
 * after the word "background" the dialog names the command and schedule it stops and deletes, and the thread keeps a
 * "Rewound to here" line. A steer, a note and a wake have no `⋯`.
 * Rewind files (Work): after the word "diff", Rewind on that message checks the files first ("Checking files…") and
 * lists the four the agent wrote, src/log.ts "changed since the agent wrote it": it opens on "Conversation only", and a
 * file choice reads "Restore and overwrite". "Files only" keeps the thread and the box and leaves "Files restored · 4
 * files"; a second Rewind there finds no files ("The agent changed no files after this message."). The fake agent
 * keeps its files beside the session's updates.jsonl, so they come back after a restart; a fork's copy never has any.
 * Fork (the thread bar `⋯`, a sent message's "Fork with this answer", /fork, "Fork conversation" in the palette) copies
 * the conversation in 0.8 s in the fake agent ("Forking…"): the copy opens with "Fork of …" on top and a Fork tag in the
 * list, its toast's "Back to the original" goes back; in Work the toast names the shared folder.
 * Recap (the thread bar `⋯`, /recap, "Recap conversation" in the palette) shows "Writing a recap…" for 4 s in the fake
 * agent, then a Recap row; before the first finished answer the agent has none ("No recap this time."). The live Work
 * session holds one stored recap after its third answer.
 * Thinking: the fake agent reports mlx-agent-v0.3.6 and lists levels, so the model menu has a Thinking row: Off · Low ·
 * Medium · High on qwen3.5-4b and the Qwen3.8-27B builds, Low · High on qwen3.5-9b, "This model has no levels." on the
 * long-named model. A pick shows on the chip ("· Low") and is set in the agent before the next answer; switching
 * qwen3.5-4b at Medium to qwen3.5-9b steps it to Low.
 * The context ring opens the Context panel (also "Show context and usage" in the palette). The live Work session reads
 * 61K; a conversation the agent holds shows the fake agent's parts, a cold one "Send a message to see the parts.".
 * Hooks: the live Work session's first answer has the session's failed start hook (`global/env`), `rm -rf build`
 * "Blocked · by a hook", a stop hook's failure that counts on one row ("5 times · last …"), and its third answer a
 * hook's note. The words "hookstop", "hookblock" and (Work) "hookdeny" play them live, with the "Running 2 hooks" line
 * after 1 s; "hookblock" leaves only the hook's row and "Stopped by a hook.", and its text comes back to the box, with
 * the files it carried.
 * A retry, a picture and a hook row of a running answer show before its next call, else at its end, as on disk; the
 * words "retry" and "picture" play them. While a fork or a recap runs, Enter queues and the round button has no Stop.
 * Files: qwen3.5-4b and qwen3.5-9b see images, so they take pictures in Chat and PDFs (up to 20 pages, drawn into page
 * images in the composer) in both modes; every other model refuses both. With `MLX_DEV_FLIGHT=chat`, a Work message
 * with a PDF and the word "pdfwait" waits 40 s before any work, so its wait line names the pages it reads.
 * The PDF tool (Work): a PDF of up to 320 pages goes to the host page by page as it is drawn; one past the message's
 * room reads "40 pages · read by the PDF tool". The Work word "pdftool" asks to start `read-pdf` on it (the card reads
 * "Read report.pdf with the PDF tool · 40 pages · 10 reader agents"); Allow runs the block "Reading report.pdf": one
 * reader after another, 4 pages each, each row with the page it opened; pages 9-12 fail twice, so the end reads "36 of
 * 40 pages read · … · pages 9-12 not read". The Work word "pdfnot" answers without reading the PDF: the turn ends with
 * "The pages of report.pdf were not read." and Read it now, which starts the run as your own `/workflow` turn ("You
 * started reading report.pdf"); its pages 5-8 fail once and are read at the end.
 * Pictures in tool results: the Work words "picture" and "readpdf" read a picture and three pages of a 12-page PDF; the
 * read row reads "1 picture · …" or "3 of 12 pages · …" and opens to tiles read from disk, and a tile opens the viewer
 * (← →, − + 0, Esc). The live Work session's 2nd answer reads a screenshot and a PDF the same way.
 * A message typed during Compact now, a fork, a rewind or a recap goes when that op ends, even with older rows held:
 * Stop "slow" with a row queued, run Compact now, type and press Enter; it goes after 6 s and the held row stays.
 * The queue: send "slow" (Chat 30 s, Work 45 s), then type and press Enter; the rows wait under the composer and go one
 * by one when it ends. Stop holds them (Send next sends them); after a restart they come back held. In Work a row's
 * Send now (or ⌘Enter) goes into "slow": the agent reads it after the wait, and a Stop before that drops it. With five
 * rows waiting Queue is off ("5 messages wait…") and ⌘Enter still goes into "slow"; once it ends the rows lose Send
 * now and go by themselves.
 * The Work word "background" leaves a dev server, a build watch, a subagent and a schedule running: Delete reads "A
 * command runs here. Stop it first.", and after a Web Search key change "A change applies when the running work ends."
 * Its wake rows read "Woken by subagent: Find the flaky test" and "A monitor ended: Watch the build log" from their
 * first line. Stop on the dev server's or the build watch's rail row asks first; the row reads "Command · Stopping…"
 * (or "Monitor · Stopping…") and leaves, its thread row reads "Stopped in the background", and the agent wakes to hear
 * of it; a row offers Stop only while the host's tracker of the running agent holds its task. Its subagent row ticks
 * on the rail and in the thread ("1 error · turn 3 · 86% of context · 28.2K tokens"), then reads "Quiet for 0:20" from
 * 38 s until it ends. From 4 s the build watch prints seven lines, 1.5 s apart: its rail row reads "Monitor · 7 events"
 * with the newest line under it, its thread row "In the background · 7 events" with Events (7) holding the last five,
 * and then the agent wakes once to read them: "A monitor reported: Watch the build log". The Work word "ended" ends two
 * commands with no wake; one turn reports both: "2 commands ended: …".
 * The schedule of "background" ("Check the deploy") fires 25 s in: its rail row reads "Running now" while the run it
 * started (a Subagent row) runs, 20 s; then its "Next" time is 5 min after the fire, as the fire line says.
 * Delete on its row asks first ("Delete this schedule?"), reads "Removing…" and
 * the row leaves. After the agent stops (Restart, or an app restart) the row reads "Paused" and still offers Delete,
 * which loads the conversation first ("Opening…"); while the engine restarted under live commands Delete is off.
 * A cold conversation the agent opens slowly: send the Work word "lateopen", then Restart the agent (or the app).
 * Rewind, Compact now, Recap or a schedule's Delete there loads it, and the fake agent answers that load 40 s late:
 * after 30 s the op ends with "The agent is slow to open this conversation. Try again." while the agent and its
 * background commands go on; the late load lands at 40 s and the next try runs at once. A message sent meanwhile
 * waits on the same load with a turn's own deadline (30 s from the send), as a turn always has.
 * Type `/` in the composer: Compact, Context, Fork, Recap and Rewind in both modes; in Work also /goal, /loop,
 * /deep-research and /workflow, and once the conversation is loaded the fixture's skills (commit, review-pr) and
 * workflow (release).
 * /goal (alone or pause, resume…), /workflow and /deep-research answer as the agent's own host turns; /goal with an
 * objective is a model turn; /yolo or any unknown name reads "No such command here.". The Work word "noloop" turns the
 * agent's scheduler off for that conversation: /loop leaves the list and reads "No such command here." too.
 * A goal: in Work send `/goal Fix the flaky cache tests`. The strip above the composer reads "Working · 0:05 · 7k
 * tokens" with Pause, the thread keeps one "Goal · Fix the flaky cache tests · …" row after the message, and the rail
 * (when open) has a Goal section after Needs you. The typed answer runs two rounds (16 s), then the agent starts two
 * more by itself, 1.5 s apart ("Started by the goal loop", 9 s each), and the strip ends at "Done · 0:34 · 33k tokens"
 * with Clear. A message queued meanwhile waits ("Queued · 1 · Runs when the goal pauses.") and goes once it is done; a
 * row's Send now goes into the running round. Pause (or Stop) pauses it ("Paused", Resume and Clear); Resume sends
 * `/goal resume`, Clear `/goal clear`; a message before the goal's row cannot be rewound while it is open.
 * The Work word "mcp": the MCP server "github" asks for input with eight fields (a menu of six repositories, a needed
 * title of 3 to 80 characters, a long Details box, labels, a priority from 1 to 5, a date, an email, a switch). A
 * refused Send marks each wrong field; Send, Decline or Stop leaves "github asked for input · sent" (or declined, or
 * closed with no answer) in the thread, also after a reload. The Auto row stays on while the card is open.
 * The Work word "signin": "github" asks you to open its sign-in page; the card shows the whole address and "Site:
 * github.com". Open page (600 ms after the card shows) opens it in the browser and leaves "github asked you to open a
 * page · opened"; 3 s later the server says it is done: "github · finished in the browser".
 * The Work word "oddlink": a sign-in page on an `xn--` host carries "This address uses letters that can look like
 * other letters. Check the site name.", and a second request with no id reads as a card that cannot open; Decline
 * closes it.
 * The Work word "helperform": the workflow run "deep-research" starts, and 3 s later its helper's MCP server "github"
 * asks for a repository. The rail row reads "Workflow deep-research" and the composer keeps its own placeholder, as
 * the ask is the run's, not the typed answer's.
 * The Work word "twoagents" (Allow its launch): the run "research-advisory" starts two agents, verify:claims and
 * verify:fix-vectors, that tick every 2 s for 2 min. Each row has Stop (X from the row): its confirm opens on "Keep
 * running", and the row then reads "Stopped by you" while the other runs on; the block's and the rail row's Stop end
 * the run. The rail lists both agents under the run row; under 1296 px wide the dock strip above the composer reads
 * "research-advisory · Step 2 of 3: Verify · 2 running" with Stop, and ⌘K lists them under "Agents".
 * Never allow: the Work folder starts with two rules (`git push`, `npm publish`), so the permission chip's menu has
 * "Never allowed · 2", which lists them with a remove button each. The Work word "leftpad": Deny's arrow offers "Never
 * allow npm install…" (scopes `npm install` or every word); kept, a toast offers Undo, the row reads "Never allowed ·
 * npm install", the note row "Your rule · never allow npm install" with Remove rule, and the agent goes on with pnpm.
 * Send "leftpad" again: the app denies it with no card. The rules live in `chat/never-allow.json` of the fixture home.
 * The Work word "wrapped" asks for `sudo env CI=1 bash -c 'npm publish --tag next'`: the `npm publish` rule denies it
 * with no card, as it does through any of sudo, env, command, exec, nohup, time, nice, timeout and sh/bash/zsh -c.
 * The Work word "agentview" starts the subagent "Find the flaky test" for 2.5 min. Its row (rail or thread) opens its
 * Agent view, `?agent=<id>` in the address: at 1448 px and wider in the rail's column, from 1296 px a drawer over the
 * rail, else a full-height drawer. Its lines stream in over about 40 s, the prompt first. The agent stores a message
 * from the field 0.8 s after it is accepted and sends no live line of it, so the view shows it once it opens again or
 * at the agent's end. Escape closes the view and focus goes back to the row.
 */

import { randomBytes, randomUUID } from 'node:crypto';
import { closeSync, existsSync, mkdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { mkdir, rename } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

import { createServer } from 'vite-plus';

import { HARNESS_LOAD_MS, startHarnessEngine } from '../__test__/helpers/fake-engine.js';
import { startHarnessProviders } from '../__test__/helpers/fake-providers.js';
import { chatPaths } from '../src/chat/paths.js';
import { takeNamespaceLock } from '../src/prefix-cache.js';
import { createDashboardRuntime } from '../src/runtime.js';
import {
  assertFixtureTree,
  CATALOG_LOCATED,
  DEFAULT_HOME,
  FIXTURE_COMMAND,
  fixtureArgs,
  fixtureExists,
  fixturePaths,
  fixturePrefixNamespaces,
  type LiveProcess,
  refreshRuntime,
  resetHome,
  writeFixtureHome,
  writeLiveFile,
} from './dev-fixture.js';
import { devHarness, harnessAppEnds, harnessChatOptions, harnessTempRoot } from './dev/harness-plugin.js';

const UI_ROOT = join(import.meta.dirname, '../ui');
/** `MLX_DEV_WORK_DEEP=1`: the folder dialog answers a folder whose encoded path is too long for the agent. */
const DEEP_FOLDER = [
  'a-project-folder-with-a-long-name-one',
  'a-project-folder-with-a-long-name-two',
  'a-project-folder-with-a-long-name-three',
  'a-project-folder-with-a-long-name-four',
  'a-project-folder-with-a-long-name-five',
  'a-project-folder-with-a-long-name-six',
];
/** Rewrites the heartbeat well inside its promised 5 s interval, so the process reads as running. */
const HEARTBEAT_MS = 2_000;

function tickerValues(): LiveProcess['ticker'] {
  const tps = Number(process.env.MLX_DEV_TICKER_TPS ?? '');
  const memGb = Number(process.env.MLX_DEV_TICKER_MEM_GB ?? '');
  if (process.env.MLX_DEV_TICKER_TPS === undefined || !Number.isFinite(tps)) return undefined;
  const mem = process.env.MLX_DEV_TICKER_MEM_GB !== undefined && Number.isFinite(memGb) ? memGb : 27;
  return { outputTps: tps, memActive: Math.round(mem * 1024 ** 3) };
}

function flightValue(value: string | undefined): LiveProcess['flight'] {
  return value === 'one-model' || value === 'phases' || value === 'chat' ? value : 'two-models';
}

/** `<done>/<steps>` of `MLX_DEV_LOAD_KERNELS`, or null. */
function kernelSteps(value: string | undefined): { stepsDone: number; steps: number } | null {
  if (value === undefined) return null;
  const [done, steps] = value.split('/').map((part) => Number.parseInt(part, 10));
  if (done === undefined || steps === undefined || !Number.isFinite(done) || !Number.isFinite(steps)) return null;
  return { stepsDone: done, steps };
}

function diskCacheValue(value: string | undefined): LiveProcess['diskCache'] {
  return value === 'native' || value === 'off' || value === 'older' ? value : 'whole-prompt';
}

/** The Work folder's never-allow rules on the first start; later starts keep the rules the user changed. */
function seedNeverRules(chatRoot: string, folder: string): void {
  const file = chatPaths(chatRoot).neverAllow;
  if (existsSync(file)) return;
  mkdirSync(chatRoot, { recursive: true });
  const rule = (words: string[], ago: number) => ({
    id: `n_${randomBytes(16).toString('hex')}`,
    words,
    at: Date.now() - ago,
  });
  const rules = [rule(['git', 'push'], 2 * 86_400_000), rule(['npm', 'publish'], 3_600_000)];
  writeFileSync(file, `${JSON.stringify({ version: 1, folders: { [folder]: rules } })}\n`, { mode: 0o600 });
}

async function main(): Promise<void> {
  const { home, fresh, port } = fixtureArgs(process.argv.slice(2), DEFAULT_HOME);
  if (fresh) resetHome(home);
  const paths = fixturePaths(home);
  const startedAt = Date.now() - 6 * 3_600_000;
  const engineStates = process.env.MLX_DEV_ENGINE_STATES === '1';
  const live: LiveProcess = {
    pid: process.pid,
    startedAt,
    engineStates,
    ticker: tickerValues(),
    flight: flightValue(process.env.MLX_DEV_FLIGHT),
    diskCache: diskCacheValue(process.env.MLX_DEV_DISK_CACHE),
  };
  if (fixtureExists(home)) {
    assertFixtureTree(home);
    refreshRuntime(home, live);
  } else {
    process.stdout.write(`writing fixture home ${home}\n`);
    const summary = writeFixtureHome(home, { engineStates });
    live.startedAt = summary.live.startedAt;
  }
  const heartbeat = setInterval(() => writeLiveFile(home, live), HEARTBEAT_MS);
  let prefixLock: number | null = null;
  try {
    prefixLock = takeNamespaceLock(join(paths.prefixCacheRoot, fixturePrefixNamespaces()[0]!));
  } catch {
    prefixLock = null;
  }

  const providers = await startHarnessProviders();
  const loadMs = Number.parseInt(process.env.MLX_DEV_LOAD_MS ?? '', 10);
  const hidden = process.env.MLX_DEV_LOAD_HIDDEN === '1';
  const kernels = kernelSteps(process.env.MLX_DEV_LOAD_KERNELS);
  const engine = await startHarnessEngine({
    loadMs: Number.isFinite(loadMs) && loadMs >= 0 ? loadMs : HARNESS_LOAD_MS,
    unloadBusy: process.env.MLX_DEV_UNLOAD_BUSY === '1',
    onState: (state) => {
      live.loading =
        hidden || state.loading === null
          ? []
          : [
              kernels === null
                ? { ...state.loading, phase: 'weights', stepsDone: 0, steps: 0 }
                : { ...state.loading, phase: 'kernels', ...kernels },
            ];
      live.chatModels = state.resident === null ? [] : [state.resident];
      writeLiveFile(home, live);
    },
  });
  const chat = {
    ...harnessChatOptions(home, providers.baseUrls, engine.url),
    tempRoots: harnessTempRoot(home),
    appEnds: Promise.resolve(harnessAppEnds(process.env.MLX_DEV_APP_END, home, Date.now())),
  };
  const workFolder = join(home, 'work', ...(process.env.MLX_DEV_WORK_DEEP === '1' ? DEEP_FOLDER : []));
  mkdirSync(workFolder, { recursive: true });
  seedNeverRules(chat.root ?? join(home, 'chat'), realpathSync(workFolder));
  const devTrash = join(dirname(paths.modelsDir), '.dev-trash');
  const runtime = createDashboardRuntime({
    ...paths,
    codingAgents: { home, env: {} },
    prepareDelegation: async () => FIXTURE_COMMAND,
    chat,
    // The harness never reaches the real Trash: a cleanup's old entry moves into a folder beside the fixture models.
    moveToTrash: async (path) => {
      const into = join(devTrash, randomUUID());
      await mkdir(into, { recursive: true });
      await rename(path, join(into, basename(path)));
    },
  });

  let server: Awaited<ReturnType<typeof createServer>>;
  try {
    server = await createServer({
      root: UI_ROOT,
      configFile: join(UI_ROOT, 'vite.config.ts'),
      plugins: [
        devHarness(runtime, { chat, folders: { work: workFolder, model: join(home, CATALOG_LOCATED.picked) } }),
      ],
      server: { host: '127.0.0.1', port, strictPort: true },
    });
    await server.listen();
  } catch (err) {
    clearInterval(heartbeat);
    await runtime.close();
    await providers.close();
    await engine.close();
    throw err;
  }
  process.stdout.write(`Portato dev harness: http://127.0.0.1:${port}/  (home ${home})\n`);

  let stopping = false;
  const stop = async (): Promise<void> => {
    if (stopping) return;
    stopping = true;
    clearInterval(heartbeat);
    if (prefixLock !== null) closeSync(prefixLock);
    try {
      await server.close();
    } finally {
      await runtime.close();
      await providers.close();
      await engine.close();
      rmSync(join(paths.runtimeDir, `live-${process.pid}.json`), { force: true });
      process.exit(0);
    }
  };
  process.on('SIGINT', () => void stop());
  process.on('SIGTERM', () => void stop());
}

main().catch((err: unknown) => {
  process.stderr.write(`dev harness failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
  process.exit(1);
});
