// Runs entirely on the test Mac with the signed app's bundled Node and native addon.
// No network, package installation, private conversation reads, or model conversion.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { execFileSync, spawn } = require('node:child_process');
const sha = (data) => crypto.createHash('sha256').update(data).digest('hex');
const manifest = require('./manifest.json');
const fixtureBytes = fs.readFileSync(path.join(__dirname, 'messages.json'));
if (sha(fixtureBytes) !== manifest.messagesSha256) throw new Error('Benchmark fixture hash mismatch');
const messages = JSON.parse(fixtureBytes);
const args = process.argv.slice(2);
const readCommand = (exe, args) => {
  try { return execFileSync(exe, args, { encoding: 'utf8', timeout: 10000 }).trim(); }
  catch { return null; }
};
function systemSnapshot() {
  return {
    swap: readCommand('/usr/sbin/sysctl', ['vm.swapusage']),
    thermal: readCommand('/usr/bin/pmset', ['-g', 'therm']),
    vmCounters: Object.fromEntries((readCommand('/usr/bin/vm_stat', []) ?? '').split('\n').flatMap(line => {
      const match = line.match(/^([^:]+):\s*(\d+)\./);
      return match && /^(Pages free|Pages inactive|Pages wired down|Pages occupied by compressor|Pageins|Pageouts|Swapins|Swapouts)$/.test(match[1]) ? [[match[1], Number(match[2])]] : [];
    })),
  };
}
function appPaths(app) {
  app = path.resolve(app);
  const resources = path.join(app, 'Contents', 'Resources');
  return {
    executable: path.join(app, 'Contents', 'MacOS', 'mlx-node'),
    addon: path.join(resources, 'native', 'mlx-core.darwin-arm64.node'),
    package: path.join(resources, 'app', 'package.json'),
  };
}
function save(file, report) { fs.writeFileSync(file, JSON.stringify(report, null, 2) + '\n'); }
async function hashFile(file) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}
async function worker(app, modelPath, output, mode) {
  const core = require(appPaths(app).addon);
  const config = {
    cacheOwnerId: 'm3-portable-check', cacheRootOwnerId: 'm3-portable-check',
    temperature: 0, maxNewTokens: 64, enableMtp: mode === 'mtp',
    reasoningEffort: 'medium', reportPerformance: true,
    maxConsecutiveTokens: 0, maxNgramRepeats: 0,
  };
  const report = { config, before: systemSnapshot(), switches: { MLX_PORTABLE_KQUANT: process.env.MLX_PORTABLE_KQUANT } };
  let model;
  try {
    console.log(`Loading the model for ${mode}...`);
    const started = performance.now();
    model = await core.Qwen35Model.load(modelPath);
    report.loadMs = performance.now() - started;
    report.loadMemory = core.memoryStats();
    report.contextLimits = model.contextLimits();
    report.hasMtpWeights = model.hasMtpWeights();
    if (!model.hasBlockPagedCache()) throw new Error('This check requires the production paged cache');
    if (config.enableMtp && !report.hasMtpWeights) throw new Error('MTP weights are missing');
    const warmup = [messages[0], { role: 'user', content: 'Review this code:\n' + messages.filter(m => m.role === 'tool').map(m => m.content).join('\n').slice(0, 2600) }];
    await model.chatSessionStart(warmup, { ...config, maxNewTokens: 8, enableMtp: false });
    await model.resetCaches();
    const summarize = (result, input, wallMs) => ({
      wallMs, inputSha256: sha(JSON.stringify(input)), outputSha256: sha(result.rawText),
      promptTokens: result.promptTokens, cachedTokens: result.cachedTokens,
      newTokens: result.promptTokens - result.cachedTokens, generatedTokens: result.numTokens,
      finishReason: result.finishReason, performance: result.performance,
      memory: core.memoryStats(),
    });
    console.log('Measuring a fresh code-review prompt...');
    let start = performance.now();
    const cold = await model.chatSessionStart(messages, config);
    report.cold = summarize(cold, messages, performance.now() - start);
    save(output, report);
    const continuation = [...messages,
      { role: 'assistant', content: cold.text, reasoningContent: cold.thinking ?? '', thinkingEnabled: cold.thinkingEnabled },
      { role: 'user', content: 'Continue the review, taking these additional constraints into account:\n' +
        Array.from({ length: 75 }, (_, i) => `${i + 1}. Preserve public behavior, check error handling, and explain a concrete regression test.`).join('\n') },
    ];
    console.log('Measuring a cached follow-up with about 1.5k new tokens...');
    start = performance.now();
    const warm = await model.chatSessionContinue(continuation, config);
    report.continuation = summarize(warm, continuation, performance.now() - start);
    report.valid = cold.cachedTokens === 0 && warm.cachedTokens > 0 && cold.numTokens === 64 && warm.numTokens === 64;
    if (!report.valid) throw new Error('The expected cache/token counts did not hold; keep this report for diagnosis');
    report.after = systemSnapshot();
    for (const [label, turn] of [['Fresh prompt', report.cold], ['Cached follow-up', report.continuation]]) {
      const perf = turn.performance;
      console.log(`${label}: prefill ${perf?.prefillTokensPerSecond?.toFixed(1) ?? 'unknown'} tokens/s; decode ${perf?.decodeTokensPerSecond?.toFixed(1) ?? 'unknown'} tokens/s`);
    }
  } catch (error) {
    report.error = String(error);
    throw error;
  } finally {
    save(output, report);
    if (model) await model.resetCaches();
  }
}
async function main(app, modelPath, outputDir) {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') throw new Error('Use this check on an Apple Silicon Mac');
  if (!app || !modelPath || !outputDir) throw new Error('Expected APP MODEL.gguf OUTPUT_DIRECTORY');
  const paths = appPaths(path.resolve(app));
  for (const file of Object.values(paths)) if (!fs.existsSync(file)) throw new Error(`App file missing: ${file}`);
  if (fs.statSync(modelPath).size !== manifest.modelBytes) throw new Error('Select the original Qwen3.8-27B-UD-Q4_K_XL.gguf checkpoint');
  if (os.totalmem() < manifest.modelBytes + 12 * 1024 ** 3) throw new Error('Not enough physical memory for this benchmark');
  // A second model process can exceed a 36 GB machine's safe working set.
  const processes = readCommand('/bin/ps', ['-axo', 'pid=,rss=,comm=']) ?? '';
  const largeProcesses = processes.split('\n').flatMap(line => {
    const match = line.trim().match(/^(\d+)\s+(\d+)\s+(.+)$/);
    return match && Number(match[1]) !== process.pid && Number(match[2]) >= 8 * 1024 ** 2 ? [Number(match[1])] : [];
  });
  if (os.totalmem() <= 40 * 1024 ** 3 && largeProcesses.length) {
    throw new Error('Another large process is using at least 8 GiB. Quit model apps and agents before this check. PIDs: ' + largeProcesses.join(', '));
  }
  console.log('Checking the checkpoint fingerprint...');
  if (await hashFile(modelPath) !== manifest.modelSha256) throw new Error('Checkpoint hash differs from the investigated model');
  fs.mkdirSync(outputDir, { recursive: true });
  const file = path.join(outputDir, 'M3-results.json');
  const report = {
    createdAt: new Date().toISOString(), device: os.cpus()[0]?.model,
    memoryBytes: os.totalmem(), macOS: readCommand('/usr/bin/sw_vers', ['-productVersion']),
    appVersion: JSON.parse(fs.readFileSync(paths.package, 'utf8')).version,
    nativeSha256: await hashFile(paths.addon), fixture: manifest,
    protocol: 'One excluded short warmup; 64 greedy output tokens per turn; medium reasoning; fresh process per arm; native automatic memory limits.',
    runs: [],
  };
  let gpu = readCommand('/usr/sbin/system_profiler', ['SPDisplaysDataType', '-json']);
  try { report.gpu = JSON.parse(gpu).SPDisplaysDataType.map(d => ({ name: d.sppci_model, cores: d.sppci_cores ?? d.spdisplays_cores })); } catch {}
  const arms = [{ name: 'test3-ar', portable: '1', mode: 'ar' }, { name: 'stock-ar', portable: '0', mode: 'ar' }, { name: 'test3-mtp', portable: '1', mode: 'mtp' }];
  for (const arm of arms) {
    console.log(`\n${report.runs.length + 1}/3: ${arm.name}`);
    const childOutput = path.join(outputDir, `${arm.name}.json`);
    const log = fs.openSync(path.join(outputDir, `${arm.name}.log`), 'w');
    const env = { ...process.env };
    // Never inherit tuning, cache-size, or architecture overrides from a shell.
    for (const key of Object.keys(env)) if (/^(MLX_|NAPI_RS_)/.test(key)) delete env[key];
    Object.assign(env, { ELECTRON_RUN_AS_NODE: '1', MLX_PORTABLE_KQUANT: arm.portable, MLX_NODE_LOG: 'info', MLX_NODE_LOG_FILE: path.join(outputDir, `${arm.name}-inference.jsonl`) });
    let code;
    try {
      code = await new Promise((resolve, reject) => {
        const child = spawn(paths.executable, [__filename, '--worker', app, modelPath, childOutput, arm.mode], { env, stdio: ['ignore', 'inherit', log] });
        child.once('error', reject);
        child.once('exit', (code, signal) => resolve(signal ? `signal:${signal}` : code));
        const timeout = setTimeout(() => child.kill('SIGTERM'), 15 * 60 * 1000);
        child.once('exit', () => clearTimeout(timeout));
        child.once('error', () => clearTimeout(timeout));
      });
    } finally { fs.closeSync(log); }
    const result = fs.existsSync(childOutput) ? JSON.parse(fs.readFileSync(childOutput, 'utf8')) : { error: 'No child result; see log' };
    const nativeLog = fs.readFileSync(path.join(outputDir, `${arm.name}.log`), 'utf8');
    result.portableKernelMessages = nativeLog.split('\n').filter(line => line.includes('portable K-quant prefill'));
    report.runs.push({ ...arm, exitCode: code, result });
    save(file, report);
    if (code !== 0) throw new Error(`${arm.name} failed; partial results are saved in ${file}`);
  }
  const [on, off] = report.runs.map(r => r.result);
  report.arSameOutputs = ['cold', 'continuation'].every(turn => on[turn].inputSha256 === off[turn].inputSha256 && on[turn].outputSha256 === off[turn].outputSha256 && on[turn].cachedTokens === off[turn].cachedTokens);
  report.notes = ['MTP and AR are separate modes; cross-mode output equality is not asserted.', 'This is a native-runtime check, not an oMLX benchmark or the private agent conversation.'];
  save(file, report);
  console.log(`\nComplete. Send M3-results.json from:\n${outputDir}`);
}
(async () => {
  if (args[0] === '--self-test') { console.log(`Fixture verified: ${manifest.fixture}, ${messages.length} messages`); return; }
  if (args[0] === '--worker') await worker(...args.slice(1));
  else await main(...args);
})().catch(error => { console.error(String(error)); process.exitCode = 1; });
