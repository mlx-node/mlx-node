// A stdio MCP server for the real-agent tests of MCP input: one tool, `file_issue`, whose call asks the client
// for input with `elicitation/create` and returns what came back. Plain Node, no imports from the repo.
//
// Usage: node elicit-mcp-server.mjs <config.json>
// config: { message, requestedSchema, record } — `record` receives one JSON line per message the client sent.
// Link mode: { mode: 'url', url, complete } — each call asks to open `url` under the id `link-<n>`; with `complete`,
// once any answer is back the server says the browser part is done (`notifications/elicitation/complete`).

import { appendFileSync, readFileSync } from 'node:fs';

const config = JSON.parse(readFileSync(process.argv[2], 'utf8'));
let nextId = 1;
let calls = 0;
const waiting = new Map();

function send(message) {
  process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', ...message })}\n`);
}

function record(entry) {
  if (config.record) appendFileSync(config.record, `${JSON.stringify(entry)}\n`);
}

function ask(method, params) {
  const id = `server-${nextId++}`;
  send({ id, method, params });
  return new Promise((resolve) => waiting.set(id, resolve));
}

async function onRequest(msg) {
  switch (msg.method) {
    case 'initialize':
      record({ kind: 'initialize', capabilities: msg.params?.capabilities ?? null });
      send({
        id: msg.id,
        result: {
          protocolVersion: msg.params?.protocolVersion ?? '2025-11-25',
          capabilities: { tools: {} },
          serverInfo: { name: 'probe', version: '1.0.0' },
        },
      });
      return;
    case 'ping':
      send({ id: msg.id, result: {} });
      return;
    case 'tools/list':
      send({
        id: msg.id,
        result: {
          tools: [
            {
              name: 'file_issue',
              description: 'File an issue. It asks the user for the details.',
              inputSchema: { type: 'object', properties: {} },
            },
          ],
        },
      });
      return;
    case 'tools/call': {
      calls += 1;
      const elicitationId = `link-${calls}`;
      const answer = await ask(
        'elicitation/create',
        config.mode === 'url'
          ? { mode: 'url', message: config.message, url: config.url, elicitationId }
          : { mode: 'form', message: config.message, requestedSchema: config.requestedSchema },
      );
      record({ kind: 'elicit-answer', answer });
      if (config.mode === 'url' && config.complete) {
        send({ method: 'notifications/elicitation/complete', params: { elicitationId } });
        record({ kind: 'complete-sent', elicitationId });
      }
      send({ id: msg.id, result: { content: [{ type: 'text', text: `elicit: ${JSON.stringify(answer)}` }] } });
      return;
    }
    default:
      if (msg.id !== undefined) send({ id: msg.id, error: { code: -32601, message: 'Method not found' } });
  }
}

let partial = '';
process.stdin.on('data', (chunk) => {
  partial += chunk.toString();
  let nl = partial.indexOf('\n');
  while (nl !== -1) {
    const line = partial.slice(0, nl);
    partial = partial.slice(nl + 1);
    nl = partial.indexOf('\n');
    if (line.trim() === '') continue;
    const msg = JSON.parse(line);
    if (msg.method === undefined && waiting.has(msg.id)) {
      const resolve = waiting.get(msg.id);
      waiting.delete(msg.id);
      resolve(msg.result ?? { error: msg.error });
      continue;
    }
    if (msg.method !== undefined && msg.id !== undefined) void onRequest(msg);
  }
});
