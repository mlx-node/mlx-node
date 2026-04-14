#!/usr/bin/env node
/**
 * Chat API Example with Tool Calling
 *
 * Demonstrates the `ChatSession` API for conversational AI with tools.
 * The session tracks its own history, KV cache, and tool-call state on
 * the native side, so the JS caller only has to ship the first user
 * prompt + a fan-out of `sendToolResult` calls — one per executed tool.
 *
 * Workflow:
 * 1. Define tools with OpenAI-compatible format (via createToolDefinition)
 * 2. `session.send(userPrompt, { config: { tools, ... } })` — returns a
 *    ChatResult with structured tool calls and thinking already parsed.
 * 3. For each valid tool call, execute it and feed the result back via
 *    `session.sendToolResult(call.id, result, { config: { tools, ... } })`.
 *    The session handles appending the tool response to the live cache
 *    and re-invoking the model for a follow-up reply.
 *
 * Usage:
 *   yarn oxnode examples/tool-use-example.ts [model-path]
 *
 * Arguments:
 *   model-path  Path to the model directory (default: .cache/models/qwen3.5-4B-mlx-bf16)
 *
 * Environment:
 *   MODEL_PATH  Alternative way to specify model path
 */

import { resolve } from 'node:path';

import { ChatSession, createToolDefinition, Qwen35Model } from '@mlx-node/lm';

// Get model path from CLI args, environment, or default
const DEFAULT_MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3.5-4B-mlx-bf16');
const MODEL_PATH = process.argv[2] || process.env.MODEL_PATH || DEFAULT_MODEL_PATH;

// Define available tools using the createToolDefinition helper
// This automatically handles JSON.stringify() for the properties field
const tools = [
  createToolDefinition(
    'fetch_url',
    'Fetch content from a URL. Returns the response text. Use this to get data from the web.',
    {
      url: {
        type: 'string',
        description: 'The URL to fetch',
      },
      method: {
        type: 'string',
        description: 'HTTP method (GET, POST, etc.). Defaults to GET.',
        enum: ['GET', 'POST', 'PUT', 'DELETE'],
      },
    },
    ['url'],
  ),
  createToolDefinition('get_current_time', 'Get the current date and time in various formats'),
];

// Tool implementations using Node.js fetch
async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case 'fetch_url': {
      const url = args.url as string;
      const method = (args.method as string) || 'GET';
      console.log(`  [FETCH] ${url} (${method})...`);
      try {
        const response = await fetch(url, {
          method,
          headers: { 'User-Agent': 'mlx-node-tool-example/1.0' },
        });
        const text = await response.text();
        // Truncate long responses
        const truncated = text.length > 500 ? text.substring(0, 500) + '... [truncated]' : text;
        return JSON.stringify({
          status: response.status,
          statusText: response.statusText,
          body: truncated,
        });
      } catch (error) {
        return JSON.stringify({
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    case 'get_current_time': {
      const now = new Date();
      return JSON.stringify({
        iso: now.toISOString(),
        local: now.toLocaleString(),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        unix: Math.floor(now.getTime() / 1000),
      });
    }
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}

async function runToolConversation(session: ChatSession<Qwen35Model>, userPrompt: string) {
  console.log('='.repeat(75));
  console.log(`User: ${userPrompt}`);
  console.log('='.repeat(75));

  // Turn 1: hand the user prompt + tool definitions to the session.
  // The session routes this through `chatSessionStart` on turn 0 and
  // returns a parsed ChatResult.
  console.log('\n[->] Generating response with tools...');
  const result = await session.send(userPrompt, {
    config: {
      tools,
      maxNewTokens: 32768,
      temperature: 0.7,
    },
  });

  // Show thinking if the model reasoned before responding
  if (result.thinking) {
    console.log(`\n[THINK] ${result.thinking}`);
  }
  console.log(`\n[AI] ${result.text}`);
  console.log(`[INFO] Finish reason: ${result.finishReason}`);

  // Check for tool calls — they're already parsed.
  const validCalls = result.toolCalls.filter((tc) => tc.status === 'ok');
  if (validCalls.length > 0) {
    console.log(`\n[TOOL] Found ${validCalls.length} tool call(s):`);

    // Execute each tool call and feed the result back through the
    // session. The session owns the conversation history now — no
    // manual message-array plumbing required.
    let finalResult = result;
    for (const call of validCalls) {
      console.log(`   - ${call.name}(${JSON.stringify(call.arguments)})`);

      const toolResult = await executeTool(call.name, call.arguments as Record<string, unknown>);
      const displayResult = toolResult.length > 200 ? toolResult.substring(0, 200) + '...' : toolResult;
      console.log(`   [<-] ${displayResult}`);

      console.log('\n[->] Generating follow-up response with tool result...');
      finalResult = await session.sendToolResult(call.id, toolResult, {
        config: {
          tools,
          maxNewTokens: 2048,
          temperature: 0.9,
        },
      });

      if (finalResult.thinking) {
        console.log(`\n[THINK] ${finalResult.thinking}`);
      }
      console.log(`\n[AI] ${finalResult.text}`);
    }

    console.log(`\n[AI] Final response: ${finalResult.text}`);

    // Log any parsing errors
    for (const call of result.toolCalls) {
      if (call.status !== 'ok') {
        console.log(`   [WARN] ${call.name || '(unknown)'}: ${call.status} - ${call.error}`);
      }
    }
  } else {
    console.log('\n[NOTE] No tool calls detected - direct response.');
  }

  console.log('\n');
}

async function main() {
  console.log('+' + '-'.repeat(58) + '+');
  console.log('|   Chat API Example with Tool Calling                     |');
  console.log('+' + '-'.repeat(58) + '+\n');

  console.log(`Loading model from: ${MODEL_PATH}\n`);
  const model = await Qwen35Model.load(MODEL_PATH);
  console.log('[OK] Model loaded\n');

  // Example prompts that should trigger tool use
  const prompts = ['What time is it right now?', 'Can you fetch https://httpbin.org/json and tell me what it returns?'];

  // Fresh session per prompt so each prompt starts from a clean cache.
  for (const prompt of prompts) {
    const session = new ChatSession(model);
    await runToolConversation(session, prompt);
  }

  console.log('+' + '-'.repeat(58) + '+');
  console.log('|   Example Complete                                       |');
  console.log('+' + '-'.repeat(58) + '+\n');
}

main().catch((error) => {
  console.error('\n[ERROR] Example failed!');
  console.error('Error:', error.message);
  console.error('\nStack trace:', error.stack);
  process.exit(1);
});
