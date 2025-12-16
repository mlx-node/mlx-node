#!/usr/bin/env node
/**
 * Chat API Example with Tool Calling
 *
 * Demonstrates how to use the model.chat() API for conversational AI with tools.
 * The model returns structured responses with tool calls and thinking extracted.
 *
 * This example shows the streamlined workflow:
 * 1. Define tools with OpenAI-compatible format (using createToolDefinition helper)
 * 2. Call model.chat() - returns ChatResult with structured tool calls and thinking
 * 3. Execute tools using the parsed arguments (already JS objects!)
 * 4. Continue conversation with tool results
 *
 * Usage:
 *   yarn oxnode examples/tool-use-example.ts [model-path]
 *
 * Arguments:
 *   model-path  Path to the model directory (default: .cache/models/qwen3-0.6b-mlx-bf16)
 *
 * Environment:
 *   MODEL_PATH  Alternative way to specify model path
 */

import { resolve } from 'node:path';
import { ModelLoader, formatToolResponse, createToolDefinition } from '@mlx-node/lm';

// Get model path from CLI args, environment, or default
const DEFAULT_MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3-0.6b-mlx-bf16');
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

async function runToolConversation(model: Awaited<ReturnType<typeof ModelLoader.loadPretrained>>, userPrompt: string) {
  console.log('='.repeat(75));
  console.log(`User: ${userPrompt}`);
  console.log('='.repeat(75));

  let messages = [{ role: 'user', content: userPrompt }];

  // Use chat() API - returns ChatResult with structured tool calls and thinking
  console.log('\n[->] Generating response with tools...');
  const result = await model.chat(messages, {
    tools,
    maxNewTokens: 2048,
    temperature: 0.7,
  });

  // Show thinking if the model reasoned before responding
  if (result.thinking) {
    console.log(`\n[THINK] ${result.thinking}`);
  }
  console.log(`\n[AI] ${result.text}`);
  console.log(`[INFO] Finish reason: ${result.finishReason}`);

  // Check for tool calls - they're already parsed!
  if (result.toolCalls.length > 0) {
    console.log(`\n[TOOL] Found ${result.toolCalls.length} tool call(s):`);

    for (const call of result.toolCalls) {
      if (call.status === 'ok') {
        // Arguments are already a JS object - no JSON.parse needed!
        console.log(`   - ${call.name}(${JSON.stringify(call.arguments)})`);
        console.log(`     ID: ${call.id}`);

        // Execute the tool
        const toolResult = await executeTool(call.name, call.arguments);
        const displayResult = toolResult.length > 200 ? toolResult.substring(0, 200) + '...' : toolResult;
        console.log(`\n[<-] Tool result: ${displayResult}`);

        // Continue conversation with tool result
        messages = [
          ...messages,
          { role: 'assistant', content: result.rawText }, // Use rawText to preserve tool_call tags
          { role: 'user', content: formatToolResponse(toolResult) },
        ];

        // Generate final response using chat() API
        console.log('\n[->] Generating final response...');
        const finalResult = await model.chat(messages, {
          tools,
          maxNewTokens: 2048,
          temperature: 0.7,
          topP: 0.9,
        });

        // Show thinking if model reasoned through the tool result
        if (finalResult.thinking) {
          console.log(`\n[THINK] ${finalResult.thinking}`);
        }
        console.log(`\n[AI] Final response: ${finalResult.text}`);
      } else {
        // Handle parsing errors gracefully
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
  const model = await ModelLoader.loadPretrained(MODEL_PATH);
  console.log('[OK] Model loaded\n');

  // Example prompts that should trigger tool use
  const prompts = ['What time is it right now?', 'Can you fetch https://httpbin.org/json and tell me what it returns?'];

  for (const prompt of prompts) {
    await runToolConversation(model, prompt);
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
