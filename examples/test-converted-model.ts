import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { ModelLoader, formatToolResponse, createToolDefinition, type ChatConfig, type ChatMessage } from '@mlx-node/lm';
import { input } from '@inquirer/prompts';

import { initLspService, executeToolCall, type LspToolArgs } from './grpo/lsp';
import { githubToolReward } from './grpo/github-tool-reward';

const DEFAULT_MODEL_PATH = resolve(process.cwd(), 'outputs', 'sft-octokit', 'checkpoint-300');

// LSP tool definition for querying Octokit API documentation
const lspTool = createToolDefinition(
  'lsp',
  'Query Octokit API documentation for method parameters or return types',
  {
    method: { type: 'string', description: 'Method path, e.g., "octokit.rest.pulls.list"' },
    kind: { type: 'string', enum: ['parameters', 'return'], description: 'What to query: parameters or return type' },
  },
  ['method', 'kind'],
);

// run_js tool definition (already implicitly supported via system prompt)
const runJsTool = createToolDefinition(
  'run_js',
  'Execute JavaScript code that uses the Octokit API',
  {
    code: { type: 'string', description: 'JavaScript code to execute' },
  },
  ['code'],
);

// Parse CLI arguments
const { values } = parseArgs({
  options: {
    help: {
      type: 'boolean',
      short: 'h',
      default: false,
    },
    model: {
      type: 'string',
      short: 'm',
      default: DEFAULT_MODEL_PATH,
    },
    interactive: {
      type: 'boolean',
      short: 'i',
      default: false,
    },
    iterations: {
      type: 'string',
      short: 'n',
      default: '1',
    },
  },
  allowPositionals: true,
});

if (values.help) {
  console.log(`
Usage: yarn oxnode examples/test-converted-model.ts [options]

Options:
  -h, --help        Show this help message
  -i, --interactive Run in interactive mode

Examples:
  yarn oxnode examples/test-converted-model.ts                 # Run generation tests
`);
  process.exit(0);
}

console.log('╔════════════════════════════════════════════════════════╗');
console.log('║   Testing Converted MLX Model (bf16)                   ║');
console.log('╚════════════════════════════════════════════════════════╝\n');

console.log(`Loading model from: ${values.model}`);
console.log('(Tokenizer will be loaded automatically)\n');

// Load model (tokenizer is loaded automatically)
const model = await ModelLoader.loadPretrained(values.model);

console.log('Model and tokenizer loaded');
console.log(`Config: tie_word_embeddings=${model.getConfig().tieWordEmbeddings}\n`);

const lspService = await initLspService();

const prompts = [
  `Use \`head\` parameter to filter the pull requests, The head value should be \`\${owner}:\${currentBranch}\`.
This pr should be opened.
Check \`lsp\` tool for the API docs. Check the parameters description carefully.
Return the prs array.`,
];

const systemPrompt = `You are a GitHub API code generator assistant, the assistant in the following conversation is yourself.
You generate \`run_js\` tool calls for GitHub related tasks with octokit.js API.

### Tips:
- If you are handling the tasks related to pull request, you always need to find the pull request first, then pass the pull request number as the parameter to the following \`octokit\` API.
- Call the \`lsp\` tool early will save your time.

### Code format for \`run_js\`:
\`\`\`js
import { owner, repo, currentBranch, octokit } from './utils'

export default async function() { ... }
\`\`\`

\`utils\` is a predefined helper module you can use it directly without any concern, **owner, repo, currentBranch is known**.

For example:
- owner: "Brooooooklyn"
- repo: "mlx-node"
- currentBranch: "fix-issue"

### Task:
- For any \`octokit\` logic, you need to use \`lsp\` tool to get the API docs first, your knowledge is outdated.
- You can call \`lsp\` multiple times to get the API docs before you generate the JavaScript code.
- You need to implement full logic in your JavaScript in \`run_js\` tool rather than leave it in-complete.
- Always \`export default async function() { ... }\`. **Return** the result.
`;

const numIterations = parseInt(values.iterations!, 10);
const allScores: number[] = [];

for (let iter = 1; iter <= numIterations; iter++) {
  for (const prompt of prompts) {
    if (numIterations === 1) {
      console.log('-'.repeat(60));
      console.log(`Prompt: "${prompt}"`);
      console.log('-'.repeat(60));
    }

    // Generate using the simple message-based API
    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt },
    ];
    const startTime = Date.now();
    const chatOptions: ChatConfig = {
      maxNewTokens: 1536,
      temperature: 0.6,
      topP: 0.95,
      topK: 20,
      returnLogprobs: false,
      tools: [lspTool, runJsTool],
    };
    let result = await model.chat(messages, chatOptions);

    if (numIterations === 1) console.log(result.rawText);

    let generatedToken = 0;
    let toolCalls = 0;
    let validationScore = 0;

    while (result.finishReason === 'tool_calls') {
      generatedToken += result.numTokens;
      const [toolCall] = result.toolCalls;
      if (toolCall.status === 'ok') {
        toolCalls++;
        if (toolCall.name === 'lsp' && toolCalls <= 5) {
          const lspArgs = toolCall.arguments as unknown as LspToolArgs;
          const lspResult = executeToolCall(lspService, lspArgs);
          messages.push({
            role: 'assistant',
            content: result.text,
            reasoningContent: result.thinking ?? void 0,
            toolCallId: toolCall.id,
            toolCalls: [
              {
                ...toolCall,
                arguments: JSON.stringify(toolCall.arguments),
              },
            ],
          });
          messages.push({ role: 'tool', content: formatToolResponse(lspResult) });
          if (numIterations === 1) {
            console.log(toolCall.rawContent);
            console.log(lspResult);
          }
          result = await model.chat(messages, chatOptions);
        } else if (toolCall.name === 'run_js') {
          const validation = await githubToolReward(
            [
              {
                prompt,
                // @ts-expect-error
                completion: result,
              },
            ],
            'get-pr',
          );
          validationScore = validation[0];
          if (validationScore >= 7) {
            if (numIterations === 1) {
              console.log('Validation passed!');
              console.log(result.rawText);
            }
            break;
          } else if (values.interactive) {
            console.log('Validation failed!');
            console.log(result.rawText);
            const userReject = await input({
              message: 'Reply',
            });
            messages.push({ role: 'assistant', content: result.rawText });
            messages.push({ role: 'user', content: userReject });
            result = await model.chat(messages, chatOptions);
          } else {
            break;
          }
        } else {
          break;
        }
      } else {
        if (numIterations === 1) console.error(`Tool call failed: ${toolCall.name}`, toolCall.error);
      }
    }
    generatedToken += result.numTokens;
    messages.push({ role: 'assistant', content: result.rawText });
    if (numIterations === 1) console.log(result.rawText);
    const duration = Date.now() - startTime;
    const tokensPerSecond = (generatedToken / duration) * 1000;

    allScores.push(validationScore);

    if (numIterations === 1) {
      console.log(
        `\nGenerated (${generatedToken} tokens, ${duration}ms, ${tokensPerSecond.toFixed(2)} tokens/s), Code score: ${validationScore}`,
      );
      console.log('');
    } else {
      console.log(`Run ${iter}: ${validationScore}`);
    }
  }
}

if (numIterations > 1) {
  const avg = allScores.reduce((a, b) => a + b, 0) / allScores.length;
  const score9 = allScores.filter((s) => s >= 9).length;
  const score55 = allScores.filter((s) => s >= 5 && s < 9).length;
  const scoreLow = allScores.filter((s) => s < 5).length;
  console.log('\n' + '='.repeat(60));
  console.log(`Summary (${numIterations} runs):`);
  console.log(`  Average score: ${avg.toFixed(2)}`);
  console.log(`  Score >= 9: ${score9} (${((score9 / numIterations) * 100).toFixed(1)}%)`);
  console.log(`  Score 5-8.9: ${score55} (${((score55 / numIterations) * 100).toFixed(1)}%)`);
  console.log(`  Score < 5: ${scoreLow} (${((scoreLow / numIterations) * 100).toFixed(1)}%)`);
  console.log('='.repeat(60));
}

console.log('╔════════════════════════════════════════════════════════╗');
console.log('║   Test Complete                                        ║');
console.log('╚════════════════════════════════════════════════════════╝\n');
