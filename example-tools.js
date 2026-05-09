const BASE_URL = 'http://localhost:8080/v1';
const MODEL = 'qwen3.5-9b';

async function callResponses(input, options = {}) {
  const res = await fetch(`${BASE_URL}/responses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      input,
      reasoning: { effort: 'none' },
      ...options,
    }),
  });
  return res.json();
}

// Define tools
const tools = [
  {
    type: 'function',
    name: 'get_weather',
    description: 'Get the current weather for a given location.',
    parameters: {
      type: 'object',
      properties: {
        location: { type: 'string', description: 'City name, e.g. "San Francisco"' },
        unit: { type: 'string', enum: ['celsius', 'fahrenheit'], description: 'Temperature unit' },
      },
      required: ['location'],
    },
  },
  {
    type: 'function',
    name: 'search_web',
    description: 'Search the web for information.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query' },
      },
      required: ['query'],
    },
  },
];

// Fake tool implementations
function executeToolCall(name, args) {
  switch (name) {
    case 'get_weather':
      return JSON.stringify({
        temperature: 18,
        condition: 'sunny',
        location: args.location,
        unit: args.unit ?? 'celsius',
      });
    case 'search_web':
      return JSON.stringify({ results: [`Result 1 for "${args.query}"`, `Result 2 for "${args.query}"`] });
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}

async function main() {
  // Step 1: Send a message with tools
  console.log('--- Step 1: Ask a question that needs tools ---');
  const response1 = await callResponses('What is the weather in Tokyo and San Francisco?', { tools });

  console.log('Status:', response1.status);
  console.log(
    'Output items:',
    response1.output.map((o) => `${o.type}${o.name ? ` (${o.name})` : ''}`),
  );

  // Step 2: If the model made tool calls, execute them and send results back
  const functionCalls = response1.output.filter((o) => o.type === 'function_call');

  if (functionCalls.length === 0) {
    console.log('No tool calls — model answered directly:');
    console.log(response1.output_text);
    return;
  }

  console.log(`\n--- Step 2: Execute ${functionCalls.length} tool call(s) ---`);

  // Build input items: echo back the function calls + provide results
  const toolResultItems = [];
  for (const fc of functionCalls) {
    const args = JSON.parse(fc.arguments);
    const result = executeToolCall(fc.name, args);
    console.log(`  ${fc.name}(${JSON.stringify(args)}) => ${result}`);

    // Echo the function_call back
    toolResultItems.push({
      type: 'function_call',
      id: fc.id,
      call_id: fc.call_id,
      name: fc.name,
      arguments: fc.arguments,
    });

    // Provide the result
    toolResultItems.push({
      type: 'function_call_output',
      call_id: fc.call_id,
      output: result,
    });
  }

  // Step 3: Send tool results back using previous_response_id
  console.log('\n--- Step 3: Send tool results back ---');
  const response2 = await callResponses(toolResultItems, {
    tools,
    previous_response_id: response1.id,
  });

  console.log('Status:', response2.status);
  console.log('Response:', response2.output_text);
  console.log('\nUsage:', response2.usage);
}

main().catch(console.error);
