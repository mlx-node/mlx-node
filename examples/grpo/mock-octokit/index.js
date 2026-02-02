/**
 * Mock Octokit for Execution-Based Reward Validation
 *
 * This mock package intercepts all Octokit API calls and:
 * 1. Asserts actual params match expected patterns from scenario snapshot
 * 2. Records assertion results (match/mismatch)
 * 3. Returns mocked responses
 * 4. Outputs results via stdout for parent process to capture
 *
 * Environment variables:
 * - SCENARIO_SNAPSHOT: JSON string of ScenarioSnapshot
 * - MOCK_CONTEXT: JSON string with { owner, repo, branch } for pattern matching
 */

// Load scenario snapshot from environment
let snapshot = null;
let context = { owner: 'test-owner', repo: 'test-repo', branch: 'test-feature' };
let callIndex = 0; // Current position in expectedCalls array
let actualCallsMade = 0; // Actual API calls made by model
let matchedRequiredCalls = 0; // Required calls that were successfully matched
let assertionResults = [];
let lastResult = null;

try {
  if (process.env.SCENARIO_SNAPSHOT) {
    snapshot = JSON.parse(process.env.SCENARIO_SNAPSHOT);
  }
  if (process.env.MOCK_CONTEXT) {
    context = JSON.parse(process.env.MOCK_CONTEXT);
  }
} catch (e) {
  console.error('[mock-octokit] Failed to parse env:', e.message);
}

/**
 * Validate that the execution result contains all expected return keys.
 * Catches destructuring bugs where model writes `const { comments }`
 * instead of `const { data: comments }`, resulting in undefined values.
 */
function validateResult(result, expectedReturnKeys) {
  if (!expectedReturnKeys || expectedReturnKeys.length === 0) {
    return { valid: true, skipped: true };
  }

  if (typeof result !== 'object' || result === null) {
    return { valid: false, error: 'Result is not an object' };
  }

  // Error results are valid (model handled edge case)
  if (result.error) {
    return { valid: true, isErrorCase: true };
  }

  const missing = [];
  const undefinedKeys = [];

  for (const key of expectedReturnKeys) {
    if (!(key in result)) {
      missing.push(key);
    } else if (result[key] === undefined) {
      undefinedKeys.push(key); // Catches destructuring bugs!
    }
  }

  if (missing.length > 0 || undefinedKeys.length > 0) {
    return { valid: false, missing, undefinedKeys };
  }

  return { valid: true };
}

/**
 * Check if actual value matches expected pattern.
 * Patterns:
 * - '{owner}' matches context.owner
 * - '{repo}' matches context.repo
 * - '{branch}' matches context.branch
 * - '{any}' matches any value
 * - '{contains:xyz}' matches if value contains 'xyz'
 * - Exact values must match exactly
 */
function matchesPattern(actual, expected) {
  if (expected === '{any}') {
    return true;
  }

  if (expected === '{owner}') {
    return actual === context.owner;
  }

  if (expected === '{repo}') {
    return actual === context.repo;
  }

  if (expected === '{branch}') {
    return actual === context.branch;
  }

  if (typeof expected === 'string' && expected.startsWith('{owner}:')) {
    // Pattern like '{owner}:{branch}' for head parameter
    const expectedValue = `${context.owner}:${context.branch}`;
    return actual === expectedValue;
  }

  if (typeof expected === 'string' && expected.startsWith('{contains:')) {
    const searchStr = expected.slice(10, -1);
    return typeof actual === 'string' && actual.includes(searchStr);
  }

  // Exact match
  return actual === expected;
}

/**
 * Check if actual params match expected params pattern
 */
function paramsMatch(actualParams, expectedParams) {
  const mismatches = [];

  for (const [key, expectedValue] of Object.entries(expectedParams)) {
    const actualValue = actualParams[key];

    if (!matchesPattern(actualValue, expectedValue)) {
      mismatches.push({
        key,
        expected: expectedValue,
        actual: actualValue,
      });
    }
  }

  return {
    matched: mismatches.length === 0,
    mismatches,
  };
}

/**
 * Find the next expected call that matches the method.
 * Skips optional calls that don't match.
 */
function findExpectedCall(method) {
  if (!snapshot || !snapshot.expectedCalls) {
    return null;
  }

  // Try current index first
  while (callIndex < snapshot.expectedCalls.length) {
    const expected = snapshot.expectedCalls[callIndex];

    if (expected.method === method) {
      return expected;
    }

    // Skip optional calls
    if (expected.optional) {
      callIndex++;
      continue;
    }

    // Method mismatch on non-optional call
    return null;
  }

  return null;
}

/**
 * Record an API call and return the mocked response
 */
function recordAndRespond(method, actualParams) {
  actualCallsMade++;
  const expectedCall = findExpectedCall(method);

  if (!expectedCall) {
    // Unexpected call
    assertionResults.push({
      matched: false,
      callIndex,
      method,
      expected: null,
      actual: { method, params: actualParams },
      error: `Unexpected API call: ${method}`,
    });

    // Return a generic error response
    const error = new Error(`Unexpected API call: ${method}`);
    error.status = 400;
    throw error;
  }

  // Check params
  const matchResult = paramsMatch(actualParams, expectedCall.expectedParams);

  assertionResults.push({
    matched: matchResult.matched,
    callIndex,
    method,
    expected: {
      method: expectedCall.method,
      params: expectedCall.expectedParams,
    },
    actual: {
      method,
      params: actualParams,
    },
    mismatches: matchResult.mismatches,
    isRequired: !expectedCall.optional,
  });

  // Track matched required calls
  if (matchResult.matched && !expectedCall.optional) {
    matchedRequiredCalls++;
  }

  // Move to next expected call
  callIndex++;

  // Return mocked response
  return expectedCall.response;
}

/**
 * REST API Proxy
 * Intercepts calls like: octokit.rest.pulls.list(params)
 */
class RestProxy {
  constructor() {
    return new Proxy(this, {
      get: (_, namespace) => {
        // e.g., octokit.rest.pulls
        return new Proxy(
          {},
          {
            get: (__, methodName) => {
              // e.g., octokit.rest.pulls.list
              return async (params = {}) => {
                if (typeof namespace === 'string' && typeof methodName === 'string') {
                  const method = `rest.${namespace}.${methodName}`;
                  return recordAndRespond(method, params);
                }
                return null;
              };
            },
          },
        );
      },
    });
  }
}

/**
 * GraphQL Proxy
 * Intercepts calls like: octokit.graphql(query, variables)
 */
async function graphqlProxy(query, variables = {}) {
  const params = { query, ...variables };
  return recordAndRespond('graphql', params);
}

/**
 * Main Octokit class
 */
export class Octokit {
  constructor() {
    // Ignore auth options - we're mocking
    this.rest = new RestProxy();
    this.graphql = graphqlProxy;
  }
}

/**
 * Store the result from the generated code's default export
 */
export function setLastResult(result) {
  lastResult = result;
}

/**
 * Get assertion results for output
 */
export function getAssertionResults() {
  return assertionResults;
}

/**
 * Output results when process exits
 * This is captured by the parent process
 *
 * NOTE: Using 'exit' instead of 'beforeExit' because:
 * - 'beforeExit' does NOT fire when process.exit() is called
 * - 'exit' fires even on process.exit(), enabling output capture on errors
 * - 'exit' only allows synchronous operations (console.log is sync, so this works)
 */
process.on('exit', () => {
  // Only output if we haven't already (avoid double output)
  if (process.env.MOCK_OUTPUT_DONE) return;
  process.env.MOCK_OUTPUT_DONE = '1';

  // Read execution result from global (set by wrapper)
  const executionResult = typeof global !== 'undefined' ? global.__EXECUTION_RESULT__ : lastResult;

  // Validate result against expected return keys
  const resultValidation = validateResult(executionResult, snapshot?.expectedReturnKeys || []);

  // Count required (non-optional) expected calls
  const requiredCalls = snapshot?.expectedCalls?.filter((c) => !c.optional) || [];
  const requiredCallCount = requiredCalls.length;

  // Success requires:
  // 1. All assertions matched (no param mismatches)
  // 2. All required expected calls were actually matched (not just skipped over)
  // 3. Result contains all expected return keys with valid values
  const allAssertionsMatched = assertionResults.every((a) => a.matched);
  const allRequiredCallsMade = matchedRequiredCalls >= requiredCallCount;
  const success = allAssertionsMatched && allRequiredCallsMade && resultValidation.valid;

  const output = {
    success,
    result: executionResult,
    assertions: assertionResults,
    callCount: actualCallsMade, // Actual API calls made, not array index
    expectedCallCount: snapshot?.expectedCalls?.length || 0,
    requiredCallCount,
    matchedRequiredCalls,
    allAssertionsMatched,
    allRequiredCallsMade,
    resultValidation,
  };

  // Output as JSON on a special line for parent to parse
  console.log('__MOCK_OUTPUT__' + JSON.stringify(output));
});

export default { Octokit };
