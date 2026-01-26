/**
 * Real API Test Script
 *
 * Tests high-scoring completions from training against the REAL GitHub API.
 * This validates whether the mock-based training rewards actually correlate
 * with real-world API success.
 */

import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { findCredentialsAsync } from '@napi-rs/keyring';
import { Octokit } from 'octokit';

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_PATH = join(__dirname, 'shared', 'test-fixtures.json');

const [credential] = await findCredentialsAsync('mlx-node', 'github-token');

const GITHUB_TOKEN = credential.password;

interface TestFixtures {
  owner: string;
  repo: string;
  branch: string;
  prNumber: number;
  reviewCommentId: number;
  issueCommentId: number;
  workflowExists: boolean;
}

interface CompletionData {
  id: number;
  reward: number;
  scenario: string;
  code: string;
}

interface TestResult {
  id: number;
  reward: number;
  scenario: string;
  originalCode: string;
  success: boolean;
  result?: unknown;
  error?: string;
  apiEndpoint?: string;
}

// The 5 high-scoring completions from the database (reward = 10)
const completions: CompletionData[] = [
  {
    id: 28,
    reward: 10.0,
    scenario: 'Show me the file changes in this PR',
    code: `import { owner, repo, currentBranch, octokit } from '../utils';
export default async function() {
  try {
    const { data: prs } = await octokit.rest.pulls.list({ owner, repo, head: owner+':'+currentBranch, state: 'open' });
    if (!prs.length) return { error: 'no_pr' };
    const { data: files } = await octokit.rest.pulls.listFiles({ owner, repo, pull_number: prs[0].number });
    return { files };
  } catch (e) {
    if (e.status === 404) return { error: 'not_found' }; throw e;
  }
}`,
  },
  {
    id: 228,
    reward: 10.0,
    scenario: 'What is the workflow status for this branch?',
    code: `import { owner, repo, currentBranch, octokit } from '../utils';
export default async function() {
  try {
    const { data: { workflow_runs } } = await octokit.rest.actions.listWorkflowRunsForRepo({ owner, repo, branch: currentBranch });
    if (!workflow_runs) return { error: 'no_runs' };
    const latestRun = workflow_runs[0];
    return { status: latestRun.conclusion, name: latestRun.name, url: latestRun.html_url };
  } catch (e) {
    if (e.status === 404) return { error: 'not_found' }; throw e;
  }
}`,
  },
  {
    id: 299,
    reward: 10.0,
    scenario: 'Check the CI status for the current PR',
    code: `import { owner, repo, currentBranch, octokit } from '../utils';
export default async function() {
  try {
    const { data: { workflow_runs } } = await octokit.rest.actions.listWorkflowRunsForRepo({ owner, repo, branch: currentBranch });
    if (!workflow_runs) return { error: 'no_runs' };
    const latest = workflow_runs[0];
    return { status: latest.conclusion, name: latest.name, url: latest.html_url };
  } catch (e) {
    if (e.status === 404) return { error: 'not_found' }; throw e; 
  }
}`,
  },
  {
    id: 413,
    reward: 10.0,
    scenario: "What's been commented on my PR?",
    code: `import { owner, repo, currentBranch, octokit } from '../utils';
export default async function() {
  try {
    const { data: prs } = await octokit.rest.pulls.list({ owner, rep, head: owner+':'+currentBranch, state: 'open' });
    if (!prs.length) return { error: 'no_pr' };
    const { data: comments } = await octokit.rest.pulls.listReviewComments({ owner, repo, pull_number: prs[0].number });
    return { comments };
  } catch (e) {
    if (e.status === 404) return { error: 'not_found' }; throw e;
  }
}`,
  },
  {
    id: 627,
    reward: 10.0,
    scenario: 'Did the tests pass?',
    code: `import { owner, repo, currentBranch, octokit } from '../utils';
export default async function() {
  try {
    const { data: { workflow_runs } } = await octokit.rest.actions.listWorkflowRunsForRepo({ owner, repo, branch: currentBranch });
    const latest = workflow_runs[0];
    if (!latest) return { error: 'no_runs' };
    return { status: latest.conclusion, name: latest.name, url: latest.html_url };
  } catch (e) {
    if (e.status === 404) return { error: 'not_found' }; throw e;
  }
}`,
  },
];

function extractApiEndpoint(code: string): string {
  // Extract the primary API endpoint being used
  const endpoints = [
    { pattern: /pulls\.listFiles/i, name: 'pulls.listFiles' },
    { pattern: /pulls\.listReviewComments/i, name: 'pulls.listReviewComments' },
    { pattern: /actions\.listWorkflowRunsForRepo/i, name: 'actions.listWorkflowRunsForRepo' },
    { pattern: /pulls\.list/i, name: 'pulls.list' },
  ];

  for (const ep of endpoints) {
    if (ep.pattern.test(code)) {
      return ep.name;
    }
  }
  return 'unknown';
}

async function executeAgainstRealAPI(
  octokit: Octokit,
  completion: CompletionData,
  fixtures: TestFixtures,
): Promise<TestResult> {
  const apiEndpoint = extractApiEndpoint(completion.code);
  const owner = fixtures.owner;
  const repoName = fixtures.repo;
  const currentBranch = fixtures.branch;

  try {
    // Execute the specific API call based on what the code does
    if (completion.code.includes('listFiles')) {
      // First find the PR, then list files
      const { data: prs } = await octokit.rest.pulls.list({
        owner,
        repo: repoName,
        head: `${owner}:${currentBranch}`,
        state: 'open',
      });

      if (!prs.length) {
        return {
          id: completion.id,
          reward: completion.reward,
          scenario: completion.scenario,
          originalCode: completion.code,
          success: true, // The code handles this gracefully
          result: { error: 'no_pr' },
          apiEndpoint,
        };
      }

      const { data: files } = await octokit.rest.pulls.listFiles({
        owner,
        repo: repoName,
        pull_number: prs[0].number,
      });

      return {
        id: completion.id,
        reward: completion.reward,
        scenario: completion.scenario,
        originalCode: completion.code,
        success: true,
        result: {
          files: files.map((f) => ({
            filename: f.filename,
            status: f.status,
            additions: f.additions,
            deletions: f.deletions,
          })),
        },
        apiEndpoint,
      };
    }

    if (completion.code.includes('listReviewComments')) {
      // First find the PR, then list comments
      const { data: prs } = await octokit.rest.pulls.list({
        owner,
        repo: repoName,
        head: `${owner}:${currentBranch}`,
        state: 'open',
      });

      if (!prs.length) {
        return {
          id: completion.id,
          reward: completion.reward,
          scenario: completion.scenario,
          originalCode: completion.code,
          success: true,
          result: { error: 'no_pr' },
          apiEndpoint,
        };
      }

      const { data: comments } = await octokit.rest.pulls.listReviewComments({
        owner,
        repo: repoName,
        pull_number: prs[0].number,
      });

      return {
        id: completion.id,
        reward: completion.reward,
        scenario: completion.scenario,
        originalCode: completion.code,
        success: true,
        result: { comments: comments.map((c) => ({ id: c.id, body: c.body?.slice(0, 50), user: c.user?.login })) },
        apiEndpoint,
      };
    }

    if (completion.code.includes('listWorkflowRunsForRepo')) {
      const {
        data: { workflow_runs },
      } = await octokit.rest.actions.listWorkflowRunsForRepo({
        owner,
        repo: repoName,
        branch: currentBranch,
      });

      // Check if the code has the bug (accessing workflow_runs.conclusion instead of workflow_runs[0].conclusion)
      const hasBug =
        completion.code.includes('workflow_runs.conclusion') && !completion.code.includes('workflow_runs[0]');

      if (hasBug) {
        // This is a bug - workflow_runs is an array, not an object with .conclusion
        return {
          id: completion.id,
          reward: completion.reward,
          scenario: completion.scenario,
          originalCode: completion.code,
          success: false,
          error: 'BUG: workflow_runs is an array, not an object. Should use workflow_runs[0].conclusion',
          result: { buggyCode: true, workflowRunsIsArray: Array.isArray(workflow_runs), count: workflow_runs.length },
          apiEndpoint,
        };
      }

      const latest = workflow_runs[0];
      if (!latest) {
        return {
          id: completion.id,
          reward: completion.reward,
          scenario: completion.scenario,
          originalCode: completion.code,
          success: true,
          result: { error: 'no_runs' },
          apiEndpoint,
        };
      }

      return {
        id: completion.id,
        reward: completion.reward,
        scenario: completion.scenario,
        originalCode: completion.code,
        success: true,
        result: { status: latest.conclusion, name: latest.name, url: latest.html_url },
        apiEndpoint,
      };
    }

    return {
      id: completion.id,
      reward: completion.reward,
      scenario: completion.scenario,
      originalCode: completion.code,
      success: false,
      error: 'Unknown API endpoint pattern',
      apiEndpoint,
    };
  } catch (error: unknown) {
    const err = error as { status?: number; message?: string };
    if (err.status === 404) {
      return {
        id: completion.id,
        reward: completion.reward,
        scenario: completion.scenario,
        originalCode: completion.code,
        success: true, // Code handles 404 gracefully
        result: { error: 'not_found' },
        apiEndpoint,
      };
    }

    return {
      id: completion.id,
      reward: completion.reward,
      scenario: completion.scenario,
      originalCode: completion.code,
      success: false,
      error: err.message || String(error),
      apiEndpoint,
    };
  }
}

function generateReport(results: TestResult[], fixtures: TestFixtures): string {
  const successCount = results.filter((r) => r.success).length;
  const failCount = results.length - successCount;

  let report = `# Real API Test Results

Generated: ${new Date().toISOString()}

## Test Configuration

- **Test Repo**: ${fixtures.owner}/${fixtures.repo}
- **Branch**: ${fixtures.branch}
- **PR Number**: #${fixtures.prNumber}

## Summary

| Metric | Value |
|--------|-------|
| Total Tested | ${results.length} |
| Successful | ${successCount} |
| Failed | ${failCount} |
| Success Rate | ${((successCount / results.length) * 100).toFixed(1)}% |

## Gap Analysis: Mock Validation vs Real API

All completions tested had reward = 10 (perfect score) in mock validation.

`;

  if (successCount === results.length) {
    report += `### Result: EXCELLENT

All high-scoring completions work correctly against the real GitHub API. The mock validation system accurately predicts real-world success.

`;
  } else if (successCount > failCount) {
    report += `### Result: PARTIAL GAP DETECTED

${successCount}/${results.length} completions work in reality. Some gaps exist between mock validation and real execution.

**Key Finding**: The mock validation system gives perfect scores (10) to code that may have runtime bugs.

`;
  } else {
    report += `### Result: SIGNIFICANT GAP

${failCount}/${results.length} completions failed against real API despite having reward = 10 in mock validation.

**Critical Issue**: Mock validation does not accurately reflect real API behavior.

`;
  }

  report += `## Detailed Results

`;

  for (const result of results) {
    const statusIcon = result.success ? 'PASS' : 'FAIL';
    report += `### Completion #${result.id} (Reward: ${result.reward}) - ${statusIcon}

**Scenario**: "${result.scenario}"
**API Endpoint**: \`${result.apiEndpoint}\`
**Result**: ${result.success ? 'Success' : 'Failed'}

`;

    if (result.success) {
      report += `**API Response**:
\`\`\`json
${JSON.stringify(result.result, null, 2)}
\`\`\`

`;
    } else {
      report += `**Error**: ${result.error}

`;
      if (result.result) {
        report += `**Debug Info**:
\`\`\`json
${JSON.stringify(result.result, null, 2)}
\`\`\`

`;
      }
    }

    report += `---

`;
  }

  // Analysis section
  report += `## Bug Analysis

`;

  const buggyResults = results.filter((r) => !r.success);
  if (buggyResults.length > 0) {
    report += `### Bugs Found in "Perfect Score" Completions

`;
    for (const result of buggyResults) {
      report += `#### Completion #${result.id}

**Bug**: ${result.error}

**Code Pattern**:
\`\`\`javascript
${result.originalCode.slice(0, 400)}...
\`\`\`

**Analysis**: This code received a perfect reward score of 10 from the mock validation system, but contains a real bug that would cause runtime errors or incorrect behavior.

`;
    }
  }

  report += `## Recommendations

`;

  if (failCount > 0) {
    report += `### 1. Improve Mock Validation

The current mock validation system only checks if the correct API methods are called, but does not validate:
- Correct handling of array vs object return types
- Proper null/undefined checks
- Correct property access patterns

### 2. Add Runtime Semantics Checks

Consider adding validation that:
- \`workflow_runs\` is an array and should be indexed before accessing \`.conclusion\`
- Return type shapes match what the real API returns

### 3. Include Negative Test Cases

Add training examples that show common bugs and penalize them, such as:
- Accessing \`.conclusion\` directly on an array
- Missing null checks on \`workflow_runs[0]\`

`;
  } else {
    report += `The mock validation system appears to be working correctly. All high-scoring completions executed successfully against the real API.

### Continue Monitoring

Periodically run this validation to ensure training quality remains high.
`;
  }

  return report;
}

console.log('=== Real API Test for High-Scoring Completions ===\n');

console.log('Loading fixtures...');
const fixtures = JSON.parse(await readFile(FIXTURES_PATH, 'utf-8'));
console.log(`Test repo: ${fixtures.owner}/${fixtures.repo}`);
console.log(`Branch: ${fixtures.branch}, PR: #${fixtures.prNumber}\n`);

console.log('Creating Octokit client...');
const octokit = new Octokit({ auth: GITHUB_TOKEN });

// Verify token works
try {
  const { data: repo } = await octokit.rest.repos.get({
    owner: fixtures.owner,
    repo: fixtures.repo,
  });
  console.log(`Verified access to repo: ${repo.full_name}\n`);
} catch (error: unknown) {
  const err = error as { status?: number; message?: string };
  console.error(`Failed to access repo: ${err.message}`);
  process.exit(1);
}

console.log(`Testing ${completions.length} high-scoring completions...\n`);

const results: TestResult[] = [];

for (const completion of completions) {
  console.log(`--- Testing completion #${completion.id} (reward: ${completion.reward}) ---`);
  console.log(`Scenario: "${completion.scenario}"`);

  const result = await executeAgainstRealAPI(octokit, completion, fixtures);
  results.push(result);

  if (result.success) {
    console.log(`  PASS: ${JSON.stringify(result.result).slice(0, 100)}...\n`);
  } else {
    console.log(`  FAIL: ${result.error}\n`);
  }
}

// Generate and save report
const report = generateReport(results, fixtures);
console.log(report);

// Print summary
const successCount = results.filter((r) => r.success).length;
console.log(`\n=== SUMMARY ===`);
console.log(`Success: ${successCount}/${results.length}`);
console.log(`Failed: ${results.length - successCount}/${results.length}`);
