/**
 * Test script for the execution-based reward system
 */

import { getScenarioSnapshot, getAllSnapshotIds } from './scenario-snapshots.js';
import { executeGitHubCode } from './execute-github-code.js';
import { EXAMPLE_VALID_CODE } from './code-validator.js';

async function main() {
  console.log('=== Testing Scenario Snapshots ===');
  console.log('All scenario IDs:', getAllSnapshotIds());

  const snapshot = getScenarioSnapshot('get-inline-comments');
  console.log('\nget-inline-comments snapshot:');
  console.log('  Category:', snapshot?.category);
  console.log('  Expected calls:', snapshot?.expectedCalls?.length);
  snapshot?.expectedCalls?.forEach((call, i) => {
    console.log(`  Call ${i + 1}: ${call.method}`);
  });

  console.log('\n=== Testing Execution Helper ===');
  console.log('Executing example valid code...');

  const result = await executeGitHubCode(EXAMPLE_VALID_CODE, 'get-inline-comments', {
    timeoutMs: 10000,
  });

  console.log('\nExecution result:');
  console.log('  Success:', result.success);
  console.log('  Error:', result.error || 'none');
  console.log('  Error type:', result.errorType || 'none');
  console.log('  Time:', result.executionTimeMs, 'ms');
  console.log('  All assertions matched:', result.allAssertionsMatched);

  if (result.assertions) {
    console.log('\nAssertions:');
    result.assertions.forEach((a, i) => {
      console.log(`  ${i + 1}. ${a.method}: ${a.matched ? 'MATCHED' : 'MISMATCH'}`);
      if (!a.matched && a.mismatches) {
        a.mismatches.forEach((m) => {
          console.log(`     - ${m.key}: expected ${JSON.stringify(m.expected)}, got ${JSON.stringify(m.actual)}`);
        });
      }
    });
  }

  // Test "do nothing" code - should fail since no API calls made
  console.log('\n=== Testing "Do Nothing" Code ===');
  const doNothingCode = `import { Repository } from '@napi-rs/simple-git';
import { Octokit } from 'octokit';

// Does NOT make any API calls
export default async function() {
  return { message: 'I did nothing' };
}`;

  const doNothingResult = await executeGitHubCode(doNothingCode, 'get-inline-comments', {
    timeoutMs: 10000,
  });

  console.log('\nDo-nothing result:');
  console.log('  Success:', doNothingResult.success);
  console.log('  All required calls made:', doNothingResult.allRequiredCallsMade);
  console.log('  Call count:', doNothingResult.callCount, 'of', doNothingResult.requiredCallCount, 'required');

  if (doNothingResult.success) {
    console.log('  ❌ ERROR: Do-nothing code should NOT succeed!');
  } else {
    console.log('  ✓ Correctly rejected do-nothing code');
  }

  // Test "optional-only" code - should fail since required call not made
  // This tests the fix for the callIndex bug
  console.log('\n=== Testing "Optional-Only" Code ===');
  const optionalOnlyCode = `import { Repository } from '@napi-rs/simple-git';
import { Octokit } from 'octokit';

const repo = new Repository(process.cwd());
const remotes = repo.remotes();
const origin = remotes.find(r => r.name() === 'origin');
const remoteUrl = origin ? origin.url() : '';
const match = remoteUrl.match(/github\\.com[:/]([^/]+)\\/([^/.]+)/);
const owner = match ? match[1] : '';
const repoName = match ? match[2] : '';
const headReference = repo.head();
const currentBranch = headReference.shorthand();

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

// Only makes the OPTIONAL call, not the REQUIRED createReplyForReviewComment
export default async function() {
  const { data: prs } = await octokit.rest.pulls.list({
    owner,
    repo: repoName,
    head: \`\${owner}:\${currentBranch}\`,
    state: 'open',
  });
  return { prs };
}`;

  const optionalOnlyResult = await executeGitHubCode(optionalOnlyCode, 'reply-inline-comment', {
    timeoutMs: 10000,
  });

  console.log('\nOptional-only result:');
  console.log('  Success:', optionalOnlyResult.success);
  console.log(
    '  Matched required calls:',
    optionalOnlyResult.matchedRequiredCalls,
    'of',
    optionalOnlyResult.requiredCallCount,
  );
  console.log('  All required calls made:', optionalOnlyResult.allRequiredCallsMade);
  console.log('  Actual API calls:', optionalOnlyResult.callCount);

  if (optionalOnlyResult.success) {
    console.log('  ❌ ERROR: Optional-only code should NOT succeed!');
  } else {
    console.log('  ✓ Correctly rejected optional-only code (required call missing)');
  }

  // Test "wrong method" code - should fail with wrong_api error type
  console.log('\n=== Testing "Wrong Method" Code ===');
  const wrongMethodCode = `import { Repository } from '@napi-rs/simple-git';
import { Octokit } from 'octokit';

const repo = new Repository(process.cwd());
const remotes = repo.remotes();
const origin = remotes.find(r => r.name() === 'origin');
const remoteUrl = origin ? origin.url() : '';
const match = remoteUrl.match(/github\\.com[:/]([^/]+)\\/([^/.]+)/);
const owner = match ? match[1] : '';
const repoName = match ? match[2] : '';

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

// Calls completely wrong method - issues API instead of pulls API
export default async function() {
  const { data: issues } = await octokit.rest.issues.listForRepo({
    owner,
    repo: repoName,
  });
  return { issues };
}`;

  const wrongMethodResult = await executeGitHubCode(wrongMethodCode, 'get-inline-comments', {
    timeoutMs: 10000,
  });

  console.log('\nWrong-method result:');
  console.log('  Success:', wrongMethodResult.success);
  console.log('  Error type:', wrongMethodResult.errorType);
  console.log('  Assertions:', wrongMethodResult.assertions?.length || 0);

  if (wrongMethodResult.success) {
    console.log('  ❌ ERROR: Wrong-method code should NOT succeed!');
  } else if (wrongMethodResult.errorType === 'wrong_api') {
    console.log('  ✓ Correctly identified as wrong_api');
  } else {
    console.log('  ⚠ Error type is:', wrongMethodResult.errorType, '(expected wrong_api)');
  }

  console.log('\n=== Test Complete ===');
}

main().catch(console.error);
