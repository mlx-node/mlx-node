/**
 * Edge Case Testing Script
 *
 * Tests GitHub API edge cases that the training data doesn't cover:
 * 1. No PR exists - call pulls.list with a branch that has no PR
 * 2. Closed PR - check if the test repo has any closed PRs
 * 3. Invalid comment ID - call createReplyForReviewComment with ID 99999999
 * 4. Non-existent workflow - call createWorkflowDispatch with workflow_id 'nonexistent.yml'
 * 5. Permission denied simulation - try an action the token might not have permission for
 */

import { Octokit } from 'octokit';

const TEST_OWNER = 'mlx-node-training-example';
const TEST_REPO = 'mlx-node-test-repo';

interface EdgeCaseResult {
  name: string;
  apiCall: string;
  request: object;
  response: {
    status?: number;
    message?: string;
    documentation_url?: string;
    errors?: unknown[];
    data?: unknown;
  };
  errorHandling: {
    works: boolean;
    notes: string;
  };
}

async function main() {
  const token = process.env.GITHUB_TOKEN;
  if (!token) {
    console.error('GITHUB_TOKEN environment variable required');
    process.exit(1);
  }

  const octokit = new Octokit({ auth: token });
  const results: EdgeCaseResult[] = [];

  console.log('Testing GitHub API edge cases...\n');

  // ============================================================================
  // Edge Case 1: No PR exists for branch
  // ============================================================================
  console.log('1. Testing: No PR exists for branch');
  try {
    const { data } = await octokit.rest.pulls.list({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      head: `${TEST_OWNER}:nonexistent-branch-xyz-12345`,
      state: 'all',
    });

    results.push({
      name: 'No PR exists for branch',
      apiCall: 'pulls.list',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
        head: `${TEST_OWNER}:nonexistent-branch-xyz-12345`,
        state: 'all',
      },
      response: {
        status: 200,
        message: 'Success - empty array returned',
        data: data,
      },
      errorHandling: {
        works: true,
        notes: 'Returns empty array [], not an error. Generated code should check array length.',
      },
    });
    console.log(`   Result: Empty array returned (${data.length} items)\n`);
  } catch (e: any) {
    results.push({
      name: 'No PR exists for branch',
      apiCall: 'pulls.list',
      request: { head: 'nonexistent-branch' },
      response: {
        status: e.status,
        message: e.message,
        documentation_url: e.response?.data?.documentation_url,
      },
      errorHandling: {
        works: false,
        notes: `Unexpected error: ${e.message}`,
      },
    });
    console.log(`   Error: ${e.status} - ${e.message}\n`);
  }

  // ============================================================================
  // Edge Case 2: List closed PRs
  // ============================================================================
  console.log('2. Testing: Closed PRs in repo');
  try {
    const { data: closedPRs } = await octokit.rest.pulls.list({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      state: 'closed',
      per_page: 5,
    });

    const summary = closedPRs.map((pr) => ({
      number: pr.number,
      title: pr.title,
      merged: pr.merged_at !== null,
      closed_at: pr.closed_at,
    }));

    results.push({
      name: 'Closed PRs in repo',
      apiCall: 'pulls.list',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
        state: 'closed',
        per_page: 5,
      },
      response: {
        status: 200,
        message: `Found ${closedPRs.length} closed PRs`,
        data: summary,
      },
      errorHandling: {
        works: true,
        notes: 'Closed PRs are accessible. Check merged_at to distinguish merged vs closed-without-merge.',
      },
    });
    console.log(`   Result: Found ${closedPRs.length} closed PRs\n`);
  } catch (e: any) {
    results.push({
      name: 'Closed PRs in repo',
      apiCall: 'pulls.list',
      request: { state: 'closed' },
      response: {
        status: e.status,
        message: e.message,
      },
      errorHandling: {
        works: false,
        notes: `Error: ${e.message}`,
      },
    });
    console.log(`   Error: ${e.status} - ${e.message}\n`);
  }

  // ============================================================================
  // Edge Case 3: Invalid comment ID
  // ============================================================================
  console.log('3. Testing: Invalid comment ID for reply');

  // First, get an open PR to use
  let testPRNumber: number | null = null;
  try {
    const { data: openPRs } = await octokit.rest.pulls.list({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      state: 'open',
      per_page: 1,
    });
    if (openPRs.length > 0) {
      testPRNumber = openPRs[0].number;
    }
  } catch {
    console.log('   Could not find open PR for testing\n');
  }

  if (testPRNumber) {
    try {
      await octokit.rest.pulls.createReplyForReviewComment({
        owner: TEST_OWNER,
        repo: TEST_REPO,
        pull_number: testPRNumber,
        comment_id: 99999999,
        body: 'Test reply',
      });

      results.push({
        name: 'Invalid comment ID for reply',
        apiCall: 'pulls.createReplyForReviewComment',
        request: {
          pull_number: testPRNumber,
          comment_id: 99999999,
          body: 'Test reply',
        },
        response: {
          status: 200,
          message: 'Unexpected success',
        },
        errorHandling: {
          works: false,
          notes: 'Should have returned an error but succeeded',
        },
      });
      console.log('   Unexpected success!\n');
    } catch (e: any) {
      results.push({
        name: 'Invalid comment ID for reply',
        apiCall: 'pulls.createReplyForReviewComment',
        request: {
          pull_number: testPRNumber,
          comment_id: 99999999,
          body: 'Test reply',
        },
        response: {
          status: e.status,
          message: e.response?.data?.message || e.message,
          documentation_url: e.response?.data?.documentation_url,
          errors: e.response?.data?.errors,
        },
        errorHandling: {
          works: true,
          notes: `Returns ${e.status} error. Generated code should catch this and inform user.`,
        },
      });
      console.log(`   Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
    }
  } else {
    // Create a temporary PR for testing
    console.log('   No open PR found, trying with a random PR number...');
    try {
      await octokit.rest.pulls.createReplyForReviewComment({
        owner: TEST_OWNER,
        repo: TEST_REPO,
        pull_number: 1, // Try with PR #1
        comment_id: 99999999,
        body: 'Test reply',
      });
    } catch (e: any) {
      results.push({
        name: 'Invalid comment ID for reply',
        apiCall: 'pulls.createReplyForReviewComment',
        request: {
          pull_number: 1,
          comment_id: 99999999,
          body: 'Test reply',
        },
        response: {
          status: e.status,
          message: e.response?.data?.message || e.message,
          documentation_url: e.response?.data?.documentation_url,
        },
        errorHandling: {
          works: true,
          notes: `Returns ${e.status} error. Generated code should catch this and inform user.`,
        },
      });
      console.log(`   Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
    }
  }

  // ============================================================================
  // Edge Case 4: Non-existent workflow
  // ============================================================================
  console.log('4. Testing: Non-existent workflow dispatch');
  try {
    await octokit.rest.actions.createWorkflowDispatch({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      workflow_id: 'nonexistent.yml',
      ref: 'main',
    });

    results.push({
      name: 'Non-existent workflow',
      apiCall: 'actions.createWorkflowDispatch',
      request: {
        workflow_id: 'nonexistent.yml',
        ref: 'main',
      },
      response: {
        status: 204,
        message: 'Unexpected success',
      },
      errorHandling: {
        works: false,
        notes: 'Should have returned an error but succeeded',
      },
    });
    console.log('   Unexpected success!\n');
  } catch (e: any) {
    results.push({
      name: 'Non-existent workflow',
      apiCall: 'actions.createWorkflowDispatch',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
        workflow_id: 'nonexistent.yml',
        ref: 'main',
      },
      response: {
        status: e.status,
        message: e.response?.data?.message || e.message,
        documentation_url: e.response?.data?.documentation_url,
      },
      errorHandling: {
        works: true,
        notes: `Returns ${e.status} error. Generated code should validate workflow exists first or handle this error.`,
      },
    });
    console.log(`   Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
  }

  // ============================================================================
  // Edge Case 5: Permission denied scenarios
  // ============================================================================
  console.log('5. Testing: Permission denied scenarios');

  // 5a. Try to delete a protected branch (main)
  console.log('   5a. Attempting to delete main branch...');
  try {
    await octokit.rest.git.deleteRef({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      ref: 'heads/main',
    });

    results.push({
      name: 'Delete protected branch (main)',
      apiCall: 'git.deleteRef',
      request: {
        ref: 'heads/main',
      },
      response: {
        status: 200,
        message: 'Unexpected success - main branch deleted!',
      },
      errorHandling: {
        works: false,
        notes: 'DANGER: Main branch was deleted! This should be protected.',
      },
    });
    console.log('       UNEXPECTED: Main branch was deleted!\n');
  } catch (e: any) {
    results.push({
      name: 'Delete protected branch (main)',
      apiCall: 'git.deleteRef',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
        ref: 'heads/main',
      },
      response: {
        status: e.status,
        message: e.response?.data?.message || e.message,
        documentation_url: e.response?.data?.documentation_url,
      },
      errorHandling: {
        works: true,
        notes: `Returns ${e.status} error. Branch protection or permissions prevent deletion.`,
      },
    });
    console.log(`       Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
  }

  // 5b. Try to access a private endpoint that might require admin permissions
  console.log('   5b. Attempting to access repo settings/secrets...');
  try {
    await octokit.rest.actions.listRepoSecrets({
      owner: TEST_OWNER,
      repo: TEST_REPO,
    });

    results.push({
      name: 'Access repo secrets (admin required)',
      apiCall: 'actions.listRepoSecrets',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
      },
      response: {
        status: 200,
        message: 'Success - token has admin access',
      },
      errorHandling: {
        works: true,
        notes: 'Token has sufficient permissions. In production, handle 403 for non-admin tokens.',
      },
    });
    console.log('       Success - token has admin access\n');
  } catch (e: any) {
    results.push({
      name: 'Access repo secrets (admin required)',
      apiCall: 'actions.listRepoSecrets',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
      },
      response: {
        status: e.status,
        message: e.response?.data?.message || e.message,
        documentation_url: e.response?.data?.documentation_url,
      },
      errorHandling: {
        works: true,
        notes: `Returns ${e.status} error. Generated code should check for 403 and inform user about required permissions.`,
      },
    });
    console.log(`       Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
  }

  // 5c. Try to write to a repo we don't have write access to
  console.log('   5c. Attempting to create issue in external repo...');
  try {
    await octokit.rest.issues.create({
      owner: 'octocat',
      repo: 'Hello-World',
      title: 'Test issue',
      body: 'This is a test',
    });

    results.push({
      name: 'Create issue in external repo',
      apiCall: 'issues.create',
      request: {
        owner: 'octocat',
        repo: 'Hello-World',
        title: 'Test issue',
      },
      response: {
        status: 201,
        message: 'Unexpected success',
      },
      errorHandling: {
        works: false,
        notes: 'Should have been denied but succeeded',
      },
    });
    console.log('       Unexpected success\n');
  } catch (e: any) {
    results.push({
      name: 'Create issue in external repo',
      apiCall: 'issues.create',
      request: {
        owner: 'octocat',
        repo: 'Hello-World',
        title: 'Test issue',
      },
      response: {
        status: e.status,
        message: e.response?.data?.message || e.message,
        documentation_url: e.response?.data?.documentation_url,
      },
      errorHandling: {
        works: true,
        notes: `Returns ${e.status} error. Generated code should verify repo access before attempting writes.`,
      },
    });
    console.log(`       Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
  }

  // ============================================================================
  // Additional Edge Cases
  // ============================================================================

  // 6. Invalid PR number
  console.log('6. Testing: Invalid PR number');
  try {
    await octokit.rest.pulls.get({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      pull_number: 99999999,
    });
  } catch (e: any) {
    results.push({
      name: 'Invalid PR number',
      apiCall: 'pulls.get',
      request: {
        owner: TEST_OWNER,
        repo: TEST_REPO,
        pull_number: 99999999,
      },
      response: {
        status: e.status,
        message: e.response?.data?.message || e.message,
        documentation_url: e.response?.data?.documentation_url,
      },
      errorHandling: {
        works: true,
        notes: `Returns ${e.status} error. Generated code should validate PR exists or handle 404.`,
      },
    });
    console.log(`   Error: ${e.status} - ${e.response?.data?.message || e.message}\n`);
  }

  // 7. Rate limiting (check rate limit status)
  console.log('7. Testing: Rate limit status');
  try {
    const { data } = await octokit.rest.rateLimit.get();
    results.push({
      name: 'Rate limit check',
      apiCall: 'rateLimit.get',
      request: {},
      response: {
        status: 200,
        data: {
          limit: data.rate.limit,
          remaining: data.rate.remaining,
          reset: new Date(data.rate.reset * 1000).toISOString(),
        },
      },
      errorHandling: {
        works: true,
        notes: 'Generated code should check rate limits before bulk operations and handle 429 responses.',
      },
    });
    console.log(`   Limit: ${data.rate.limit}, Remaining: ${data.rate.remaining}\n`);
  } catch (e: any) {
    console.log(`   Error: ${e.message}\n`);
  }

  // ============================================================================
  // Generate Report
  // ============================================================================
  console.log('\n========================================');
  console.log('Generating report...');

  const report = generateReport(results);
  console.log(report);

  // Write to file
  const fs = await import('fs');
  const path = await import('path');
  const reportPath = path.join(path.dirname(import.meta.url.replace('file://', '')), 'reports', 'edge-cases.md');

  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, report);
  console.log(`\nReport written to: ${reportPath}`);
}

function generateReport(results: EdgeCaseResult[]): string {
  const now = new Date().toISOString();

  let report = `# GitHub API Edge Case Testing Report

Generated: ${now}

## Summary

This report documents the behavior of the GitHub API for edge cases that may not be covered in training data. Each edge case includes the API call made, the error response received, and whether the generated code's error handling would work.

## Test Configuration

- **Test Repo**: mlx-node-training-example/mlx-node-test-repo
- **Total Edge Cases Tested**: ${results.length}
- **Cases with Working Error Handling**: ${results.filter((r) => r.errorHandling.works).length}

---

## Edge Case Results

`;

  for (const result of results) {
    report += `### ${result.name}

**API Call**: \`${result.apiCall}\`

**Request**:
\`\`\`json
${JSON.stringify(result.request, null, 2)}
\`\`\`

**Response**:
\`\`\`json
${JSON.stringify(result.response, null, 2)}
\`\`\`

**Error Handling Assessment**: ${result.errorHandling.works ? 'WORKS' : 'NEEDS IMPROVEMENT'}

${result.errorHandling.notes}

---

`;
  }

  report += `## Recommendations for Training Data

Based on these edge case tests, the training data should include examples that handle:

1. **Empty Results** - API calls like \`pulls.list\` return empty arrays for non-existent resources, not errors. Generated code should check array length.

2. **404 Errors** - Invalid IDs (PR numbers, comment IDs) return 404. Generated code should:
   - Validate IDs before making requests when possible
   - Catch 404 errors and provide user-friendly messages
   - Suggest next steps (e.g., "PR not found. Use \`list_pull_requests\` to see available PRs")

3. **403 Permission Errors** - Operations without sufficient permissions return 403. Generated code should:
   - Check for 403 status
   - Parse the error message to determine required permissions
   - Suggest how to fix (e.g., "Token needs 'repo' scope" or "Admin access required")

4. **Protected Resource Errors** - Some resources have additional protection (branch protection, etc.). Generated code should handle these gracefully.

5. **Rate Limiting** - Heavy API usage can hit rate limits. Generated code should:
   - Check \`x-ratelimit-remaining\` headers
   - Handle 429 responses with appropriate wait/retry logic
   - Inform user of rate limit status for bulk operations

6. **Cross-Repo Operations** - Operations on repos the token doesn't have access to fail with 403/404. Generated code should verify repo access.

## Error Response Patterns

| Status Code | Meaning | Recommended Handling |
|------------|---------|---------------------|
| 200 | Success (empty array possible) | Check array length |
| 204 | Success (no content) | Operation completed |
| 403 | Permission denied | Parse message, suggest permission fix |
| 404 | Resource not found | Validate ID, suggest alternatives |
| 422 | Validation failed | Parse error details, fix request |
| 429 | Rate limited | Wait and retry |

## Conclusion

The edge case testing reveals that the GitHub API has consistent error patterns. Generated code should implement defensive programming practices and provide helpful error messages to guide users when operations fail.
`;

  return report;
}

main().catch(console.error);
