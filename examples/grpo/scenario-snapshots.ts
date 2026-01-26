/**
 * Scenario Snapshots for Execution-Based Reward Validation
 *
 * Pre-generated expected API calls and responses for all 10 GitHub tool scenarios.
 * Used during training to:
 * 1. Assert model's API calls match expected patterns
 * 2. Return mocked responses without hitting real API
 * 3. Score correctness based on param matching
 */

export interface ExpectedApiCall {
  /** Method name: 'rest.pulls.list', 'rest.pulls.listReviewComments', 'graphql', etc */
  method: string;
  /** Expected params - values can be exact matches or patterns like '{owner}' */
  expectedParams: Record<string, unknown>;
  /** Mocked response to return */
  response: unknown;
  /** If true, match is optional (for calls that may or may not happen) */
  optional?: boolean;
}

export interface ScenarioSnapshot {
  /** Base scenario ID without variation suffix (e.g., 'get-inline-comments') */
  scenarioId: string;
  /** Category for grouping */
  category: 'read' | 'write' | 'workflow';
  /** Expected API calls in order */
  expectedCalls: ExpectedApiCall[];
  /** Expected return shape from the default export function */
  expectedReturnKeys: string[];
}

// Mock data used across scenarios
const MOCK_PR = {
  id: 123456789,
  node_id: 'PR_kwDOQxAyvc67QIjC',
  number: 123,
  title: 'Test PR for training',
  body: 'This PR adds new features for testing.',
  state: 'open',
  draft: false,
  merged: false,
  head: { ref: 'test-feature', sha: 'abc123def456' },
  base: { ref: 'main' },
  html_url: 'https://github.com/test-owner/test-repo/pull/123',
  user: { login: 'test-owner', id: 12345 },
  additions: 25,
  deletions: 10,
  changed_files: 3,
};

const MOCK_REVIEW_COMMENTS = [
  {
    id: 1001,
    body: 'Consider adding error handling here',
    path: 'src/index.ts',
    line: 42,
    user: { login: 'reviewer1', id: 99 },
    created_at: '2024-01-15T10:30:00Z',
    in_reply_to_id: null,
    diff_hunk: '@@ -40,5 +40,10 @@\n+  const result = await fetch(url);',
    commit_id: 'abc123def456',
  },
  {
    id: 1002,
    body: 'Nice refactoring!',
    path: 'src/utils.ts',
    line: 15,
    user: { login: 'reviewer2', id: 100 },
    created_at: '2024-01-15T11:00:00Z',
    in_reply_to_id: null,
    diff_hunk: '@@ -10,3 +10,8 @@\n+  return formatData(input);',
    commit_id: 'abc123def456',
  },
];

const MOCK_ISSUE_COMMENTS = [
  {
    id: 2001,
    body: 'LGTM! Ready to merge.',
    user: { login: 'approver', id: 101 },
    created_at: '2024-01-15T12:00:00Z',
  },
  {
    id: 2002,
    body: 'Can you add tests for the new function?',
    user: { login: 'reviewer1', id: 99 },
    created_at: '2024-01-15T09:00:00Z',
  },
];

const MOCK_FILES = [
  {
    sha: 'abc123def456789abcdef1234567890abcdef12',
    filename: 'src/index.ts',
    additions: 25,
    deletions: 10,
    changes: 35,
    status: 'modified',
    patch: '@@ -10,5 +10,20 @@\n-old code\n+new improved code',
  },
  {
    sha: 'def456789abcdef1234567890abcdef12345678',
    filename: 'src/new-feature.ts',
    additions: 50,
    deletions: 0,
    changes: 50,
    status: 'added',
    patch: '@@ -0,0 +1,50 @@\n+// New feature implementation',
  },
  {
    sha: '789abcdef1234567890abcdef1234567890abcd',
    filename: 'src/deprecated.ts',
    additions: 0,
    deletions: 30,
    changes: 30,
    status: 'removed',
    patch: '@@ -1,30 +0,0 @@\n-// Removed deprecated code',
  },
];

const MOCK_WORKFLOW_RUNS = [
  {
    id: 9001,
    name: 'CI',
    status: 'completed',
    conclusion: 'success',
    html_url: 'https://github.com/test-owner/test-repo/actions/runs/9001',
    head_branch: 'test-feature',
    head_sha: 'abc123def456789',
    event: 'pull_request',
    run_number: 42,
    created_at: '2024-01-15T10:00:00Z',
  },
  {
    id: 9000,
    name: 'Lint',
    status: 'completed',
    conclusion: 'failure',
    html_url: 'https://github.com/test-owner/test-repo/actions/runs/9000',
    head_branch: 'test-feature',
    head_sha: 'abc123def456789',
    event: 'pull_request',
    run_number: 41,
    created_at: '2024-01-15T09:30:00Z',
  },
];

/**
 * All scenario snapshots indexed by base scenario ID.
 * The ID in dataset is like 'get-inline-comments-0', but we match on base 'get-inline-comments'.
 */
export const SCENARIO_SNAPSHOTS: Record<string, ScenarioSnapshot> = {
  // ============================================================================
  // READ OPERATIONS (4 scenarios)
  // ============================================================================

  'get-pr': {
    scenarioId: 'get-pr',
    category: 'read',
    expectedCalls: [
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
      },
    ],
    expectedReturnKeys: ['length'],
  },

  'get-inline-comments': {
    scenarioId: 'get-inline-comments',
    category: 'read',
    expectedCalls: [
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
      },
      {
        method: 'rest.pulls.listReviewComments',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          pull_number: 123,
        },
        response: { data: MOCK_REVIEW_COMMENTS },
      },
    ],
    expectedReturnKeys: ['comments', 'prNumber'],
  },

  'get-comments': {
    scenarioId: 'get-comments',
    category: 'read',
    expectedCalls: [
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
      },
      {
        method: 'rest.issues.listComments',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          issue_number: 123,
        },
        response: { data: MOCK_ISSUE_COMMENTS },
      },
    ],
    expectedReturnKeys: ['comments', 'prNumber'],
  },

  'get-file-diff': {
    scenarioId: 'get-file-diff',
    category: 'read',
    expectedCalls: [
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
      },
      {
        method: 'rest.pulls.listFiles',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          pull_number: 123,
        },
        response: { data: MOCK_FILES },
      },
    ],
    expectedReturnKeys: ['files', 'prNumber'],
  },

  // ============================================================================
  // WRITE OPERATIONS (3 scenarios)
  // ============================================================================

  'reply-inline-comment': {
    scenarioId: 'reply-inline-comment',
    category: 'write',
    expectedCalls: [
      // Note: Some implementations may first list the PR
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
        optional: true,
      },
      {
        method: 'rest.pulls.createReplyForReviewComment',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          pull_number: '{any}', // Could be 123 or from param
          comment_id: '{any}', // From prompt
          body: '{any}', // From prompt
        },
        response: {
          data: {
            id: 1003,
            body: 'Reply created',
            user: { login: 'assistant', id: 200 },
            created_at: new Date().toISOString(),
          },
        },
      },
    ],
    expectedReturnKeys: ['reply'],
  },

  'resolve-inline-comment': {
    scenarioId: 'resolve-inline-comment',
    category: 'write',
    expectedCalls: [
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
        optional: true,
      },
      {
        method: 'graphql',
        expectedParams: {
          query: '{contains:resolveReviewThread}',
          threadId: '{any}',
        },
        response: {
          resolveReviewThread: {
            thread: {
              id: 'thread_123',
              isResolved: true,
            },
          },
        },
      },
    ],
    expectedReturnKeys: ['resolved', 'threadId'],
  },

  'batch-resolve-comments': {
    scenarioId: 'batch-resolve-comments',
    category: 'write',
    expectedCalls: [
      {
        method: 'graphql',
        expectedParams: {
          query: '{contains:resolveReviewThread}',
        },
        response: {
          resolveReviewThread: {
            thread: {
              id: 'thread_123',
              isResolved: true,
            },
          },
        },
        // This call may happen multiple times
      },
    ],
    expectedReturnKeys: ['resolved'],
  },

  'change-pr-title': {
    scenarioId: 'change-pr-title',
    category: 'write',
    expectedCalls: [
      {
        method: 'rest.pulls.list',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          state: 'open',
          head: '{owner}:{branch}',
        },
        response: { data: [MOCK_PR] },
      },
      {
        method: 'rest.pulls.update',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          pull_number: 123,
          title: '{any}', // New title from prompt
        },
        response: {
          data: {
            ...MOCK_PR,
            title: 'Updated PR Title',
          },
        },
      },
    ],
    expectedReturnKeys: ['updated', 'title', 'prNumber'],
  },

  // ============================================================================
  // WORKFLOW OPERATIONS (3 scenarios)
  // ============================================================================

  'check-workflow-status': {
    scenarioId: 'check-workflow-status',
    category: 'workflow',
    expectedCalls: [
      {
        method: 'rest.actions.listWorkflowRunsForRepo',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          branch: '{branch}',
        },
        response: { data: { workflow_runs: MOCK_WORKFLOW_RUNS } },
      },
    ],
    expectedReturnKeys: ['runs'],
  },

  'get-ci-outputs': {
    scenarioId: 'get-ci-outputs',
    category: 'workflow',
    expectedCalls: [
      {
        method: 'rest.actions.listWorkflowRunsForRepo',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          branch: '{branch}',
        },
        response: { data: { workflow_runs: MOCK_WORKFLOW_RUNS } },
      },
      {
        method: 'rest.actions.downloadWorkflowRunLogs',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          run_id: '{any}', // Usually 9001 (first run)
        },
        response: {
          // Mock response structure matching Octokit's actual return type
          // Real API returns binary zip data; we use base64 for JSON serialization
          status: 200,
          url: 'https://api.github.com/repos/{owner}/{repo}/actions/runs/9001/logs',
          headers: {
            'content-type': 'application/zip',
            'content-length': '1024',
          },
          // Base64-encoded mock zip content (models should decode if needed)
          data: Buffer.from(
            'Mock CI log content:\n\nJob: test\nStatus: success\nDuration: 45s\n\n=== Output ===\nAll tests passed.',
          ).toString('base64'),
        },
      },
    ],
    expectedReturnKeys: ['runId', 'workflow', 'conclusion', 'logsSize'],
  },

  'trigger-workflow': {
    scenarioId: 'trigger-workflow',
    category: 'workflow',
    expectedCalls: [
      {
        method: 'rest.actions.createWorkflowDispatch',
        expectedParams: {
          owner: '{owner}',
          repo: '{repo}',
          workflow_id: '{any}', // From prompt
          ref: '{branch}',
        },
        response: { status: 204, data: null },
      },
    ],
    expectedReturnKeys: ['triggered', 'workflow', 'ref'],
  },
};

/**
 * Get snapshot for a scenario ID.
 * Handles both base IDs (e.g., 'get-inline-comments') and
 * variation IDs (e.g., 'get-inline-comments-0').
 */
export function getScenarioSnapshot(scenarioId: string): ScenarioSnapshot | null {
  // Try exact match first
  if (SCENARIO_SNAPSHOTS[scenarioId]) {
    return SCENARIO_SNAPSHOTS[scenarioId];
  }

  // Try base ID (remove trailing -N suffix)
  const baseId = scenarioId.replace(/-\d+$/, '');
  if (SCENARIO_SNAPSHOTS[baseId]) {
    return SCENARIO_SNAPSHOTS[baseId];
  }

  return null;
}

/**
 * Get all snapshot IDs
 */
export function getAllSnapshotIds(): string[] {
  return Object.keys(SCENARIO_SNAPSHOTS);
}

/**
 * Serialize snapshot for passing to subprocess via env var
 */
export function serializeSnapshot(snapshot: ScenarioSnapshot): string {
  return JSON.stringify(snapshot);
}

/**
 * Deserialize snapshot from env var
 */
export function deserializeSnapshot(json: string): ScenarioSnapshot {
  return JSON.parse(json) as ScenarioSnapshot;
}
