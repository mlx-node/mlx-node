/**
 * GitHub Tool Training Dataset
 *
 * Contains 10 scenario types for training a model to interact with GitHub via Octokit:
 * 1. Get inline comments - Fetch review comments from a PR
 * 2. Get comments - Fetch general PR/issue comments
 * 3. Reply to inline comments - Add reply to existing review comment
 * 4. Resolve inline comments - Mark review thread as resolved
 * 5. Batch resolve comments - Resolve multiple review threads at once
 * 6. Get CI outputs - Download workflow logs, return metadata
 * 7. Change PR title - Update PR title
 * 8. Get file diff - Get diff for specific files in PR
 * 9. Check workflow status - Get status of GitHub Actions workflows
 * 10. Trigger workflows - Dispatch workflow runs
 *
 * Each scenario has multiple variations with different phrasings.
 */

import { GET_PR_PROMPTS } from '../sft/prompts';

export type ScenarioCategory = 'read' | 'write' | 'workflow';

export interface GitHubScenario {
  id: string;
  category: ScenarioCategory;
  description: string;
  prompt: string;
  expectedApiCall: string;
  expectedApiMethod: string;
  expectedReturnShape: string;
}

// ============================================================================
// Random ID Generators (prevents model from memorizing ID-to-scenario mappings)
// ============================================================================

/**
 * Generate a random comment ID (5-6 digits)
 */
function generateRandomCommentId(): number {
  return Math.floor(Math.random() * 900000) + 100000; // 100000-999999
}

/**
 * Generate a random GraphQL thread ID
 */
function generateRandomThreadId(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let suffix = '';
  for (let i = 0; i < 12; i++) {
    suffix += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return `PRRT_kwDO${suffix}`;
}

/**
 * Generate multiple random thread IDs for batch operations
 */
function generateRandomThreadIds(count: number): string[] {
  const ids: string[] = [];
  for (let i = 0; i < count; i++) {
    ids.push(generateRandomThreadId());
  }
  return ids;
}

// ============================================================================
// Placeholder Values for Realistic Prompts
// ============================================================================

const REPLY_MESSAGES = [
  'Fixed in the latest commit',
  'Good catch, will update this',
  'Thanks for the review!',
  'Agreed, refactoring now',
  'Done, please take another look',
  'Sounds good, updating now',
];
const PR_TITLES = [
  'feat: add user authentication',
  'fix: resolve race condition in data loader',
  'refactor: simplify error handling',
  'docs: update API documentation',
  'chore: upgrade dependencies',
];
// Use filenames (not display names) - GitHub API requires workflow filename or ID
const WORKFLOW_NAMES = ['ci.yml', 'test.yml', 'lint.yml', 'deploy.yml', 'build.yml'];
const FILENAMES = ['src/index.ts', 'src/utils.ts', 'lib/api.ts', 'tests/main.test.ts'];

/**
 * Fill placeholders in a prompt with realistic values
 */
function fillPlaceholders(prompt: string, index: number): string {
  return prompt
    .replace('{commentId}', String(generateRandomCommentId()))
    .replace('{threadId}', generateRandomThreadId())
    .replace('{threadIds}', generateRandomThreadIds(3 + Math.floor(Math.random() * 3)).join(' '))
    .replace('{message}', REPLY_MESSAGES[index % REPLY_MESSAGES.length])
    .replace('{newTitle}', PR_TITLES[index % PR_TITLES.length])
    .replace('{workflowName}', WORKFLOW_NAMES[index % WORKFLOW_NAMES.length])
    .replace('{filename}', FILENAMES[index % FILENAMES.length]);
}

// ============================================================================
// Scenario Templates
// ============================================================================

/**
 * Template for generating scenario variations
 */
interface ScenarioTemplate {
  id: string;
  category: ScenarioCategory;
  description: string;
  prompts: string[];
  expectedApiCall: string;
  expectedApiMethod: string;
  expectedReturnShape: string;
}

const SCENARIO_TEMPLATES: ScenarioTemplate[] = [
  {
    id: 'get-pr',
    category: 'read',
    description: 'Get pull requests for the current branch',
    prompts: GET_PR_PROMPTS,
    expectedApiCall: 'octokit.rest.pulls.list',
    expectedApiMethod: 'GET /repos/{owner}/{repo}/pulls',
    expectedReturnShape: 'PullRequest[]',
  },
  // 1. Get inline comments (review comments)
  {
    id: 'get-inline-comments',
    category: 'read',
    description: 'Get inline review comments from a PR',
    prompts: [
      // Direct requests
      'Get all inline comments for the current PR',
      'Fetch review comments from the current pull request',
      'List all code review comments on this PR',
      'Show me the inline comments on the current PR',
      // Questions
      'What inline comments are there on the current pull request?',
      'Are there any review comments on my PR?',
      'Can you get the code comments from my pull request?',
      // Contextual
      "Get the review comments for this branch's PR",
      'Fetch all code comments from the current PR',
      'I need to see the feedback on my code changes',
      'Check what reviewers said about my PR',
      // Technical
      'Fetch PR review comments using the GitHub API',
      'Get the listReviewComments data for this PR',
      // Casual
      "What's been commented on my PR?",
      'Show PR code feedback',
    ],
    expectedApiCall: 'octokit.rest.pulls.listReviewComments',
    expectedApiMethod: 'GET /repos/{owner}/{repo}/pulls/{pull_number}/comments',
    expectedReturnShape: '{ comments: ReviewComment[] }',
  },

  // 2. Get comments (general PR/issue comments)
  {
    id: 'get-comments',
    category: 'read',
    description: 'Get general comments from a PR or issue',
    prompts: [
      // Direct requests
      'Get all comments from the current PR',
      'Fetch the conversation comments on this pull request',
      'List all discussion comments on the current PR',
      'Show me the PR comments',
      // Questions
      'What comments are there on this pull request?',
      'What did people say on my PR?',
      'Are there any comments on this pull request?',
      // Contextual
      'Get the discussion thread for the current PR',
      'Show the PR conversation history',
      'I want to see the discussion on my PR',
      // Technical
      'Fetch issue comments for the PR',
      'Get the listComments data for this PR',
      // Casual
      "What's the conversation on my PR?",
      'Show PR discussion',
    ],
    expectedApiCall: 'octokit.rest.issues.listComments',
    expectedApiMethod: 'GET /repos/{owner}/{repo}/issues/{issue_number}/comments',
    expectedReturnShape: '{ comments: IssueComment[] }',
  },

  // 3. Reply to inline comments
  {
    id: 'reply-inline-comment',
    category: 'write',
    description: 'Reply to an existing inline review comment',
    prompts: [
      // No quotes around {message} - avoids JSON escaping issues in model output
      'Reply to inline comment {commentId} with: {message}',
      'Add a reply to review comment {commentId} saying: {message}',
      'Respond to the inline comment {commentId} with: {message}',
      'Post a reply to comment {commentId}: {message}',
      'Answer inline comment {commentId}: {message}',
      // More natural phrasing
      'Reply to comment {commentId} with: {message}',
      'On comment {commentId}, say: {message}',
      'Comment {commentId} needs reply: {message}',
      // Technical
      'Create a reply to review comment {commentId}, body: {message}',
      'Use createReplyForReviewComment for comment {commentId}: {message}',
      // Action-oriented
      'Send to inline comment {commentId}: {message}',
      'Respond to {commentId}: {message}',
    ],
    expectedApiCall: 'octokit.rest.pulls.createReplyForReviewComment',
    expectedApiMethod: 'POST /repos/{owner}/{repo}/pulls/{pull_number}/comments/{comment_id}/replies',
    expectedReturnShape: '{ reply: ReviewComment }',
  },

  // 4. Resolve inline comments
  {
    id: 'resolve-inline-comment',
    category: 'write',
    description: 'Resolve/dismiss an inline review thread',
    prompts: [
      // Direct requests
      'Resolve inline comment thread {threadId}',
      'Mark review thread {threadId} as resolved',
      'Dismiss the review comment thread {threadId}',
      'Close the inline discussion for thread {threadId}',
      'Resolve all comments in thread {threadId}',
      // Action-oriented
      'Set thread {threadId} to resolved',
      'Complete the review thread {threadId}',
      'Mark {threadId} as done',
      // Technical
      'Use GraphQL to resolve thread {threadId}',
      'Run resolveReviewThread mutation for {threadId}',
      // Casual
      'Close thread {threadId}',
      'Finish discussion {threadId}',
    ],
    expectedApiCall: 'octokit.graphql',
    expectedApiMethod: 'POST /graphql (resolveReviewThread mutation)',
    expectedReturnShape: '{ resolved: boolean, threadId: string }',
  },

  // 5. Batch resolve inline comments
  {
    id: 'batch-resolve-comments',
    category: 'write',
    description: 'Resolve multiple review threads at once',
    prompts: [
      'Resolve these threads: {threadIds}',
      'Mark all these threads as resolved: {threadIds}',
      'Batch resolve threads {threadIds}',
      'Resolve threads {threadIds}',
      'Close all these review threads: {threadIds}',
      'Mark threads {threadIds} as done',
      'Resolve multiple threads: {threadIds}',
      'Bulk resolve: {threadIds}',
      'Finish these discussions: {threadIds}',
      'Complete threads {threadIds}',
      'Dismiss threads {threadIds}',
      'Mark resolved: {threadIds}',
    ],
    expectedApiCall: 'octokit.graphql',
    expectedApiMethod: 'POST /graphql (resolveReviewThread mutation, multiple calls)',
    expectedReturnShape: '{ resolved: Array<{ threadId: string, success: boolean }> }',
  },

  // 6. Get CI outputs (downloads workflow run logs)
  {
    id: 'get-ci-outputs',
    category: 'workflow',
    description: 'Download GitHub Actions workflow logs',
    prompts: [
      // Direct requests
      'Get the CI output for the current PR',
      'Download the workflow logs for this PR',
      'Fetch the GitHub Actions logs for the latest run',
      'Get the CI build logs',
      'Download the test output from the latest workflow run',
      // With workflow name (no quotes)
      'Show me the GitHub Actions log for workflow {workflowName}',
      'Get logs from the {workflowName} workflow',
      'Download {workflowName} workflow output',
      // Action-oriented
      'Retrieve workflow run logs',
      'Get the CI output and return it',
      'Pull the latest workflow logs',
      // Technical
      'Download workflow run logs using downloadWorkflowRunLogs',
      'Fetch the Actions run artifacts',
      // Casual
      'What did CI output?',
      'Show CI logs',
    ],
    expectedApiCall: 'octokit.rest.actions.downloadWorkflowRunLogs',
    expectedApiMethod: 'GET /repos/{owner}/{repo}/actions/runs/{run_id}/logs',
    expectedReturnShape: '{ runId: number, workflow: string, conclusion: string, logsSize: number }',
  },

  // 7. Change PR title
  {
    id: 'change-pr-title',
    category: 'write',
    description: 'Update the title of a pull request',
    prompts: [
      // No quotes around {newTitle} - avoids JSON escaping issues
      'Change the PR title to: {newTitle}',
      'Update the pull request title to: {newTitle}',
      'Rename this PR to: {newTitle}',
      'Set the PR title to: {newTitle}',
      'Change the current PR title to: {newTitle}',
      // Imperative
      'Update PR title to: {newTitle}',
      'New title for PR: {newTitle}',
      'Make the PR title: {newTitle}',
      // Contextual
      'The PR should be titled: {newTitle}',
      'I want to rename the PR to: {newTitle}',
      // Technical
      'Use pulls.update to set title: {newTitle}',
      'PATCH the PR with title: {newTitle}',
    ],
    expectedApiCall: 'octokit.rest.pulls.update',
    expectedApiMethod: 'PATCH /repos/{owner}/{repo}/pulls/{pull_number}',
    expectedReturnShape: '{ updated: boolean, title: string, prNumber: number }',
  },

  // 8. Get file diff
  {
    id: 'get-file-diff',
    category: 'read',
    description: 'Get the diff for specific files in a PR',
    prompts: [
      // Direct requests
      'Get the diff for the current PR',
      'Show me the file changes in this PR',
      'What files changed in this pull request?',
      'Show the code changes for this PR',
      'Get the patch for the current PR',
      // With filename (no quotes)
      'Get the diff for file {filename} in the current PR',
      'Show changes to {filename}',
      'What changed in {filename}?',
      // Questions
      'Which files were modified in this PR?',
      'What are the additions and deletions in this PR?',
      // Technical
      'List PR files using pulls.listFiles',
      'Get the file patches for this pull request',
      // Casual
      'Show the diff',
      "What's changed?",
    ],
    expectedApiCall: 'octokit.rest.pulls.listFiles',
    expectedApiMethod: 'GET /repos/{owner}/{repo}/pulls/{pull_number}/files',
    expectedReturnShape: '{ files: Array<{ filename: string, additions: number, deletions: number, patch: string }> }',
  },

  // 9. Check workflow status
  {
    id: 'check-workflow-status',
    category: 'workflow',
    description: 'Get the status of GitHub Actions workflows',
    prompts: [
      // Direct requests
      'Check the CI status for the current PR',
      'What is the workflow status for this branch?',
      'Get the CI status for the current branch',
      'Show me the workflow run status',
      // Questions
      'Are the GitHub Actions passing?',
      'Is CI green?',
      'Did the tests pass?',
      'Check if the tests are passing',
      // Contextual
      "What's the status of my workflows?",
      'How is CI doing on this branch?',
      // Technical
      'List workflow runs for this branch',
      'Get listWorkflowRunsForRepo status',
      // Casual
      'CI status?',
      'Tests passing?',
    ],
    expectedApiCall: 'octokit.rest.actions.listWorkflowRunsForRepo',
    expectedApiMethod: 'GET /repos/{owner}/{repo}/actions/runs',
    expectedReturnShape: '{ runs: Array<{ name: string, status: string, conclusion: string, url: string }> }',
  },

  // 10. Trigger workflows
  {
    id: 'trigger-workflow',
    category: 'workflow',
    description: 'Trigger a GitHub Actions workflow',
    prompts: [
      // No quotes - avoids JSON escaping issues
      'Trigger the {workflowName} workflow',
      'Run the GitHub Action {workflowName}',
      'Dispatch the workflow {workflowName}',
      'Start the CI workflow {workflowName}',
      'Execute the {workflowName} workflow on this branch',
      // Imperative
      'Trigger a workflow run for {workflowName}',
      'Kick off the {workflowName} workflow',
      'Start {workflowName} workflow',
      // Technical
      'Create workflow dispatch for {workflowName}',
      'Use createWorkflowDispatch for {workflowName}',
      // Casual
      'Run {workflowName} CI job',
      'Fire off {workflowName} workflow',
    ],
    expectedApiCall: 'octokit.rest.actions.createWorkflowDispatch',
    expectedApiMethod: 'POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches',
    expectedReturnShape: '{ triggered: boolean, workflow: string, ref: string }',
  },
];

// ============================================================================
// Scenario Generation
// ============================================================================

/**
 * Generate a single scenario from a template
 */
function generateScenario(template: ScenarioTemplate, promptIndex: number): GitHubScenario {
  const rawPrompt = template.prompts[promptIndex % template.prompts.length];
  // Fill any placeholders with realistic values
  const prompt = fillPlaceholders(rawPrompt, promptIndex);

  return {
    id: `${template.id}-${promptIndex}`,
    category: template.category,
    description: template.description,
    prompt,
    expectedApiCall: template.expectedApiCall,
    expectedApiMethod: template.expectedApiMethod,
    expectedReturnShape: template.expectedReturnShape,
  };
}

/**
 * Fisher-Yates shuffle helper
 */
function shuffleArray<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Interleave arrays: [a1, a2], [b1, b2], [c1] → [a1, b1, c1, a2, b2]
 */
function interleaveArrays<T>(...arrays: T[][]): T[] {
  const result: T[] = [];
  const maxLen = Math.max(...arrays.map((a) => a.length));
  for (let i = 0; i < maxLen; i++) {
    for (const arr of arrays) {
      if (i < arr.length) {
        result.push(arr[i]);
      }
    }
  }
  return result;
}

/**
 * Generate a curriculum-ordered dataset with interleaved categories
 * Distribution: 25% read, 45% write, 30% workflow
 * Interleaving: read, write, workflow, read, write, workflow, ...
 */
export function generateCurriculumDataset(numExamples: number): GitHubScenario[] {
  // Calculate proportions - focus more on harder write operations
  const readCount = Math.floor(numExamples * 0.25); // 25% (easiest)
  const writeCount = Math.floor(numExamples * 0.45); // 45% (hardest, needs more practice)
  const workflowCount = numExamples - readCount - writeCount; // 30%

  // Separate templates by category
  const readTemplates = SCENARIO_TEMPLATES.filter((t) => t.category === 'read');
  const writeTemplates = SCENARIO_TEMPLATES.filter((t) => t.category === 'write');
  const workflowTemplates = SCENARIO_TEMPLATES.filter((t) => t.category === 'workflow');

  // Generate scenarios for each category
  const readScenarios: GitHubScenario[] = [];
  for (let i = 0; i < readCount; i++) {
    const template = readTemplates[i % readTemplates.length];
    const promptIndex = Math.floor(i / readTemplates.length);
    readScenarios.push(generateScenario(template, promptIndex));
  }

  const writeScenarios: GitHubScenario[] = [];
  for (let i = 0; i < writeCount; i++) {
    const template = writeTemplates[i % writeTemplates.length];
    const promptIndex = Math.floor(i / writeTemplates.length);
    writeScenarios.push(generateScenario(template, promptIndex));
  }

  const workflowScenarios: GitHubScenario[] = [];
  for (let i = 0; i < workflowCount; i++) {
    const template = workflowTemplates[i % workflowTemplates.length];
    const promptIndex = Math.floor(i / workflowTemplates.length);
    workflowScenarios.push(generateScenario(template, promptIndex));
  }

  // Shuffle within each category for variety
  shuffleArray(readScenarios);
  shuffleArray(writeScenarios);
  shuffleArray(workflowScenarios);

  // Interleave categories: read, write, workflow, read, write, workflow, ...
  return interleaveArrays(readScenarios, writeScenarios, workflowScenarios);
}

/**
 * Generate a shuffled dataset
 */
export function generateShuffledDataset(numExamples: number): GitHubScenario[] {
  const curriculum = generateCurriculumDataset(numExamples);

  // Fisher-Yates shuffle
  for (let i = curriculum.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [curriculum[i], curriculum[j]] = [curriculum[j], curriculum[i]];
  }

  return curriculum;
}

/**
 * Get all scenario templates
 */
export function getAllTemplates(): ScenarioTemplate[] {
  return [...SCENARIO_TEMPLATES];
}

/**
 * Count scenarios by category
 */
export function countByCategory(scenarios: GitHubScenario[]): Record<ScenarioCategory, number> {
  const counts: Record<ScenarioCategory, number> = {
    read: 0,
    write: 0,
    workflow: 0,
  };

  for (const scenario of scenarios) {
    counts[scenario.category]++;
  }

  return counts;
}

// Export templates
export const SCENARIO_TEMPLATE_LIST = SCENARIO_TEMPLATES;

// Export individual template getters
export const READ_TEMPLATES = SCENARIO_TEMPLATES.filter((t) => t.category === 'read');
export const WRITE_TEMPLATES = SCENARIO_TEMPLATES.filter((t) => t.category === 'write');
export const WORKFLOW_TEMPLATES = SCENARIO_TEMPLATES.filter((t) => t.category === 'workflow');

/**
 * Get all scenario IDs for selection UI
 */
export function getAllScenarioIds(): string[] {
  return SCENARIO_TEMPLATES.map((t) => t.id);
}

/**
 * Get scenario info for display in selector
 */
export interface ScenarioInfo {
  id: string;
  category: ScenarioCategory;
  description: string;
  apiMethod: string;
  numPromptVariations: number;
}

export function getAllScenarioInfo(): ScenarioInfo[] {
  return SCENARIO_TEMPLATES.map((t) => ({
    id: t.id,
    category: t.category,
    description: t.description,
    apiMethod: t.expectedApiCall.split('.').pop() || t.expectedApiCall,
    numPromptVariations: t.prompts.length,
  }));
}

/**
 * Generate a dataset focused on a single scenario type.
 * All examples will be variations of the same API call.
 * This makes training easier for small models.
 */
export function generateSingleScenarioDataset(scenarioId: string, numExamples: number): GitHubScenario[] {
  const template = SCENARIO_TEMPLATES.find((t) => t.id === scenarioId);
  if (!template) {
    throw new Error(`Unknown scenario ID: ${scenarioId}. Valid IDs: ${getAllScenarioIds().join(', ')}`);
  }

  const scenarios: GitHubScenario[] = [];
  for (let i = 0; i < numExamples; i++) {
    // Cycle through all prompt variations
    const promptIndex = i % template.prompts.length;
    scenarios.push(generateScenario(template, promptIndex));
  }

  // Shuffle for variety in training order
  return shuffleArray(scenarios);
}

/**
 * Interleave scenarios by their base ID for balanced training order
 */
function interleaveByScenarioId(scenarios: GitHubScenario[], ids: string[]): GitHubScenario[] {
  const byId = new Map<string, GitHubScenario[]>();
  for (const id of ids) {
    byId.set(
      id,
      scenarios.filter((s) => s.id.startsWith(id + '-')),
    );
  }

  const result: GitHubScenario[] = [];
  const maxLen = Math.max(...[...byId.values()].map((a) => a.length));

  for (let i = 0; i < maxLen; i++) {
    for (const id of ids) {
      const arr = byId.get(id)!;
      if (i < arr.length) result.push(arr[i]);
    }
  }
  return result;
}

/**
 * Generate a dataset combining multiple scenario types.
 * Examples are interleaved for balanced training.
 */
export function generateMultiScenarioDataset(scenarioIds: string[], totalExamples: number): GitHubScenario[] {
  if (scenarioIds.length === 0) {
    throw new Error('At least one scenario ID required');
  }

  const perScenario = Math.ceil(totalExamples / scenarioIds.length);
  const allScenarios: GitHubScenario[] = [];

  for (const id of scenarioIds) {
    allScenarios.push(...generateSingleScenarioDataset(id, perScenario));
  }

  // Interleave scenarios for balanced training
  return interleaveByScenarioId(allScenarios, scenarioIds).slice(0, totalExamples);
}
