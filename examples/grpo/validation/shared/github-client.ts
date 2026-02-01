/**
 * GitHub Client for Validation Scripts
 *
 * Wrapper around Octokit for testing against the real GitHub API.
 * Uses the test repo: mlx-node-training-example/mlx-node-test-repo
 */

import { Octokit } from 'octokit';

export const TEST_OWNER = 'mlx-node-training-example';
export const TEST_REPO = 'mlx-node-test-repo';

export interface GitHubClientConfig {
  token?: string;
  owner?: string;
  repo?: string;
}

export class GitHubClient {
  private octokit: Octokit;
  readonly owner: string;
  readonly repo: string;

  constructor(config: GitHubClientConfig = {}) {
    const token = config.token || process.env.GITHUB_TOKEN;
    if (!token) {
      throw new Error('GITHUB_TOKEN is required. Set it as an environment variable or pass in config.');
    }

    this.octokit = new Octokit({ auth: token });
    this.owner = config.owner || TEST_OWNER;
    this.repo = config.repo || TEST_REPO;
  }

  // ============================================================================
  // Repository Info
  // ============================================================================

  async getRepo() {
    const { data } = await this.octokit.rest.repos.get({
      owner: this.owner,
      repo: this.repo,
    });
    return data;
  }

  async getDefaultBranch(): Promise<string> {
    const repo = await this.getRepo();
    return repo.default_branch;
  }

  // ============================================================================
  // Branch Operations
  // ============================================================================

  async getBranch(branch: string) {
    try {
      const { data } = await this.octokit.rest.repos.getBranch({
        owner: this.owner,
        repo: this.repo,
        branch,
      });
      return data;
    } catch (e: any) {
      if (e.status === 404) return null;
      throw e;
    }
  }

  async createBranch(name: string, fromBranch?: string): Promise<string> {
    const baseBranch = fromBranch || (await this.getDefaultBranch());
    const { data: ref } = await this.octokit.rest.git.getRef({
      owner: this.owner,
      repo: this.repo,
      ref: `heads/${baseBranch}`,
    });

    await this.octokit.rest.git.createRef({
      owner: this.owner,
      repo: this.repo,
      ref: `refs/heads/${name}`,
      sha: ref.object.sha,
    });

    return name;
  }

  async deleteBranch(name: string): Promise<void> {
    try {
      await this.octokit.rest.git.deleteRef({
        owner: this.owner,
        repo: this.repo,
        ref: `heads/${name}`,
      });
    } catch (e: any) {
      if (e.status !== 404) throw e;
    }
  }

  // ============================================================================
  // File Operations
  // ============================================================================

  async createOrUpdateFile(path: string, content: string, message: string, branch: string): Promise<{ sha: string }> {
    // Check if file exists
    let existingSha: string | undefined;
    try {
      const { data } = await this.octokit.rest.repos.getContent({
        owner: this.owner,
        repo: this.repo,
        path,
        ref: branch,
      });
      if (!Array.isArray(data) && data.type === 'file') {
        existingSha = data.sha;
      }
    } catch (e: any) {
      if (e.status !== 404) throw e;
    }

    const { data } = await this.octokit.rest.repos.createOrUpdateFileContents({
      owner: this.owner,
      repo: this.repo,
      path,
      message,
      content: Buffer.from(content).toString('base64'),
      branch,
      sha: existingSha,
    });

    return { sha: data.content?.sha || '' };
  }

  // ============================================================================
  // Pull Request Operations
  // ============================================================================

  async listPRs(options: { state?: 'open' | 'closed' | 'all'; head?: string } = {}) {
    const { data } = await this.octokit.rest.pulls.list({
      owner: this.owner,
      repo: this.repo,
      state: options.state || 'open',
      head: options.head,
    });
    return data;
  }

  async createPR(title: string, head: string, base?: string, body?: string) {
    const baseBranch = base || (await this.getDefaultBranch());
    const { data } = await this.octokit.rest.pulls.create({
      owner: this.owner,
      repo: this.repo,
      title,
      head,
      base: baseBranch,
      body: body || '',
    });
    return data;
  }

  async getPR(number: number) {
    const { data } = await this.octokit.rest.pulls.get({
      owner: this.owner,
      repo: this.repo,
      pull_number: number,
    });
    return data;
  }

  async updatePR(number: number, updates: { title?: string; body?: string; state?: 'open' | 'closed' }) {
    const { data } = await this.octokit.rest.pulls.update({
      owner: this.owner,
      repo: this.repo,
      pull_number: number,
      ...updates,
    });
    return data;
  }

  async listPRFiles(number: number) {
    const { data } = await this.octokit.rest.pulls.listFiles({
      owner: this.owner,
      repo: this.repo,
      pull_number: number,
    });
    return data;
  }

  // ============================================================================
  // Review Comments (Inline Code Comments)
  // ============================================================================

  async listReviewComments(prNumber: number) {
    const { data } = await this.octokit.rest.pulls.listReviewComments({
      owner: this.owner,
      repo: this.repo,
      pull_number: prNumber,
    });
    return data;
  }

  async createReviewComment(prNumber: number, body: string, commitId: string, path: string, line: number) {
    const { data } = await this.octokit.rest.pulls.createReviewComment({
      owner: this.owner,
      repo: this.repo,
      pull_number: prNumber,
      body,
      commit_id: commitId,
      path,
      line,
      side: 'RIGHT',
    });
    return data;
  }

  async replyToReviewComment(prNumber: number, commentId: number, body: string) {
    const { data } = await this.octokit.rest.pulls.createReplyForReviewComment({
      owner: this.owner,
      repo: this.repo,
      pull_number: prNumber,
      comment_id: commentId,
      body,
    });
    return data;
  }

  // ============================================================================
  // Issue Comments (General PR/Issue Comments)
  // ============================================================================

  async listIssueComments(issueNumber: number) {
    const { data } = await this.octokit.rest.issues.listComments({
      owner: this.owner,
      repo: this.repo,
      issue_number: issueNumber,
    });
    return data;
  }

  async createIssueComment(issueNumber: number, body: string) {
    const { data } = await this.octokit.rest.issues.createComment({
      owner: this.owner,
      repo: this.repo,
      issue_number: issueNumber,
      body,
    });
    return data;
  }

  // ============================================================================
  // Workflow Operations
  // ============================================================================

  async listWorkflows() {
    const { data } = await this.octokit.rest.actions.listRepoWorkflows({
      owner: this.owner,
      repo: this.repo,
    });
    return data.workflows;
  }

  async listWorkflowRuns(branch?: string) {
    const { data } = await this.octokit.rest.actions.listWorkflowRunsForRepo({
      owner: this.owner,
      repo: this.repo,
      branch,
      per_page: 10,
    });
    return data.workflow_runs;
  }

  async triggerWorkflow(workflowId: string | number, ref: string, inputs?: Record<string, string>) {
    await this.octokit.rest.actions.createWorkflowDispatch({
      owner: this.owner,
      repo: this.repo,
      workflow_id: workflowId,
      ref,
      inputs,
    });
    return { triggered: true };
  }

  async downloadWorkflowLogs(runId: number) {
    const { data } = await this.octokit.rest.actions.downloadWorkflowRunLogs({
      owner: this.owner,
      repo: this.repo,
      run_id: runId,
    });
    return data;
  }

  // ============================================================================
  // GraphQL Operations
  // ============================================================================

  async resolveReviewThread(threadId: string) {
    const mutation = `
      mutation($threadId: ID!) {
        resolveReviewThread(input: { threadId: $threadId }) {
          thread {
            id
            isResolved
          }
        }
      }
    `;

    const result = await this.octokit.graphql<{
      resolveReviewThread: {
        thread: { id: string; isResolved: boolean };
      };
    }>(mutation, { threadId });

    return result.resolveReviewThread.thread;
  }

  async getReviewThreadId(_commentId: number): Promise<string | null> {
    // Get the node ID for a review comment to use in GraphQL
    const _query = `
      query($owner: String!, $repo: String!, $number: Int!) {
        repository(owner: $owner, name: $repo) {
          pullRequest(number: $number) {
            reviewThreads(first: 100) {
              nodes {
                id
                comments(first: 1) {
                  nodes {
                    databaseId
                  }
                }
              }
            }
          }
        }
      }
    `;

    // This is a simplified approach - in practice you'd need the PR number
    // For now, return null and let the caller handle it
    return null;
  }

  // ============================================================================
  // Raw Octokit Access
  // ============================================================================

  get raw(): Octokit {
    return this.octokit;
  }
}

/**
 * Create a GitHub client with the test repo configuration
 */
export function createTestClient(token?: string): GitHubClient {
  return new GitHubClient({
    token,
    owner: TEST_OWNER,
    repo: TEST_REPO,
  });
}
