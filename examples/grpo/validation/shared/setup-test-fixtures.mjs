#!/usr/bin/env node
/**
 * Setup Test Fixtures for GitHub Validation
 *
 * Creates the necessary resources in the test repo for validation testing:
 * 1. Test branch (from main if not exists)
 * 2. Simple file change on that branch
 * 3. PR from test branch to main
 * 4. Review comment (inline code comment) on the PR
 * 5. Issue comment (general discussion) on the PR
 * 6. Workflow file (.github/workflows/test.yml) if not exists
 *
 * Saves created resource IDs to test-fixtures.json
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Octokit } from 'octokit';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEST_OWNER = 'mlx-node-training-example';
const TEST_REPO = 'mlx-node-test-repo';
const TEST_BRANCH = 'test-fixture-branch';
const TEST_FILE_PATH = 'test-fixtures/sample.ts';
const WORKFLOW_FILE_PATH = '.github/workflows/test.yml';

const token = process.env.GITHUB_TOKEN;
if (!token) {
  console.error('GITHUB_TOKEN is required');
  process.exit(1);
}

const octokit = new Octokit({ auth: token });

async function getBranch(branch) {
  try {
    const { data } = await octokit.rest.repos.getBranch({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      branch,
    });
    return data;
  } catch (e) {
    if (e.status === 404) return null;
    throw e;
  }
}

async function getDefaultBranch() {
  const { data } = await octokit.rest.repos.get({
    owner: TEST_OWNER,
    repo: TEST_REPO,
  });
  return data.default_branch;
}

async function createBranch(name, fromBranch) {
  const baseBranch = fromBranch || (await getDefaultBranch());
  const { data: ref } = await octokit.rest.git.getRef({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    ref: `heads/${baseBranch}`,
  });

  await octokit.rest.git.createRef({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    ref: `refs/heads/${name}`,
    sha: ref.object.sha,
  });

  return name;
}

async function createOrUpdateFile(filePath, content, message, branch) {
  let existingSha;
  try {
    const { data } = await octokit.rest.repos.getContent({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      path: filePath,
      ref: branch,
    });
    if (!Array.isArray(data) && data.type === 'file') {
      existingSha = data.sha;
    }
  } catch (e) {
    if (e.status !== 404) throw e;
  }

  const { data } = await octokit.rest.repos.createOrUpdateFileContents({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    path: filePath,
    message,
    content: Buffer.from(content).toString('base64'),
    branch,
    sha: existingSha,
  });

  return { sha: data.content?.sha || '' };
}

async function main() {
  console.log('Setting up test fixtures...\n');

  const fixtures = {
    owner: TEST_OWNER,
    repo: TEST_REPO,
    createdAt: new Date().toISOString(),
  };

  // Step 1: Check if test branch exists, create if not
  console.log(`1. Checking for test branch: ${TEST_BRANCH}`);
  let branch = await getBranch(TEST_BRANCH);
  if (branch) {
    console.log(`   Branch exists: ${TEST_BRANCH}`);
  } else {
    console.log(`   Creating branch from main...`);
    await createBranch(TEST_BRANCH);
    console.log(`   Created branch: ${TEST_BRANCH}`);
  }
  fixtures.branch = TEST_BRANCH;

  // Step 2: Create or update a test file on the branch
  console.log(`\n2. Creating test file: ${TEST_FILE_PATH}`);
  const fileContent = `/**
 * Sample Test File
 *
 * This file is used for GitHub API validation testing.
 * Last updated: ${new Date().toISOString()}
 */

export function sampleFunction(): string {
  return 'Hello from test fixtures!';
}

export function add(a: number, b: number): number {
  return a + b;
}

export const VERSION = '1.0.0';
`;

  const fileResult = await createOrUpdateFile(
    TEST_FILE_PATH,
    fileContent,
    `test: update sample file for validation testing`,
    TEST_BRANCH,
  );
  console.log(`   File created/updated with SHA: ${fileResult.sha}`);
  fixtures.fileSha = fileResult.sha;

  // Step 3: Check for existing PR or create new one
  console.log(`\n3. Checking for existing PR from ${TEST_BRANCH}...`);
  const { data: existingPRs } = await octokit.rest.pulls.list({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    state: 'open',
  });
  let pr = existingPRs.find((p) => p.head.ref === TEST_BRANCH);

  if (pr) {
    console.log(`   Found existing PR #${pr.number}: ${pr.title}`);
  } else {
    console.log(`   Creating new PR...`);
    const { data } = await octokit.rest.pulls.create({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      title: 'Test Fixtures PR for Validation',
      head: TEST_BRANCH,
      base: 'main',
      body: `## Description

This PR contains test fixtures for GitHub API validation testing.

### Changes
- Added/updated sample.ts with test functions

### Purpose
Used by the mlx-node-training validation scripts to test GitHub API interactions.

---
*This PR is automatically managed by setup-test-fixtures.mjs*`,
    });
    pr = data;
    console.log(`   Created PR #${pr.number}`);
  }
  fixtures.prNumber = pr.number;
  fixtures.prNodeId = pr.node_id;

  // Step 4: Create a review comment (inline code comment)
  console.log(`\n4. Creating review comment on PR #${pr.number}...`);

  // Get the PR to find the head commit
  const { data: prDetails } = await octokit.rest.pulls.get({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    pull_number: pr.number,
  });
  const headCommit = prDetails.head.sha;

  // Check for existing review comments
  const { data: existingReviewComments } = await octokit.rest.pulls.listReviewComments({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    pull_number: pr.number,
  });
  let reviewComment = existingReviewComments.find((c) => c.body.includes('Test review comment for validation'));

  if (reviewComment) {
    console.log(`   Found existing review comment ID: ${reviewComment.id}`);
  } else {
    console.log(`   Creating new review comment...`);
    // Create a review comment on line 8 of the test file
    const { data } = await octokit.rest.pulls.createReviewComment({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      pull_number: pr.number,
      body: `**Test review comment for validation**

This is an inline code comment used for testing the GitHub API review comment functionality.

- Created by: setup-test-fixtures.mjs
- Timestamp: ${new Date().toISOString()}`,
      commit_id: headCommit,
      path: TEST_FILE_PATH,
      line: 8, // Line number for the sampleFunction
      side: 'RIGHT',
    });
    reviewComment = data;
    console.log(`   Created review comment ID: ${reviewComment.id}`);
  }
  fixtures.reviewCommentId = reviewComment.id;

  // Step 5: Create an issue comment (general discussion)
  console.log(`\n5. Creating issue comment on PR #${pr.number}...`);

  // Check for existing issue comments
  const { data: existingIssueComments } = await octokit.rest.issues.listComments({
    owner: TEST_OWNER,
    repo: TEST_REPO,
    issue_number: pr.number,
  });
  let issueComment = existingIssueComments.find((c) => c.body?.includes('Test issue comment for validation'));

  if (issueComment) {
    console.log(`   Found existing issue comment ID: ${issueComment.id}`);
  } else {
    console.log(`   Creating new issue comment...`);
    const { data } = await octokit.rest.issues.createComment({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      issue_number: pr.number,
      body: `## Test issue comment for validation

This is a general discussion comment on the PR used for testing the GitHub API issue comment functionality.

### Details
- Created by: \`setup-test-fixtures.mjs\`
- Timestamp: ${new Date().toISOString()}
- Purpose: Validation testing for mlx-node-training

---
*This comment is automatically managed.*`,
    });
    issueComment = data;
    console.log(`   Created issue comment ID: ${issueComment.id}`);
  }
  fixtures.issueCommentId = issueComment.id;

  // Step 6: Check/create workflow file
  console.log(`\n6. Checking for workflow file: ${WORKFLOW_FILE_PATH}`);

  try {
    // Try to get the workflow file from main branch
    const defaultBranch = await getDefaultBranch();
    await octokit.rest.repos.getContent({
      owner: TEST_OWNER,
      repo: TEST_REPO,
      path: WORKFLOW_FILE_PATH,
      ref: defaultBranch,
    });
    console.log(`   Workflow file already exists on ${defaultBranch}`);
    fixtures.workflowExists = true;
  } catch (e) {
    if (e.status === 404) {
      console.log(`   Workflow file not found, creating...`);
      const workflowContent = `name: Test Workflow

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Echo test
        run: |
          echo "Test workflow running!"
          echo "Repository: \${{ github.repository }}"
          echo "Ref: \${{ github.ref }}"
          echo "SHA: \${{ github.sha }}"
`;

      const defaultBranch = await getDefaultBranch();
      await createOrUpdateFile(
        WORKFLOW_FILE_PATH,
        workflowContent,
        'ci: add test workflow for validation',
        defaultBranch,
      );
      console.log(`   Created workflow file on ${defaultBranch}`);
      fixtures.workflowExists = true;
    } else {
      console.log(`   Error checking workflow: ${e.message}`);
      fixtures.workflowExists = false;
    }
  }

  // Save fixtures to JSON file
  const outputPath = path.join(__dirname, 'test-fixtures.json');
  fs.writeFileSync(outputPath, JSON.stringify(fixtures, null, 2));
  console.log(`\n========================================`);
  console.log(`Test fixtures saved to: ${outputPath}`);
  console.log(`========================================\n`);
  console.log(JSON.stringify(fixtures, null, 2));

  return fixtures;
}

main().catch((error) => {
  console.error('Error setting up test fixtures:', error);
  process.exit(1);
});
