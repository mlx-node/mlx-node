export const SYSTEM_PROMPT = `You are a GitHub API code generator.

WORKFLOW:
First, use the lsp tool to query API parameters.
Then, use the run_js tool to execute your code.

TOOLS:
- lsp: Query API documentation before writing code
- run_js: Execute JavaScript code after checking API docs

CODE FORMAT:
import { owner, repo, currentBranch, octokit } from './utils'

export default async function() {
  const result = await octokit.rest.pulls.list({
    owner,
    repo,
    state: 'open',
    head: \`\${owner}:\${currentBranch}\`
  })
  return result.data
}

VARIABLES:
- owner: repository owner
- repo: repository name
- currentBranch: current git branch
- octokit: authenticated Octokit client
`;

export const GET_PR_PROMPTS = [
  "List open PRs for the current branch. Query the API first, then write the code. Use state='open' and head filter.",
  "Get open pull requests filtered by current branch using pulls.list API. Use state='open' and head filter.",
  "Query pulls.list to check parameters, then write code with state='open' and head filter.",
  "Check the pulls.list API parameters, then generate code to list open PRs. Use state='open' and head filter.",
  "Get open PRs for this branch. Use lsp first, then run_js. Use state='open' and head filter.",
  "List PRs for current branch. Query API, then write code. Use state='open' and head filter.",
  "Get PRs with state='open' and head=owner:currentBranch. Check API first. Use state='open' and head filter.",
  "I need to list open pull requests for the current branch. Use state='open' and head filter.",
];
