# GitHub Tool Training with GRPO

Train Qwen3 models to use GitHub's Octokit API via Group Relative Policy Optimization (GRPO).

## Quick Start

```bash
# Basic training run
yarn oxnode examples/grpo/train-github-tool.ts

# With TUI visualization
./target/release/mlx-train --script examples/grpo/train-github-tool.ts
```

## CLI Options

| Flag                     | Description                            | Default             |
| ------------------------ | -------------------------------------- | ------------------- |
| `-t, --target <id>`      | Training scenario(s) - can be repeated | All scenarios       |
| `-n, --num-examples <n>` | Number of training examples            | 50                  |
| `--epochs <n>`           | Number of training epochs              | 3                   |
| `--dry-run`              | Preview prompts without training       | false               |
| `--model <path>`         | Path to model directory                | ./models/qwen3-0.6b |

## Model Output Format

The model generates **only** the function body. Pre-defined variables are available:

```javascript
// Pre-defined (always available):
// - owner: GitHub repository owner
// - repo: GitHub repository name
// - branch: Current git branch
// - octokit: Authenticated Octokit client

// Model generates:
export default async function () {
  const { data: prs } = await octokit.rest.pulls.list({
    owner,
    repo,
    head: owner + ':' + branch,
    state: 'open',
  });
  if (!prs.length) return { error: 'no_pr' };

  const { data: comments } = await octokit.rest.pulls.listReviewComments({
    owner,
    repo,
    pull_number: prs[0].number,
  });
  return { comments };
}
```

## Reward Scoring

| Score  | Gate        | Meaning                       |
| ------ | ----------- | ----------------------------- |
| 0-2    | Format      | No valid tool call            |
| 2-4    | Syntax      | Parse errors, missing export  |
| 4-6    | API         | Wrong or missing API call     |
| 5.5-7  | Execution   | Runtime error or wrong result |
| **7+** | **Success** | **All gates passed**          |

**Bonus points (7-10):**

- Thinking quality: +0-1.5 (concise, relevant reasoning)
- Error handling: +0.5 (try-catch in function)
- Typed errors: +0.5 (checks like `error.status === 404`)
- Conciseness: +0.5 (function body < 80 words)

## Available Scenarios

### Read Operations

| ID                    | Description                  | API                        |
| --------------------- | ---------------------------- | -------------------------- |
| `get-comments`        | Get PR conversation comments | `issues.listComments`      |
| `get-inline-comments` | Get inline review comments   | `pulls.listReviewComments` |
| `get-file-diff`       | Get changed files in PR      | `pulls.listFiles`          |

### Write Operations

| ID                       | Description              | API                                 |
| ------------------------ | ------------------------ | ----------------------------------- |
| `reply-inline-comment`   | Reply to review comment  | `pulls.createReplyForReviewComment` |
| `resolve-inline-comment` | Resolve review thread    | GraphQL mutation                    |
| `batch-resolve-comments` | Resolve multiple threads | GraphQL mutation                    |
| `change-pr-title`        | Update PR title          | `pulls.update`                      |

### Workflow Operations

| ID                      | Description      | API                               |
| ----------------------- | ---------------- | --------------------------------- |
| `check-workflow-status` | Get CI status    | `actions.listWorkflowRunsForRepo` |
| `trigger-workflow`      | Trigger workflow | `actions.createWorkflowDispatch`  |
| `get-ci-outputs`        | Download CI logs | `actions.downloadWorkflowRunLogs` |

## Examples

```bash
# Train on specific scenario
yarn oxnode examples/grpo/train-github-tool.ts -t get-comments

# Train on multiple scenarios
yarn oxnode examples/grpo/train-github-tool.ts -t get-comments -t get-inline-comments

# Train with more examples
yarn oxnode examples/grpo/train-github-tool.ts -n 100 --epochs 5

# Preview without training
yarn oxnode examples/grpo/train-github-tool.ts --dry-run
```

## Files

| File                     | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| `train-github-tool.ts`   | Main training script                      |
| `github-dataset.ts`      | Scenario templates and dataset generation |
| `code-validator.ts`      | JavaScript code validation                |
| `execute-github-code.ts` | Sandboxed code execution                  |

## Output

Training outputs are saved to `./outputs/grpo-github-tool/`:

- `outputs.db` - SQLite database with generations and metrics
- `checkpoints/` - Model checkpoints
