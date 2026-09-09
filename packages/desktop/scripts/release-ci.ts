export const CI_WAIT_TIMEOUT_MS = 3 * 60 * 60 * 1_000;
export const CI_POLL_INTERVAL_MS = 30_000;

interface WorkflowRun {
  id: number;
  head_sha: string;
  head_branch: string;
  event: string;
  path: string;
  status: string;
  conclusion: string | null;
  html_url: string;
}

/** Only a completed main-push CI run for this exact commit can expose an update. */
export async function waitForReleaseCI(commit: string, github: (args: string[]) => string): Promise<void> {
  if (!/^[a-f0-9]{40}$/.test(commit)) throw new Error('Release CI requires a full commit SHA');
  const deadline = Date.now() + CI_WAIT_TIMEOUT_MS;
  const endpoint = `repos/{owner}/{repo}/actions/workflows/ci.yml/runs?head_sha=${commit}&branch=main&event=push&per_page=100`;
  let lastStatus = '';
  while (true) {
    // Do not filter for success in the API: an older successful run must not
    // hide a newer failed run or an in-progress rerun of the tagged commit.
    const response = JSON.parse(github(['api', endpoint])) as { workflow_runs: WorkflowRun[] };
    const run = response.workflow_runs
      .filter(
        (run) =>
          run.head_sha === commit &&
          run.head_branch === 'main' &&
          run.event === 'push' &&
          run.path === '.github/workflows/ci.yml',
      )
      .sort((a, b) => b.id - a.id)[0];
    if (run?.status === 'completed') {
      if (run.conclusion !== 'success') {
        throw new Error(
          `Release blocked: CI for ${commit} concluded ${run.conclusion ?? 'without a result'} (${run.html_url})`,
        );
      }
      console.log(`Release CI passed: ${run.html_url}`);
      return;
    }
    if (Date.now() >= deadline) {
      throw new Error(`Timed out waiting for successful main-branch CI for ${commit}; the release remains a draft`);
    }
    const status = run ? `${run.id}: ${run.status}` : 'waiting for the main-push run to appear';
    if (status !== lastStatus) {
      console.log(`Release CI: ${status}`);
      lastStatus = status;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, CI_POLL_INTERVAL_MS));
  }
}
