import { AsyncEntry } from '@napi-rs/keyring';
import { Repository } from '@napi-rs/simple-git';
import { Octokit } from 'octokit';

const keyringEntry = new AsyncEntry('mlx-node', 'github-token');
// keyring v2 rejects on store errors where v1 returned null.
const GITHUB_TOKEN = await keyringEntry.getPassword().catch(() => null);

if (!GITHUB_TOKEN) {
  throw new Error('GITHUB_TOKEN is not set');
}

const _repo = new Repository(process.cwd());
export const currentBranch = _repo.head().shorthand();
const remoteUrl = _repo.findRemote('origin')?.url() || '';

function parseGitHub(url: string) {
  const s = url.replace('.git', '').replace('git@github.com:', '/').split('/');
  return [s.at(-2)!, s.at(-1)!];
}

export const [owner, repo] = parseGitHub(remoteUrl);
export const octokit = new Octokit({ auth: GITHUB_TOKEN });
