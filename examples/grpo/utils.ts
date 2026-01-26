import { Repository } from '@napi-rs/simple-git';
import { AsyncEntry } from '@napi-rs/keyring';
import { Octokit } from 'octokit';

const keyringEntry = new AsyncEntry('mlx-node', 'github-token');
const GITHUB_TOKEN = await keyringEntry.getPassword();

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
