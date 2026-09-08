/** Draft-first release lifecycle, invoked only by the desktop release workflow. */
import { execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';

import { prepareDraftRelease, publishDesktopRelease } from './release-publication.js';
import { versionFromTag } from './release-version.js';

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    tag: { type: 'string' },
    dmg: { type: 'string' },
    zip: { type: 'string' },
    blockmap: { type: 'string' },
    manifest: { type: 'string' },
  },
});
const tag = values.tag;
if (!tag) throw new Error('A release tag is required for publication');
versionFromTag(tag);
// A manual dispatch must build the tag, not whatever branch was selected in
// the Actions UI. --verify-tag below also requires it to exist on GitHub.
const revision = (ref: string): string =>
  execFileSync('git', ['rev-parse', '--verify', ref], { encoding: 'utf8' }).trim();
if (revision(`refs/tags/${tag}^{commit}`) !== revision('HEAD')) {
  throw new Error(`The checkout does not match release tag ${tag}`);
}
const github = (args: string[]): string => execFileSync('gh', args, { encoding: 'utf8' });
if (positionals[0] === 'prepare') {
  prepareDraftRelease(tag, github);
} else if (positionals[0] === 'publish' && values.dmg && values.zip && values.blockmap && values.manifest) {
  publishDesktopRelease(tag, [values.dmg, values.zip, values.blockmap, values.manifest], github);
} else {
  throw new Error(
    'usage: publish-release.ts <prepare|publish> --tag <tag> [--dmg <dmg> --zip <zip> --blockmap <blockmap> --manifest <manifest>]',
  );
}
