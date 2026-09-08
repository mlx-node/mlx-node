/** Create electron-updater metadata from the actual packaged app and final ZIP. */
import { execFileSync } from 'node:child_process';
import { appendFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

import { createUpdateArtifacts } from './update-artifacts.js';

const { values } = parseArgs({
  options: {
    app: { type: 'string' },
    zip: { type: 'string' },
    'staging-percentage': { type: 'string', default: '100' },
  },
});
if (!values.app || !values.zip) {
  throw new Error('usage: generate-update.ts --app <app> --zip <zip> [--staging-percentage <0-100>]');
}
const readPlist = (key: string): string =>
  execFileSync('plutil', ['-extract', key, 'raw', '-o', '-', join(values.app!, 'Contents', 'Info.plist')], {
    encoding: 'utf-8',
  }).trim();
const result = await createUpdateArtifacts({
  zipPath: values.zip,
  version: readPlist('CFBundleShortVersionString'),
  minimumSystemVersion: readPlist('LSMinimumSystemVersion'),
  stagingPercentage: Number(values['staging-percentage']),
});
if (process.env.GITHUB_OUTPUT) {
  appendFileSync(process.env.GITHUB_OUTPUT, `manifest=${result.manifestPath}\nblockmap=${result.blockmapPath}\n`);
}
console.log(`Created ${result.manifestPath} and ${result.blockmapPath}`);
