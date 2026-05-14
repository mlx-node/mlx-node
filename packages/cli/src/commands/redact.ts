import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { PrivacyFilter } from '@mlx-node/privacy';

function printHelp() {
  console.log(`
Redact PII from text using a privacy-filter model

Usage:
  mlx redact --model <path> [options]

Required Arguments:
  --model, -m <path>        Path to a privacy-filter model directory
                            (must contain config.json, model.safetensors,
                            tokenizer.json, viterbi_calibration.json)

Optional Arguments:
  --input, -i <path>        Input text file (defaults to stdin)
  --output, -o <path>       Output file for redacted text (defaults to stdout)
  --replacement <string>    Replacement string. The sentinel value "label"
                            (the default) substitutes "[<label>]" for each
                            detected span. Any other string is inserted
                            verbatim.
  --labels <csv>            Comma-separated allowlist of labels to redact.
                            Entities with labels outside this list are
                            left in place. Example:
                              --labels private_email,private_person
  --threshold <float>       Minimum mean per-token probability for an
                            entity to be kept (default: 0.5)
  --json                    Emit the entities sidecar as JSON. With
                            --output, writes to "<output>.entities.json".
                            Without --output, writes to stderr.
  --help, -h                Show this help message

Examples:
  mlx redact -m .cache/models/privacy-filter -i input.txt -o redacted.txt
  cat input.txt | mlx redact -m .cache/models/privacy-filter > redacted.txt
  mlx redact -m .cache/models/privacy-filter -i input.txt -o out.txt --json
  mlx redact -m .cache/models/privacy-filter -i input.txt --labels private_email
  mlx redact -m .cache/models/privacy-filter -i input.txt --replacement '[REDACTED]'
`);
}

function readStdin(): Promise<string> {
  return new Promise((resolvePromise, rejectPromise) => {
    const chunks: Buffer[] = [];
    process.stdin.on('data', (chunk: Buffer) => chunks.push(chunk));
    process.stdin.on('end', () => resolvePromise(Buffer.concat(chunks).toString('utf-8')));
    process.stdin.on('error', rejectPromise);
  });
}

export async function run(argv: string[]) {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      model: { type: 'string', short: 'm' },
      input: { type: 'string', short: 'i' },
      output: { type: 'string', short: 'o' },
      replacement: { type: 'string' },
      labels: { type: 'string' },
      threshold: { type: 'string' },
      json: { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  if (!args.model) {
    console.error('Error: --model is required\n');
    console.error('Use --help for usage information');
    process.exit(1);
  }

  let threshold: number | undefined;
  if (args.threshold !== undefined) {
    const parsed = Number(args.threshold);
    if (!Number.isFinite(parsed)) {
      console.error('Error: --threshold must be a finite number');
      process.exit(1);
    }
    threshold = parsed;
  }

  const labels = args.labels
    ? args.labels
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    : undefined;

  // The sentinel string "label" means "replace with [<label_name>]".
  // Any other value is passed through verbatim as the replacement string.
  // When the flag is omitted entirely, the redactor defaults to "label".
  const replacement = args.replacement ?? 'label';

  const modelPath = resolve(args.model);

  const inputText = args.input ? await readFile(resolve(args.input), 'utf-8') : await readStdin();

  const pf = await PrivacyFilter.load(modelPath);
  // `labels` is widened to `string[]` from the CLI; the runtime filter in
  // `redactImpl` only checks set membership, so any non-PrivacyLabel strings
  // simply match nothing — safe to cast.
  const { redacted, entities } = await pf.redact(inputText, {
    replacement,
    labels: labels as never,
    threshold,
  });

  if (args.output) {
    const outputPath = resolve(args.output);
    await writeFile(outputPath, redacted, 'utf-8');
    if (args.json) {
      await writeFile(`${outputPath}.entities.json`, JSON.stringify(entities, null, 2), 'utf-8');
    }
  } else {
    process.stdout.write(redacted);
    if (args.json) {
      process.stderr.write(`${JSON.stringify(entities, null, 2)}\n`);
    }
  }
}
