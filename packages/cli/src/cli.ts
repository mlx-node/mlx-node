#!/usr/bin/env node

import pkgJson from '../package.json' with { type: 'json' };

const args = process.argv.slice(2);
const command = args[0];
const subcommand = args[1];

function printHelp() {
  console.log(`
mlx - MLX-Node CLI v${pkgJson.version}

Usage:
  mlx <command> [options]

Commands:
  download model     Download a model from HuggingFace
  download dataset   Download a dataset from HuggingFace
  convert            Convert model weights to MLX format
  calibrate          Calibrate FP8 activation amax for --q-recipe nvidia models
  redact             Redact PII from text using a privacy-filter model
  launch claude      Start a local server and spawn Claude Code pointed at it

Options:
  -h, --help         Show this help message
  -v, --version      Show version number

Examples:
  mlx download model -m Qwen/Qwen3-0.6B
  mlx download dataset -d openai/gsm8k
  mlx convert -i ~/.mlx-node/models/qwen3-0.6b -o ~/.mlx-node/models/qwen3-0.6b-mlx -d bf16
  mlx calibrate -i ./qwen3.6-27b-nvidia-mxfp4-mlx --dataset ~/.cache/nvidia-calib/cnn_nemotron_v2_calib.jsonl
  mlx redact -m .cache/models/privacy-filter -i input.txt -o redacted.txt
  mlx launch claude
`);
}

async function main() {
  if (!command || command === '--help' || command === '-h') {
    printHelp();
    return;
  }

  if (command === '--version' || command === '-v') {
    console.log(pkgJson.version);
    return;
  }

  switch (command) {
    case 'download': {
      // Delegate the whole arg list to the download dispatcher; the token is a
      // download-wide credential, so flags like `--set-token` are handled
      // there rather than being interpreted as a subcommand at this level.
      const { run } = await import('./commands/download.js');
      await run(args.slice(1));
      break;
    }

    case 'convert': {
      const rest = args.slice(1);
      const { run } = await import('./commands/convert.js');
      await run(rest);
      break;
    }

    case 'calibrate': {
      const rest = args.slice(1);
      const { run } = await import('./commands/calibrate.js');
      await run(rest);
      break;
    }

    case 'redact': {
      const rest = args.slice(1);
      const { run } = await import('./commands/redact.js');
      await run(rest);
      break;
    }

    case 'launch': {
      const rest = args.slice(2);
      if (!subcommand || subcommand === '--help' || subcommand === '-h') {
        console.log(`
Usage:
  mlx launch claude     Start a local server and spawn Claude Code pointed at it

Run mlx launch <subcommand> --help for more information.
`);
        return;
      }
      if (subcommand === 'claude') {
        const { run } = await import('./commands/launch-claude/index.js');
        await run(rest);
      } else {
        console.error(`Unknown launch subcommand: ${subcommand}`);
        console.error('Available: claude');
        process.exit(1);
      }
      break;
    }

    default:
      console.error(`Unknown command: ${command}`);
      printHelp();
      process.exit(1);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
