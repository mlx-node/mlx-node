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

Options:
  -h, --help         Show this help message
  -v, --version      Show version number

Examples:
  mlx download model -m Qwen/Qwen3-0.6B
  mlx download dataset -d openai/gsm8k
  mlx convert -i .cache/models/qwen3-0.6b -o .cache/models/qwen3-0.6b-mlx -d bf16
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
      if (!subcommand || subcommand === '--help' || subcommand === '-h') {
        console.log(`
Usage:
  mlx download model     Download a model from HuggingFace
  mlx download dataset   Download a dataset from HuggingFace

Run mlx download <subcommand> --help for more information.
`);
        return;
      }

      const rest = args.slice(2);

      if (subcommand === 'model') {
        const { run } = await import('./commands/download-model.js');
        await run(rest);
      } else if (subcommand === 'dataset') {
        const { run } = await import('./commands/download-dataset.js');
        await run(rest);
      } else {
        console.error(`Unknown download subcommand: ${subcommand}`);
        console.error('Available: model, dataset');
        process.exit(1);
      }
      break;
    }

    case 'convert': {
      const rest = args.slice(1);
      const { run } = await import('./commands/convert.js');
      await run(rest);
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
