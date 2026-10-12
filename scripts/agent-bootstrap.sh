#!/usr/bin/env bash
# Bootstrap the mlx-node/grok-build fork and build the `mlx-agent` binary.
#
# The fork lives as a gitignored checkout at <repo>/grok-build. `mlx agent`
# resolves the binary in this order (see packages/cli/src/commands/agent/grok-build.ts):
#   1. MLX_AGENT_BIN env var
#   2. <repo>/grok-build/target/release/mlx-agent   (this script's output)
#   3. <app>/Contents/Resources/native/mlx-agent    (bundled desktop app)
#   4. ~/.mlx-node/bin/mlx-agent                    (verified release download)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORK_DIR="${REPO_ROOT}/grok-build"
FORK_URL="${MLX_AGENT_FORK_URL:-https://github.com/mlx-node/grok-build.git}"
FORK_BRANCH="${MLX_AGENT_FORK_BRANCH:-mlx}"

if [ ! -d "${FORK_DIR}/.git" ]; then
  echo "Cloning ${FORK_URL} (branch ${FORK_BRANCH}) into ${FORK_DIR}"
  git clone --depth 1 --branch "${FORK_BRANCH}" "${FORK_URL}" "${FORK_DIR}"
else
  echo "grok-build checkout already present at ${FORK_DIR}"
fi

command -v protoc >/dev/null || {
  echo "error: protoc is required (brew install protobuf)" >&2
  exit 1
}

cd "${FORK_DIR}"
# dotslash-managed protoc is not vendored; PATH protoc works fine.
PROTOC="$(command -v protoc)" cargo build -p xai-grok-pager-bin --release

echo "Built ${FORK_DIR}/target/release/mlx-agent"
