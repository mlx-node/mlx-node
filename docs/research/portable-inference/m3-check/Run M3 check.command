#!/bin/zsh
set -eu
TASK_DIR="${0:A:h}"
print 'Quit mlx-node and its coding agent before running this check.'
print 'The check runs locally, uses public code, and sends nothing over the network.'
TASK_APP=$(/usr/bin/osascript -e 'POSIX path of (choose file with prompt "Select the signed mlx-node test app (test3)" of type {"com.apple.application-bundle"})')
TASK_MODEL=$(/usr/bin/osascript -e 'POSIX path of (choose file with prompt "Select Qwen3.8-27B-UD-Q4_K_XL.gguf")')
TASK_OUTPUT="$HOME/Desktop/mlx-M3-check-$(date +%Y%m%d-%H%M%S)"
export ELECTRON_RUN_AS_NODE=1
if "$TASK_APP/Contents/MacOS/mlx-node" "$TASK_DIR/run.cjs" "$TASK_APP" "$TASK_MODEL" "$TASK_OUTPUT"; then
  /usr/bin/open "$TASK_OUTPUT"
else
  print 'The check stopped. Keep any partial report and logs in the output folder.'
fi
print 'Press Enter to close.'
read -r REPLY
