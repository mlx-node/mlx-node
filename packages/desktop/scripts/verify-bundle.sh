#!/usr/bin/env bash
#
# Release gate for the packaged .app.
#
# The reason this exists: `codesign --verify --deep --strict` PASSES on a bundle
# containing an unsigned Mach-O under Contents/Resources, and the notary service
# rejects that same bundle hours later. Local verification is not a predictor of
# notarization, so the release path needs the check codesign does not do —
# "every Mach-O in here carries OUR Team ID".
#
# Measured on the artifacts this app ships (2026-07-27):
#
#   codesign -dv packages/core/npm/darwin-arm64/mlx-core.darwin-arm64.node
#     Identifier=libmlx_core.dylib          <- a build artifact name, not a bundle id
#     Signature=adhoc                        <- linker-signed
#     TeamIdentifier=not set                 <- the notary refuses this
#
#   otool -l ... | grep LC_ID_DYLIB
#     name /Users/<user>/workspace/github/mlx-node/target/.../libmlx_core.dylib
#                                            <- ships a home directory to every user
#
# Both are fixed before signing (install_name_tool -id, codesign --identifier);
# this script is what proves the fix actually happened.
#
#   usage: verify-bundle.sh <path-to-.app> [--team-id ID] [--notarized]
#
# --notarized adds the post-submission checks. Omit it for a locally signed build.

set -euo pipefail

APP=""
TEAM_ID="${APPLE_TEAM_ID:-}"
NOTARIZED=0

while [ $# -gt 0 ]; do
  case "$1" in
    --team-id) TEAM_ID="$2"; shift 2 ;;
    --notarized) NOTARIZED=1; shift ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) APP="$1"; shift ;;
  esac
done

if [ -z "$APP" ] || [ ! -d "$APP" ]; then
  echo "usage: $0 <path-to-.app> [--team-id ID] [--notarized]" >&2
  exit 2
fi

fail=0
note() { printf '  %-8s %s\n' "$1" "$2"; }

echo "==> $APP"

# ---------------------------------------------------------------------------
# 1. What codesign DOES check. Necessary, and nowhere near sufficient.
# ---------------------------------------------------------------------------
echo "[1/4] codesign --verify --deep --strict"
if codesign --verify --deep --strict --verbose=4 "$APP" 2>&1 | sed 's/^/  /'; then
  note "ok" "bundle seal intact"
else
  note "FAIL" "bundle seal broken"
  fail=1
fi

# ---------------------------------------------------------------------------
# 2. The check codesign does NOT do.
#
# Scanning by `-perm +111` (the obvious approach) is WRONG: a .node is not
# necessarily executable and a .dylib usually is not, so an exec-bit filter walks
# straight past the exact files most likely to be unsigned. Every regular file is
# sniffed with `file` instead.
#
# "TeamIdentifier=not set" is only damning INSIDE our bundle. Apple's own platform
# binaries report the same thing because they are signed under Apple's authority
# rather than a Team ID — which is why this never scans outside $APP.
# ---------------------------------------------------------------------------
echo "[2/4] every Mach-O carries a Team ID"
# ONE `file` pass over the whole bundle, batched through xargs. The obvious
# implementation -- `find -exec file {} \;` per step -- spawns one process per
# file, and this bundle has ~30k of them; two such passes took over two minutes
# and were killed. Batching turns it into a handful of execs.
MACHO_LIST="$(mktemp)"
trap 'rm -f "$MACHO_LIST"' EXIT
find "$APP" -type f -print0 | xargs -0 file | sed -n 's/^\(.*\): .*Mach-O.*/\1/p' > "$MACHO_LIST"
macho_count="$(wc -l < "$MACHO_LIST" | tr -d ' ')"

while IFS= read -r f; do
  # `|| true` is load-bearing, not defensive noise. On a file that is not signed
  # AT ALL, `codesign -dv` exits non-zero; with `set -e` + `pipefail` that aborts
  # the whole script mid-scan -- no FAIL line, no count, no later steps. The one
  # case this gate exists to catch would produce the LEAST diagnostic output, and
  # an operator would read the abort as some unrelated bundle problem.
  team="$(codesign -dv "$f" 2>&1 | sed -n 's/^TeamIdentifier=//p' || true)"
  if [ -z "$team" ] || [ "$team" = "not set" ]; then
    note "FAIL" "no Team ID: ${f#"$APP"/}"
    fail=1
  elif [ -n "$TEAM_ID" ] && [ "$team" != "$TEAM_ID" ]; then
    note "FAIL" "Team ID $team != $TEAM_ID: ${f#"$APP"/}"
    fail=1
  fi
done < "$MACHO_LIST"
note "scanned" "$macho_count Mach-O files"
[ "$macho_count" -gt 0 ] || { note "FAIL" "found no Mach-O at all -- wrong path?"; fail=1; }

# ---------------------------------------------------------------------------
# 3. Build-path leak. Not a signing failure, but it ships a developer's home
#    directory (and CI layout) to every user, and it is invisible in the UI.
# ---------------------------------------------------------------------------
echo "[3/4] no build paths baked into load commands"
while IFS= read -r f; do
  # Same `|| true` reasoning: grep exits 1 when it matches nothing, which is the
  # HEALTHY case here.
  leaks="$(otool -l "$f" 2>/dev/null | sed -n 's/^ *name \(.*\) (offset.*/\1/p' \
    | grep -E '^/Users/|/target/' || true)"
  if [ -n "$leaks" ]; then
    note "FAIL" "leaks a build path: ${f#"$APP"/}"
    printf '%s\n' "$leaks" | sed 's/^/           /'
    fail=1
  fi
done < "$MACHO_LIST"

# ---------------------------------------------------------------------------
# 4. Post-notarization only. Gatekeeper assessment + a stapled ticket, so the app
#    opens on a machine that has never seen it and is offline.
# ---------------------------------------------------------------------------
echo "[4/4] notarization"
if [ "$NOTARIZED" -eq 1 ]; then
  if spctl -a -vvv -t install "$APP" 2>&1 | sed 's/^/  /'; then
    note "ok" "gatekeeper accepts"
  else
    note "FAIL" "gatekeeper rejects"
    fail=1
  fi
  if xcrun stapler validate "$APP" 2>&1 | sed 's/^/  /'; then
    note "ok" "ticket stapled"
  else
    note "FAIL" "no stapled ticket — the app needs network on first launch"
    fail=1
  fi
else
  note "skip" "pass --notarized after the notary returns"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "BUNDLE GATE: FAIL"
  exit 1
fi
echo "BUNDLE GATE: PASS"
