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
echo "[1/5] codesign --verify --deep --strict"
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
echo "[2/5] every Mach-O carries a Team ID"
# ONE `file` pass over the whole bundle, batched through xargs. The obvious
# implementation -- `find -exec file {} \;` per step -- spawns one process per
# file, and this bundle has ~30k of them; two such passes took over two minutes
# and were killed. Batching turns it into a handful of execs.
MACHO_LIST="$(mktemp)"
trap 'rm -f "$MACHO_LIST"' EXIT
# Parsing `file` output for NAMES is fiddly and two obvious forms are both wrong. `sed 's/^\(.*\): .*Mach-O.*/\1/p'` is
# WRONG here and was: `.*` is greedy, and `file` prints
#   path: Mach-O universal binary with 2 architectures: [x86_64:...] [arm64:...]
# so the capture ran past the second colon. `awk -F': '` fails too, because a
# universal binary emits one line PER ARCHITECTURE and those use a different
# separator entirely:
#   path (for architecture x86_64):<TAB>Mach-O 64-bit dynamically linked ...
# So: cut at the FIRST colon, strip a trailing " (for architecture …)", dedupe.
# Thin binaries parsed fine under both broken versions, which is exactly why this
# survived the first pass.
find "$APP" -type f -print0 | xargs -0 file \
  | awk '/Mach-O/ { i = index($0, ":"); if (i == 0) next; n = substr($0, 1, i - 1); sub(/ \(for architecture [^)]*\)$/, "", n); print n }' | sort -u > "$MACHO_LIST"
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
echo "[3/5] no build paths baked into load commands"
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
echo "[4/5] notarization"
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

# ---------------------------------------------------------------------------
# 5. LSMinimumSystemVersion vs what the binaries actually demand.
#
# @electron/packager writes Electron's template floor (12.0) regardless of what
# the payload was built against. Release builds set MACOSX_DEPLOYMENT_TARGET=26.0,
# so the addon's LC_BUILD_VERSION says minos 26.0 and dyld refuses to map it on
# 12-15 -- but LaunchServices reads the plist, so the app OPENS there and only
# then fails to load the addon. The supervisor sees a dead sidecar and restarts
# it, forever, with nothing naming the real cause.
#
# Computed over EVERY Mach-O here, while package.ts sets the plist from three
# representative binaries. Deliberately two independent derivations: if they
# agreed by construction this step would only be able to pass.
# ---------------------------------------------------------------------------
echo "[5/5] LSMinimumSystemVersion matches the binaries' build floor"
PLIST="$APP/Contents/Info.plist"
declared="$(plutil -extract LSMinimumSystemVersion raw -o - "$PLIST" 2>/dev/null || true)"
# Highest minos over the whole bundle. `sort -t. -kNn` rather than `sort -V`:
# version sort is a GNU extension that BSD sort has only sometimes had, and this
# runs on whatever macOS the runner image ships.
required="$(while IFS= read -r f; do otool -l "$f" 2>/dev/null || true; done < "$MACHO_LIST" \
  | awk '/^ *minos /{ print $2 }' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)"

if [ -z "$declared" ]; then
  note "FAIL" "Info.plist has no LSMinimumSystemVersion — packager's default would apply"
  fail=1
elif [ -z "$required" ]; then
  note "FAIL" "no LC_BUILD_VERSION on any Mach-O — cannot tell what the floor should be"
  fail=1
elif ! awk -v a="$declared" -v b="$required" 'BEGIN {
       split(a, x, "."); split(b, y, ".");
       for (i = 1; i <= 3; i++) if (x[i] + 0 != y[i] + 0) exit 1;
       exit 0 }'; then
  # Both directions are bugs: too low and it launches where it cannot run, too
  # high and it refuses to launch where it could.
  note "FAIL" "declares $declared but its binaries demand $required"
  fail=1
else
  note "ok" "declares $declared, binaries demand $required"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "BUNDLE GATE: FAIL"
  exit 1
fi
echo "BUNDLE GATE: PASS"
