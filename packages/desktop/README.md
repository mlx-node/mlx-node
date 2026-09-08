# mlx-node for macOS

Install the signed DMG from [GitHub Releases](https://github.com/mlx-node/mlx-node/releases)
by dragging mlx-node to Applications. The app requires Apple Silicon and macOS 26 or newer.

## Updates

Signed stable builds check for updates at launch and every six hours. Updates download
in the background. Use **Restart to Update…** in the tray or application menu when
you are ready to stop inference and restart. A downloaded update also applies the
next time you launch the app after quitting normally.

The same menu offers **Check for Updates…**, download progress, and a retry action if
an update fails. Models, sessions, and settings stay in their existing data directories.
Development, unsigned, and prerelease builds do not check the public update feed.

Versions released before this updater was added need one manual installation of a
release that includes it.

## Release artifacts

The [desktop release workflow](../../.github/workflows/desktop-release.yml) produces:

- `mlx-node-<version>-arm64.dmg` for installation.
- `mlx-node-<version>-darwin-arm64.zip` for automatic updates.
- `mlx-node-<version>-darwin-arm64.zip.blockmap` for differential downloads.
- `latest-mac.yml` with the update version, archive size/hash, minimum macOS, and rollout percentage.

Both contain the same Developer ID signed, notarized, and stapled app. The ZIP is
created after stapling, then extracted to verify its signature, ticket, and version
before publication. Dry runs build and verify both archives but skip notarization
and release uploads.

The app uses [electron-updater](https://www.electron.build/v26/docs/features/auto-update/)
with public GitHub Releases. macOS installation still uses Squirrel.Mac. Keep the
archive naming, bundle ID `ai.mlxnode.desktop`, and Developer ID signing identity
consistent across releases. No additional update server, client token, or signing
secret is needed. Packaging includes `Contents/Resources/app-update.yml` before signing.

The release workflow generates blockmaps from the final ZIP using electron-builder's
build-time library; it does not replace our packager or ship that library in the app.
The updater reuses a previously cached ZIP where possible and falls back to a full
download when the cache or blockmaps are unavailable. The first update normally
downloads the complete ZIP. Retain old ZIPs and blockmaps on their original releases.

Updates default to a 100% rollout. Set the repository variable
`DESKTOP_UPDATE_STAGING_PERCENTAGE` to an integer from 0 to 100 before a release
to limit automatic offers; the manual workflow has a matching input. To expand or
pause an existing rollout, edit only `stagingPercentage` in that release's
`latest-mac.yml` and replace the manifest asset. Keep all archive hashes and URLs
unchanged. Clients keep a persistent rollout ID, so increasing the percentage
includes the earlier group. A rollout change does not undo installed updates;
publish a higher fixed version to repair a bad release. Beta-channel selection is
not enabled in the desktop UI.

Packaging stamps update eligibility into the staged app manifest before signing;
the source manifest does not enable updates. The release tag, desktop manifest,
bundle version, DMG filename, ZIP filename, and update metadata must agree. The
workflow publishes the manifest after the ZIP, blockmap, and DMG are uploaded.

Before the first production rollout, verify an upgrade between signed releases,
including normal quit without relaunch, **Restart to Update…**, active inference,
and a failed differential download falling back to the full ZIP.
