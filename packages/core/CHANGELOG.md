# Changelog

## Unreleased

### Removed

- **Breaking (public API):** `gdnPrefixCheckpointLimit()` is no longer exported. It existed only so a
  test of the removed pi subagent runtime could check its concurrency cap against the native GDN prefix
  checkpoint limit; it has no remaining consumer.
