# Autoresearch Dashboard: webgpu-production

**Runs:** 5 | **Kept:** 5 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** decode_tok_s: 2.4 tok/s (#1)
**Best:** decode_tok_s: 4.1 tok/s (#5, +70.8%)

| # | commit | decode_tok_s | browser_tests | output_ok | temp>0 | ttft_ms | status | description |
|---|--------|-------------|---------------|-----------|--------|---------|--------|-------------|
| 1 | a8a9007 | 2.4 tok/s | 162/163 | yes | no | 2793ms | keep | baseline |
| 2 | 4eb7c53 | 2.4 tok/s (0%) | 162/163 | yes | no | 2834ms | keep | remove diagnostics (code quality) |
| 3 | 253869d | 2.5 tok/s (+4.2%) | 162/163 | yes | **YES** | 2796ms | keep | fix temp>0: pure Rust categorical |
| 4 | f4c1d42 | 2.8 tok/s (+16.7%) | 162/163 | yes | yes | 2739ms | keep | batch size 64→512, dedup |
| 5 | 9770666 | **4.1 tok/s (+70.8%)** | 162/163 | yes | yes | **1585ms** | keep | fuse dispatch into single RPC |
