# Autoresearch Dashboard: webgpu-production

**Runs:** 3 | **Kept:** 3 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** decode_tok_s: 2.4 tok/s (#1)
**Best:** decode_tok_s: 2.5 tok/s (#3, +4.2%)

| # | commit | decode_tok_s | browser_tests | output_ok | temp>0 | ttft_ms | status | description |
|---|--------|-------------|---------------|-----------|--------|---------|--------|-------------|
| 1 | a8a9007 | 2.4 tok/s | 162/163 | yes | no | 2793ms | keep | baseline |
| 2 | 4eb7c53 | 2.4 tok/s (0%) | 162/163 | yes | no | 2834ms | keep | remove diagnostics (code quality) |
| 3 | 253869d | 2.5 tok/s (+4.2%) | 162/163 | yes | **YES** | 2796ms | keep | fix temp>0: pure Rust categorical |
