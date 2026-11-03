//! Offline G2: renders a recorded agent request list through the target
//! tokenizer and classifies every request with the DFlash2 turn ledger under
//! today's flat rule. No weights are loaded.
//!
//! `scripts/dflash2-agent-replay.ts --offline` writes the request list, runs
//! this test, and checks its result:
//!
//! ```shell
//! MLX_DFLASH2_REPLAY_REQUESTS=/path/to/requests.json \
//! MLX_DFLASH2_REPLAY_TOKENIZER=/path/to/tokenizer.json \
//! MLX_DFLASH2_REPLAY_RESULT=/path/to/result.json \
//! cargo test -p mlx-core --test dflash2_replay_offline -- --ignored --nocapture
//! ```

use std::path::Path;

use mlx_core::models::qwen3_5::dflash2_state::replay::{RecordedRequest, replay};
use mlx_core::tokenizer::Qwen3Tokenizer;

/// The default paged block size, which the ledger hashes prompts with.
const BLOCK_SIZE: u32 = 16;

fn env_path(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

#[test]
#[ignore = "driven by scripts/dflash2-agent-replay.ts --offline"]
fn dflash2_replay_offline() {
    let requests_path = env_path("MLX_DFLASH2_REPLAY_REQUESTS");
    let tokenizer_path = env_path("MLX_DFLASH2_REPLAY_TOKENIZER");
    let result_path = env_path("MLX_DFLASH2_REPLAY_RESULT");

    let requests: Vec<RecordedRequest> = serde_json::from_str(
        &std::fs::read_to_string(&requests_path)
            .unwrap_or_else(|error| panic!("{requests_path}: {error}")),
    )
    .unwrap_or_else(|error| panic!("{requests_path}: {error}"));
    let tokenizer = Qwen3Tokenizer::from_file(Path::new(&tokenizer_path))
        .unwrap_or_else(|error| panic!("{tokenizer_path}: {error}"));

    let turns = replay(&tokenizer, BLOCK_SIZE, &requests).unwrap_or_else(|error| panic!("{error}"));
    std::fs::write(&result_path, serde_json::to_vec_pretty(&turns).unwrap())
        .unwrap_or_else(|error| panic!("{result_path}: {error}"));
    println!("replayed {} requests into {result_path}", turns.len());
}
