//! Offline replay of a recorded request sequence through the turn ledger.
//!
//! Each request is rendered the way `chat_session_start` renders it and runs
//! under today's flat rule: one model-wide live state, reused only when its
//! whole history is a prefix of the prompt. No weights are loaded.
//! `tests/dflash2_replay_offline.rs` is the entry point.

use serde::{Deserialize, Serialize};

use super::trace::{
    self, Hit, HitKind, LedgerTurn, LiveView, StateDesign, StoreView, TurnLedger, TurnObservation,
};
use crate::engine::gen_prompt::replay_point;
use crate::engine::session::SESSION_PRESERVE_THINKING;
use crate::engine::types::ChatConfig;
use crate::models::qwen3_5::dflash2_decode::live_window_prefix;
use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, ToolDefinition};

/// Token ids of unrecorded requests start here, above every real vocabulary,
/// so they share no block with a rendered prompt or with each other.
const UNRECORDED_TOKEN_BASE: u32 = 0x8000_0000;

#[derive(Deserialize)]
pub struct RecordedRequest {
    pub label: String,
    /// `None` for a request whose content was not recorded, such as a side
    /// call the agent made outside its sessions.
    pub session: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// The recorded prompt length of an unrecorded request.
    #[serde(default)]
    pub prompt_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ReplayedTurn {
    pub label: String,
    pub session: Option<String>,
    pub prompt_tokens: usize,
    /// Rows the turn generated, read from the next prompt of its session.
    /// `None` when that prompt does not extend this one or the session ends.
    pub generated_tokens: Option<usize>,
    pub gen_suffix_len: usize,
    pub replay_point: Option<usize>,
    pub hit_kind: &'static str,
    pub hit_tokens: usize,
    pub prefilled_tokens: usize,
    /// `none` when the turn resumed at its ideal, `extra_request` for an
    /// unrecorded request.
    pub miss_reason: String,
    pub shadow_hit: usize,
    pub ideal_tokens: usize,
    pub ideal_paged_tokens: usize,
    pub diverge_at: i64,
}

struct Rendered {
    prompt: Vec<u32>,
    gen_suffix_len: usize,
    replay_point: Option<usize>,
}

fn render(
    tokenizer: &Qwen3Tokenizer,
    block_size: u32,
    request: &RecordedRequest,
    unrecorded_base: &mut u32,
) -> Result<Rendered, String> {
    if request.session.is_none() {
        let len = request
            .prompt_tokens
            .filter(|&len| len > 0)
            .ok_or_else(|| {
                format!(
                    "{}: an unrecorded request needs prompt_tokens",
                    request.label
                )
            })?;
        let start = *unrecorded_base;
        *unrecorded_base = u32::try_from(len)
            .ok()
            .and_then(|len| start.checked_add(len))
            .ok_or_else(|| format!("{}: unrecorded token ids overflow", request.label))?;
        return Ok(Rendered {
            prompt: (start..*unrecorded_base).collect(),
            gen_suffix_len: 0,
            replay_point: None,
        });
    }
    let config = ChatConfig {
        tools: request.tools.clone(),
        reasoning_effort: request.reasoning_effort.clone(),
        ..Default::default()
    };
    let prompt = tokenizer
        .apply_chat_template_with_config(
            &request.messages,
            true,
            &config,
            SESSION_PRESERVE_THINKING,
        )
        .map_err(|error| format!("{}: render: {error}", request.label))?;
    let suffix = tokenizer
        .generation_suffix(&config)
        .map_err(|error| format!("{}: generation suffix: {error}", request.label))?;
    let replay_point = suffix
        .as_deref()
        .and_then(|suffix| replay_point(&prompt, suffix, block_size as usize));
    Ok(Rendered {
        gen_suffix_len: suffix.map_or(0, |suffix| suffix.len()),
        replay_point,
        prompt,
    })
}

/// The live history a turn leaves behind: its prompt plus the rows it
/// generated, minus the final `<|im_end|>`, which the DFlash2 turn samples but
/// never forwards. The generated rows are the ones the session's next prompt
/// renders after this prompt.
fn history(prompt: &[u32], next: Option<&[u32]>, im_end: Option<u32>) -> (Vec<u32>, Option<usize>) {
    let generated = next
        .filter(|next| next.starts_with(prompt))
        .and_then(|next| {
            next[prompt.len()..]
                .iter()
                .position(|&token| Some(token) == im_end)
                .map(|end| &next[prompt.len()..prompt.len() + end])
        });
    match generated {
        Some(rows) => ([prompt, rows].concat(), Some(rows.len())),
        None => (prompt.to_vec(), None),
    }
}

/// Replays `requests` in order. Every request is a DFlash2 turn on one model.
pub fn replay(
    tokenizer: &Qwen3Tokenizer,
    block_size: u32,
    requests: &[RecordedRequest],
) -> Result<Vec<ReplayedTurn>, String> {
    let mut unrecorded_base = UNRECORDED_TOKEN_BASE;
    let rendered = requests
        .iter()
        .map(|request| render(tokenizer, block_size, request, &mut unrecorded_base))
        .collect::<Result<Vec<_>, _>>()?;
    let im_end = tokenizer.im_end_id();
    let histories: Vec<(Vec<u32>, Option<usize>)> = requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            let next = request.session.as_ref().and_then(|session| {
                requests[index + 1..]
                    .iter()
                    .position(|later| later.session.as_ref() == Some(session))
                    .map(|offset| rendered[index + 1 + offset].prompt.as_slice())
            });
            history(&rendered[index].prompt, next, im_end)
        })
        .collect();

    let paged_turns: Vec<trace::ReplayTurn<'_>> = rendered
        .iter()
        .zip(&histories)
        .map(|(turn, (history, _))| trace::ReplayTurn {
            prompt: &turn.prompt,
            history,
            replay_point: turn.replay_point,
            is_dflash2: true,
        })
        .collect();
    let ideal_paged = trace::ideal_prefill(StateDesign::Paged, block_size, &paged_turns);

    let mut ledger = TurnLedger::new(block_size);
    let mut live: &[u32] = &[];
    let mut turns = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        let Rendered {
            prompt,
            gen_suffix_len,
            replay_point,
        } = &rendered[index];
        let cached = live_window_prefix(prompt, live);
        let used = if cached > 0 {
            Hit {
                kind: HitKind::Live,
                tokens: cached,
            }
        } else {
            Hit::MISS
        };
        let classification = ledger.classify(&TurnObservation {
            prompt,
            reuse_cache: true,
            used,
            live: (!live.is_empty()).then_some(LiveView {
                tokens: live,
                authoritative: true,
                context_matches: true,
            }),
            dropped: None,
            store: StoreView::default(),
        });
        let (history, generated_tokens) = &histories[index];
        let resume = classification.ideal.max(cached);
        let captures =
            trace::design_captures(StateDesign::Flat, block_size, *replay_point, resume, resume);
        ledger.record(LedgerTurn {
            history,
            prompt_len: prompt.len(),
            replay_point: *replay_point,
            is_dflash2: true,
            captures: &captures,
        });
        live = history;

        let miss_reason = if request.session.is_none() {
            String::from("extra_request")
        } else {
            classification
                .miss_reason
                .map_or_else(|| String::from("none"), |reason| reason.to_string())
        };
        turns.push(ReplayedTurn {
            label: request.label.clone(),
            session: request.session.clone(),
            prompt_tokens: prompt.len(),
            generated_tokens: *generated_tokens,
            gen_suffix_len: *gen_suffix_len,
            replay_point: *replay_point,
            hit_kind: classification.hit.kind.as_str(),
            hit_tokens: classification.hit.tokens,
            prefilled_tokens: prompt.len() - classification.hit.tokens,
            miss_reason,
            shadow_hit: classification.shadow_hit,
            ideal_tokens: classification.ideal,
            ideal_paged_tokens: ideal_paged[index].resume,
            diverge_at: classification.diverge.index,
        });
    }
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IM_END: u32 = 7;

    #[test]
    fn history_keeps_the_generated_rows_before_im_end() {
        let prompt = [1, 2, 3];
        let next = [1, 2, 3, 40, 41, IM_END, 9, 1, 2];
        let (history, generated) = history(&prompt, Some(&next), Some(IM_END));
        assert_eq!(history, [1, 2, 3, 40, 41]);
        assert_eq!(generated, Some(2));
    }

    #[test]
    fn history_is_the_prompt_when_the_next_prompt_does_not_extend_it() {
        let prompt = [1, 2, 3];
        for next in [
            None,
            Some(&[1, 2, 4, 40, IM_END][..]),
            Some(&[1, 2, 3, 40, 41][..]),
        ] {
            let (history, generated) = history(&prompt, next, Some(IM_END));
            assert_eq!(history, prompt);
            assert_eq!(generated, None);
        }
    }

    fn unrecorded(label: &str, prompt_tokens: usize) -> RecordedRequest {
        RecordedRequest {
            label: label.to_string(),
            session: None,
            messages: Vec::new(),
            tools: None,
            reasoning_effort: None,
            prompt_tokens: Some(prompt_tokens),
        }
    }

    #[test]
    fn unrecorded_requests_share_no_block() {
        let tokenizer = crate::tokenizer::audio_test_tokenizer();
        let requests = [unrecorded("x", 40), unrecorded("y", 40)];
        let turns = replay(&tokenizer, 16, &requests).unwrap();
        assert_eq!(turns.len(), 2);
        for (turn, label) in turns.iter().zip(["x", "y"]) {
            assert_eq!(turn.label, label);
            assert_eq!(turn.miss_reason, "extra_request");
            assert_eq!((turn.prompt_tokens, turn.prefilled_tokens), (40, 40));
            assert_eq!((turn.hit_tokens, turn.shadow_hit), (0, 0));
        }
        assert!(replay(&tokenizer, 16, &[unrecorded("z", 0)]).is_err());
    }
}
