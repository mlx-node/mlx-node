//! Load-time inference capabilities and request-time turn planning.
//!
//! Models declare the features wired at load time in an [`ExecutionPlan`].
//! The session engine resolves that immutable description with one request's
//! inputs into a compact [`TurnPlan`] exactly once, before prefill. The decode
//! hot loop never probes model capabilities.
//!
//! The dimensions deliberately compose:
//!
//! - image and audio inputs may coexist;
//! - paged attention describes eligible attention KV state, not every cache a
//!   hybrid model owns (sliding, convolutional, and recurrent state remains
//!   model-owned);
//! - speculative decoding decorates the target execution and may opt into
//!   paged attention, current-turn media, and live-session media independently.

/// Media kinds a loaded model (or speculative decoder) accepts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MediaCapabilities {
    pub images: bool,
    pub audio: bool,
}

impl MediaCapabilities {
    pub const NONE: Self = Self {
        images: false,
        audio: false,
    };
    pub const IMAGES: Self = Self {
        images: true,
        audio: false,
    };
    #[cfg(test)]
    pub const AUDIO: Self = Self {
        images: false,
        audio: true,
    };
    #[cfg(test)]
    pub const IMAGES_AND_AUDIO: Self = Self {
        images: true,
        audio: true,
    };

    pub const fn supports(self, requested: Self) -> bool {
        (!requested.images || self.images) && (!requested.audio || self.audio)
    }

    pub const fn union(self, other: Self) -> Self {
        Self {
            images: self.images || other.images,
            audio: self.audio || other.audio,
        }
    }

    /// Set difference (`self \\ other`) by media kind.
    pub const fn difference(self, other: Self) -> Self {
        Self {
            images: self.images && !other.images,
            audio: self.audio && !other.audio,
        }
    }

    pub const fn is_empty(self) -> bool {
        !self.images && !self.audio
    }
}

/// Load-time media admission policy for a target model.
///
/// `available` is the truthful set of encoders wired into this loaded model.
/// `backend_validated` is deliberately narrower in meaning: the engine may
/// admit those inputs only so the family's multimodal handler can preserve a
/// more specific compatibility error (for example, "vision encoder not
/// loaded"). It must never be treated as actual media capability by cache or
/// speculative-decoder admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MediaPlan {
    pub available: MediaCapabilities,
    pub backend_validated: MediaCapabilities,
}

impl MediaPlan {
    pub const NONE: Self = Self {
        available: MediaCapabilities::NONE,
        backend_validated: MediaCapabilities::NONE,
    };

    /// Construct a truthful plan from the media actually wired into the
    /// target and the broader set the engine should admit. Kinds already
    /// available are removed from `backend_validated`, so the two fields keep
    /// distinct meanings even when `admitted` includes `available`.
    pub const fn with_backend_validation(
        available: MediaCapabilities,
        admitted: MediaCapabilities,
    ) -> Self {
        Self {
            available,
            backend_validated: admitted.difference(available),
        }
    }

    /// Media accepted at the engine's pre-render boundary. Inputs admitted
    /// only through `backend_validated` still route to the multimodal handler,
    /// which owns the family-specific validation and error.
    pub const fn admitted(self) -> MediaCapabilities {
        self.available.union(self.backend_validated)
    }
}

/// Raw media carried by one turn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MediaInputs<'a> {
    pub images: &'a [Vec<u8>],
    pub audio: &'a [Vec<u8>],
}

impl MediaInputs<'_> {
    pub const fn capabilities(self) -> MediaCapabilities {
        MediaCapabilities {
            images: !self.images.is_empty(),
            audio: !self.audio.is_empty(),
        }
    }

    pub const fn is_empty(self) -> bool {
        self.images.is_empty() && self.audio.is_empty()
    }
}

/// Load-time paged-attention admission policy.
///
/// This applies only to attention layers represented by the model's paged
/// adapter. Hybrid recurrent/sliding/conv state may coexist with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PagedAttentionPlan {
    /// Whether a live paged request can serve a raw session delta.
    pub supports_delta: bool,
}

/// Speculative implementation attached to a loaded target model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpeculativeKind {
    /// In-checkpoint multi-token-prediction head.
    NativeMtp,
    /// Separately loaded assistant/draft checkpoint.
    DraftModel,
}

/// Execution lane for an admitted speculative turn.
///
/// Computed only by [`SpeculativePlan::lane`] and consulted by every
/// admission gate, so de-barriering speculation is a one-property flip
/// rather than a per-gate hunt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpeculativeLane {
    /// Whole-turn exclusive execution through the ordered command lane.
    Barrier,
    /// Admission into the continuous-batching scheduler alongside plain
    /// autoregressive rows.
    Scheduled,
}

/// Drafted rows one speculative verify cycle carries, tagged by the
/// quantity's origin so a native MTP chain depth and an external draft
/// block size can never be conflated: they are different load-time numbers
/// that only coincidentally share the `+ 1` anchor arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum SpeculativeDraftWidth {
    /// Native MTP chain depth `D` — `D` chained head steps per cycle.
    Depth(usize),
    /// External draft-model block size `L` — one drafted block per cycle
    /// (gemma4 DSpark block size, muse DFlash mask width).
    DraftBlockSize(usize),
}

/// Load-time admission policy for a speculative decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpeculativePlan {
    pub kind: SpeculativeKind,
    /// Media supplied on this turn that the proposer can consume.
    pub supported_input_media: MediaCapabilities,
    /// Media already represented by the live target session that the
    /// proposer can continue against. This is intentionally independent of
    /// current-turn input: a text delta may still sit on an image/audio KV
    /// prefix.
    pub supported_context_media: MediaCapabilities,
    /// Whether proposal/verification is implemented against the target's
    /// paged attention state.
    pub supports_paged_attention: bool,
}

impl SpeculativePlan {
    /// The one Barrier-vs-Scheduled decision every admission gate consults.
    ///
    /// Always [`SpeculativeLane::Barrier`] today. Flipping any configuration
    /// to `Scheduled` (planned behind `MLX_MTP_SCHEDULED=1`) requires the
    /// scheduler to reserve [`Self::lookahead_rows`] slots per verify cycle,
    /// roll back rejected rows per sequence, and key drafter state by owner;
    /// none of that machinery exists yet. The gemma4/muse `execute_barrier`
    /// flat-lane owner installs are cache-layout decisions (which cache
    /// representation a live owner occupies), not lane decisions, and remain
    /// family-owned regardless of this value.
    pub const fn lane(self) -> SpeculativeLane {
        SpeculativeLane::Barrier
    }

    /// Rows one speculative verify cycle appends to attention state: the
    /// anchor row plus `depth` draft rows.
    ///
    /// The single source for the speculative slot margin (vLLM's
    /// `num_lookahead_tokens`): every reserver reads this; none re-derives
    /// `depth + 1` locally.
    ///
    /// Depth-parameterized form for the engine MTP loop
    /// (`engine::mtp_turn::turn_lookahead_rows`), whose per-cycle quantity
    /// is a chain depth. Reservers sized by an external draft block cannot
    /// use it — this form has no way to know a block size — and go through
    /// [`Self::lookahead_rows_for`] with
    /// [`SpeculativeDraftWidth::DraftBlockSize`] instead.
    pub const fn lookahead_rows(self, depth: usize) -> usize {
        depth + 1
    }

    /// [`Self::lookahead_rows`], per kind: the TARGET-side verify rows one
    /// cycle appends — [`SpeculativeKind::NativeMtp`] writes `depth + 1`
    /// (anchor + `D` chained drafts), [`SpeculativeKind::DraftModel`]
    /// (DSpark, DFlash, assistant) writes `draft_block_size + 1` (anchor +
    /// `L` block drafts). The width must be the plan kind's own quantity;
    /// a mismatched pairing is a reservation-accounting bug at the call
    /// site and panics rather than sizing the pool from the wrong number.
    ///
    /// Divergence from vLLM, deliberate: vLLM's `num_lookahead_tokens`
    /// counts drafter-WRITTEN rows (DFlash reserves `K + 1` for the
    /// drafter, `config/vllm.py:622-646`) because its drafter KV is
    /// pool-resident; our drafter KV is off-pool (non-goal N1 — it dies
    /// with its owner and never registers), so the reservation counts the
    /// target's verify rows only. vLLM's harmless over-reserve for its
    /// read-only gemma4_mtp drafter is likewise not copied.
    ///
    /// No production reserver consumes it yet — the external-draft paged
    /// steppers (`SpecPagedCache::reserve_lookahead` callers) do; until
    /// then it exists so their reservations are sized by the plan property
    /// rather than a re-derived local formula (I1).
    #[allow(dead_code)]
    pub const fn lookahead_rows_for(self, width: SpeculativeDraftWidth) -> usize {
        match (self.kind, width) {
            (SpeculativeKind::NativeMtp, SpeculativeDraftWidth::Depth(depth)) => depth + 1,
            (SpeculativeKind::DraftModel, SpeculativeDraftWidth::DraftBlockSize(block)) => {
                block + 1
            }
            (SpeculativeKind::NativeMtp, SpeculativeDraftWidth::DraftBlockSize(_))
            | (SpeculativeKind::DraftModel, SpeculativeDraftWidth::Depth(_)) => {
                panic!("speculative draft width kind does not match the plan's kind")
            }
        }
    }
}

/// Immutable inference features resolved when a model is loaded.
///
/// Concrete model structs currently expose this by value from immutable
/// load-time fields (`paged_adapter`, MTP/draft presence, vision/audio
/// configuration). Keeping the description data-only makes it suitable for a
/// future model registry without coupling the engine to family types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExecutionPlan {
    pub media: MediaPlan,
    pub paged_attention: Option<PagedAttentionPlan>,
    pub speculative: Option<SpeculativePlan>,
}

impl ExecutionPlan {
    pub const TEXT_ONLY: Self = Self {
        media: MediaPlan::NONE,
        paged_attention: None,
        speculative: None,
    };
}

/// Request facts used to resolve a [`TurnPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TurnRequest {
    pub is_delta: bool,
    /// Media bytes carried by the current turn.
    pub input_media: MediaCapabilities,
    /// Media already encoded in the live session prefix. Fresh turns always
    /// resolve this as [`MediaCapabilities::NONE`].
    pub context_media: MediaCapabilities,
    pub speculative_requested: bool,
}

/// Decoder selected for one turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecoderPlan {
    Autoregressive,
    Speculative(SpeculativeKind),
}

/// Compact request-time plan consumed by the session boundary and specialized
/// whole-turn handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TurnPlan {
    pub is_delta: bool,
    pub input_media: MediaCapabilities,
    pub context_media: MediaCapabilities,
    pub use_paged_attention: bool,
    pub decoder: DecoderPlan,
}

/// Engine handler selected from the composable turn dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnPath {
    Generic,
    Paged,
    Speculative,
    Multimodal,
}

impl TurnPlan {
    /// Resolve load-time capabilities against one request.
    ///
    /// Media outside the target's admission set is rejected by the session
    /// before prompt rendering. Backend-validated media still reaches the
    /// family handler for its specific error. This function decides which
    /// compatible optional features can participate; an incompatible
    /// speculative combination falls back to the exact target autoregressive
    /// path and never drops the turn.
    pub const fn resolve(execution: ExecutionPlan, request: TurnRequest) -> Self {
        let use_paged_attention = match execution.paged_attention {
            Some(paged) => !request.is_delta || paged.supports_delta,
            None => false,
        };

        let decoder = if request.speculative_requested {
            match execution.speculative {
                Some(speculative)
                    if speculative
                        .supported_input_media
                        .supports(request.input_media)
                        && speculative
                            .supported_context_media
                            .supports(request.context_media)
                        && (!use_paged_attention || speculative.supports_paged_attention) =>
                {
                    DecoderPlan::Speculative(speculative.kind)
                }
                _ => DecoderPlan::Autoregressive,
            }
        } else {
            DecoderPlan::Autoregressive
        };

        Self {
            is_delta: request.is_delta,
            input_media: request.input_media,
            context_media: request.context_media,
            use_paged_attention,
            decoder,
        }
    }

    /// Derive the outer whole-turn handler without collapsing the plan's
    /// independent dimensions. Multimodal preparation owns its prompt merge;
    /// a paged handler owns paged AR *or* paged speculative execution; a flat
    /// speculative handler owns the remaining speculative case.
    pub const fn path(self) -> TurnPath {
        if !self.input_media.is_empty() {
            TurnPath::Multimodal
        } else if self.use_paged_attention {
            TurnPath::Paged
        } else if matches!(self.decoder, DecoderPlan::Speculative(_)) {
            TurnPath::Speculative
        } else {
            TurnPath::Generic
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGED: PagedAttentionPlan = PagedAttentionPlan {
        supports_delta: true,
    };
    const MTP_PAGED: SpeculativePlan = SpeculativePlan {
        kind: SpeculativeKind::NativeMtp,
        supported_input_media: MediaCapabilities::NONE,
        supported_context_media: MediaCapabilities::NONE,
        supports_paged_attention: true,
    };

    fn request(
        is_delta: bool,
        input_media: MediaCapabilities,
        context_media: MediaCapabilities,
        speculative_requested: bool,
    ) -> TurnRequest {
        TurnRequest {
            is_delta,
            input_media,
            context_media,
            speculative_requested,
        }
    }

    #[test]
    fn all_current_speculative_plans_are_barrier() {
        let media_shapes = [
            MediaCapabilities::NONE,
            MediaCapabilities::IMAGES,
            MediaCapabilities::AUDIO,
            MediaCapabilities::IMAGES_AND_AUDIO,
        ];
        for kind in [SpeculativeKind::NativeMtp, SpeculativeKind::DraftModel] {
            for supported_input_media in media_shapes {
                for supported_context_media in media_shapes {
                    for supports_paged_attention in [false, true] {
                        let plan = SpeculativePlan {
                            kind,
                            supported_input_media,
                            supported_context_media,
                            supports_paged_attention,
                        };
                        assert_eq!(
                            plan.lane(),
                            SpeculativeLane::Barrier,
                            "no speculative configuration may reach the scheduled lane \
                             before its admission accounting exists: {plan:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn lookahead_rows_counts_anchor_plus_drafts() {
        assert_eq!(MTP_PAGED.lookahead_rows(1), 2);
        assert_eq!(MTP_PAGED.lookahead_rows(4), 5);
    }

    #[test]
    fn lookahead_rows_per_kind() {
        let native = MTP_PAGED;
        let draft = SpeculativePlan {
            kind: SpeculativeKind::DraftModel,
            ..MTP_PAGED
        };

        // NativeMtp: anchor + D chained drafts.
        assert_eq!(
            native.lookahead_rows_for(SpeculativeDraftWidth::Depth(1)),
            2
        );
        assert_eq!(
            native.lookahead_rows_for(SpeculativeDraftWidth::Depth(4)),
            5
        );

        // DraftModel (DSpark/DFlash/assistant): anchor + L block drafts.
        assert_eq!(
            draft.lookahead_rows_for(SpeculativeDraftWidth::DraftBlockSize(7)),
            8
        );
        assert_eq!(
            draft.lookahead_rows_for(SpeculativeDraftWidth::DraftBlockSize(4)),
            5
        );

        // The two quantities are not interchangeable: sizing a kind's
        // reservation from the other kind's number must refuse, not return
        // a uniformly-`depth + 1` answer.
        assert!(
            std::panic::catch_unwind(
                || native.lookahead_rows_for(SpeculativeDraftWidth::DraftBlockSize(7))
            )
            .is_err(),
            "a NativeMtp plan accepted an external draft block size"
        );
        assert!(
            std::panic::catch_unwind(|| draft.lookahead_rows_for(SpeculativeDraftWidth::Depth(1)))
                .is_err(),
            "a DraftModel plan accepted a native MTP chain depth"
        );
    }

    #[test]
    fn media_capabilities_are_compositional() {
        assert!(MediaCapabilities::IMAGES_AND_AUDIO.supports(MediaCapabilities::IMAGES));
        assert!(MediaCapabilities::IMAGES_AND_AUDIO.supports(MediaCapabilities::IMAGES_AND_AUDIO));
        assert!(!MediaCapabilities::IMAGES.supports(MediaCapabilities {
            images: true,
            audio: true,
        }));
        assert_eq!(
            MediaCapabilities::IMAGES.union(MediaCapabilities {
                images: false,
                audio: true,
            }),
            MediaCapabilities::IMAGES_AND_AUDIO,
        );
    }

    #[test]
    fn backend_validated_media_is_admitted_without_claiming_availability() {
        let media =
            MediaPlan::with_backend_validation(MediaCapabilities::NONE, MediaCapabilities::IMAGES);
        assert!(!media.available.images);
        assert!(media.admitted().images);
        assert!(!media.admitted().audio);

        let audio_available = MediaPlan::with_backend_validation(
            MediaCapabilities::AUDIO,
            MediaCapabilities::IMAGES_AND_AUDIO,
        );
        assert_eq!(audio_available.available, MediaCapabilities::AUDIO);
        assert_eq!(audio_available.backend_validated, MediaCapabilities::IMAGES,);
    }

    #[test]
    fn delta_uses_paged_attention_only_when_admitted() {
        let mut execution = ExecutionPlan::TEXT_ONLY;
        execution.paged_attention = Some(PagedAttentionPlan {
            supports_delta: false,
        });
        let delta = TurnPlan::resolve(
            execution,
            request(
                true,
                MediaCapabilities::NONE,
                MediaCapabilities::NONE,
                false,
            ),
        );
        assert!(!delta.use_paged_attention);
        assert_eq!(delta.path(), TurnPath::Generic);

        let fresh = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::NONE,
                MediaCapabilities::NONE,
                false,
            ),
        );
        assert!(fresh.use_paged_attention);
        assert_eq!(fresh.path(), TurnPath::Paged);
    }

    #[test]
    fn paged_attention_and_mtp_compose_when_supported() {
        let execution = ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: Some(PAGED),
            speculative: Some(MTP_PAGED),
        };
        let plan = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::NONE,
                MediaCapabilities::NONE,
                true,
            ),
        );
        assert!(plan.use_paged_attention);
        assert_eq!(
            plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp)
        );
        assert_eq!(plan.path(), TurnPath::Paged);
    }

    #[test]
    fn incompatible_paged_speculation_falls_back_to_target_ar() {
        let execution = ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: Some(PAGED),
            speculative: Some(SpeculativePlan {
                supports_paged_attention: false,
                ..MTP_PAGED
            }),
        };
        let plan = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::NONE,
                MediaCapabilities::NONE,
                true,
            ),
        );
        assert_eq!(plan.decoder, DecoderPlan::Autoregressive);
        assert_eq!(plan.path(), TurnPath::Paged);
    }

    #[test]
    fn incompatible_media_speculation_keeps_multimodal_target_path() {
        let execution = ExecutionPlan {
            media: MediaPlan {
                available: MediaCapabilities::IMAGES_AND_AUDIO,
                backend_validated: MediaCapabilities::NONE,
            },
            paged_attention: None,
            speculative: Some(MTP_PAGED),
        };
        let plan = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::IMAGES_AND_AUDIO,
                MediaCapabilities::NONE,
                true,
            ),
        );
        assert_eq!(plan.decoder, DecoderPlan::Autoregressive);
        assert_eq!(plan.path(), TurnPath::Multimodal);
    }

    #[test]
    fn compatible_flat_speculation_selects_speculative_handler() {
        let execution = ExecutionPlan {
            media: MediaPlan::NONE,
            paged_attention: None,
            speculative: Some(MTP_PAGED),
        };
        let plan = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::NONE,
                MediaCapabilities::NONE,
                true,
            ),
        );
        assert_eq!(plan.path(), TurnPath::Speculative);
    }

    #[test]
    fn current_input_and_live_context_gate_speculation_independently() {
        let execution = ExecutionPlan {
            media: MediaPlan {
                available: MediaCapabilities::IMAGES,
                backend_validated: MediaCapabilities::NONE,
            },
            paged_attention: None,
            speculative: Some(SpeculativePlan {
                supported_input_media: MediaCapabilities::NONE,
                supported_context_media: MediaCapabilities::IMAGES,
                ..MTP_PAGED
            }),
        };

        // A dense-like native MTP implementation may continue a text delta
        // over an image-bearing target prefix without consuming new images.
        let supported_context = TurnPlan::resolve(
            execution,
            request(
                true,
                MediaCapabilities::NONE,
                MediaCapabilities::IMAGES,
                true,
            ),
        );
        assert_eq!(
            supported_context.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
        );
        assert_eq!(supported_context.path(), TurnPath::Speculative);

        // The same proposer still cannot consume a newly supplied image.
        let unsupported_input = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::IMAGES,
                MediaCapabilities::NONE,
                true,
            ),
        );
        assert_eq!(unsupported_input.decoder, DecoderPlan::Autoregressive);
        assert_eq!(unsupported_input.path(), TurnPath::Multimodal);

        // Conversely, an input-compatible proposer must not speculate over a
        // live media prefix unless it explicitly supports that context.
        let input_only_execution = ExecutionPlan {
            speculative: Some(SpeculativePlan {
                supported_input_media: MediaCapabilities::IMAGES,
                supported_context_media: MediaCapabilities::NONE,
                ..MTP_PAGED
            }),
            ..execution
        };
        let unsupported_context = TurnPlan::resolve(
            input_only_execution,
            request(
                true,
                MediaCapabilities::NONE,
                MediaCapabilities::IMAGES,
                true,
            ),
        );
        assert_eq!(unsupported_context.decoder, DecoderPlan::Autoregressive);
        assert_eq!(unsupported_context.path(), TurnPath::Generic);
    }

    #[test]
    fn multimodal_outer_path_keeps_paged_and_decoder_composition() {
        let execution = ExecutionPlan {
            media: MediaPlan {
                available: MediaCapabilities::IMAGES,
                backend_validated: MediaCapabilities::NONE,
            },
            paged_attention: Some(PAGED),
            speculative: Some(SpeculativePlan {
                supported_input_media: MediaCapabilities::IMAGES,
                supported_context_media: MediaCapabilities::NONE,
                ..MTP_PAGED
            }),
        };
        let plan = TurnPlan::resolve(
            execution,
            request(
                false,
                MediaCapabilities::IMAGES,
                MediaCapabilities::NONE,
                true,
            ),
        );
        assert!(plan.use_paged_attention);
        assert_eq!(
            plan.decoder,
            DecoderPlan::Speculative(SpeculativeKind::NativeMtp),
        );
        assert_eq!(plan.path(), TurnPath::Multimodal);
    }
}
