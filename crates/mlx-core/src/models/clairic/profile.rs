//! Model profile feature flags distinguishing `ClairicFlash` (`4096`) and `Clairic27B` (`5120`).
//!
//! Mirrors the class attributes on upstream `ClairicFlashModel` (`backend.py:400-422`)
//! and `Clairic27BModel` (`large.py:14-29`).

use super::head::ClairicHeadConfig;

/// Backbone hidden size identifying the Clairic 27B checkpoint (`Clairic27BModel`).
pub const LARGE_27B_HIDDEN_SIZE: i64 = 5120;

/// Runtime behavioral profile selected from `ClairicHeadConfig::hidden_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClairicProfile {
    /// Clairic Flash (9B backbone, `hidden_size == 4096`).
    Flash,
    /// Clairic 27B (`hidden_size == 5120`).
    Large27B,
}

impl ClairicProfile {
    /// Select the profile from a backbone/head `hidden_size`.
    pub const fn from_hidden_size(hidden_size: i64) -> Self {
        if hidden_size == LARGE_27B_HIDDEN_SIZE {
            Self::Large27B
        } else {
            Self::Flash
        }
    }

    /// Select the profile from a parsed `clair_head_config.json`.
    pub const fn from_head_config(config: &ClairicHeadConfig) -> Self {
        Self::from_hidden_size(config.hidden_size)
    }

    pub const fn is_flash(self) -> bool {
        matches!(self, Self::Flash)
    }

    pub const fn is_large_27b(self) -> bool {
        matches!(self, Self::Large27B)
    }

    /// `RELEASE_MODEL_ID` (`backend.py:401`, `large.py:15`).
    pub const fn release_model_id(self) -> &'static str {
        match self {
            Self::Flash => "clairic-v2",
            Self::Large27B => "clairic-27b",
        }
    }

    /// Default public `model_id` (`settings.py:14`, `large.py:54`).
    pub const fn public_model_id(self) -> &'static str {
        match self {
            Self::Flash => "clairic-flash",
            Self::Large27B => "clairic-27b",
        }
    }

    /// `COMPACT_LIST_PROMPTS` (`backend.py:402`, `large.py:16`).
    pub const fn compact_list_prompts(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `DATE_ROLE_GUIDANCE` (`backend.py:403`, `large.py:17`).
    pub const fn date_role_guidance(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `SOURCE_ROW_ELIGIBILITY` (`backend.py:405`, `large.py:18`).
    pub const fn source_row_eligibility(self) -> bool {
        matches!(self, Self::Large27B)
    }

    /// `FINANCIAL_SERVICE_GUIDANCE` (`backend.py:407`, `large.py:19`).
    pub const fn financial_service_guidance(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `FINANCIAL_FEE_GUIDANCE` (`backend.py:408`, `large.py:20`).
    pub const fn financial_fee_guidance(self) -> bool {
        matches!(self, Self::Large27B)
    }

    /// `FOCUS_SPARSE_IMAGES` (`backend.py:411`, `large.py:22`).
    pub const fn focus_sparse_images(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `FOCUS_IMAGE_PARTY_NAMES` (`backend.py:413`, `large.py:24`).
    pub const fn focus_image_party_names(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `LITERAL_IMAGE_PAYMENT_ROWS` (`backend.py:414`, `large.py:25`).
    pub const fn literal_image_payment_rows(self) -> bool {
        matches!(self, Self::Large27B)
    }

    /// `BIND_IMAGE_FINANCIAL_LINES` (`backend.py:415`, `large.py:26`).
    pub const fn bind_image_financial_lines(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `PRESERVE_IMAGE_ENTITY_NAMES` (`backend.py:416`, `large.py:23`).
    pub const fn preserve_image_entity_names(self) -> bool {
        matches!(self, Self::Large27B)
    }

    /// `BIND_IMAGE_RAGGED_ROWS` (`backend.py:418`, `large.py:28`).
    pub const fn bind_image_ragged_rows(self) -> bool {
        matches!(self, Self::Flash)
    }

    /// `IMAGE_TABLE_ROW_SCALE` (`backend.py:419`, `large.py:27`).
    pub const fn image_table_row_scale(self) -> usize {
        match self {
            Self::Flash => 2,
            Self::Large27B => 4,
        }
    }

    // ---- Shared flags enabled across both profiles (`backend.py:404-417`).

    /// `RECOVER_NULLS` (`backend.py:404`).
    pub const fn recover_nulls(self) -> bool {
        true
    }

    /// `LOCAL_SOURCE_SCOPE` (`backend.py:406`).
    pub const fn local_source_scope(self) -> bool {
        true
    }

    /// `EXPLICIT_TEMPORAL_READS` (`backend.py:409`).
    pub const fn explicit_temporal_reads(self) -> bool {
        true
    }

    /// `SPELL_VISUAL_PO_REFERENCES` (`backend.py:410`, `large.py:21`).
    pub const fn spell_visual_po_references(self) -> bool {
        true
    }

    /// `FOCUS_IMAGE_IDENTIFIERS` (`backend.py:412`).
    pub const fn focus_image_identifiers(self) -> bool {
        true
    }

    /// `BIND_IMAGE_TABLE_ROWS` (`backend.py:417`).
    pub const fn bind_image_table_rows(self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_head_config(hidden_size: i64) -> ClairicHeadConfig {
        ClairicHeadConfig {
            hidden_size,
            bins: 256,
            pointer_width: 256,
            readout: "lexical".to_string(),
            route_width: 1024,
            route_layers: 2,
            route_heads: 8,
            slot_rank: 32,
            estimate_lexical: true,
            span_trim: true,
            loop_feedback: false,
            answer_slots: true,
            answer_slot_routing: true,
            number_groups: true,
            number_pointer: true,
        }
    }

    #[test]
    fn profile_selection_from_hidden_size() {
        assert_eq!(
            ClairicProfile::from_hidden_size(4096),
            ClairicProfile::Flash
        );
        assert_eq!(
            ClairicProfile::from_hidden_size(5120),
            ClairicProfile::Large27B
        );
        assert_eq!(
            ClairicProfile::from_head_config(&sample_head_config(4096)),
            ClairicProfile::Flash
        );
        assert_eq!(
            ClairicProfile::from_head_config(&sample_head_config(5120)),
            ClairicProfile::Large27B
        );
    }

    #[test]
    fn flash_and_27b_differing_flags_match_upstream() {
        let flash = ClairicProfile::Flash;
        let large = ClairicProfile::Large27B;

        assert!(flash.is_flash());
        assert!(!flash.is_large_27b());
        assert!(!large.is_flash());
        assert!(large.is_large_27b());

        assert_eq!(flash.release_model_id(), "clairic-v2");
        assert_eq!(large.release_model_id(), "clairic-27b");
        assert_eq!(flash.public_model_id(), "clairic-flash");
        assert_eq!(large.public_model_id(), "clairic-27b");

        assert!(flash.compact_list_prompts());
        assert!(!large.compact_list_prompts());

        assert!(flash.date_role_guidance());
        assert!(!large.date_role_guidance());

        assert!(!flash.source_row_eligibility());
        assert!(large.source_row_eligibility());

        assert!(flash.financial_service_guidance());
        assert!(!large.financial_service_guidance());

        assert!(!flash.financial_fee_guidance());
        assert!(large.financial_fee_guidance());

        assert!(flash.focus_sparse_images());
        assert!(!large.focus_sparse_images());

        assert!(flash.focus_image_party_names());
        assert!(!large.focus_image_party_names());

        assert!(!flash.literal_image_payment_rows());
        assert!(large.literal_image_payment_rows());

        assert!(flash.bind_image_financial_lines());
        assert!(!large.bind_image_financial_lines());

        assert!(!flash.preserve_image_entity_names());
        assert!(large.preserve_image_entity_names());

        assert!(flash.bind_image_ragged_rows());
        assert!(!large.bind_image_ragged_rows());

        assert_eq!(flash.image_table_row_scale(), 2);
        assert_eq!(large.image_table_row_scale(), 4);

        for profile in [flash, large] {
            assert!(profile.recover_nulls());
            assert!(profile.local_source_scope());
            assert!(profile.explicit_temporal_reads());
            assert!(profile.spell_visual_po_references());
            assert!(profile.focus_image_identifiers());
            assert!(profile.bind_image_table_rows());
        }
    }
}
