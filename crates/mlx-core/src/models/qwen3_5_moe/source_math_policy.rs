//! Per-loaded-model diagnostic arithmetic. Raw operator tests keep their
//! legacy environment-driven FFI, while production never rereads it.
use napi::bindgen_prelude::*;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceMoePolicy {
    pub(crate) body: bool,
    pub(crate) residual: bool,
    pub(crate) routing: bool,
    pub(crate) mixer: bool,
    pub(crate) qk: bool,
    pub(crate) head: bool,
}
impl SourceMoePolicy {
    pub(crate) fn capture() -> Result<Self> {
        let body = std::env::var_os("MLX_SPLASH_MOE_SOURCE_MATH");
        let residual = std::env::var_os("MLX_SPLASH_MOE_SOURCE_RESIDUAL");
        let routing = std::env::var_os("MLX_SPLASH_MOE_SOURCE_ROUTING");
        let mixer = std::env::var_os("MLX_QWEN35_MOE_SOURCE_MIXER");
        let head = std::env::var_os("MLX_QWEN35_MOE_SOURCE_HEAD");
        Self::from_values(
            body.as_deref(),
            residual.as_deref(),
            routing.as_deref(),
            true,
        )
        .map(|policy| {
            policy
                .with_mixer_value(mixer.as_deref())
                .with_head_value(head.as_deref())
        })
    }
    pub(crate) fn from_values(
        body: Option<&std::ffi::OsStr>,
        residual: Option<&std::ffi::OsStr>,
        routing: Option<&std::ffi::OsStr>,
        residual_supported: bool,
    ) -> Result<Self> {
        let enabled = |v: Option<&std::ffi::OsStr>| v == Some(std::ffi::OsStr::new("1"));
        let residual = enabled(residual);
        let routing = enabled(routing);
        if residual && !residual_supported {
            return Err(Error::from_reason(
                "source MoE residual fusion is not linked in this build",
            ));
        }
        Ok(Self {
            body: enabled(body) || residual || routing,
            residual,
            routing,
            mixer: false,
            qk: false,
            head: false,
        })
    }
    // Mixer arithmetic is independent of expert-body/routing/norm diagnostics.
    pub(crate) fn with_mixer_value(mut self, value: Option<&std::ffi::OsStr>) -> Self {
        self.mixer = value == Some(std::ffi::OsStr::new("1"));
        self
    }

    pub(crate) fn validate_mixer_profile(
        self,
        config: &super::config::Qwen3_5MoeConfig,
    ) -> Result<()> {
        if self.mixer
            && (config.hidden_size != 2048
                || config.num_layers != 40
                || config.num_experts != 256
                || config.num_experts_per_tok != 8
                || config
                    .moe_intermediate_size
                    .unwrap_or(config.intermediate_size)
                    != 512
                || config.num_heads != 16
                || config.num_kv_heads != 2
                || config.head_dim != 256
                || config.linear_num_key_heads != 16
                || config.linear_num_value_heads != 32
                || config.linear_key_head_dim != 128
                || config.linear_value_head_dim != 128
                || config.full_attention_interval != 4)
        {
            return Err(Error::from_reason(
                "source mixer diagnostic requires measured H2048/L40/E256/top8/I512 attention geometry",
            ));
        }
        Ok(())
    }

    pub(crate) fn with_qk_requested(mut self, enabled: bool) -> Self {
        self.qk = enabled;
        self
    }

    pub(crate) fn with_head_value(mut self, value: Option<&std::ffi::OsStr>) -> Self {
        self.head = value == Some(std::ffi::OsStr::new("1"));
        self
    }

    pub(crate) fn validate_head_profile(
        self,
        config: &super::config::Qwen3_5MoeConfig,
    ) -> Result<()> {
        if self.head
            && (config.hidden_size != 2048
                || config.vocab_size != 248320
                || config.tie_word_embeddings
                || config.num_layers != 40
                || config.num_experts != 256
                || config.num_experts_per_tok != 8
                || config
                    .moe_intermediate_size
                    .unwrap_or(config.intermediate_size)
                    != 512)
        {
            return Err(Error::from_reason(
                "source head diagnostic requires untied H2048/V248320/L40/E256/top8/I512 target",
            ));
        }
        Ok(())
    }

    pub(crate) fn active(self) -> bool {
        self.body || self.residual || self.routing || self.mixer || self.qk || self.head
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    #[test]
    fn source_moe_policy_is_load_captured_and_unknown_values_are_off() {
        let one = Some(OsStr::new("1"));
        let captured = SourceMoePolicy::from_values(one, None, None, false).unwrap();
        assert!(captured.body && captured.active());
        for flag in [
            None,
            Some(OsStr::new("0")),
            Some(OsStr::new("true")),
            Some(OsStr::new("")),
        ] {
            assert_eq!(
                SourceMoePolicy::from_values(flag, None, None, false).unwrap(),
                SourceMoePolicy::default()
            );
        }
        assert!(captured.body);
        assert!(SourceMoePolicy::from_values(None, one, None, false).is_err());
        assert_eq!(
            SourceMoePolicy::from_values(None, one, None, true).unwrap(),
            SourceMoePolicy {
                body: true,
                residual: true,
                routing: false,
                mixer: false,
                qk: false,
                head: false
            }
        );
    }
    #[test]
    fn source_routing_implies_source_body_without_residual_fusion() {
        let one = Some(OsStr::new("1"));
        let captured = SourceMoePolicy::from_values(None, None, one, true).unwrap();
        assert_eq!(
            captured,
            SourceMoePolicy {
                body: true,
                residual: false,
                routing: true,
                mixer: false,
                qk: false,
                head: false
            }
        );
        assert!(captured.active());
        for value in [
            None,
            Some(OsStr::new("0")),
            Some(OsStr::new("true")),
            Some(OsStr::new("")),
        ] {
            assert_eq!(
                SourceMoePolicy::from_values(None, None, value, true).unwrap(),
                SourceMoePolicy::default()
            );
        }
        assert_eq!(
            captured,
            SourceMoePolicy::from_values(Some(OsStr::new("0")), None, one, true).unwrap()
        );
        let both = SourceMoePolicy::from_values(None, one, one, true).unwrap();
        assert!(both.body && both.residual && both.routing);
    }
    #[cfg(unix)]
    #[test]
    fn source_moe_policy_non_utf8_is_not_an_enable_request() {
        use std::os::unix::ffi::OsStrExt;
        assert!(
            !SourceMoePolicy::from_values(Some(OsStr::from_bytes(&[0xff])), None, None, false)
                .unwrap()
                .active()
        );
    }
    #[test]
    fn source_mixer_policy_is_independent_immutable_and_fail_closed() {
        let one = Some(OsStr::new("1"));
        let mixer = SourceMoePolicy::default().with_mixer_value(one);
        assert!(mixer.mixer && mixer.active());
        assert!(!mixer.body && !mixer.residual && !mixer.routing);
        for value in [
            None,
            Some(OsStr::new("0")),
            Some(OsStr::new("true")),
            Some(OsStr::new("")),
        ] {
            assert_eq!(
                SourceMoePolicy::default().with_mixer_value(value),
                SourceMoePolicy::default()
            );
            assert!(
                mixer.mixer,
                "changing another load must not affect captured policy"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(
                !SourceMoePolicy::default()
                    .with_mixer_value(Some(OsStr::from_bytes(&[0xff])))
                    .active()
            );
        }
        let combined = SourceMoePolicy::from_values(one, one, one, true)
            .unwrap()
            .with_mixer_value(one);
        assert!(combined.body && combined.residual && combined.routing && combined.mixer);
    }

    #[test]
    fn source_head_policy_is_independent_and_load_captured() {
        let head = SourceMoePolicy::default().with_head_value(Some(OsStr::new("1")));
        assert!(head.head && head.active());
        assert!(!head.body && !head.residual && !head.routing && !head.mixer && !head.qk);
        for value in [
            None,
            Some(OsStr::new("0")),
            Some(OsStr::new("true")),
            Some(OsStr::new("")),
        ] {
            assert_eq!(
                SourceMoePolicy::default().with_head_value(value),
                SourceMoePolicy::default()
            );
            assert!(head.head);
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(
                !SourceMoePolicy::default()
                    .with_head_value(Some(OsStr::from_bytes(&[0xff])))
                    .active()
            );
        }
        let all = SourceMoePolicy::from_values(
            Some(OsStr::new("1")),
            Some(OsStr::new("1")),
            Some(OsStr::new("1")),
            true,
        )
        .unwrap()
        .with_mixer_value(Some(OsStr::new("1")))
        .with_qk_requested(true)
        .with_head_value(Some(OsStr::new("1")));
        assert!(all.head && all.body && all.residual && all.routing && all.mixer && all.qk);
    }

    #[test]
    fn source_head_rejects_unmeasured_or_tied_target_before_loading() {
        let config: super::super::config::Qwen3_5MoeConfig =
            serde_json::from_value(serde_json::json!({
                "vocab_size":248320,"hidden_size":2048,"num_layers":40,"num_heads":16,
                "num_kv_heads":2,"intermediate_size":512,"rms_norm_eps":1e-6,"head_dim":256,
                "tie_word_embeddings":false,"max_position_embeddings":32768,"pad_token_id":0,
                "eos_token_id":1,"bos_token_id":2,"num_experts":256,"num_experts_per_tok":8
            }))
            .unwrap();
        let policy = SourceMoePolicy::default().with_head_value(Some(OsStr::new("1")));
        policy.validate_head_profile(&config).unwrap();
        for field in 0..7 {
            let mut bad = config.clone();
            match field {
                0 => bad.hidden_size = 1024,
                1 => bad.vocab_size = 248321,
                2 => bad.tie_word_embeddings = true,
                3 => bad.num_layers = 41,
                4 => bad.num_experts = 128,
                5 => bad.num_experts_per_tok = 4,
                _ => bad.moe_intermediate_size = Some(1024),
            }
            assert!(policy.validate_head_profile(&bad).is_err());
            SourceMoePolicy::default()
                .validate_head_profile(&bad)
                .unwrap();
        }
    }
}
