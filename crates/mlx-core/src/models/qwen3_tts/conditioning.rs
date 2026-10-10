//! Released Qwen3 12-Hz capability contract, independent of checkpoint names.
use super::{config::ModelConfig, model::Options};
use napi::{Error, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VoiceMode {
    Preset,
    Reference,
    Description,
}

#[derive(Clone, Copy)]
pub struct Profile {
    pub voice: VoiceMode,
    pub instruct: bool,
}

pub fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|s| !s.trim().is_empty())
}

impl Profile {
    pub fn resolve(config: &ModelConfig) -> Self {
        let voice = match config.tts_model_type.as_str() {
            "base" => VoiceMode::Reference,
            "voice_design" => VoiceMode::Description,
            _ => VoiceMode::Preset,
        };
        Self {
            voice,
            instruct: config.tts_model_size.as_deref() == Some("1b7")
                && matches!(voice, VoiceMode::Preset | VoiceMode::Description),
        }
    }
    pub fn metadata(self) -> serde_json::Value {
        serde_json::json!([{
            "voice": match self.voice { VoiceMode::Preset => "preset", VoiceMode::Reference => "reference", VoiceMode::Description => "description" },
            "instruct": if self.instruct { "supported" } else { "unsupported" }
        }])
    }
    pub fn validate_model(self) -> Result<()> {
        if self.voice == VoiceMode::Description && !self.instruct {
            return Err(Error::from_reason(
                "Unknown VoiceDesign instruction profile",
            ));
        }
        Ok(())
    }
    pub fn validate(self, options: &Options) -> Result<()> {
        let valid = match self.voice {
            VoiceMode::Preset => {
                nonempty(options.voice.as_deref()).is_some()
                    && options.prepared_voice_id.is_none()
                    && options.voice_description.is_none()
            }
            VoiceMode::Reference => {
                nonempty(options.prepared_voice_id.as_deref()).is_some()
                    && options.voice.is_none()
                    && options.voice_description.is_none()
            }
            VoiceMode::Description => {
                nonempty(options.voice_description.as_deref()).is_some()
                    && options.voice.is_none()
                    && options.prepared_voice_id.is_none()
            }
        };
        if !valid {
            return Err(Error::from_reason(
                "Voice condition is incompatible with this TTS model",
            ));
        }
        if !self.instruct
            && (nonempty(options.instruct.as_deref()).is_some()
                || self.voice == VoiceMode::Description)
        {
            return Err(Error::from_reason(
                "Instruction control is unsupported by this TTS profile",
            ));
        }
        Ok(())
    }
}

/// VoiceDesign exposes two application concepts, but has one learned channel.
pub fn instruction(options: &Options) -> Option<String> {
    let parts: Vec<_> = [
        nonempty(options.voice_description.as_deref()),
        nonempty(options.instruct.as_deref()),
    ]
    .into_iter()
    .flatten()
    .collect();
    (!parts.is_empty()).then(|| parts.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen3_tts_instruction_capabilities_fail_closed() {
        let mut config: ModelConfig =
            serde_json::from_str(include_str!("fixtures/base.json")).unwrap();
        for (variant, size, expected) in [
            ("base", "1b7", false),
            ("custom_voice", "0b6", false),
            ("custom_voice", "1b7", true),
            ("voice_design", "1b7", true),
            ("custom_voice", "unknown", false),
        ] {
            config.tts_model_type = variant.into();
            config.tts_model_size = Some(size.into());
            assert_eq!(Profile::resolve(&config).instruct, expected);
        }
        config.tts_model_size = None;
        assert!(!Profile::resolve(&config).instruct);
    }
    #[test]
    fn qwen3_tts_unknown_description_profile_is_not_loadable() {
        assert!(
            Profile {
                voice: VoiceMode::Description,
                instruct: false
            }
            .validate_model()
            .is_err()
        );
        assert!(
            Profile {
                voice: VoiceMode::Preset,
                instruct: false
            }
            .validate_model()
            .is_ok()
        );
        let profile = Profile {
            voice: VoiceMode::Reference,
            instruct: false,
        };
        let mut options = Options {
            prepared_voice_id: Some("owned".into()),
            ..Default::default()
        };
        profile.validate(&options).unwrap();
        options.instruct = Some("Calm.".into());
        assert!(profile.validate(&options).is_err());
        options.instruct = Some("  ".into());
        profile.validate(&options).unwrap();
    }
    #[test]
    fn qwen3_tts_description_survives_style_clear_and_rejects_mixed_voices() {
        let profile = Profile {
            voice: VoiceMode::Description,
            instruct: true,
        };
        let mut options = Options {
            voice_description: Some("A low voice.".into()),
            instruct: Some("Calm.".into()),
            ..Default::default()
        };
        assert_eq!(
            instruction(&options).as_deref(),
            Some("A low voice.\nCalm.")
        );
        profile.validate(&options).unwrap();
        options.instruct = Some("  ".into());
        assert_eq!(instruction(&options).as_deref(), Some("A low voice."));
        options.voice = Some("vivian".into());
        assert!(profile.validate(&options).is_err());
    }
}
