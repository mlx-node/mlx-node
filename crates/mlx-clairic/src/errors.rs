//! Clairic error taxonomy and cancellation helpers matching upstream `clairic_runtime/errors.py`.

use std::{
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

use thiserror::Error;

/// Prefix for HTTP 400 validation errors (`ClairRequestError`).
pub const INVALID_REQUEST_PREFIX: &str = "Invalid Clairic request: ";
/// Prefix for concurrency/queue saturation errors (`ClairQueueFull`).
pub const QUEUE_FULL_PREFIX: &str = "Clairic queue full: ";
/// Prefix for runtime / inference failures (`ClairInferenceError`).
pub const INFERENCE_ERROR_PREFIX: &str = "Clairic inference error: ";
/// Canonical message for cancelled Clairic requests.
pub const CANCELLED_MESSAGE: &str = "Clairic inference cancelled";

/// Result alias using [`ClairicError`].
pub type Result<T> = std::result::Result<T, ClairicError>;

/// Structured Clairic error variants corresponding to upstream `errors.py`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ClairicError {
    /// A request the model cannot serve (HTTP 400).
    #[error("{INVALID_REQUEST_PREFIX}{0}")]
    InvalidRequest(String),
    /// Request rejected because the serving queue is full (HTTP 429 / 503).
    #[error("{QUEUE_FULL_PREFIX}{0}")]
    QueueFull(String),
    /// Internal inference or runtime failure (HTTP 500).
    #[error("{INFERENCE_ERROR_PREFIX}{0}")]
    Inference(String),
    /// Unprefixed operational error (grammar, schema validation, tokenizer, head config).
    #[error("{0}")]
    Message(String),
    /// Caller cancelled an in-flight request.
    #[error("{CANCELLED_MESSAGE}")]
    Cancelled,
}

impl ClairicError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    pub fn queue_full(message: impl Into<String>) -> Self {
        Self::QueueFull(message.into())
    }

    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference(message.into())
    }

    pub fn from_reason(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    pub fn cancelled() -> Self {
        Self::Cancelled
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

/// Construct a validation error (`Invalid Clairic request: ...`).
pub fn invalid_request(message: impl fmt::Display) -> ClairicError {
    ClairicError::InvalidRequest(message.to_string())
}

/// Construct a queue-full error (`Clairic queue full: ...`).
pub fn queue_full(message: impl fmt::Display) -> ClairicError {
    ClairicError::QueueFull(message.to_string())
}

/// Construct an inference error (`Clairic inference error: ...`).
pub fn inference_error(message: impl fmt::Display) -> ClairicError {
    ClairicError::Inference(message.to_string())
}

/// Construct a cancellation error (`Clairic inference cancelled`).
pub fn cancelled() -> ClairicError {
    ClairicError::Cancelled
}

/// Fail with [`cancelled()`] if the cancellation flag has been set.
pub fn check_cancel(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::Acquire) {
        Err(cancelled())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_prefixes_and_cancel_flag() {
        assert_eq!(
            ClairicError::invalid_request("bad field").to_string(),
            "Invalid Clairic request: bad field"
        );
        assert_eq!(
            ClairicError::queue_full("busy").to_string(),
            "Clairic queue full: busy"
        );
        assert_eq!(
            ClairicError::inference("nan logits").to_string(),
            "Clairic inference error: nan logits"
        );
        assert_eq!(
            ClairicError::cancelled().to_string(),
            "Clairic inference cancelled"
        );
        assert_eq!(
            invalid_request("missing state").to_string(),
            "Invalid Clairic request: missing state"
        );
        assert_eq!(queue_full("limit").to_string(), "Clairic queue full: limit");
        assert_eq!(
            inference_error("oops").to_string(),
            "Clairic inference error: oops"
        );

        let flag = AtomicBool::new(false);
        assert!(check_cancel(&flag).is_ok());
        flag.store(true, Ordering::Release);
        let err = check_cancel(&flag).unwrap_err();
        assert_eq!(err.to_string(), CANCELLED_MESSAGE);
        assert!(err.is_cancelled());
    }
}
