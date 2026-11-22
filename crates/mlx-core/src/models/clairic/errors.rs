//! Clairic error taxonomy and cancellation helpers matching upstream `clairic_runtime/errors.py`.

use napi::{Error, Result};
use std::{
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

/// Prefix for HTTP 400 validation errors (`ClairRequestError`).
pub const INVALID_REQUEST_PREFIX: &str = "Invalid Clairic request: ";
/// Prefix for concurrency/queue saturation errors (`ClairQueueFull`).
pub const QUEUE_FULL_PREFIX: &str = "Clairic queue full: ";
/// Prefix for runtime / inference failures (`ClairInferenceError`).
pub const INFERENCE_ERROR_PREFIX: &str = "Clairic inference error: ";
/// Canonical message for cancelled Clairic requests.
pub const CANCELLED_MESSAGE: &str = "Clairic inference cancelled";

/// Structured Clairic error variants corresponding to upstream `errors.py`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClairicError {
    /// A request the model cannot serve (HTTP 400).
    InvalidRequest(String),
    /// Request rejected because the serving queue is full (HTTP 429 / 503).
    QueueFull(String),
    /// Internal inference or runtime failure (HTTP 500).
    Inference(String),
    /// Caller cancelled an in-flight request.
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

    pub fn cancelled() -> Self {
        Self::Cancelled
    }

    pub fn into_napi(self) -> Error {
        Error::from_reason(self.to_string())
    }
}

impl fmt::Display for ClairicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(msg) => write!(f, "{INVALID_REQUEST_PREFIX}{msg}"),
            Self::QueueFull(msg) => write!(f, "{QUEUE_FULL_PREFIX}{msg}"),
            Self::Inference(msg) => write!(f, "{INFERENCE_ERROR_PREFIX}{msg}"),
            Self::Cancelled => f.write_str(CANCELLED_MESSAGE),
        }
    }
}

impl std::error::Error for ClairicError {}

impl From<ClairicError> for Error {
    fn from(err: ClairicError) -> Self {
        err.into_napi()
    }
}

/// Construct a N-API validation error (`Invalid Clairic request: ...`).
pub fn invalid_request(message: impl fmt::Display) -> Error {
    Error::from_reason(format!("{INVALID_REQUEST_PREFIX}{message}"))
}

/// Construct a N-API queue-full error (`Clairic queue full: ...`).
pub fn queue_full(message: impl fmt::Display) -> Error {
    Error::from_reason(format!("{QUEUE_FULL_PREFIX}{message}"))
}

/// Construct a N-API inference error (`Clairic inference error: ...`).
pub fn inference_error(message: impl fmt::Display) -> Error {
    Error::from_reason(format!("{INFERENCE_ERROR_PREFIX}{message}"))
}

/// Construct a N-API cancellation error (`Clairic inference cancelled`).
pub fn cancelled() -> Error {
    Error::from_reason(CANCELLED_MESSAGE)
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
            invalid_request("missing state").reason,
            "Invalid Clairic request: missing state"
        );
        assert_eq!(queue_full("limit").reason, "Clairic queue full: limit");
        assert_eq!(
            inference_error("oops").reason,
            "Clairic inference error: oops"
        );

        let flag = AtomicBool::new(false);
        assert!(check_cancel(&flag).is_ok());
        flag.store(true, Ordering::Release);
        let err = check_cancel(&flag).unwrap_err();
        assert_eq!(err.reason, CANCELLED_MESSAGE);
    }
}
