//! Error types for the database module

use thiserror::Error;

/// Database operation errors
#[derive(Error, Debug)]
pub enum DbError {
    /// Connection error
    #[error("Connection error: {0}")]
    Connection(String),

    /// Schema initialization error
    #[error("Schema error: {0}")]
    Schema(String),

    /// Migration error
    #[error("Migration error: {0}")]
    Migration(String),

    /// Query execution error
    #[error("Query error: {0}")]
    Query(String),

    /// Write operation error
    #[error("Write error: {0}")]
    Write(String),

    /// Transaction error
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArg(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(String),

    /// Run not found
    #[error("Run not found: {0}")]
    RunNotFound(String),
}

impl From<sqlx::Error> for DbError {
    fn from(e: sqlx::Error) -> Self {
        DbError::Query(e.to_string())
    }
}

impl From<serde_json::Error> for DbError {
    fn from(e: serde_json::Error) -> Self {
        DbError::Serialization(e.to_string())
    }
}

impl From<std::io::Error> for DbError {
    fn from(e: std::io::Error) -> Self {
        DbError::Io(e.to_string())
    }
}
