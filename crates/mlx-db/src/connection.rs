//! Database connection management
//!
//! Provides a sync (blocking) connection wrapper.

use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;
use std::path::Path;

use crate::error::DbError;
use crate::schema::init_schema;

/// Synchronous database reader for TUI use
///
/// Uses a blocking API with a stored tokio runtime for sqlx async calls.
/// This is designed for the TUI which runs its own event loop.
pub struct SyncDbReader {
    pool: SqlitePool,
    /// Single-threaded runtime for blocking async operations
    rt: tokio::runtime::Runtime,
}

impl SyncDbReader {
    /// Open a local SQLite database (blocking)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DbError> {
        // Create a single-threaded runtime for the blocking operations
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| DbError::Connection(format!("Failed to create runtime: {}", e)))?;

        let db_url = format!("sqlite:{}?mode=rwc", path.as_ref().display());
        let pool = rt.block_on(async {
            SqlitePoolOptions::new()
                .max_connections(5)
                .connect(&db_url)
                .await
                .map_err(|e| DbError::Connection(format!("Failed to open database: {}", e)))
        })?;

        // Initialize schema
        rt.block_on(async { init_schema(&pool).await })?;

        Ok(Self { pool, rt })
    }

    /// Execute an async operation synchronously using the stored runtime
    pub fn block_on<F, T>(&self, f: F) -> Result<T, DbError>
    where
        F: std::future::Future<Output = Result<T, DbError>>,
    {
        self.rt.block_on(f)
    }

    /// Get a reference to the connection pool for passing to reader functions
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}
