//! PP-OCRv5 Text Detection Model (DBNet)
//!
//! Text line detection model based on DBNet (Differentiable Binarization Network).
//! Uses PPHGNetV2_B4 backbone + LKPAN neck + DBHead.
//! Ported from PaddleOCR PP-OCRv5_server_det.

pub mod config;
pub mod db_head;
pub mod lkpan;
pub mod model;
pub mod persistence;
pub mod postprocessing;
pub mod processing;
