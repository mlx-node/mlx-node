//! PP-OCRv5 Text Recognition Model (SVTR + CTC)
//!
//! Text recognition model that reads characters from cropped text line images.
//! Uses PPHGNetV2 backbone + SVTR (Scene Text Visual Representation) neck + CTC head.
//! Ported from PaddleOCR PP-OCRv5_server_rec.

pub mod config;
pub mod ctc_head;
pub mod dictionary;
pub mod model;
pub mod persistence;
pub mod processing;
pub mod svtr_neck;
