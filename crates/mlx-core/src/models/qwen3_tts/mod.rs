//! Native Qwen3-TTS inference, voice conditioning and streaming audio decoding.
mod codec;
mod conditioning;
mod config;
mod encoder;
mod model;
mod prefix;
mod runtime;
mod speaker;
mod tokenizer;
mod transformer;
mod weights;

fn check_cancelled(cancelled: &std::sync::atomic::AtomicBool) -> napi::Result<()> {
    if cancelled.load(std::sync::atomic::Ordering::Acquire) {
        Err(napi::Error::from_reason("TTS cancelled"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
