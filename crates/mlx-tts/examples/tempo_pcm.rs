//! Development comparison utility: little-endian float32 PCM in/out.
use mlx_tts::tempo::SpeechTempo;
use std::{env, fs, io::Write, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    if a.len() < 6 {
        return Err(
            "usage: tempo_pcm <rate> <channels> <speed> <input.f32> <output.f32> [chunk-frames]"
                .into(),
        );
    }
    let rate: u32 = a[1].parse()?;
    let channels: u32 = a[2].parse()?;
    let speed: f64 = a[3].parse()?;
    let chunk: usize = a
        .get(6)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or((rate / 10) as usize);
    if chunk == 0 {
        return Err("chunk must be positive".into());
    }
    let chunk_samples = chunk
        .checked_mul(channels as usize)
        .ok_or("chunk-frames too large")?;
    let bytes = fs::read(&a[4])?;
    if bytes.len() % 4 != 0 {
        return Err("partial float32 sample".into());
    }
    let input: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect();
    let mut tempo = SpeechTempo::new(rate, channels, speed)?;
    let mut file = fs::File::create(&a[5])?;
    let mut frames = 0usize;
    let mut nanos = 0u128;
    let mut peak = 0;
    for block in input.chunks(chunk_samples) {
        let start = Instant::now();
        let output = tempo.write(block)?;
        nanos += start.elapsed().as_nanos();
        peak = peak.max(tempo.buffered_frames() + tempo.buffered_output_frames());
        frames += output.len() / channels as usize;
        for sample in output {
            file.write_all(&sample.to_le_bytes())?;
        }
    }
    let start = Instant::now();
    let output = tempo.finish()?;
    nanos += start.elapsed().as_nanos();
    frames += output.len() / channels as usize;
    for sample in output {
        file.write_all(&sample.to_le_bytes())?;
    }
    println!(
        "input_frames={} output_frames={} dsp_ms={:.3} retained_frames_peak={}",
        input.len() / channels as usize,
        frames,
        nanos as f64 / 1e6,
        peak
    );
    Ok(())
}
