//! Microbenchmark for log-mel spectrogram extraction, the front end of
//! every ASR forward pass.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use tpt_av_voice_transcribe::MelConfig;
use tpt_av_voice_utils::signals::speech_like;

fn bench_log_mel(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_mel");
    for &secs in &[0.5f64, 2.0, 10.0] {
        let samples = speech_like(120.0, secs, 16_000, 0.5);
        let frontend = tpt_av_voice_transcribe::encoder::MelFrontend::new(MelConfig::default())
            .unwrap();
        group.bench_with_input(BenchmarkId::new("secs", secs), &secs, |bench, _| {
            bench.iter(|| frontend.log_mel(black_box(&samples)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_log_mel);
criterion_main!(benches);
