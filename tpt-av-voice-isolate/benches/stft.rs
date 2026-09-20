//! Microbenchmark for the STFT/FFT front end used by the isolator and its
//! streaming counterpart.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use tpt_av_voice_isolate::{Stft, StftConfig};
use tpt_av_voice_utils::signals::speech_like;

fn bench_stft_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("stft_forward");
    for &secs in &[0.5f64, 2.0, 5.0] {
        let samples = speech_like(120.0, secs, 16_000, 0.5);
        let stft = Stft::new(StftConfig::default()).unwrap();
        group.bench_with_input(BenchmarkId::new("secs", secs), &secs, |bench, _| {
            bench.iter(|| stft.forward(black_box(&samples)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_stft_forward);
criterion_main!(benches);
