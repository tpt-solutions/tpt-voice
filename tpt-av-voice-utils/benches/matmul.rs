//! Microbenchmarks for the dense matmul kernels in `nn.rs`, the hot path
//! for every linear layer in the transcribe/isolate/diarize models.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

use tpt_av_voice_utils::nn::{matmul, matmul_t, Rng};

fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |_| rng.uniform_range(-1.0, 1.0))
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    let mut rng = Rng::new(42);
    for &dim in &[32usize, 128, 512] {
        let a = random_matrix(&mut rng, dim, dim);
        let b = random_matrix(&mut rng, dim, dim);
        group.bench_with_input(BenchmarkId::new("square", dim), &dim, |bench, _| {
            bench.iter(|| matmul(black_box(a.view()), black_box(b.view())));
        });
    }
    group.finish();
}

fn bench_matmul_t(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_t");
    let mut rng = Rng::new(7);
    for &(rows, in_features, out_features) in &[(8usize, 256usize, 256usize), (64, 512, 512)] {
        let x = random_matrix(&mut rng, rows, in_features);
        let w = random_matrix(&mut rng, out_features, in_features);
        group.bench_with_input(
            BenchmarkId::new("linear", format!("{rows}x{in_features}x{out_features}")),
            &(rows, in_features, out_features),
            |bench, _| {
                bench.iter(|| matmul_t(black_box(x.view()), black_box(w.view())));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_matmul_t);
criterion_main!(benches);
