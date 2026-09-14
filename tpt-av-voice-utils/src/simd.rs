//! SIMD kernels with a portable scalar fallback (spec §6.2).
//!
//! Public functions dispatch at runtime: AVX2+FMA and (optionally)
//! AVX-512 on x86_64, NEON on aarch64 (baseline), scalar everywhere else
//! and for the tails. All vector paths must stay bit-comparable to the
//! scalar path within float re-association error — parity tests enforce
//! this.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps,
};

/// `a · b`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detected above.
            return unsafe { avx2_dot(a, b) };
        }
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detected above.
            return unsafe { avx512_dot(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { neon_dot(a, b) };
    }
    scalar_dot(a, b)
}

/// `y += alpha * x`.
pub fn axpy(x: &[f32], alpha: f32, y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detected above.
            return unsafe { avx2_axpy(x, alpha, y) };
        }
        #[cfg(feature = "avx512")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detected above.
            return unsafe { avx512_axpy(x, alpha, y) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { neon_axpy(x, alpha, y) };
    }
    scalar_axpy(x, alpha, y);
}

/// `Σ x[i]²`.
pub fn sum_squares(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: feature detected above.
            return unsafe { avx2_sum_squares(x) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { neon_sum_squares(x) };
    }
    scalar_sum_squares(x)
}

/// Portable reference implementations (also used by tests as ground truth).
pub fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn scalar_axpy(x: &[f32], alpha: f32, y: &mut [f32]) {
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi += alpha * xi;
    }
}

pub fn scalar_sum_squares(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

// ---------------------------------------------------------------------------
// x86_64: AVX2 / AVX-512
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
const AVX_LANES: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / AVX_LANES;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * AVX_LANES));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * AVX_LANES));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let mut lanes = [0f32; AVX_LANES];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.iter().sum();
    for i in chunks * AVX_LANES..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_axpy(x: &[f32], alpha: f32, y: &mut [f32]) {
    let n = x.len().min(y.len());
    let chunks = n / AVX_LANES;
    let va = _mm256_set1_ps(alpha);
    for i in 0..chunks {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i * AVX_LANES));
        let vy = _mm256_loadu_ps(y.as_ptr().add(i * AVX_LANES));
        _mm256_storeu_ps(y.as_mut_ptr().add(i * AVX_LANES), _mm256_add_ps(vy, _mm256_mul_ps(va, vx)));
    }
    for i in chunks * AVX_LANES..n {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_sum_squares(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks = n / AVX_LANES;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i * AVX_LANES));
        acc = _mm256_fmadd_ps(vx, vx, acc);
    }
    let mut lanes = [0f32; AVX_LANES];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.iter().sum();
    for i in chunks * AVX_LANES..n {
        sum += x[i] * x[i];
    }
    sum
}

// AVX-512 requires the `avx512` cargo feature and a toolchain with stable
// AVX-512 intrinsics (Rust ≥ 1.89). Runtime-checked, off by default.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512 {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_reduce_add_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    pub(super) const LANES: usize = 16;

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let chunks = n / LANES;
        let mut acc = _mm512_setzero_ps();
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * LANES));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * LANES));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        for i in chunks * LANES..n {
            sum += a[i] * b[i];
        }
        sum
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn axpy(x: &[f32], alpha: f32, y: &mut [f32]) {
        let n = x.len().min(y.len());
        let chunks = n / LANES;
        let va = _mm512_set1_ps(alpha);
        for i in 0..chunks {
            let vx = _mm512_loadu_ps(x.as_ptr().add(i * LANES));
            let vy = _mm512_loadu_ps(y.as_ptr().add(i * LANES));
            _mm512_storeu_ps(y.as_mut_ptr().add(i * LANES), _mm512_add_ps(vy, _mm512_mul_ps(va, vx)));
        }
        for i in chunks * LANES..n {
            y[i] += alpha * x[i];
        }
    }

    use std::arch::x86_64::{_mm512_add_ps, _mm512_mul_ps};
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_dot(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: caller has verified avx512f support at runtime.
    unsafe { avx512::dot(a, b) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_axpy(x: &[f32], alpha: f32, y: &mut [f32]) {
    // SAFETY: caller has verified avx512f support at runtime.
    unsafe { avx512::axpy(x, alpha, y) }
}

// ---------------------------------------------------------------------------
// aarch64: NEON
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddvq_f32, vfmaq_f32, vld1q_f32, vmulq_n_f32, vaddq_f32, vst1q_f32, vdupq_n_f32,
};

#[cfg(target_arch = "aarch64")]
const NEON_LANES: usize = 4;

#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / NEON_LANES;
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * NEON_LANES));
        let vb = vld1q_f32(b.as_ptr().add(i * NEON_LANES));
        acc = vfmaq_f32(acc, va, vb);
    }
    let mut sum = vaddvq_f32(acc);
    for i in chunks * NEON_LANES..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_axpy(x: &[f32], alpha: f32, y: &mut [f32]) {
    let n = x.len().min(y.len());
    let chunks = n / NEON_LANES;
    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * NEON_LANES));
        let vy = vld1q_f32(y.as_ptr().add(i * NEON_LANES));
        vst1q_f32(y.as_mut_ptr().add(i * NEON_LANES), vaddq_f32(vy, vmulq_n_f32(vx, alpha)));
    }
    for i in chunks * NEON_LANES..n {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_sum_squares(x: &[f32]) -> f32 {
    let n = x.len();
    let chunks = n / NEON_LANES;
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let vx = vld1q_f32(x.as_ptr().add(i * NEON_LANES));
        acc = vfmaq_f32(acc, vx, vx);
    }
    let mut sum = vaddvq_f32(acc);
    for i in chunks * NEON_LANES..n {
        sum += x[i] * x[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data(n: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = crate::nn::Rng::new(0xC0FFEE);
        let a: Vec<f32> = (0..n).map(|_| rng.uniform_range(-2.0, 2.0)).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.uniform_range(-2.0, 2.0)).collect();
        (a, b)
    }

    #[test]
    fn dot_matches_scalar() {
        for &n in &[0, 1, 3, 7, 8, 9, 16, 31, 100, 255, 1000] {
            let (a, b) = test_data(n);
            let fast = dot(&a, &b);
            let slow = scalar_dot(&a, &b);
            let scale = 1.0 + fast.abs().max(1.0) * 1e-4;
            assert!(
                (fast - slow).abs() < scale * 1e-4,
                "n={n}: {fast} vs {slow}"
            );
        }
    }

    #[test]
    fn axpy_matches_scalar() {
        for &n in &[0, 1, 5, 8, 13, 64, 257] {
            let (x, _) = test_data(n);
            let mut y_fast: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let mut y_slow = y_fast.clone();
            axpy(&x, 1.5, &mut y_fast);
            scalar_axpy(&x, 1.5, &mut y_slow);
            for (f, s) in y_fast.iter().zip(&y_slow) {
                assert!((f - s).abs() < 1e-5, "n={n}: {f} vs {s}");
            }
        }
    }

    #[test]
    fn sum_squares_matches_scalar() {
        for &n in &[0, 1, 7, 8, 33, 512] {
            let (x, _) = test_data(n);
            let fast = sum_squares(&x);
            let slow = scalar_sum_squares(&x);
            assert!((fast - slow).abs() < slow.max(1.0) * 1e-4, "n={n}: {fast} vs {slow}");
        }
    }
}
