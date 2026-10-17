//! Small pure-Rust neural-network building blocks shared by all model
//! crates: deterministic RNG, matrix multiplication wired to the SIMD
//! kernels, and layers (linear, layer norm, embedding, conv, pooling).
//!
//! Weights live in plain `ndarray` fields so model crates can load them
//! directly from [`crate::safetensors`] tensors without a serialization
//! framework.

use ndarray::{Array1, Array2, Array3, Array4};

/// Deterministic PRNG (xorshift64*) for reproducible weight init and
/// synthetic test data. Not cryptographically secure — by design.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
    normal_cache: Option<f32>,
}

impl Rng {
    /// Seeds the generator. `seed == 0` is remapped to a non-zero state.
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
            normal_cache: None,
        }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform float in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Uniform float in `[lo, hi)`.
    pub fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.uniform() * (hi - lo)
    }

    /// Standard-normal sample (Box–Muller, cached pair).
    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.normal_cache.take() {
            return v;
        }
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        self.normal_cache = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Matrix–vector product `A · x`.
pub fn matvec(a: ndarray::ArrayView2<f32>, x: &[f32]) -> Vec<f32> {
    assert_eq!(a.ncols(), x.len(), "matvec shape mismatch");
    let mut out = Vec::with_capacity(a.nrows());
    for row in a.rows() {
        out.push(crate::simd::dot(
            row.as_slice().expect("matrix row contiguous"),
            x,
        ));
    }
    out
}

/// Naive-but-cache-friendly matrix multiplication `(M,K) × (K,N)`.
/// Row-contiguous inputs use the `axpy` formulation; column-contiguous
/// inputs (e.g. transposed views) use the `dot` formulation; anything
/// else is copied to C order first.
pub fn matmul(a: ndarray::ArrayView2<f32>, b: ndarray::ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(a.ncols(), b.nrows(), "matmul shape mismatch");
    if b.is_standard_layout() {
        matmul_std(a, b)
    } else if b.columns().into_iter().all(|c| c.as_slice().is_some()) {
        // Columns contiguous (Fortran order / transposed view).
        let mut out = Array2::<f32>::zeros((a.nrows(), b.ncols()));
        for (mut out_row, a_row) in out.rows_mut().into_iter().zip(a.rows()) {
            let xs = a_row.as_slice().expect("x row contiguous");
            for (j, b_col) in b.columns().into_iter().enumerate() {
                out_row[j] = crate::simd::dot(xs, b_col.as_slice().expect("b column contiguous"));
            }
        }
        out
    } else {
        let b_std = Array2::from_shape_fn(b.dim(), |idx| b[idx]);
        matmul_std(a, b_std.view())
    }
}

/// `x (M,K) × wᵀ` where `w` has shape `(N,K)` — i.e. a linear layer
/// without materializing the transpose. Every row touched is contiguous,
/// so the inner loops drive [`crate::simd::dot`].
pub fn matmul_t(x: ndarray::ArrayView2<f32>, w: ndarray::ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(
        x.ncols(),
        w.ncols(),
        "matmul_t shape mismatch: x is (M,K), w must be (N,K)"
    );
    let mut out = Array2::<f32>::zeros((x.nrows(), w.nrows()));
    for (mut out_row, x_row) in out.rows_mut().into_iter().zip(x.rows()) {
        let xs = x_row.as_slice().expect("x row contiguous");
        for (j, w_row) in w.rows().into_iter().enumerate() {
            out_row[j] = crate::simd::dot(xs, w_row.as_slice().expect("w row contiguous"));
        }
    }
    out
}

fn matmul_std(a: ndarray::ArrayView2<f32>, b: ndarray::ArrayView2<f32>) -> Array2<f32> {
    debug_assert!(b.is_standard_layout());
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let mut out = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        let mut out_row = out.row_mut(i);
        let out_slice = out_row
            .as_slice_mut()
            .expect("row of a standard-layout matrix is contiguous");
        for kk in 0..k {
            let a_ik = a[[i, kk]];
            if a_ik != 0.0 {
                crate::simd::axpy(
                    b.row(kk).as_slice().expect("b row contiguous"),
                    a_ik,
                    out_slice,
                );
            }
        }
    }
    out
}

/// Affine layer: `y = x · Wᵀ + b` with `W` of shape `(out, in)`.
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
}

impl Linear {
    /// Builds a layer from weight/bias arrays.
    pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        assert_eq!(
            weight.nrows(),
            bias.len(),
            "bias length must match out_features"
        );
        Linear { weight, bias }
    }

    /// Xavier-uniform initialization for reproducible pipelines.
    pub fn with_init(rng: &mut Rng, in_features: usize, out_features: usize) -> Self {
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight = Array2::from_shape_fn((out_features, in_features), |_| {
            rng.uniform_range(-limit, limit)
        });
        let bias = Array1::zeros(out_features);
        Linear { weight, bias }
    }

    /// `(T, in) → (T, out)`.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = matmul_t(x.view(), self.weight.view());
        for mut row in out.rows_mut() {
            row += &self.bias;
        }
        out
    }
}

/// Layer normalization over the last axis.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub scale: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(scale: Array1<f32>, bias: Array1<f32>) -> Self {
        assert_eq!(scale.len(), bias.len());
        LayerNorm {
            scale,
            bias,
            eps: 1e-5,
        }
    }

    /// Unit scale, zero bias.
    pub fn unit(dim: usize) -> Self {
        LayerNorm {
            scale: Array1::ones(dim),
            bias: Array1::zeros(dim),
            eps: 1e-5,
        }
    }

    /// `(T, d) → (T, d)`, normalizing each row.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(x.dim());
        for (mut out_row, in_row) in out.rows_mut().into_iter().zip(x.rows()) {
            let n = in_row.len() as f32;
            let mean = in_row.sum() / n;
            let var = in_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            for j in 0..in_row.len() {
                let norm = (in_row[j] - mean) / (var + self.eps).sqrt();
                out_row[j] = norm * self.scale[j] + self.bias[j];
            }
        }
        out
    }
}

/// Token embedding lookup: table of shape `(n_tokens, d_model)`.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub table: Array2<f32>,
}

impl Embedding {
    pub fn new(table: Array2<f32>) -> Self {
        Embedding { table }
    }

    /// Random init (normal, σ = 0.02 — GPT-style).
    pub fn with_init(rng: &mut Rng, n_tokens: usize, d_model: usize) -> Self {
        let table = Array2::from_shape_fn((n_tokens, d_model), |_| rng.normal() * 0.02);
        Embedding { table }
    }

    /// `indices → (len(indices), d_model)`.
    pub fn forward(&self, indices: &[usize]) -> Array2<f32> {
        Array2::from_shape_fn((indices.len(), self.table.ncols()), |(i, j)| {
            self.table[[indices[i], j]]
        })
    }
}

/// GELU activation (tanh approximation, as used by Whisper/GPT).
pub fn gelu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| {
        0.5 * v * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (v + 0.044715 * v * v * v)).tanh())
    })
}

/// Exact GELU, `x · Φ(x)` via the error function. Whisper is trained with
/// this variant; the tanh approximation above differs from it by up to ~3e-4.
pub fn gelu_exact(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 0.5 * v * (1.0 + erf(v * std::f32::consts::FRAC_1_SQRT_2)))
}

/// Error function, Abramowitz & Stegun 7.1.26 (absolute error < 1.5e-7).
pub fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// ReLU activation.
pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Sigmoid activation.
pub fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Numerically stable softmax along axis 1 of a `(T, N)` matrix.
pub fn softmax_rows(x: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros(x.dim());
    for (mut out_row, in_row) in out.rows_mut().into_iter().zip(x.rows()) {
        let max = in_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = in_row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (o, e) in out_row.iter_mut().zip(exps) {
            *o = e / sum;
        }
    }
    out
}

/// Softmax over a 1-D slice.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// 1-D convolution over `(channels, time)` with `same` padding, stride 1.
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Array3<f32>, // (out_c, in_c, kernel)
    pub bias: Array1<f32>,
    /// Horizontal step between output frames (1 = dense, 2 = downsample).
    pub stride: usize,
}

impl Conv1d {
    pub fn new(weight: Array3<f32>, bias: Array1<f32>) -> Self {
        Conv1d {
            weight,
            bias,
            stride: 1,
        }
    }

    /// Xavier-uniform init.
    pub fn with_init(rng: &mut Rng, in_c: usize, out_c: usize, kernel: usize) -> Self {
        let limit = (6.0 / ((in_c + out_c) * kernel) as f32).sqrt();
        let weight =
            Array3::from_shape_fn((out_c, in_c, kernel), |_| rng.uniform_range(-limit, limit));
        let bias = Array1::zeros(out_c);
        Conv1d {
            weight,
            bias,
            stride: 1,
        }
    }

    /// `(in_c, T) → (out_c, T_out)` with `same` padding.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (in_c, t) = x.dim();
        let (out_c, w_in_c, kernel) = self.weight.dim();
        assert_eq!(in_c, w_in_c, "conv1d channel mismatch");
        assert!(self.stride > 0, "conv1d stride must be > 0");
        let pad = kernel / 2;
        let t_out = (t + 2 * pad - kernel) / self.stride + 1;
        let mut out = Array2::<f32>::zeros((out_c, t_out));
        for oc in 0..out_c {
            for tc in 0..t_out {
                let base = tc * self.stride;
                let mut sum = self.bias[oc];
                for ic in 0..in_c {
                    for k in 0..kernel {
                        let ti = base as isize + k as isize - pad as isize;
                        if ti >= 0 && (ti as usize) < t {
                            sum += self.weight[[oc, ic, k]] * x[[ic, ti as usize]];
                        }
                    }
                }
                out[[oc, tc]] = sum;
            }
        }
        out
    }
}

/// 2-D convolution over `(channels, height, width)` with `same` spatial
/// padding, stride 1.
#[derive(Debug, Clone)]
pub struct Conv2d {
    pub weight: Array4<f32>, // (out_c, in_c, kh, kw)
    pub bias: Array1<f32>,
}

impl Conv2d {
    pub fn new(weight: Array4<f32>, bias: Array1<f32>) -> Self {
        Conv2d { weight, bias }
    }

    /// Xavier-uniform init.
    pub fn with_init(rng: &mut Rng, in_c: usize, out_c: usize, kh: usize, kw: usize) -> Self {
        let limit = (6.0 / ((in_c + out_c) * kh * kw) as f32).sqrt();
        let weight =
            Array4::from_shape_fn((out_c, in_c, kh, kw), |_| rng.uniform_range(-limit, limit));
        let bias = Array1::zeros(out_c);
        Conv2d { weight, bias }
    }

    /// `(in_c, H, W) → (out_c, H, W)`.
    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let (in_c, h, w) = x.dim();
        let (out_c, _, kh, kw) = self.weight.dim();
        let (ph, pw) = (kh / 2, kw / 2);
        let mut out = Array3::<f32>::zeros((out_c, h, w));
        for oc in 0..out_c {
            for hc in 0..h {
                for wc in 0..w {
                    let mut sum = self.bias[oc];
                    for ic in 0..in_c {
                        for kh_i in 0..kh {
                            let hi = hc as isize + kh_i as isize - ph as isize;
                            if hi < 0 || hi as usize >= h {
                                continue;
                            }
                            for kw_i in 0..kw {
                                let wi = wc as isize + kw_i as isize - pw as isize;
                                if wi < 0 || wi as usize >= w {
                                    continue;
                                }
                                sum += self.weight[[oc, ic, kh_i, kw_i]]
                                    * x[[ic, hi as usize, wi as usize]];
                            }
                        }
                    }
                    out[[oc, hc, wc]] = sum;
                }
            }
        }
        out
    }
}

/// Max pooling over `(channels, h, w)` with stride = pool.
pub fn max_pool2d(x: &Array3<f32>, pool: usize) -> Array3<f32> {
    let (c, h, w) = x.dim();
    let (oh, ow) = (h / pool, w / pool);
    let mut out = Array3::<f32>::zeros((c, oh, ow));
    for ch in 0..c {
        for oh_i in 0..oh {
            for ow_i in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for hi in 0..pool {
                    for wi in 0..pool {
                        m = m.max(x[[ch, oh_i * pool + hi, ow_i * pool + wi]]);
                    }
                }
                out[[ch, oh_i, ow_i]] = m;
            }
        }
    }
    out
}

/// Nearest-neighbor upsampling by an integer factor over `(c, h, w)`.
pub fn upsample_nearest2d(x: &Array3<f32>, scale: usize) -> Array3<f32> {
    let (c, h, w) = x.dim();
    let mut out = Array3::<f32>::zeros((c, h * scale, w * scale));
    for ch in 0..c {
        for hi in 0..h * scale {
            for wi in 0..w * scale {
                out[[ch, hi, wi]] = x[[ch, hi / scale, wi / scale]];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = Rng::new(1);
        assert_ne!(Rng::new(42).next_u64(), c.next_u64());
    }

    #[test]
    fn rng_uniform_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let v = rng.uniform_range(-1.5, 2.5);
            assert!((-1.5..2.5).contains(&v));
        }
    }

    #[test]
    fn rng_normal_moments() {
        let mut rng = Rng::new(9);
        let n = 20_000;
        let sum: f64 = (0..n).map(|_| rng.normal() as f64).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "mean {mean}");
    }

    #[test]
    fn matmul_known_result() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(a.view(), b.view());
        assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
        // Non-standard layout input (transposed view): a × bᵀ, where
        // (a·bᵀ)[i,j] = dot(a_row_i, b_row_j).
        let c2 = matmul(a.view(), b.t().view());
        assert_eq!(c2, array![[17.0, 23.0], [39.0, 53.0]]);
    }

    #[test]
    fn matvec_known_result() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(matvec(a.view(), &[1.0, 1.0]), vec![3.0, 7.0]);
    }

    #[test]
    fn linear_forward() {
        let lin = Linear::new(array![[1.0, 0.0], [0.0, 1.0]], array![0.5, -0.5]);
        let out = lin.forward(&array![[1.0, 2.0]]);
        assert_eq!(out, array![[1.5, 1.5]]);
    }

    #[test]
    fn layer_norm_normalizes() {
        let ln = LayerNorm::unit(4);
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        let out = ln.forward(&x);
        let row = out.row(0);
        let mean = row.sum() / 4.0;
        assert!(mean.abs() < 1e-5);
        let var = row.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn embedding_lookup() {
        let e = Embedding::new(array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
        assert_eq!(e.forward(&[2, 0]), array![[3.0, 3.0], [1.0, 1.0]]);
    }

    #[test]
    fn activations() {
        let x = array![[-1.0, 0.0, 1.0, 2.0]];
        assert_eq!(relu(&x), array![[0.0, 0.0, 1.0, 2.0]]);
        let s = sigmoid(&x);
        assert!((s[[0, 2]] - 0.7310586).abs() < 1e-5);
        // gelu(0) == 0, gelu is ~identity for large x.
        let g = gelu(&array![[0.0, 10.0]]);
        assert!(g[[0, 0]].abs() < 1e-6 && (g[[0, 1]] - 10.0).abs() < 1e-3);
        // softmax rows sum to 1.
        let p = softmax_rows(&x);
        assert!((p.sum() - x.nrows() as f32).abs() < 1e-5);
        assert!((softmax(&[1.0, 2.0, 3.0]).iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn conv1d_identity_kernel() {
        // kernel [0,1,0] with single channel → identity (with same padding).
        let w = Array3::from_shape_fn((1, 1, 3), |(_, _, k)| if k == 1 { 1.0 } else { 0.0 });
        let conv = Conv1d::new(w, array![0.0]);
        let x = array![[1.0, 2.0, 3.0, 4.0]];
        assert_eq!(conv.forward(&x), x);
    }

    #[test]
    fn conv2d_shapes_and_bias() {
        let mut rng = Rng::new(3);
        let conv = Conv2d::with_init(&mut rng, 2, 4, 3, 3);
        let x = Array3::<f32>::zeros((2, 8, 10));
        let out = conv.forward(&x);
        assert_eq!(out.dim(), (4, 8, 10));
        // Zero input → bias everywhere.
        assert_eq!(out[[0, 0, 0]], conv.bias[0]);
    }

    #[test]
    fn pool_and_upsample() {
        let x = Array3::from_shape_fn((1, 4, 4), |(_, i, j)| (i * 4 + j) as f32);
        let pooled = max_pool2d(&x, 2);
        assert_eq!(pooled.dim(), (1, 2, 2));
        assert_eq!(pooled[[0, 0, 0]], 5.0); // max of {0,1,4,5}
        let up = upsample_nearest2d(&pooled, 2);
        assert_eq!(up.dim(), (1, 4, 4));
        assert_eq!(up[[0, 3, 3]], pooled[[0, 1, 1]]);
    }
}
