//! Speaker clustering: agglomerative hierarchical (average linkage,
//! cosine distance) and spectral clustering with eigengap-based speaker
//! count selection, plus optional min/max/expected speaker constraints.

use serde::{Deserialize, Serialize};

use tpt_av_voice_utils::VoiceError;

use crate::embedding::cosine_distance;

/// Which clustering algorithm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ClusteringAlgorithm {
    /// Average-linkage AHC with a distance threshold (default).
    #[default]
    Agglomerative,
    /// Spectral clustering on a cosine-similarity affinity.
    Spectral,
}

/// Clustering knobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Cosine distance below which segments merge (AHC).
    pub threshold: f32,
    /// Hard speaker-count floor, if known.
    pub min_speakers: Option<usize>,
    /// Hard speaker-count ceiling, if known.
    pub max_speakers: Option<usize>,
    /// Exact speaker count, if known (overrides threshold/eigengap).
    pub num_speakers: Option<usize>,
    /// Algorithm selection.
    pub algorithm: ClusteringAlgorithm,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        ClusteringConfig {
            threshold: 0.75,
            min_speakers: Some(1),
            max_speakers: Some(12),
            num_speakers: None,
            algorithm: ClusteringAlgorithm::Agglomerative,
        }
    }
}

impl ClusteringConfig {
    pub fn validate(&self) -> Result<(), VoiceError> {
        if !(0.0..=2.0).contains(&self.threshold) {
            return Err(VoiceError::config("threshold must be within [0, 2] (cosine distance)"));
        }
        if let (Some(min), Some(max)) = (self.min_speakers, self.max_speakers) {
            if min > max {
                return Err(VoiceError::config("min_speakers exceeds max_speakers"));
            }
        }
        Ok(())
    }

    fn clamp_count(&self, k: usize) -> usize {
        let mut k = k.max(1);
        if let Some(min) = self.min_speakers {
            k = k.max(min);
        }
        if let Some(max) = self.max_speakers {
            k = k.min(max);
        }
        if let Some(exact) = self.num_speakers {
            k = exact;
        }
        k.max(1)
    }
}

/// Assigns each embedding a cluster index in `0..k`.
pub fn cluster(
    embeddings: &[Vec<f32>],
    config: &ClusteringConfig,
) -> Result<Vec<usize>, VoiceError> {
    config.validate()?;
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    let assignments = match config.algorithm {
        ClusteringAlgorithm::Agglomerative => agglomerative(embeddings, config),
        ClusteringAlgorithm::Spectral => spectral(embeddings, config),
    };
    Ok(assignments)
}

/// Average-linkage agglomerative clustering. Merges proceed while the
/// closest pair distance is within the threshold and the speaker-count
/// constraints allow it; an explicit `num_speakers` forces merges down
/// to exactly that count.
fn agglomerative(embeddings: &[Vec<f32>], config: &ClusteringConfig) -> Vec<usize> {
    let n = embeddings.len();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    let cluster_dist = |a: &[usize], b: &[usize]| -> f32 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &i in a {
            for &j in b {
                sum += cosine_distance(&embeddings[i], &embeddings[j]);
                count += 1;
            }
        }
        if count == 0 { f32::INFINITY } else { sum / count as f32 }
    };

    let closest_pair = |clusters: &Vec<Vec<usize>>| -> (usize, usize, f32) {
        let (mut best_a, mut best_b) = (0usize, if clusters.len() > 1 { 1 } else { 0 });
        let mut best_d = f32::INFINITY;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let d = cluster_dist(&clusters[i], &clusters[j]);
                if d < best_d {
                    best_d = d;
                    best_a = i;
                    best_b = j;
                }
            }
        }
        (best_a, best_b, best_d)
    };

    // Floor on cluster count while threshold merging is allowed.
    let soft_floor = config
        .num_speakers
        .or(config.min_speakers)
        .unwrap_or(1)
        .clamp(1, n);
    while clusters.len() > soft_floor {
        let (a, b, d) = closest_pair(&clusters);
        if d > config.threshold {
            break;
        }
        let merged = clusters.remove(b);
        clusters[a].extend(merged);
    }
    // Hard counts win over the threshold.
    let hard_target = config.clamp_count(n).min(n);
    while clusters.len() > hard_target {
        let (a, b, _) = closest_pair(&clusters);
        let merged = clusters.remove(b);
        let a = a.min(clusters.len().saturating_sub(1));
        clusters[a].extend(merged);
    }

    let mut labels = vec![0usize; n];
    for (k, members) in clusters.iter().enumerate() {
        for &i in members {
            labels[i] = k;
        }
    }
    labels
}

/// Spectral clustering: cosine-similarity affinity, normalized Laplacian
/// (symmetric), Jacobi eigendecomposition, k-means on the top
/// eigenvectors. k comes from `num_speakers` or the eigengap heuristic,
/// clamped by min/max.
fn spectral(embeddings: &[Vec<f32>], config: &ClusteringConfig) -> Vec<usize> {
    let n = embeddings.len();
    if n == 1 {
        return vec![0];
    }
    let sigma = 0.25f32;
    // Affinity matrix.
    let mut affinity = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let d = cosine_distance(&embeddings[i], &embeddings[j]);
                affinity[i][j] = (-(d * d) / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    // Normalized Laplacian L = I − D^{-1/2} A D^{-1/2}.
    let degree: Vec<f32> = affinity.iter().map(|row| row.iter().sum()).collect();
    let mut laplacian = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            let dij = if degree[i] > 1e-12 && degree[j] > 1e-12 {
                affinity[i][j] / (degree[i] * degree[j]).sqrt()
            } else {
                0.0
            };
            laplacian[i][j] = if i == j { 1.0 - dij } else { -dij };
        }
    }

    let max_k = config.clamp_count(n).min(n.saturating_sub(1)).max(1);
    let (eigenvalues, eigenvectors) = jacobi_eigen(&laplacian, max_k);

    // Eigengap heuristic over the smallest eigenvalues.
    let k = match config.num_speakers {
        Some(exact) => exact.clamp(1, max_k),
        None => {
            let mut k = 1usize;
            let mut best_gap = f32::NEG_INFINITY;
            for i in 0..eigenvalues.len().saturating_sub(1) {
                let gap = eigenvalues[i + 1] - eigenvalues[i];
                if gap > best_gap {
                    best_gap = gap;
                    k = i + 1;
                }
            }
            k.clamp(1, max_k)
        }
    };

    // Rows: first k eigenvectors (smallest eigenvalues), row-normalized.
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f32> = (0..k).map(|v| eigenvectors[v][i]).collect();
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        rows.push(row);
    }
    let labels = kmeans(&rows, k, 50);
    // Relabel deterministically by first appearance.
    relabel_by_first_appearance(&labels, n)
}

/// Symmetric-matrix eigendecomposition via cyclic Jacobi rotations;
/// returns the `k` smallest eigenvalues (ascending) and their
/// eigenvectors (as `vectors[eigen_index][node]`).
fn jacobi_eigen(matrix: &[Vec<f32>], k: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f32>> = matrix.to_vec();
    let mut v: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..100 {
        // Largest off-diagonal magnitude.
        let (mut p, mut q) = (0usize, 0usize);
        let mut off = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > off {
                    off = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-9 {
            break;
        }
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        let theta = 0.5 * (aqq - app) / apq;
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // Rotate.
        for i in 0..n {
            let (aip, aiq) = (a[i][p], a[i][q]);
            a[i][p] = c * aip - s * aiq;
            a[i][q] = s * aip + c * aiq;
        }
        for j in 0..n {
            let (apj, aqj) = (a[p][j], a[q][j]);
            a[p][j] = c * apj - s * aqj;
            a[q][j] = s * apj + c * aqj;
        }
        for i in 0..n {
            let (vip, viq) = (v[i][p], v[i][q]);
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }
    // Collect eigenpairs, sort by eigenvalue ascending.
    let mut pairs: Vec<(f32, Vec<f32>)> = (0..n).map(|i| (a[i][i], (0..n).map(|r| v[r][i]).collect())).collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(pairs.len());
    let values = pairs[..take].iter().map(|(val, _)| *val).collect();
    let vectors = pairs[..take].iter().map(|(_, vec)| vec.clone()).collect();
    (values, vectors)
}

/// Plain k-means with deterministic seeding (spread indices).
fn kmeans(points: &[Vec<f32>], k: usize, iterations: usize) -> Vec<usize> {
    let n = points.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }
    let k = k.min(n);
    let dim = points[0].len();
    // Deterministic spread seeding.
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|c| points[(c * n / k).min(n - 1)].clone())
        .collect();
    let mut labels = vec![0usize; n];
    for _ in 0..iterations {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for (c, cent) in centroids.iter().enumerate() {
                let d: f32 = p.iter().zip(cent).map(|(x, y)| (x - y) * (x - y)).sum();
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        // Update.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            counts[labels[i]] += 1;
            for (d, &x) in p.iter().enumerate() {
                sums[labels[i]][d] += x;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
        if !changed {
            break;
        }
    }
    labels
}

/// Remaps labels to ascending order of first appearance.
fn relabel_by_first_appearance(labels: &[usize], n: usize) -> Vec<usize> {
    let mut map = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(n);
    let mut next = 0usize;
    for &l in labels {
        let entry = *map.entry(l).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        out.push(entry);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::l2_normalize;

    fn point(dir: &[f32]) -> Vec<f32> {
        let mut v = dir.to_vec();
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn config_validation() {
        assert!(ClusteringConfig::default().validate().is_ok());
        let bad = ClusteringConfig { threshold: 3.0, ..Default::default() };
        assert!(bad.validate().is_err());
        let bad = ClusteringConfig {
            min_speakers: Some(4),
            max_speakers: Some(2),
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn ahc_separates_two_groups() {
        let data = vec![
            point(&[1.0, 0.1, 0.0]),
            point(&[0.9, 0.0, 0.1]),
            point(&[0.0, 1.0, 0.1]),
            point(&[0.1, 0.9, 0.0]),
        ];
        let labels = cluster(&data, &ClusteringConfig { threshold: 0.3, ..Default::default() }).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn ahc_respects_exact_count() {
        let data: Vec<Vec<f32>> = (0..6)
            .map(|i| point(&[i as f32 * 0.3, 1.0 - i as f32 * 0.1, 0.5]))
            .collect();
        let labels = cluster(
            &data,
            &ClusteringConfig { num_speakers: Some(3), ..Default::default() },
        )
        .unwrap();
        assert_eq!(labels.iter().collect::<std::collections::HashSet<_>>().len(), 3);
    }

    #[test]
    fn spectral_separates_two_groups() {
        let data = vec![
            point(&[1.0, 0.05, 0.0, 0.0]),
            point(&[0.95, 0.0, 0.05, 0.0]),
            point(&[0.0, 0.0, 1.0, 0.05]),
            point(&[0.0, 0.05, 0.95, 0.0]),
        ];
        let labels = cluster(
            &data,
            &ClusteringConfig { algorithm: ClusteringAlgorithm::Spectral, ..Default::default() },
        )
        .unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn empty_and_single() {
        assert!(cluster(&[], &ClusteringConfig::default()).unwrap().is_empty());
        let one = cluster(&[point(&[1.0, 0.0])], &ClusteringConfig::default()).unwrap();
        assert_eq!(one, vec![0]);
    }

    #[test]
    fn jacobi_finds_known_eigenpairs() {
        // diag(1, 3, 2) has eigenvalues 1, 2, 3.
        let m = vec![vec![1.0, 0.0, 0.0], vec![0.0, 3.0, 0.0], vec![0.0, 0.0, 2.0]];
        let (values, _) = jacobi_eigen(&m, 3);
        assert!((values[0] - 1.0).abs() < 1e-4);
        assert!((values[1] - 2.0).abs() < 1e-4);
        assert!((values[2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn relabel_is_deterministic() {
        let out = relabel_by_first_appearance(&[3, 3, 1, 1, 0], 5);
        assert_eq!(out, vec![0, 0, 1, 1, 2]);
    }
}
