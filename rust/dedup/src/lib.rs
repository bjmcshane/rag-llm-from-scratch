//! Corpus deduplication: exact (normalized-text hash) + near-duplicate (MinHash LSH).
//!
//! MinHash: a document becomes a set of word n-gram "shingles". For each of `num_perm` hash
//! functions we keep the minimum hash over all shingles; two documents agree on any given slot
//! with probability equal to their Jaccard similarity.
//!
//! LSH: the signature is cut into `bands` bands of `rows` values. Documents sharing any whole
//! band become candidates, and a candidate only counts as a duplicate if its estimated Jaccard
//! (fraction of equal signature slots) is at least `threshold`.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use rayon::prelude::*;
use xxhash_rust::xxh3::{xxh3_64, xxh3_64_with_seed};

const MERSENNE_61: u64 = (1 << 61) - 1;

/// (a * x + b) mod 2^61 - 1, the classic universal hash family used for MinHash permutations.
fn permute(a: u64, b: u64, x: u64) -> u64 {
    let prod = (a as u128) * (x as u128) + (b as u128);
    let folded = (prod & MERSENNE_61 as u128) + (prod >> 61);
    (folded as u64 % MERSENNE_61) as u64
}

/// Lowercase and collapse whitespace so trivial formatting changes don't defeat dedup.
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

struct MinHasher {
    ngram: usize,
    coeffs: Vec<(u64, u64)>,
}

impl MinHasher {
    fn new(num_perm: usize, seed: u64) -> Self {
        let coeffs = (0..num_perm as u64)
            .map(|i| {
                let a = xxh3_64_with_seed(&i.to_le_bytes(), seed) % (MERSENNE_61 - 1) + 1;
                let b = xxh3_64_with_seed(&i.to_le_bytes(), seed ^ 0x9E37_79B9_7F4A_7C15) % MERSENNE_61;
                (a, b)
            })
            .collect();
        Self { ngram: 5, coeffs }
    }

    fn signature(&self, normalized: &str) -> Vec<u32> {
        let words: Vec<&str> = normalized.split(' ').filter(|w| !w.is_empty()).collect();
        let shingles: HashSet<u64> = if words.len() < self.ngram {
            std::iter::once(xxh3_64(normalized.as_bytes())).collect()
        } else {
            words
                .windows(self.ngram)
                .map(|w| xxh3_64(w.join(" ").as_bytes()))
                .collect()
        };
        self.coeffs
            .iter()
            .map(|&(a, b)| {
                shingles
                    .iter()
                    .map(|&s| permute(a, b, s & MERSENNE_61))
                    .min()
                    .unwrap_or(u64::MAX) as u32
            })
            .collect()
    }
}

/// Streaming deduplicator. Feed documents in order; the first copy is kept, later copies are
/// reported as duplicates. Memory: ~4 bytes * num_perm per unique document.
#[pyclass]
struct Deduper {
    hasher: MinHasher,
    bands: usize,
    rows: usize,
    threshold: f64,
    exact: HashSet<u64>,
    buckets: Vec<HashMap<u64, Vec<u32>>>,
    signatures: Vec<Vec<u32>>,
    #[pyo3(get)]
    exact_dupes: u64,
    #[pyo3(get)]
    near_dupes: u64,
}

impl Deduper {
    fn band_key(&self, sig: &[u32], band: usize) -> u64 {
        let slice = &sig[band * self.rows..(band + 1) * self.rows];
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        xxh3_64(&bytes)
    }

    fn jaccard(a: &[u32], b: &[u32]) -> f64 {
        a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
    }

    /// Check one precomputed document and record it if it's new. Returns true if duplicate.
    fn check_and_insert(&mut self, exact_key: u64, sig: Vec<u32>) -> bool {
        if !self.exact.insert(exact_key) {
            self.exact_dupes += 1;
            return true;
        }
        let keys: Vec<u64> = (0..self.bands).map(|b| self.band_key(&sig, b)).collect();
        let mut seen = HashSet::new();
        for (band, key) in keys.iter().enumerate() {
            if let Some(ids) = self.buckets[band].get(key) {
                for &id in ids {
                    if seen.insert(id)
                        && Self::jaccard(&sig, &self.signatures[id as usize]) >= self.threshold
                    {
                        self.near_dupes += 1;
                        return true;
                    }
                }
            }
        }
        let id = self.signatures.len() as u32;
        for (band, key) in keys.into_iter().enumerate() {
            self.buckets[band].entry(key).or_default().push(id);
        }
        self.signatures.push(sig);
        false
    }
}

#[pymethods]
impl Deduper {
    /// bands * rows = number of MinHash permutations. The defaults (16 x 8 = 128) make documents
    /// with Jaccard >= ~0.7 very likely to collide; `threshold` then confirms the match.
    #[new]
    #[pyo3(signature = (bands=16, rows=8, threshold=0.8, ngram=5, seed=42))]
    fn new(bands: usize, rows: usize, threshold: f64, ngram: usize, seed: u64) -> Self {
        let mut hasher = MinHasher::new(bands * rows, seed);
        hasher.ngram = ngram.max(1);
        Self {
            hasher,
            bands,
            rows,
            threshold,
            exact: HashSet::new(),
            buckets: (0..bands).map(|_| HashMap::new()).collect(),
            signatures: Vec::new(),
            exact_dupes: 0,
            near_dupes: 0,
        }
    }

    /// True if `text` duplicates a document seen earlier; otherwise remembers it and returns False.
    fn is_duplicate(&mut self, py: Python<'_>, text: &str) -> bool {
        let (key, sig) = py.detach(|| {
            let norm = normalize(text);
            (xxh3_64(norm.as_bytes()), self.hasher.signature(&norm))
        });
        self.check_and_insert(key, sig)
    }

    /// Batch version: MinHash signatures are computed in parallel on all cores (GIL released),
    /// then checked in order. Returns one bool per text, True = duplicate.
    fn filter_batch(&mut self, py: Python<'_>, texts: Vec<String>) -> Vec<bool> {
        let hasher = &self.hasher;
        let prepared: Vec<(u64, Vec<u32>)> = py.detach(|| {
            texts
                .par_iter()
                .map(|t| {
                    let norm = normalize(t);
                    (xxh3_64(norm.as_bytes()), hasher.signature(&norm))
                })
                .collect()
        });
        prepared
            .into_iter()
            .map(|(key, sig)| self.check_and_insert(key, sig))
            .collect()
    }

    /// Number of unique documents kept so far.
    fn __len__(&self) -> usize {
        self.signatures.len()
    }
}

#[pymodule]
fn corpus_dedup(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Deduper>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deduper() -> Deduper {
        Deduper::new(16, 8, 0.8, 5, 42)
    }

    const DOC: &str = "the quick brown fox jumps over the lazy dog while the farmer watches from \
                       the porch and drinks a cup of coffee in the warm morning sun of late summer";

    #[test]
    fn exact_duplicate_after_normalization() {
        let mut d = deduper();
        let n = normalize(DOC);
        assert!(!d.check_and_insert(xxh3_64(n.as_bytes()), d.hasher.signature(&n)));
        let n2 = normalize(&DOC.to_uppercase().replace(' ', "  \n"));
        assert!(d.check_and_insert(xxh3_64(n2.as_bytes()), d.hasher.signature(&n2)));
        assert_eq!(d.exact_dupes, 1);
    }

    #[test]
    fn near_duplicate_detected_unrelated_kept() {
        let mut d = deduper();
        let add = |d: &mut Deduper, t: &str| {
            let n = normalize(t);
            d.check_and_insert(xxh3_64(n.as_bytes()), d.hasher.signature(&n))
        };
        assert!(!add(&mut d, DOC));
        assert!(add(&mut d, &format!("{DOC} today")));
        assert!(!add(&mut d, "an entirely different document about compilers, register \
                              allocation, instruction scheduling and loop unrolling in llvm"));
        assert_eq!(d.near_dupes, 1);
    }
}
