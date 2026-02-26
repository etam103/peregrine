use crate::cpu_pool::pool_get;
use crate::tensor::Tensor;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Xoshiro256++ PRNG
// ---------------------------------------------------------------------------

static GLOBAL_STATE: Mutex<Option<Xoshiro256PlusPlus>> = Mutex::new(None);

struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Initialise from a single u64 seed using SplitMix64.
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for item in &mut s {
            sm = sm.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *item = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ---------------------------------------------------------------------------
// Helper: run a closure with the global PRNG (lazily initialised with seed 42)
// ---------------------------------------------------------------------------

fn with_rng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Xoshiro256PlusPlus) -> R,
{
    let mut state = GLOBAL_STATE.lock().unwrap();
    if state.is_none() {
        *state = Some(Xoshiro256PlusPlus::new(42));
    }
    f(state.as_mut().unwrap())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Seed the global PRNG.
pub fn seed(s: u64) {
    *GLOBAL_STATE.lock().unwrap() = Some(Xoshiro256PlusPlus::new(s));
}

/// Uniform random tensor in `[low, high)`.
pub fn uniform(shape: &[usize], low: f32, high: f32, requires_grad: bool) -> Tensor {
    let n: usize = shape.iter().product();
    let data = with_rng(|rng| {
        let mut buf = pool_get(n);
        let range = high - low;
        for i in 0..n {
            buf[i] = low + range * rng.next_f32();
        }
        buf
    });
    Tensor::new(data, shape.to_vec(), requires_grad)
}

/// Normal (Gaussian) random tensor via Box-Muller.
pub fn normal(shape: &[usize], mean: f32, std: f32, requires_grad: bool) -> Tensor {
    let n: usize = shape.iter().product();
    let data = with_rng(|rng| {
        // Allocate for pairs (round up to even)
        let alloc_n = n + (n & 1);
        let mut buf = pool_get(alloc_n);
        let mut i = 0;
        while i < alloc_n {
            let u1 = rng.next_f32().max(1e-10);
            let u2 = rng.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            buf[i] = mean + std * r * theta.cos();
            buf[i + 1] = mean + std * r * theta.sin();
            i += 2;
        }
        buf.truncate(n);
        buf
    });
    Tensor::new(data, shape.to_vec(), requires_grad)
}

/// Random integers in `[low, high)` stored as f32.
pub fn randint(shape: &[usize], low: i64, high: i64) -> Tensor {
    assert!(high > low, "randint requires high > low");
    let n: usize = shape.iter().product();
    let range = (high - low) as u64;
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| (low + (rng.next_u64() % range) as i64) as f32)
            .collect()
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Bernoulli random tensor (1.0 with probability `p`, 0.0 otherwise).
pub fn bernoulli(shape: &[usize], p: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| if rng.next_f32() < p { 1.0 } else { 0.0 })
            .collect()
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Truncated normal: values outside `[a, b]` are resampled.
pub fn truncated_normal(shape: &[usize], mean: f32, std: f32, a: f32, b: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            loop {
                let u1 = rng.next_f32().max(1e-10);
                let u2 = rng.next_f32();
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                let sample = mean + std * r * theta.cos();
                if sample >= a && sample <= b {
                    v.push(sample);
                    break;
                }
            }
        }
        v
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Gumbel distribution: `loc - scale * ln(-ln(uniform))`.
pub fn gumbel(shape: &[usize], loc: f32, scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| {
                let u = rng.next_f32().max(1e-10).min(1.0 - 1e-7);
                loc - scale * (-u.ln()).ln()
            })
            .collect()
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Categorical sampling using the Gumbel-max trick.
///
/// `logits` has shape `[batch, num_classes]`. Returns a `[batch, num_samples]`
/// tensor of class indices (as f32).
pub fn categorical(logits: &Tensor, num_samples: usize) -> Tensor {
    let shape = logits.shape();
    assert!(
        shape.len() == 2,
        "categorical expects 2D logits [batch, num_classes]"
    );
    let batch = shape[0];
    let num_classes = shape[1];
    let logit_data = logits.data();

    let mut result = Vec::with_capacity(batch * num_samples);
    with_rng(|rng| {
        for b in 0..batch {
            for _ in 0..num_samples {
                let mut best_class = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for c in 0..num_classes {
                    let u = rng.next_f32().max(1e-10).min(1.0 - 1e-7);
                    let g = -(-u.ln()).ln();
                    let perturbed = logit_data[b * num_classes + c] + g;
                    if perturbed > best_val {
                        best_val = perturbed;
                        best_class = c;
                    }
                }
                result.push(best_class as f32);
            }
        }
    });
    Tensor::new(result, vec![batch, num_samples], false)
}

/// Laplace distribution: `loc - scale * sign(u-0.5) * ln(1 - 2|u-0.5|)`.
pub fn laplace(shape: &[usize], loc: f32, scale: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| {
                let u = rng.next_f32() - 0.5;
                let sign = if u >= 0.0 { 1.0f32 } else { -1.0 };
                loc - scale * sign * (1.0 - 2.0 * u.abs()).max(1e-10).ln()
            })
            .collect()
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Random permutation of `0..n` (Fisher-Yates shuffle).
pub fn permutation(n: usize) -> Tensor {
    let mut data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    with_rng(|rng| {
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            data.swap(i, j);
        }
    });
    Tensor::new(data, vec![n], false)
}

// ---------------------------------------------------------------------------
// Convenience methods on Tensor
// ---------------------------------------------------------------------------

impl Tensor {
    /// Uniform random tensor in [0, 1).
    pub fn rand(shape: &[usize], requires_grad: bool) -> Tensor {
        uniform(shape, 0.0, 1.0, requires_grad)
    }

    /// Uniform random tensor with the same shape as `self`.
    pub fn rand_like(&self) -> Tensor {
        let shape = self.shape();
        uniform(&shape, 0.0, 1.0, false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_determinism() {
        seed(123);
        let a = uniform(&[100], 0.0, 1.0, false).data();
        seed(123);
        let b = uniform(&[100], 0.0, 1.0, false).data();
        assert_eq!(a, b);
    }

    #[test]
    fn test_uniform_range() {
        seed(99);
        let t = uniform(&[1000], 2.0, 5.0, false);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 2.0 && x < 5.0));
    }

    #[test]
    fn test_normal_shape() {
        seed(7);
        let t = normal(&[4, 3], 0.0, 1.0, false);
        assert_eq!(t.shape(), vec![4, 3]);
        assert_eq!(t.data().len(), 12);
    }

    #[test]
    fn test_randint_range() {
        seed(42);
        let t = randint(&[500], 0, 10);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0 && x < 10.0));
    }

    #[test]
    fn test_bernoulli_values() {
        seed(0);
        let t = bernoulli(&[1000], 0.5);
        let d = t.data();
        assert!(d.iter().all(|&x| x == 0.0 || x == 1.0));
        let ones: f32 = d.iter().sum();
        // Roughly half should be 1s (very loose check).
        assert!(ones > 300.0 && ones < 700.0);
    }

    #[test]
    fn test_truncated_normal_bounds() {
        seed(42);
        let t = truncated_normal(&[500], 0.0, 1.0, -1.5, 1.5);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= -1.5 && x <= 1.5));
    }

    #[test]
    fn test_permutation() {
        seed(10);
        let t = permutation(10);
        let mut d = t.data();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(d, expected);
    }

    #[test]
    fn test_categorical_shape() {
        seed(42);
        let logits = Tensor::new(vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5], vec![2, 3], false);
        let samples = categorical(&logits, 5);
        assert_eq!(samples.shape(), vec![2, 5]);
    }

    #[test]
    fn test_gumbel_shape() {
        seed(42);
        let t = gumbel(&[3, 4], 0.0, 1.0);
        assert_eq!(t.shape(), vec![3, 4]);
        assert_eq!(t.data().len(), 12);
    }

    #[test]
    fn test_laplace_shape() {
        seed(42);
        let t = laplace(&[5, 5], 0.0, 1.0);
        assert_eq!(t.shape(), vec![5, 5]);
        assert_eq!(t.data().len(), 25);
    }

    #[test]
    fn test_tensor_rand() {
        seed(42);
        let t = Tensor::rand(&[3, 3], false);
        assert_eq!(t.shape(), vec![3, 3]);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_tensor_rand_like() {
        seed(42);
        let base = Tensor::zeros(&[2, 4], false);
        let t = base.rand_like();
        assert_eq!(t.shape(), vec![2, 4]);
    }
}
