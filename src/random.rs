use crate::cpu_pool::{pool_get, pool_recycle};
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
        let scale = 1.0 / (1u32 << 24) as f32;
        // Generate 2 floats per u64 for throughput
        let pairs = n / 2;
        for i in 0..pairs {
            let u = rng.next_u64();
            buf[2 * i] = low + range * ((u >> 40) as f32 * scale);
            buf[2 * i + 1] = low + range * (((u >> 16) & 0xFFFFFF) as f32 * scale);
        }
        if n & 1 != 0 {
            buf[n - 1] = low + range * rng.next_f32();
        }
        buf
    });
    Tensor::new(data, shape.to_vec(), requires_grad)
}

/// Normal (Gaussian) random tensor via Box-Muller.
pub fn normal(shape: &[usize], mean: f32, std: f32, requires_grad: bool) -> Tensor {
    let n: usize = shape.iter().product();
    let data = with_rng(|rng| {
        let alloc_n = n + (n & 1);
        let pairs = alloc_n / 2;
        let mut u1_buf = pool_get(pairs);
        let mut u2_buf = pool_get(pairs);
        // Generate uniform pairs
        for i in 0..pairs {
            u1_buf[i] = rng.next_f32().max(1e-10);
            u2_buf[i] = rng.next_f32();
        }
        #[cfg(target_os = "macos")]
        {
            // Vectorized Box-Muller via Accelerate
            extern "C" {
                fn vvlogf(r: *mut f32, x: *const f32, n: *const i32);
                fn vvcosf(r: *mut f32, x: *const f32, n: *const i32);
                fn vvsinf(r: *mut f32, x: *const f32, n: *const i32);
            }
            let np = pairs as i32;
            // r = sqrt(-2 * ln(u1))
            unsafe { vvlogf(u1_buf.as_mut_ptr(), u1_buf.as_ptr(), &np); }
            for i in 0..pairs { u1_buf[i] = (-2.0 * u1_buf[i]).sqrt(); }
            // theta = 2*PI*u2
            let two_pi = 2.0 * std::f32::consts::PI;
            for i in 0..pairs { u2_buf[i] *= two_pi; }
            // cos(theta), sin(theta)
            let mut cos_buf = pool_get(pairs);
            let mut sin_buf = pool_get(pairs);
            unsafe {
                vvcosf(cos_buf.as_mut_ptr(), u2_buf.as_ptr(), &np);
                vvsinf(sin_buf.as_mut_ptr(), u2_buf.as_ptr(), &np);
            }
            // Interleave: buf[2i] = mean + std * r * cos, buf[2i+1] = mean + std * r * sin
            let mut buf = pool_get(alloc_n);
            for i in 0..pairs {
                let r = u1_buf[i];
                buf[2 * i] = mean + std * r * cos_buf[i];
                buf[2 * i + 1] = mean + std * r * sin_buf[i];
            }
            pool_recycle(cos_buf);
            pool_recycle(sin_buf);
            pool_recycle(u1_buf);
            pool_recycle(u2_buf);
            buf.truncate(n);
            buf
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mut buf = pool_get(alloc_n);
            for i in 0..pairs {
                let r = (-2.0 * u1_buf[i].ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2_buf[i];
                buf[2 * i] = mean + std * r * theta.cos();
                buf[2 * i + 1] = mean + std * r * theta.sin();
            }
            pool_recycle(u1_buf);
            pool_recycle(u2_buf);
            buf.truncate(n);
            buf
        }
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
        let mut out = vec![0.0f32; n];
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let vp = unsafe { vdupq_n_f32(p) };
            let vone = unsafe { vdupq_n_f32(1.0) };
            let vzero = unsafe { vdupq_n_f32(0.0) };
            let chunks = n / 4;
            for i in 0..chunks {
                let off = i * 4;
                // Generate 4 uniform floats
                let u = [rng.next_f32(), rng.next_f32(), rng.next_f32(), rng.next_f32()];
                unsafe {
                    let vu = vld1q_f32(u.as_ptr());
                    // vcltq_f32: 1s mask where u < p
                    let mask = vcltq_f32(vu, vp);
                    let result = vbslq_f32(mask, vone, vzero);
                    vst1q_f32(out.as_mut_ptr().add(off), result);
                }
            }
            for i in (chunks * 4)..n {
                out[i] = if rng.next_f32() < p { 1.0 } else { 0.0 };
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..n {
                out[i] = if rng.next_f32() < p { 1.0 } else { 0.0 };
            }
        }
        out
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

/// Exponential distribution via inverse CDF: `-log(uniform) / rate`.
pub fn exponential(shape: &[usize], rate: f32) -> Tensor {
    assert!(rate > 0.0, "exponential requires rate > 0");
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        (0..n)
            .map(|_| {
                let u = rng.next_f32().max(1e-10);
                -u.ln() / rate
            })
            .collect()
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Gamma distribution via Marsaglia-Tsang method.
///
/// For `alpha >= 1`, uses the direct Marsaglia-Tsang rejection method.
/// For `alpha < 1`, uses the transformation: `gamma(alpha) = gamma(alpha+1) * U^(1/alpha)`.
pub fn gamma(shape: &[usize], alpha: f32, beta: f32) -> Tensor {
    assert!(alpha > 0.0, "gamma requires alpha > 0");
    assert!(beta > 0.0, "gamma requires beta > 0");
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(gamma_sample(rng, alpha, beta));
        }
        v
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Internal: sample a single gamma variate using Marsaglia-Tsang.
fn gamma_sample(rng: &mut Xoshiro256PlusPlus, alpha: f32, beta: f32) -> f32 {
    if alpha < 1.0 {
        // gamma(alpha) = gamma(alpha+1) * U^(1/alpha)
        let g = gamma_sample_ge1(rng, alpha + 1.0);
        let u = rng.next_f32().max(1e-10);
        return g * u.powf(1.0 / alpha) / beta;
    }
    gamma_sample_ge1(rng, alpha) / beta
}

/// Internal: Marsaglia-Tsang for alpha >= 1.
fn gamma_sample_ge1(rng: &mut Xoshiro256PlusPlus, alpha: f32) -> f32 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Generate normal(0,1) via Box-Muller
        let u1 = rng.next_f32().max(1e-10);
        let u2 = rng.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let x = r * theta.cos();

        let v_inner = 1.0 + c * x;
        if v_inner <= 0.0 {
            continue;
        }
        let v = v_inner * v_inner * v_inner;
        let u = rng.next_f32().max(1e-10);
        // Acceptance criterion
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Beta distribution via two independent gamma samples.
///
/// `x = gamma(a, 1)`, `y = gamma(b, 1)`, result = `x / (x + y)`.
pub fn beta(shape: &[usize], a: f32, b: f32) -> Tensor {
    assert!(a > 0.0, "beta requires a > 0");
    assert!(b > 0.0, "beta requires b > 0");
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let x = gamma_sample(rng, a, 1.0);
            let y = gamma_sample(rng, b, 1.0);
            v.push(x / (x + y));
        }
        v
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Poisson distribution via Knuth's algorithm (for small lambda < 30).
///
/// Counts uniform products until the product drops below `exp(-lambda)`.
/// Returns values as f32.
pub fn poisson(shape: &[usize], lambda: f32) -> Tensor {
    assert!(lambda >= 0.0, "poisson requires lambda >= 0");
    let n: usize = shape.iter().product();
    let data: Vec<f32> = with_rng(|rng| {
        let l = (-lambda).exp();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let mut k = 0u32;
            let mut p = 1.0f32;
            loop {
                p *= rng.next_f32();
                if p < l {
                    break;
                }
                k += 1;
            }
            v.push(k as f32);
        }
        v
    });
    Tensor::new(data, shape.to_vec(), false)
}

/// Multinomial sampling from a probability distribution.
///
/// `probs` has shape `[batch, num_classes]`. Returns `[batch, num_samples]`
/// tensor of class indices (as f32).
///
/// With `replacement=true`: normalize probs, use cumulative sum + uniform search.
/// With `replacement=false`: sequential removal (sample and zero out chosen class).
pub fn multinomial(probs: &Tensor, num_samples: usize, replacement: bool) -> Tensor {
    let shape = probs.shape();
    assert!(
        shape.len() == 2,
        "multinomial expects 2D probs [batch, num_classes]"
    );
    let batch = shape[0];
    let num_classes = shape[1];
    assert!(num_samples > 0, "num_samples must be > 0");
    if !replacement {
        assert!(
            num_samples <= num_classes,
            "without replacement, num_samples must be <= num_classes"
        );
    }

    let prob_data = probs.data();
    let mut result = Vec::with_capacity(batch * num_samples);

    with_rng(|rng| {
        for b in 0..batch {
            let row_start = b * num_classes;
            // Normalize probabilities for this batch row
            let mut row_probs: Vec<f32> = prob_data[row_start..row_start + num_classes].to_vec();
            let sum: f32 = row_probs.iter().sum();
            assert!(sum > 0.0, "multinomial: row probabilities must sum to > 0");
            for p in row_probs.iter_mut() {
                *p /= sum;
            }

            if replacement {
                // Build cumulative sum once
                let mut cdf = Vec::with_capacity(num_classes);
                let mut cumsum = 0.0f32;
                for &p in &row_probs {
                    cumsum += p;
                    cdf.push(cumsum);
                }
                // Ensure last element is exactly 1.0 to avoid edge cases
                if let Some(last) = cdf.last_mut() {
                    *last = 1.0;
                }

                for _ in 0..num_samples {
                    let u = rng.next_f32();
                    // Binary search for the class
                    let mut lo = 0usize;
                    let mut hi = num_classes;
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if cdf[mid] <= u {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    result.push(lo.min(num_classes - 1) as f32);
                }
            } else {
                // Without replacement: sequential removal
                let mut probs_left = row_probs.clone();
                for _ in 0..num_samples {
                    // Build CDF from remaining probabilities
                    let total: f32 = probs_left.iter().sum();
                    let mut cdf = Vec::with_capacity(num_classes);
                    let mut cumsum = 0.0f32;
                    for &p in &probs_left {
                        cumsum += p / total;
                        cdf.push(cumsum);
                    }
                    if let Some(last) = cdf.last_mut() {
                        *last = 1.0;
                    }

                    let u = rng.next_f32();
                    let mut lo = 0usize;
                    let mut hi = num_classes;
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if cdf[mid] <= u {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    let chosen = lo.min(num_classes - 1);
                    result.push(chosen as f32);
                    // Zero out the chosen class so it can't be selected again
                    probs_left[chosen] = 0.0;
                }
            }
        }
    });

    Tensor::new(result, vec![batch, num_samples], false)
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
        // Hold the global lock for both seed+generate pairs to prevent
        // concurrent tests from consuming PRNG state in between.
        let (a, b) = {
            let mut state = GLOBAL_STATE.lock().unwrap();
            *state = Some(Xoshiro256PlusPlus::new(123));
            let rng = state.as_mut().unwrap();
            let a: Vec<f32> = (0..100).map(|_| rng.next_f32()).collect();
            *state = Some(Xoshiro256PlusPlus::new(123));
            let rng = state.as_mut().unwrap();
            let b: Vec<f32> = (0..100).map(|_| rng.next_f32()).collect();
            (a, b)
        };
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

    // -----------------------------------------------------------------------
    // Exponential distribution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exponential_shape() {
        seed(42);
        let t = exponential(&[3, 4], 1.0);
        assert_eq!(t.shape(), vec![3, 4]);
        assert_eq!(t.data().len(), 12);
    }

    #[test]
    fn test_exponential_positive() {
        seed(42);
        let t = exponential(&[1000], 2.0);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0), "exponential samples must be non-negative");
    }

    #[test]
    fn test_exponential_mean() {
        seed(42);
        let rate = 2.0;
        let t = exponential(&[10000], rate);
        let d = t.data();
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        let expected = 1.0 / rate;
        assert!(
            (mean - expected).abs() < 0.05,
            "exponential mean should be ~{}, got {}",
            expected,
            mean
        );
    }

    // -----------------------------------------------------------------------
    // Gamma distribution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gamma_shape() {
        seed(42);
        let t = gamma(&[5, 3], 2.0, 1.0);
        assert_eq!(t.shape(), vec![5, 3]);
        assert_eq!(t.data().len(), 15);
    }

    #[test]
    fn test_gamma_positive() {
        seed(42);
        let t = gamma(&[1000], 2.0, 1.0);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0), "gamma samples must be non-negative");
    }

    #[test]
    fn test_gamma_mean() {
        seed(42);
        let alpha = 3.0;
        let beta = 2.0;
        let t = gamma(&[10000], alpha, beta);
        let d = t.data();
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        let expected = alpha / beta;
        assert!(
            (mean - expected).abs() < 0.1,
            "gamma(alpha={}, beta={}) mean should be ~{}, got {}",
            alpha, beta, expected, mean
        );
    }

    #[test]
    fn test_gamma_small_alpha() {
        // alpha < 1 uses the boosting trick
        seed(42);
        let alpha = 0.5;
        let beta = 1.0;
        let t = gamma(&[5000], alpha, beta);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0));
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        let expected = alpha / beta;
        assert!(
            (mean - expected).abs() < 0.1,
            "gamma(alpha={}, beta={}) mean should be ~{}, got {}",
            alpha, beta, expected, mean
        );
    }

    // -----------------------------------------------------------------------
    // Beta distribution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_beta_shape() {
        seed(42);
        let t = beta(&[4, 5], 2.0, 5.0);
        assert_eq!(t.shape(), vec![4, 5]);
        assert_eq!(t.data().len(), 20);
    }

    #[test]
    fn test_beta_range() {
        seed(42);
        let t = beta(&[5000], 2.0, 5.0);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0 && x <= 1.0), "beta samples must be in [0, 1]");
    }

    #[test]
    fn test_beta_mean() {
        seed(42);
        let a = 2.0;
        let b = 5.0;
        let t = beta(&[10000], a, b);
        let d = t.data();
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        let expected = a / (a + b);
        assert!(
            (mean - expected).abs() < 0.05,
            "beta(a={}, b={}) mean should be ~{}, got {}",
            a, b, expected, mean
        );
    }

    // -----------------------------------------------------------------------
    // Poisson distribution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_shape() {
        seed(42);
        let t = poisson(&[3, 4], 5.0);
        assert_eq!(t.shape(), vec![3, 4]);
        assert_eq!(t.data().len(), 12);
    }

    #[test]
    fn test_poisson_non_negative_integers() {
        seed(42);
        let t = poisson(&[1000], 3.0);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0 && x == x.floor()), "poisson samples must be non-negative integers");
    }

    #[test]
    fn test_poisson_mean() {
        seed(42);
        let lambda = 5.0;
        let t = poisson(&[10000], lambda);
        let d = t.data();
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        assert!(
            (mean - lambda).abs() < 0.3,
            "poisson(lambda={}) mean should be ~{}, got {}",
            lambda, lambda, mean
        );
    }

    // -----------------------------------------------------------------------
    // Multinomial distribution tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multinomial_shape() {
        seed(42);
        let probs = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4], false);
        let t = multinomial(&probs, 5, true);
        assert_eq!(t.shape(), vec![1, 5]);
    }

    #[test]
    fn test_multinomial_valid_indices() {
        seed(42);
        let probs = Tensor::new(vec![0.25, 0.25, 0.25, 0.25], vec![1, 4], false);
        let t = multinomial(&probs, 100, true);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0 && x < 4.0 && x == x.floor()));
    }

    #[test]
    fn test_multinomial_without_replacement_unique() {
        seed(42);
        let probs = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![1, 4], false);
        let t = multinomial(&probs, 4, false);
        let d = t.data();
        // All 4 classes should be selected exactly once
        let mut sorted: Vec<f32> = d.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "without replacement: all samples should be unique");
    }

    #[test]
    fn test_multinomial_batch() {
        seed(42);
        let probs = Tensor::new(
            vec![0.5, 0.5, 0.3, 0.7],
            vec![2, 2],
            false,
        );
        let t = multinomial(&probs, 3, true);
        assert_eq!(t.shape(), vec![2, 3]);
        let d = t.data();
        assert!(d.iter().all(|&x| x >= 0.0 && x < 2.0));
    }

    #[test]
    fn test_multinomial_biased() {
        // Class 2 has overwhelming probability
        seed(42);
        let probs = Tensor::new(vec![0.001, 0.001, 0.998], vec![1, 3], false);
        let t = multinomial(&probs, 100, true);
        let d = t.data();
        let count_2 = d.iter().filter(|&&x| x == 2.0).count();
        assert!(count_2 > 80, "class 2 should be selected most often, got {}", count_2);
    }
}
