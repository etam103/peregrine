//! Hand-tuned NEON intrinsic kernels for aarch64 (Apple Silicon).
//!
//! Each kernel processes 4 f32 elements per iteration via `float32x4_t`,
//! with a scalar tail for `len % 4 != 0`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ========================================================================
// Phase 1: Forward elementwise kernels
// ========================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vaddq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vsubq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i] - b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_div_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vdivq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i] / b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_neg_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vnegq_f32(va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = -a[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_abs_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vabsq_f32(va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].abs();
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_scale_f32(a: &[f32], s: f32, out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let vs = vdupq_n_f32(s);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(va, vs));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i] * s;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_relu_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vmaxq_f32(va, zero));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].max(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_add_inplace_f32(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(a.as_mut_ptr().add(off), vaddq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        a[i] += b[i];
    }
}

// ========================================================================
// Phase 1: Backward-specific kernels
// ========================================================================

/// relu backward: out[i] = if input[i] > 0 { grad[i] } else { 0 }
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_relu_backward_f32(input: &[f32], grad: &[f32], out: &mut [f32]) {
    let len = input.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(input.as_ptr().add(off));
            let vg = vld1q_f32(grad.as_ptr().add(off));
            let mask = vcgtq_f32(vi, zero);
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vg, zero));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if input[i] > 0.0 { grad[i] } else { 0.0 };
    }
}

// ========================================================================
// Phase 2: Polynomial exp and transcendental kernels
// ========================================================================

/// Fast vectorized exp using Cephes-style polynomial approximation.
/// Relative error ~1.2e-7 over the full float32 range.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn fast_exp_f32x4(x: float32x4_t) -> float32x4_t {
    // Clamp input to prevent overflow/underflow
    let min_val = vdupq_n_f32(-87.3365);
    let max_val = vdupq_n_f32(88.7228);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    // exp(x) = 2^(x * log2(e)) = 2^n * 2^r where n is integer, r is fraction
    let log2e = vdupq_n_f32(1.4426950409);
    let ln2_hi = vdupq_n_f32(0.6931471806);
    let ln2_lo = vdupq_n_f32(1.1730463525e-7);

    // n = round(x * log2e)
    let t = vmulq_f32(x, log2e);
    let n = vrndnq_f32(t); // round to nearest integer

    // r = x - n * ln2 (two-step for precision)
    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n, ln2_hi)), vmulq_f32(n, ln2_lo));

    // Horner polynomial for 2^r - 1 on [-0.5, 0.5]
    // Coefficients from Cephes: p0 + r*(p1 + r*(p2 + r*(p3 + r*(p4 + r*p5))))
    let p0 = vdupq_n_f32(1.0);
    let p1 = vdupq_n_f32(1.0);
    let p2 = vdupq_n_f32(0.5);
    let p3 = vdupq_n_f32(0.16666666);
    let p4 = vdupq_n_f32(0.041666666);
    let p5 = vdupq_n_f32(0.008333333);

    // Evaluate polynomial via Horner's method
    let mut y = vmlaq_f32(p4, p5, r);
    y = vmlaq_f32(p3, y, r);
    y = vmlaq_f32(p2, y, r);
    y = vmlaq_f32(p1, y, r);
    y = vmlaq_f32(p0, y, r);

    // Reconstruct 2^n by adding n to the exponent field of the f32
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ni, vdupq_n_s32(127))));
    vmulq_f32(y, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_exp_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), fast_exp_f32x4(va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].exp();
    }
}

/// sigmoid(x) = 1 / (1 + exp(-x)), fused for efficiency
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sigmoid_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let neg_x = vnegq_f32(vx);
            let exp_neg_x = fast_exp_f32x4(neg_x);
            let denom = vaddq_f32(one, exp_neg_x);
            vst1q_f32(out.as_mut_ptr().add(off), vdivq_f32(one, denom));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = 1.0 / (1.0 + (-a[i]).exp());
    }
}

/// tanh(x) = 2*sigmoid(2x) - 1
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_tanh_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let two_x = vmulq_f32(two, vx);
            let neg_2x = vnegq_f32(two_x);
            let exp_neg = fast_exp_f32x4(neg_2x);
            let denom = vaddq_f32(one, exp_neg);
            let sig2x = vdivq_f32(one, denom);
            let result = vsubq_f32(vmulq_f32(two, sig2x), one);
            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].tanh();
    }
}

/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_gelu_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    unsafe {
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let coeff = vdupq_n_f32(0.044715);
        let s2p = vdupq_n_f32(sqrt_2_over_pi);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let x3 = vmulq_f32(vmulq_f32(vx, vx), vx);
            let inner = vmulq_f32(s2p, vaddq_f32(vx, vmulq_f32(coeff, x3)));
            // tanh via 2*sigmoid(2x)-1
            let two_inner = vmulq_f32(two, inner);
            let neg_2inner = vnegq_f32(two_inner);
            let exp_neg = fast_exp_f32x4(neg_2inner);
            let sig = vdivq_f32(one, vaddq_f32(one, exp_neg));
            let tanh_val = vsubq_f32(vmulq_f32(two, sig), one);
            let result = vmulq_f32(half, vmulq_f32(vx, vaddq_f32(one, tanh_val)));
            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        let inner_val = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        out[i] = 0.5 * x * (1.0 + inner_val.tanh());
    }
}

// ========================================================================
// Phase 5: Additional backward-specific kernels
// ========================================================================

/// abs backward: out[i] = if input[i] >= 0 { grad[i] } else { -grad[i] }
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_abs_backward_f32(input: &[f32], grad: &[f32], out: &mut [f32]) {
    let len = input.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(input.as_ptr().add(off));
            let vg = vld1q_f32(grad.as_ptr().add(off));
            let neg_g = vnegq_f32(vg);
            let mask = vcgeq_f32(vi, zero); // input >= 0
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vg, neg_g));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if input[i] >= 0.0 { grad[i] } else { -grad[i] };
    }
}

/// tanh backward: out[i] = grad[i] * (1 - tanh_out[i]^2)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_tanh_backward_f32(tanh_out: &[f32], grad: &[f32], out: &mut [f32]) {
    let len = tanh_out.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let vy = vld1q_f32(tanh_out.as_ptr().add(off));
            let vg = vld1q_f32(grad.as_ptr().add(off));
            let y2 = vmulq_f32(vy, vy);
            let sech2 = vsubq_f32(one, y2);
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vg, sech2));
        }
    }
    for i in (chunks * 4)..len {
        let y = tanh_out[i];
        out[i] = grad[i] * (1.0 - y * y);
    }
}

/// sigmoid backward: out[i] = grad[i] * sigmoid_out[i] * (1 - sigmoid_out[i])
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sigmoid_backward_f32(sigmoid_out: &[f32], grad: &[f32], out: &mut [f32]) {
    let len = sigmoid_out.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let vs = vld1q_f32(sigmoid_out.as_ptr().add(off));
            let vg = vld1q_f32(grad.as_ptr().add(off));
            let one_minus_s = vsubq_f32(one, vs);
            let ds = vmulq_f32(vs, one_minus_s);
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vg, ds));
        }
    }
    for i in (chunks * 4)..len {
        let s = sigmoid_out[i];
        out[i] = grad[i] * s * (1.0 - s);
    }
}

// ========================================================================
// Phase 4: NEON Adam step
// ========================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn adam_step_f32(
    data: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    bc1: f32,
    bc2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled_wd: bool,
) {
    let len = data.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, m.len());
    debug_assert_eq!(len, v.len());
    let chunks = len / 4;

    unsafe {
        let vbeta1 = vdupq_n_f32(beta1);
        let vbeta2 = vdupq_n_f32(beta2);
        let v1_minus_b1 = vdupq_n_f32(1.0 - beta1);
        let v1_minus_b2 = vdupq_n_f32(1.0 - beta2);
        let vlr = vdupq_n_f32(lr);
        let vbc1 = vdupq_n_f32(bc1);
        let vbc2 = vdupq_n_f32(bc2);
        let veps = vdupq_n_f32(eps);
        let vwd = vdupq_n_f32(weight_decay);
        let vlr_wd = vdupq_n_f32(lr * weight_decay);

        for i in 0..chunks {
            let off = i * 4;
            let vd = vld1q_f32(data.as_ptr().add(off));
            let vg_raw = vld1q_f32(grad.as_ptr().add(off));
            let vm = vld1q_f32(m.as_ptr().add(off));
            let vv = vld1q_f32(v.as_ptr().add(off));

            // Apply L2 regularization to grad if not decoupled
            let vg = if !decoupled_wd && weight_decay != 0.0 {
                vmlaq_f32(vg_raw, vwd, vd)
            } else {
                vg_raw
            };

            // m = beta1 * m + (1 - beta1) * g
            let new_m = vmlaq_f32(vmulq_f32(vbeta1, vm), v1_minus_b1, vg);
            // v = beta2 * v + (1 - beta2) * g * g
            let g2 = vmulq_f32(vg, vg);
            let new_v = vmlaq_f32(vmulq_f32(vbeta2, vv), v1_minus_b2, g2);

            vst1q_f32(m.as_mut_ptr().add(off), new_m);
            vst1q_f32(v.as_mut_ptr().add(off), new_v);

            // Bias-corrected: m_hat = m / bc1, v_hat = v / bc2
            let m_hat = vdivq_f32(new_m, vbc1);
            let v_hat = vdivq_f32(new_v, vbc2);

            // Fast approximate sqrt via vrsqrteq_f32 + one Newton step
            let v_hat_eps = vaddq_f32(v_hat, veps);
            let rsqrt_est = vrsqrteq_f32(v_hat_eps);
            let rsqrt_refined = vmulq_f32(
                rsqrt_est,
                vrsqrtsq_f32(vmulq_f32(v_hat_eps, rsqrt_est), rsqrt_est),
            );
            // sqrt(v_hat+eps) = (v_hat+eps) * rsqrt(v_hat+eps)
            let sqrt_v = vmulq_f32(v_hat_eps, rsqrt_refined);

            // update = lr * m_hat / sqrt_v
            let update = vmulq_f32(vlr, vdivq_f32(m_hat, sqrt_v));

            // Decoupled weight decay
            let mut new_d = vsubq_f32(vd, update);
            if decoupled_wd && weight_decay != 0.0 {
                new_d = vsubq_f32(new_d, vmulq_f32(vlr_wd, vd));
            }

            vst1q_f32(data.as_mut_ptr().add(off), new_d);
        }
    }

    // Scalar tail
    for j in (chunks * 4)..len {
        let mut g = grad[j];
        if weight_decay != 0.0 && !decoupled_wd {
            g += weight_decay * data[j];
        }
        m[j] = beta1 * m[j] + (1.0 - beta1) * g;
        v[j] = beta2 * v[j] + (1.0 - beta2) * g * g;
        let m_hat = m[j] / bc1;
        let v_hat = v[j] / bc2;
        if decoupled_wd && weight_decay != 0.0 {
            data[j] -= lr * weight_decay * data[j];
        }
        data[j] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ========================================================================
// Phase 6: Performance optimization kernels
// ========================================================================

/// leaky_relu: out[i] = if x > 0 { x } else { alpha * x }
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_leaky_relu_f32(a: &[f32], alpha: f32, out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let valpha = vdupq_n_f32(alpha);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let mask = vcgtq_f32(vx, zero); // x > 0
            let scaled = vmulq_f32(valpha, vx);
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vx, scaled));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if a[i] > 0.0 { a[i] } else { alpha * a[i] };
    }
}

/// leaky_relu backward: out[i] = if input > 0 { grad } else { alpha * grad }
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_leaky_relu_backward_f32(input: &[f32], grad: &[f32], alpha: f32, out: &mut [f32]) {
    let len = input.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let valpha = vdupq_n_f32(alpha);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(input.as_ptr().add(off));
            let vg = vld1q_f32(grad.as_ptr().add(off));
            let mask = vcgtq_f32(vi, zero);
            let scaled = vmulq_f32(valpha, vg);
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vg, scaled));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if input[i] > 0.0 { grad[i] } else { alpha * grad[i] };
    }
}

/// elu: out[i] = if x > 0 { x } else { alpha * (exp(x) - 1) }
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_elu_f32(a: &[f32], alpha: f32, out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let valpha = vdupq_n_f32(alpha);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let mask = vcgtq_f32(vx, zero); // x > 0
            let exp_x = fast_exp_f32x4(vx);
            let neg_branch = vmulq_f32(valpha, vsubq_f32(exp_x, one));
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vx, neg_branch));
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        out[i] = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
    }
}

/// elu backward: out[i] = if x > 0 { grad } else { grad * alpha * exp(x) }
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_elu_backward_f32(input: &[f32], grad: &[f32], alpha: f32, out: &mut [f32]) {
    let len = input.len();
    debug_assert_eq!(len, grad.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let valpha = vdupq_n_f32(alpha);
        for i in 0..chunks {
            let off = i * 4;
            let vi = vld1q_f32(input.as_ptr().add(off));
            let vg = vld1q_f32(grad.as_ptr().add(off));
            let mask = vcgtq_f32(vi, zero);
            let exp_x = fast_exp_f32x4(vi);
            let neg_branch = vmulq_f32(vg, vmulq_f32(valpha, exp_x));
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vg, neg_branch));
        }
    }
    for i in (chunks * 4)..len {
        let x = input[i];
        out[i] = if x > 0.0 { grad[i] } else { grad[i] * alpha * x.exp() };
    }
}

/// silu: out[i] = x * sigmoid(x)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_silu_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let neg_x = vnegq_f32(vx);
            let exp_neg_x = fast_exp_f32x4(neg_x);
            let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg_x));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vx, sigmoid));
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        out[i] = x * sig;
    }
}

/// maximum: out[i] = max(a[i], b[i])
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_maximum_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vmaxq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].max(b[i]);
    }
}

/// minimum: out[i] = min(a[i], b[i])
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_minimum_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vminq_f32(va, vb));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].min(b[i]);
    }
}

/// clip: out[i] = clamp(a[i], min, max)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_clip_f32(a: &[f32], min_val: f32, max_val: f32, out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let vmin = vdupq_n_f32(min_val);
        let vmax = vdupq_n_f32(max_val);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let clamped = vmaxq_f32(vminq_f32(vx, vmax), vmin);
            vst1q_f32(out.as_mut_ptr().add(off), clamped);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].clamp(min_val, max_val);
    }
}

/// square: out[i] = a[i] * a[i]
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_square_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(va, va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i] * a[i];
    }
}

/// reciprocal: out[i] = 1.0 / a[i] (via vrecpeq + Newton refinement)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_reciprocal_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let est = vrecpeq_f32(va);
            // One Newton-Raphson step: est = est * (2 - a * est)
            let refined = vmulq_f32(est, vrecpsq_f32(va, est));
            vst1q_f32(out.as_mut_ptr().add(off), refined);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].recip();
    }
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::*;

    #[cfg(target_arch = "aarch64")]
    fn assert_approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for i in 0..a.len() {
            let diff = (a[i] - b[i]).abs();
            let rel = if b[i].abs() > 1e-10 { diff / b[i].abs() } else { diff };
            assert!(
                rel <= tol || diff <= tol,
                "mismatch at [{}]: got {}, expected {}, diff={}, rel={}",
                i, a[i], b[i], diff, rel
            );
        }
    }

    // Use non-multiple-of-4 lengths to test scalar tails
    const TEST_LEN: usize = 67;

    #[cfg(target_arch = "aarch64")]
    fn make_test_data() -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.1).collect();
        let b: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.7 + 1.0).collect();
        (a, b)
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_add() {
        let (a, b) = make_test_data();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_add_f32(&a, &b, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_sub() {
        let (a, b) = make_test_data();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_sub_f32(&a, &b, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_mul() {
        let (a, b) = make_test_data();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_mul_f32(&a, &b, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_div() {
        let (a, b) = make_test_data();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x / y).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_div_f32(&a, &b, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_neg() {
        let (a, _) = make_test_data();
        let expected: Vec<f32> = a.iter().map(|x| -x).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_neg_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_abs() {
        let (a, _) = make_test_data();
        let expected: Vec<f32> = a.iter().map(|x| x.abs()).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_abs_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_scale() {
        let (a, _) = make_test_data();
        let s = 2.5;
        let expected: Vec<f32> = a.iter().map(|x| x * s).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_scale_f32(&a, s, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_relu() {
        let (a, _) = make_test_data();
        let expected: Vec<f32> = a.iter().map(|x| x.max(0.0)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_relu_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_add_inplace() {
        let (a_orig, b) = make_test_data();
        let mut a = a_orig.clone();
        let expected: Vec<f32> = a_orig.iter().zip(&b).map(|(x, y)| x + y).collect();
        vec_add_inplace_f32(&mut a, &b);
        assert_approx_eq(&a, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_relu_backward() {
        let (input, _) = make_test_data();
        let grad: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.3).collect();
        let expected: Vec<f32> = input.iter().zip(&grad)
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_relu_backward_f32(&input, &grad, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_exp() {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.1).collect();
        let expected: Vec<f32> = a.iter().map(|&x| x.exp()).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_exp_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_exp_edge_cases() {
        let a: Vec<f32> = vec![0.0, -100.0, 88.0, -87.0, 1.0, -1.0, 0.5];
        let expected: Vec<f32> = a.iter().map(|x| x.exp()).collect();
        let mut out = vec![0.0f32; a.len()];
        vec_exp_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 2e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_sigmoid() {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.2).collect();
        let expected: Vec<f32> = a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_sigmoid_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_tanh() {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.15).collect();
        let expected: Vec<f32> = a.iter().map(|x| x.tanh()).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_tanh_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_gelu() {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.1).collect();
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let expected: Vec<f32> = a.iter().map(|&x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_gelu_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_abs_backward() {
        let (input, _) = make_test_data();
        let grad: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.3).collect();
        let expected: Vec<f32> = input.iter().zip(&grad)
            .map(|(&x, &g)| if x >= 0.0 { g } else { -g }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_abs_backward_f32(&input, &grad, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_tanh_backward() {
        let tanh_out: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.02).collect();
        let grad: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.1).collect();
        let expected: Vec<f32> = tanh_out.iter().zip(&grad)
            .map(|(&y, &g)| g * (1.0 - y * y)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_tanh_backward_f32(&tanh_out, &grad, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_sigmoid_backward() {
        let sig_out: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 / TEST_LEN as f32).collect();
        let grad: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.1).collect();
        let expected: Vec<f32> = sig_out.iter().zip(&grad)
            .map(|(&s, &g)| g * s * (1.0 - s)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_sigmoid_backward_f32(&sig_out, &grad, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_adam_step() {
        let n = 67;
        let mut data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let grad: Vec<f32> = (0..n).map(|i| (i as f32 - 33.0) * 0.01).collect();
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];

        // Reference scalar computation
        let mut data_ref = data.clone();
        let mut m_ref = m.clone();
        let mut v_ref = v.clone();
        let (beta1, beta2, lr, eps, wd) = (0.9, 0.999, 0.001, 1e-8, 0.01);
        let bc1 = 1.0 - 0.9f32.powi(1);
        let bc2 = 1.0 - 0.999f32.powi(1);

        for j in 0..n {
            let g = grad[j] + wd * data_ref[j]; // L2
            m_ref[j] = beta1 as f32 * m_ref[j] + (1.0 - beta1 as f32) * g;
            v_ref[j] = beta2 as f32 * v_ref[j] + (1.0 - beta2 as f32) * g * g;
            let m_hat = m_ref[j] / bc1;
            let v_hat = v_ref[j] / bc2;
            data_ref[j] -= lr as f32 * m_hat / (v_hat.sqrt() + eps);
        }

        adam_step_f32(
            &mut data, &grad, &mut m, &mut v,
            beta1, beta2, lr, bc1, bc2, eps, wd, false,
        );
        // Allow slightly larger tolerance due to fast rsqrt approximation
        assert_approx_eq(&data, &data_ref, 1e-3);
        assert_approx_eq(&m, &m_ref, 1e-5);
        assert_approx_eq(&v, &v_ref, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_leaky_relu() {
        let (a, _) = make_test_data();
        let alpha = 0.01;
        let expected: Vec<f32> = a.iter().map(|&x| if x > 0.0 { x } else { alpha * x }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_leaky_relu_f32(&a, alpha, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_leaky_relu_backward() {
        let (input, _) = make_test_data();
        let grad: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.3).collect();
        let alpha = 0.01;
        let expected: Vec<f32> = input.iter().zip(&grad)
            .map(|(&x, &g)| if x > 0.0 { g } else { alpha * g }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_leaky_relu_backward_f32(&input, &grad, alpha, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_elu() {
        let (a, _) = make_test_data();
        let alpha = 1.0;
        let expected: Vec<f32> = a.iter().map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_elu_f32(&a, alpha, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_elu_backward() {
        let (input, _) = make_test_data();
        let grad: Vec<f32> = (0..TEST_LEN).map(|i| i as f32 * 0.3).collect();
        let alpha = 1.0;
        let expected: Vec<f32> = input.iter().zip(&grad)
            .map(|(&x, &g)| if x > 0.0 { g } else { g * alpha * x.exp() }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_elu_backward_f32(&input, &grad, alpha, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_silu() {
        let (a, _) = make_test_data();
        let expected: Vec<f32> = a.iter().map(|&x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        }).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_silu_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-5);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_maximum() {
        let (a, b) = make_test_data();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_maximum_f32(&a, &b, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_minimum() {
        let (a, b) = make_test_data();
        let expected: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.min(y)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_minimum_f32(&a, &b, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_clip() {
        let (a, _) = make_test_data();
        let min_val = -1.0;
        let max_val = 1.0;
        let expected: Vec<f32> = a.iter().map(|&x| x.clamp(min_val, max_val)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_clip_f32(&a, min_val, max_val, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_square() {
        let (a, _) = make_test_data();
        let expected: Vec<f32> = a.iter().map(|&x| x * x).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_square_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-6);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_reciprocal() {
        let (_, b) = make_test_data(); // b values are all positive
        let expected: Vec<f32> = b.iter().map(|&x| x.recip()).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_reciprocal_f32(&b, &mut out);
        assert_approx_eq(&out, &expected, 1e-3); // vrecpe + 1 Newton step ~1e-3 precision
    }
}
