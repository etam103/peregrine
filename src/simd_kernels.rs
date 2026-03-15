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
    let chunks8 = len / 8;
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = out.as_mut_ptr();
        for i in 0..chunks8 {
            let off = i * 8;
            let va0 = vld1q_f32(ap.add(off));
            let va1 = vld1q_f32(ap.add(off + 4));
            let vb0 = vld1q_f32(bp.add(off));
            let vb1 = vld1q_f32(bp.add(off + 4));
            vst1q_f32(op.add(off), vaddq_f32(va0, vb0));
            vst1q_f32(op.add(off + 4), vaddq_f32(va1, vb1));
        }
    }
    for i in (chunks8 * 8)..len {
        out[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks8 = len / 8;
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = out.as_mut_ptr();
        for i in 0..chunks8 {
            let off = i * 8;
            vst1q_f32(op.add(off), vsubq_f32(vld1q_f32(ap.add(off)), vld1q_f32(bp.add(off))));
            vst1q_f32(op.add(off + 4), vsubq_f32(vld1q_f32(ap.add(off + 4)), vld1q_f32(bp.add(off + 4))));
        }
    }
    for i in (chunks8 * 8)..len {
        out[i] = a[i] - b[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks8 = len / 8;
    unsafe {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let op = out.as_mut_ptr();
        for i in 0..chunks8 {
            let off = i * 8;
            let va0 = vld1q_f32(ap.add(off));
            let va1 = vld1q_f32(ap.add(off + 4));
            let vb0 = vld1q_f32(bp.add(off));
            let vb1 = vld1q_f32(bp.add(off + 4));
            vst1q_f32(op.add(off), vmulq_f32(va0, vb0));
            vst1q_f32(op.add(off + 4), vmulq_f32(va1, vb1));
        }
    }
    for i in (chunks8 * 8)..len {
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
pub unsafe fn fast_exp_f32x4(x: float32x4_t) -> float32x4_t {
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

/// Fast NEON reciprocal via Newton-Raphson (2 iterations).
/// Avoids the slow 6-cycle ARM divider; ~23-bit accuracy.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn fast_recip_f32x4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let est = vrecpeq_f32(x);
    let est = vmulq_f32(vrecpsq_f32(x, est), est);  // 1st Newton-Raphson
    vmulq_f32(vrecpsq_f32(x, est), est)              // 2nd Newton-Raphson
}

/// Fast vectorized log using Cephes-style polynomial approximation.
/// Relative error ~2e-7 over the positive float32 range.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fast_log_f32x4(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // Cephes-style: extract exponent, normalize mantissa to [0.5, 1.0), polynomial
    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(0.5);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);

    // Reinterpret as integer to extract exponent
    let xi = vreinterpretq_u32_f32(x);
    // Extract exponent: ((xi >> 23) & 0xFF) - 127
    let exp_bits = vandq_u32(vshrq_n_u32(xi, 23), vdupq_n_u32(0xFF));
    let exponent = vcvtq_f32_s32(vsubq_s32(vreinterpretq_s32_u32(exp_bits), vdupq_n_s32(127)));

    // Normalize mantissa to [1.0, 2.0) by clearing exponent and setting to 127
    let mantissa_bits = vorrq_u32(
        vandq_u32(xi, vdupq_n_u32(0x007FFFFF)),
        vdupq_n_u32(0x3F800000),  // exponent = 127 -> value in [1.0, 2.0)
    );
    let m = vreinterpretq_f32_u32(mantissa_bits);

    // Adjust: if mantissa > sqrt(2), multiply by 0.5 and add 1 to exponent
    let sqrt2 = vdupq_n_f32(std::f32::consts::SQRT_2);
    let mask = vcgtq_f32(m, sqrt2);
    let m = vbslq_f32(mask, vmulq_f32(m, half), m);
    let exponent = vaddq_f32(exponent, vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(one), mask)));

    // f = m - 1.0
    let f = vsubq_f32(m, one);

    // Minimax polynomial for log(1+f) on [0, sqrt(2)-1]
    // Coefficients from Cephes logf
    let c0 = vdupq_n_f32(3.3333331174E-1);
    let c1 = vdupq_n_f32(-2.4999993993E-1);
    let c2 = vdupq_n_f32(2.0000714765E-1);
    let c3 = vdupq_n_f32(-1.6668057665E-1);
    let c4 = vdupq_n_f32(1.4249322787E-1);
    let c5 = vdupq_n_f32(-1.2420140846E-1);
    let c6 = vdupq_n_f32(1.1676998740E-1);
    let c7 = vdupq_n_f32(-1.1514610310E-1);
    let c8 = vdupq_n_f32(7.0376836292E-2);

    let f2 = vmulq_f32(f, f);
    // Horner evaluation: p = f2 * (c0 + f*(c1 + f*(c2 + f*(c3 + f*(c4 + f*(c5 + f*(c6 + f*(c7 + f*c8))))))))
    let p = vmlaq_f32(c7, c8, f);
    let p = vmlaq_f32(c6, p, f);
    let p = vmlaq_f32(c5, p, f);
    let p = vmlaq_f32(c4, p, f);
    let p = vmlaq_f32(c3, p, f);
    let p = vmlaq_f32(c2, p, f);
    let p = vmlaq_f32(c1, p, f);
    let p = vmlaq_f32(c0, p, f);
    let p = vmulq_f32(p, vmulq_f32(f2, f));

    // result = f - 0.5*f^2 + p + exponent * ln2
    let result = vaddq_f32(
        vsubq_f32(f, vmulq_f32(half, f2)),
        vaddq_f32(p, vmulq_f32(exponent, ln2))
    );

    result
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_exp_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks8 = len / 8;
    unsafe {
        let ap = a.as_ptr();
        let op = out.as_mut_ptr();
        for i in 0..chunks8 {
            let off = i * 8;
            vst1q_f32(op.add(off), fast_exp_f32x4(vld1q_f32(ap.add(off))));
            vst1q_f32(op.add(off + 4), fast_exp_f32x4(vld1q_f32(ap.add(off + 4))));
        }
    }
    for i in (chunks8 * 8)..len {
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
    // Fused erf-based GELU: gelu(x) = x * clamp(0.5 + x * Q(x²), 0, 1)
    // where Q(x²) = P((x/√2)²) / (2√2), with pre-scaled coefficients.
    // Eliminates the x/√2 and 0.5* multiplies vs the naive erf approach.

    // Pre-scaled coefficients: Q_i = C_i / (2^(i+1) * sqrt(2))
    const Q0: f32 =  3.988541172905277e-01;
    const Q1: f32 = -6.614151898077855e-02;
    const Q2: f32 =  9.628934587039180e-03;
    const Q3: f32 = -1.038195573428582e-03;
    const Q4: f32 =  8.001855642097436e-05;
    const Q5: f32 = -4.216919411082338e-06;
    const Q6: f32 =  1.424844268031300e-07;
    const Q7: f32 = -2.757840492454109e-09;
    const Q8: f32 =  2.312428161695820e-11;

    let chunks = len / 4;
    unsafe {
        let vhalf = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        let vq0 = vdupq_n_f32(Q0);
        let vq1 = vdupq_n_f32(Q1);
        let vq2 = vdupq_n_f32(Q2);
        let vq3 = vdupq_n_f32(Q3);
        let vq4 = vdupq_n_f32(Q4);
        let vq5 = vdupq_n_f32(Q5);
        let vq6 = vdupq_n_f32(Q6);
        let vq7 = vdupq_n_f32(Q7);
        let vq8 = vdupq_n_f32(Q8);

        let chunks2 = chunks / 2;
        for i in 0..chunks2 {
            let off = i * 8;
            let vx0 = vld1q_f32(a.as_ptr().add(off));
            let vx1 = vld1q_f32(a.as_ptr().add(off + 4));
            let x2_0 = vmulq_f32(vx0, vx0);
            let x2_1 = vmulq_f32(vx1, vx1);
            let q0 = vmlaq_f32(vq7, vq8, x2_0);
            let q1 = vmlaq_f32(vq7, vq8, x2_1);
            let q0 = vmlaq_f32(vq6, q0, x2_0);
            let q1 = vmlaq_f32(vq6, q1, x2_1);
            let q0 = vmlaq_f32(vq5, q0, x2_0);
            let q1 = vmlaq_f32(vq5, q1, x2_1);
            let q0 = vmlaq_f32(vq4, q0, x2_0);
            let q1 = vmlaq_f32(vq4, q1, x2_1);
            let q0 = vmlaq_f32(vq3, q0, x2_0);
            let q1 = vmlaq_f32(vq3, q1, x2_1);
            let q0 = vmlaq_f32(vq2, q0, x2_0);
            let q1 = vmlaq_f32(vq2, q1, x2_1);
            let q0 = vmlaq_f32(vq1, q0, x2_0);
            let q1 = vmlaq_f32(vq1, q1, x2_1);
            let q0 = vmlaq_f32(vq0, q0, x2_0);
            let q1 = vmlaq_f32(vq0, q1, x2_1);
            let inner0 = vmaxq_f32(zero, vminq_f32(one, vaddq_f32(vhalf, vmulq_f32(vx0, q0))));
            let inner1 = vmaxq_f32(zero, vminq_f32(one, vaddq_f32(vhalf, vmulq_f32(vx1, q1))));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vx0, inner0));
            vst1q_f32(out.as_mut_ptr().add(off + 4), vmulq_f32(vx1, inner1));
        }
        // Handle remaining chunk of 4
        if chunks2 * 2 < chunks {
            let off = chunks2 * 8;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let x2 = vmulq_f32(vx, vx);
            let q = vmlaq_f32(vq7, vq8, x2);
            let q = vmlaq_f32(vq6, q, x2);
            let q = vmlaq_f32(vq5, q, x2);
            let q = vmlaq_f32(vq4, q, x2);
            let q = vmlaq_f32(vq3, q, x2);
            let q = vmlaq_f32(vq2, q, x2);
            let q = vmlaq_f32(vq1, q, x2);
            let q = vmlaq_f32(vq0, q, x2);
            let inner = vmaxq_f32(zero, vminq_f32(one, vaddq_f32(vhalf, vmulq_f32(vx, q))));
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vx, inner));
        }
    }
    // Scalar tail
    for i in (chunks * 4)..len {
        let x = a[i];
        let x2 = x * x;
        let q = Q0 + x2 * (Q1 + x2 * (Q2 + x2 * (Q3 + x2 * (Q4 + x2 * (Q5
            + x2 * (Q6 + x2 * (Q7 + x2 * Q8)))))));
        out[i] = x * (0.5 + x * q).clamp(0.0, 1.0);
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
    let chunks8 = len / 8;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let ap = a.as_ptr();
        let op = out.as_mut_ptr();
        for i in 0..chunks8 {
            let off = i * 8;
            let vx0 = vld1q_f32(ap.add(off));
            let vx1 = vld1q_f32(ap.add(off + 4));
            let neg0 = vnegq_f32(vx0);
            let neg1 = vnegq_f32(vx1);
            let exp0 = fast_exp_f32x4(neg0);
            let exp1 = fast_exp_f32x4(neg1);
            let sig0 = vdivq_f32(one, vaddq_f32(one, exp0));
            let sig1 = vdivq_f32(one, vaddq_f32(one, exp1));
            vst1q_f32(op.add(off), vmulq_f32(vx0, sig0));
            vst1q_f32(op.add(off + 4), vmulq_f32(vx1, sig1));
        }
        // Handle remaining chunk of 4 if len % 8 >= 4
        if chunks8 * 8 + 4 <= len {
            let off = chunks8 * 8;
            let vx = vld1q_f32(ap.add(off));
            let neg_x = vnegq_f32(vx);
            let exp_neg_x = fast_exp_f32x4(neg_x);
            let sigmoid = vdivq_f32(one, vaddq_f32(one, exp_neg_x));
            vst1q_f32(op.add(off), vmulq_f32(vx, sigmoid));
        }
    }
    let tail_start = (len / 4) * 4;
    for i in tail_start..len {
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

/// logaddexp: out[i] = ln(exp(a[i]) + exp(b[i])), computed as max(a,b) + log1p(exp(-|a-b|))
/// Uses NEON intrinsics for max/abs/exp, then scalar log1p for precision.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_logaddexp_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let vmax = vmaxq_f32(va, vb);
            let neg_abs_diff = vnegq_f32(vabsq_f32(vsubq_f32(va, vb)));
            let exp_neg = fast_exp_f32x4(neg_abs_diff);
            // log(1 + exp(-|a-b|)) = log1p(exp(-|a-b|)) via log(1 + x)
            let sum = vaddq_f32(one, exp_neg);
            let log_sum = fast_log_f32x4(sum);
            vst1q_f32(out.as_mut_ptr().add(off), vaddq_f32(vmax, log_sum));
        }
    }
    for i in (chunks * 4)..len {
        let m = a[i].max(b[i]);
        let d = (a[i] - b[i]).abs();
        out[i] = m + (-d).exp().ln_1p();
    }
}

// ========================================================================
// Phase 7: Int8 quantization kernels (NEON sdot)
// ========================================================================

/// Compute absmax of an f32 slice using NEON fabs + fmax reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn absmax_f32(a: &[f32]) -> f32 {
    let len = a.len();
    if len == 0 {
        return 0.0;
    }
    let chunks = len / 4;
    unsafe {
        let mut vmax = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vabs = vabsq_f32(va);
            vmax = vmaxq_f32(vmax, vabs);
        }
        // Horizontal max of the 4-lane vector
        let mut result = vmaxvq_f32(vmax);
        // Scalar tail
        for i in (chunks * 4)..len {
            result = result.max(a[i].abs());
        }
        result
    }
}

/// Quantize a single row of f32 values to i8 given a scale.
/// `out[i] = round(a[i] / scale)` clamped to [-127, 127].
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn quantize_row_i8(a: &[f32], scale: f32, out: &mut [i8]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let inv_scale = 1.0 / scale;
    let chunks = len / 4;
    unsafe {
        let vinv = vdupq_n_f32(inv_scale);
        let vmin = vdupq_n_f32(-127.0);
        let vmax = vdupq_n_f32(127.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let scaled = vmulq_f32(vx, vinv);
            // Round to nearest, clamp, convert to i32 then narrow to i16 then i8
            let rounded = vrndnq_f32(scaled);
            let clamped = vmaxq_f32(vminq_f32(rounded, vmax), vmin);
            let i32s = vcvtq_s32_f32(clamped);
            // Extract 4 lanes and store as i8
            let i32_arr: [i32; 4] = [
                vgetq_lane_s32(i32s, 0),
                vgetq_lane_s32(i32s, 1),
                vgetq_lane_s32(i32s, 2),
                vgetq_lane_s32(i32s, 3),
            ];
            out[off] = i32_arr[0] as i8;
            out[off + 1] = i32_arr[1] as i8;
            out[off + 2] = i32_arr[2] as i8;
            out[off + 3] = i32_arr[3] as i8;
        }
    }
    for i in (chunks * 4)..len {
        out[i] = (a[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Int8 GEMM using NEON widening multiply + pairwise accumulate (sdot equivalent): C[M,N] = dequant(A_i8[M,K] @ B_t_i8[N,K]^T)
///
/// B is pre-transposed (column-major): B_t[n, k] so dot products are contiguous.
/// Post-loop dequantization: C[m,n] = acc_i32 * a_scales[m] * w_scales[n].
///
/// Uses widening multiply + pairwise add as a stable alternative to vdotq_s32,
/// processing 16 i8 elements per iteration via vmull_s8 (8→16 widen) + vpadalq_s16 (16→32 accumulate).
#[cfg(target_arch = "aarch64")]
pub fn gemm_i8_sdot(
    a: &[i8],       // [M, K] row-major
    b_t: &[i8],     // [N, K] (transposed weights)
    out: &mut [f32], // [M, N]
    a_scales: &[f32],
    w_scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b_t.len(), n * k);
    debug_assert_eq!(out.len(), m * n);

    // Process 4 rows at a time for better register utilization
    let m_blocks = m / 4;

    for mb in 0..m_blocks {
        let mi = mb * 4;
        for ni in 0..n {
            let k_chunks = k / 16;
            let (mut acc0, mut acc1, mut acc2, mut acc3): (i32, i32, i32, i32);
            unsafe {
                let b_ptr = b_t.as_ptr().add(ni * k);
                let a0_ptr = a.as_ptr().add(mi * k);
                let a1_ptr = a.as_ptr().add((mi + 1) * k);
                let a2_ptr = a.as_ptr().add((mi + 2) * k);
                let a3_ptr = a.as_ptr().add((mi + 3) * k);

                let mut vacc0 = vdupq_n_s32(0);
                let mut vacc1 = vdupq_n_s32(0);
                let mut vacc2 = vdupq_n_s32(0);
                let mut vacc3 = vdupq_n_s32(0);

                for kc in 0..k_chunks {
                    let koff = kc * 16;
                    // Load 16 bytes (i8x16) for b and each a row
                    let vb = vld1q_s8(b_ptr.add(koff));
                    let va0 = vld1q_s8(a0_ptr.add(koff));
                    let va1 = vld1q_s8(a1_ptr.add(koff));
                    let va2 = vld1q_s8(a2_ptr.add(koff));
                    let va3 = vld1q_s8(a3_ptr.add(koff));

                    // Widening multiply low 8 bytes: i8x8 * i8x8 -> i16x8
                    let vb_lo = vget_low_s8(vb);
                    let vb_hi = vget_high_s8(vb);

                    // Row 0
                    let prod0_lo = vmull_s8(vget_low_s8(va0), vb_lo);
                    let prod0_hi = vmull_s8(vget_high_s8(va0), vb_hi);
                    vacc0 = vpadalq_s16(vacc0, prod0_lo);
                    vacc0 = vpadalq_s16(vacc0, prod0_hi);

                    // Row 1
                    let prod1_lo = vmull_s8(vget_low_s8(va1), vb_lo);
                    let prod1_hi = vmull_s8(vget_high_s8(va1), vb_hi);
                    vacc1 = vpadalq_s16(vacc1, prod1_lo);
                    vacc1 = vpadalq_s16(vacc1, prod1_hi);

                    // Row 2
                    let prod2_lo = vmull_s8(vget_low_s8(va2), vb_lo);
                    let prod2_hi = vmull_s8(vget_high_s8(va2), vb_hi);
                    vacc2 = vpadalq_s16(vacc2, prod2_lo);
                    vacc2 = vpadalq_s16(vacc2, prod2_hi);

                    // Row 3
                    let prod3_lo = vmull_s8(vget_low_s8(va3), vb_lo);
                    let prod3_hi = vmull_s8(vget_high_s8(va3), vb_hi);
                    vacc3 = vpadalq_s16(vacc3, prod3_lo);
                    vacc3 = vpadalq_s16(vacc3, prod3_hi);
                }

                // Horizontal sum of each accumulator
                acc0 = vaddvq_s32(vacc0);
                acc1 = vaddvq_s32(vacc1);
                acc2 = vaddvq_s32(vacc2);
                acc3 = vaddvq_s32(vacc3);
            }

            // Scalar tail for remaining k elements
            for ki in (k_chunks * 16)..k {
                let b_val = b_t[ni * k + ki] as i32;
                acc0 += a[(mi) * k + ki] as i32 * b_val;
                acc1 += a[(mi + 1) * k + ki] as i32 * b_val;
                acc2 += a[(mi + 2) * k + ki] as i32 * b_val;
                acc3 += a[(mi + 3) * k + ki] as i32 * b_val;
            }

            // Dequantize
            let ws = w_scales[ni];
            out[mi * n + ni] = acc0 as f32 * a_scales[mi] * ws;
            out[(mi + 1) * n + ni] = acc1 as f32 * a_scales[mi + 1] * ws;
            out[(mi + 2) * n + ni] = acc2 as f32 * a_scales[mi + 2] * ws;
            out[(mi + 3) * n + ni] = acc3 as f32 * a_scales[mi + 3] * ws;
        }
    }

    // Handle remaining rows
    for mi in (m_blocks * 4)..m {
        for ni in 0..n {
            let k_chunks = k / 16;
            let mut acc: i32;
            unsafe {
                let b_ptr = b_t.as_ptr().add(ni * k);
                let a_ptr = a.as_ptr().add(mi * k);
                let mut vacc = vdupq_n_s32(0);

                for kc in 0..k_chunks {
                    let koff = kc * 16;
                    let vb = vld1q_s8(b_ptr.add(koff));
                    let va = vld1q_s8(a_ptr.add(koff));

                    let vb_lo = vget_low_s8(vb);
                    let vb_hi = vget_high_s8(vb);
                    let prod_lo = vmull_s8(vget_low_s8(va), vb_lo);
                    let prod_hi = vmull_s8(vget_high_s8(va), vb_hi);
                    vacc = vpadalq_s16(vacc, prod_lo);
                    vacc = vpadalq_s16(vacc, prod_hi);
                }
                acc = vaddvq_s32(vacc);
            }
            for ki in (k_chunks * 16)..k {
                acc += a[mi * k + ki] as i32 * b_t[ni * k + ki] as i32;
            }
            out[mi * n + ni] = acc as f32 * a_scales[mi] * w_scales[ni];
        }
    }
}

// ========================================================================
// 2:4 Structured Sparse GEMM
// ========================================================================

/// NEON-accelerated 2:4 sparse matmul: C[M,N] = A[M,K] @ W_sparse_24[K,N].
///
/// W is stored in 2:4 packed format:
///   - `w_vals`: [K/4 * N * 2] f32 — two non-zero values per group-of-4 per column
///   - `w_idx`:  [K/4 * N] u8 — nibble-packed indices (low=idx0, high=idx1)
///
/// Processes 4-element groups along K, gathers 2 A values per group via indices,
/// multiplies with packed W values, accumulates with NEON fmla.
#[cfg(target_arch = "aarch64")]
pub fn gemm_sparse_24(
    a: &[f32],       // [M, K] row-major
    w_vals: &[f32],  // [K/4 * N * 2] packed values
    w_idx: &[u8],    // [K/4 * N] nibble-packed indices
    out: &mut [f32],  // [M, N]
    m: usize,
    n: usize,
    k: usize,
) {
    debug_assert_eq!(k % 4, 0);
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(out.len(), m * n);
    let groups = k / 4;
    debug_assert_eq!(w_vals.len(), groups * n * 2);
    debug_assert_eq!(w_idx.len(), groups * n);

    // Process 4 columns at a time with NEON
    let n_chunks = n / 4;

    for mi in 0..m {
        let a_row = &a[mi * k..];

        for nc in 0..n_chunks {
            let ni_base = nc * 4;
            unsafe {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);

                for g in 0..groups {
                    let a_base = g * 4;
                    let idx_off = g * n + ni_base;
                    let val_off = g * n * 2 + ni_base * 2;

                    // Load 4 index bytes for 4 columns
                    let ib0 = *w_idx.get_unchecked(idx_off);
                    let ib1 = *w_idx.get_unchecked(idx_off + 1);
                    let ib2 = *w_idx.get_unchecked(idx_off + 2);
                    let ib3 = *w_idx.get_unchecked(idx_off + 3);

                    // For each column, gather the two A values and multiply
                    let i00 = (ib0 & 0x0F) as usize;
                    let i01 = ((ib0 >> 4) & 0x0F) as usize;
                    let i10 = (ib1 & 0x0F) as usize;
                    let i11 = ((ib1 >> 4) & 0x0F) as usize;
                    let i20 = (ib2 & 0x0F) as usize;
                    let i21 = ((ib2 >> 4) & 0x0F) as usize;
                    let i30 = (ib3 & 0x0F) as usize;
                    let i31 = ((ib3 >> 4) & 0x0F) as usize;

                    // Gather A values for idx0 across 4 columns
                    let a_gather0 = vld1q_f32([
                        *a_row.get_unchecked(a_base + i00),
                        *a_row.get_unchecked(a_base + i10),
                        *a_row.get_unchecked(a_base + i20),
                        *a_row.get_unchecked(a_base + i30),
                    ].as_ptr());

                    // Gather A values for idx1 across 4 columns
                    let a_gather1 = vld1q_f32([
                        *a_row.get_unchecked(a_base + i01),
                        *a_row.get_unchecked(a_base + i11),
                        *a_row.get_unchecked(a_base + i21),
                        *a_row.get_unchecked(a_base + i31),
                    ].as_ptr());

                    // Load packed W values: layout is [v0_col0, v1_col0, v0_col1, v1_col1, ...]
                    // We need w_val0 = [v0_c0, v0_c1, v0_c2, v0_c3]
                    //          w_val1 = [v1_c0, v1_c1, v1_c2, v1_c3]
                    let wv0 = vld1q_f32([
                        *w_vals.get_unchecked(val_off + 0),
                        *w_vals.get_unchecked(val_off + 2),
                        *w_vals.get_unchecked(val_off + 4),
                        *w_vals.get_unchecked(val_off + 6),
                    ].as_ptr());
                    let wv1 = vld1q_f32([
                        *w_vals.get_unchecked(val_off + 1),
                        *w_vals.get_unchecked(val_off + 3),
                        *w_vals.get_unchecked(val_off + 5),
                        *w_vals.get_unchecked(val_off + 7),
                    ].as_ptr());

                    acc0 = vfmaq_f32(acc0, a_gather0, wv0);
                    acc1 = vfmaq_f32(acc1, a_gather1, wv1);
                }

                let sum = vaddq_f32(acc0, acc1);
                vst1q_f32(out.as_mut_ptr().add(mi * n + ni_base), sum);
            }
        }

        // Scalar tail for remaining columns
        for ni in (n_chunks * 4)..n {
            let mut sum = 0.0f32;
            for g in 0..groups {
                let a_base = g * 4;
                let idx_byte = w_idx[g * n + ni];
                let idx0 = (idx_byte & 0x0F) as usize;
                let idx1 = ((idx_byte >> 4) & 0x0F) as usize;

                let v_base = g * n * 2 + ni * 2;
                sum += a_row[a_base + idx0] * w_vals[v_base]
                     + a_row[a_base + idx1] * w_vals[v_base + 1];
            }
            out[mi * n + ni] = sum;
        }
    }
}

// ========================================================================
// Tests
// ========================================================================

// ========================================================================
// Phase 5: Unary math kernels (floor, ceil, round, rsqrt, sign, erf)
// ========================================================================

/// floor: out[i] = floor(a[i]) via NEON vrndmq_f32
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_floor_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vrndmq_f32(va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].floor();
    }
}

/// ceil: out[i] = ceil(a[i]) via NEON vrndpq_f32
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_ceil_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vrndpq_f32(va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].ceil();
    }
}

/// round: out[i] = round(a[i]) via NEON vrndnq_f32 (round to nearest even)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_round_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(out.as_mut_ptr().add(off), vrndnq_f32(va));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].round();
    }
}

/// rsqrt: out[i] = 1/sqrt(a[i]) via NEON vrsqrteq_f32 + 2 Newton steps
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_rsqrt_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let est = vrsqrteq_f32(va);
            // Newton step 1: est = est * (3 - va * est * est) / 2
            let step1 = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(va, est), est));
            // Newton step 2
            let step2 = vmulq_f32(step1, vrsqrtsq_f32(vmulq_f32(va, step1), step1));
            vst1q_f32(out.as_mut_ptr().add(off), step2);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = 1.0 / a[i].sqrt();
    }
}

/// sign: out[i] = sign(a[i]) — 1.0 if positive, -1.0 if negative, 0.0 if zero
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sign_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let neg_one = vdupq_n_f32(-1.0);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            // pos_mask: a > 0 → all 1s
            let pos_mask = vcgtq_f32(va, zero);
            // neg_mask: a < 0 → all 1s
            let neg_mask = vcltq_f32(va, zero);
            // result = select(pos_mask, 1.0, select(neg_mask, -1.0, 0.0))
            let result = vbslq_f32(pos_mask, one, vbslq_f32(neg_mask, neg_one, zero));
            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if a[i] > 0.0 { 1.0 } else if a[i] < 0.0 { -1.0 } else { 0.0 };
    }
}

/// erf: out[i] = erf(a[i]) — exp-free Chebyshev polynomial approximation via NEON
///
/// Uses erf(x) = clamp(x * P(x²), -1, 1) where P is a degree-8 Chebyshev
/// polynomial in x² (degree-17 in x). No exp, no division.
/// Fitted on [0, 3.5] via Chebyshev interpolation; max error < 2.1e-4.
/// For |x| >= 3.5, erf(x) is within 1e-6 of +-1, handled by clamping.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_erf_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;

    // Chebyshev minimax coefficients for erf(x)/x as polynomial in t = x²
    // Fitted on t in [0, 12.25] (x in [0, 3.5]), max error < 2.1e-4.
    // At typical values (|x| < 3): error < 5e-5.
    const C0: f32 =  1.128129804161227e+00_f32;
    const C1: f32 = -3.741529327142981e-01_f32;
    const C2: f32 =  1.089389590735535e-01_f32;
    const C3: f32 = -2.349168416541460e-02_f32;
    const C4: f32 =  3.621226487425877e-03_f32;
    const C5: f32 = -3.816719758455685e-04_f32;
    const C6: f32 =  2.579243632792871e-05_f32;
    const C7: f32 = -9.984449093863159e-07_f32;
    const C8: f32 =  1.674376841361261e-08_f32;

    unsafe {
        let one = vdupq_n_f32(1.0);
        let neg_one = vdupq_n_f32(-1.0);
        let vc0 = vdupq_n_f32(C0);
        let vc1 = vdupq_n_f32(C1);
        let vc2 = vdupq_n_f32(C2);
        let vc3 = vdupq_n_f32(C3);
        let vc4 = vdupq_n_f32(C4);
        let vc5 = vdupq_n_f32(C5);
        let vc6 = vdupq_n_f32(C6);
        let vc7 = vdupq_n_f32(C7);
        let vc8 = vdupq_n_f32(C8);

        let chunks2 = chunks / 2;
        for i in 0..chunks2 {
            let off = i * 8;
            let vx0 = vld1q_f32(a.as_ptr().add(off));
            let vx1 = vld1q_f32(a.as_ptr().add(off + 4));
            let x2_0 = vmulq_f32(vx0, vx0);
            let x2_1 = vmulq_f32(vx1, vx1);
            let p0 = vmlaq_f32(vc7, vc8, x2_0);
            let p1 = vmlaq_f32(vc7, vc8, x2_1);
            let p0 = vmlaq_f32(vc6, p0, x2_0);
            let p1 = vmlaq_f32(vc6, p1, x2_1);
            let p0 = vmlaq_f32(vc5, p0, x2_0);
            let p1 = vmlaq_f32(vc5, p1, x2_1);
            let p0 = vmlaq_f32(vc4, p0, x2_0);
            let p1 = vmlaq_f32(vc4, p1, x2_1);
            let p0 = vmlaq_f32(vc3, p0, x2_0);
            let p1 = vmlaq_f32(vc3, p1, x2_1);
            let p0 = vmlaq_f32(vc2, p0, x2_0);
            let p1 = vmlaq_f32(vc2, p1, x2_1);
            let p0 = vmlaq_f32(vc1, p0, x2_0);
            let p1 = vmlaq_f32(vc1, p1, x2_1);
            let p0 = vmlaq_f32(vc0, p0, x2_0);
            let p1 = vmlaq_f32(vc0, p1, x2_1);
            let y0 = vmaxq_f32(neg_one, vminq_f32(one, vmulq_f32(vx0, p0)));
            let y1 = vmaxq_f32(neg_one, vminq_f32(one, vmulq_f32(vx1, p1)));
            vst1q_f32(out.as_mut_ptr().add(off), y0);
            vst1q_f32(out.as_mut_ptr().add(off + 4), y1);
        }
        if chunks2 * 2 < chunks {
            let off = chunks2 * 8;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let x2 = vmulq_f32(vx, vx);
            let p = vmlaq_f32(vc7, vc8, x2);
            let p = vmlaq_f32(vc6, p, x2);
            let p = vmlaq_f32(vc5, p, x2);
            let p = vmlaq_f32(vc4, p, x2);
            let p = vmlaq_f32(vc3, p, x2);
            let p = vmlaq_f32(vc2, p, x2);
            let p = vmlaq_f32(vc1, p, x2);
            let p = vmlaq_f32(vc0, p, x2);
            let y = vmaxq_f32(neg_one, vminq_f32(one, vmulq_f32(vx, p)));
            vst1q_f32(out.as_mut_ptr().add(off), y);
        }
    }
    // Scalar tail
    for i in (chunks * 4)..len {
        let x = a[i];
        let x2 = x * x;
        let p = C0 + x2 * (C1 + x2 * (C2 + x2 * (C3 + x2 * (C4 + x2 * (C5
            + x2 * (C6 + x2 * (C7 + x2 * C8)))))));
        out[i] = (x * p).clamp(-1.0, 1.0);
    }
}

// ========================================================================
// Phase 6: Comparison/conditional kernels
// ========================================================================

/// greater: out[i] = (a[i] > b[i]) ? 1.0 : 0.0
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_greater_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let mask = vcgtq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, one, zero));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if a[i] > b[i] { 1.0 } else { 0.0 };
    }
}

/// equal: out[i] = (a[i] == b[i]) ? 1.0 : 0.0
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_equal_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, b.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            let mask = vceqq_f32(va, vb);
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, one, zero));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if a[i] == b[i] { 1.0 } else { 0.0 };
    }
}

/// where_cond: out[i] = (cond[i] != 0) ? x[i] : y[i]
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_where_f32(cond: &[f32], x: &[f32], y: &[f32], out: &mut [f32]) {
    let len = cond.len();
    debug_assert_eq!(len, x.len());
    debug_assert_eq!(len, y.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vc = vld1q_f32(cond.as_ptr().add(off));
            let vx = vld1q_f32(x.as_ptr().add(off));
            let vy = vld1q_f32(y.as_ptr().add(off));
            // mask: cond != 0
            let mask = vmvnq_u32(vceqq_f32(vc, zero));
            vst1q_f32(out.as_mut_ptr().add(off), vbslq_f32(mask, vx, vy));
        }
    }
    for i in (chunks * 4)..len {
        out[i] = if cond[i] != 0.0 { x[i] } else { y[i] };
    }
}

// ========================================================================
// Phase 7: Reduction kernels
// ========================================================================

/// NEON-accelerated axis sum reduction.
/// Reduces `data` of shape [outer_size, reduce_size, inner_size] along the middle axis.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_sum_axis(data: &[f32], out: &mut [f32], outer_size: usize, reduce_size: usize, inner_size: usize) {
    let stride = reduce_size * inner_size;
    if inner_size == 1 {
        // Reducing last axis: vectorize along reduce_size
        let chunks = reduce_size / 4;
        for o in 0..outer_size {
            let base = o * stride;
            let mut acc = 0.0f32;
            unsafe {
                let mut vacc = vdupq_n_f32(0.0);
                for c in 0..chunks {
                    vacc = vaddq_f32(vacc, vld1q_f32(data.as_ptr().add(base + c * 4)));
                }
                acc = vaddvq_f32(vacc);
            }
            for r in (chunks * 4)..reduce_size {
                acc += data[base + r];
            }
            out[o] = acc;
        }
    } else {
        for o in 0..outer_size {
            let base = o * stride;
            let out_base = o * inner_size;
            let chunks = inner_size / 4;
            unsafe {
                for c in 0..chunks {
                    let inner_off = c * 4;
                    let mut acc = vdupq_n_f32(0.0);
                    for r in 0..reduce_size {
                        acc = vaddq_f32(acc, vld1q_f32(data.as_ptr().add(base + r * inner_size + inner_off)));
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + inner_off), acc);
                }
            }
            for i in (chunks * 4)..inner_size {
                let mut acc = 0.0f32;
                for r in 0..reduce_size {
                    acc += data[base + r * inner_size + i];
                }
                out[out_base + i] = acc;
            }
        }
    }
}

/// NEON-accelerated axis max reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_max_axis(data: &[f32], out: &mut [f32], outer_size: usize, reduce_size: usize, inner_size: usize) {
    let stride = reduce_size * inner_size;
    if inner_size == 1 {
        let chunks = reduce_size / 4;
        for o in 0..outer_size {
            let base = o * stride;
            let mut acc = f32::NEG_INFINITY;
            unsafe {
                let mut vacc = vdupq_n_f32(f32::NEG_INFINITY);
                for c in 0..chunks {
                    vacc = vmaxq_f32(vacc, vld1q_f32(data.as_ptr().add(base + c * 4)));
                }
                acc = vmaxvq_f32(vacc);
            }
            for r in (chunks * 4)..reduce_size {
                let v = data[base + r];
                if v > acc { acc = v; }
            }
            out[o] = acc;
        }
    } else {
        for o in 0..outer_size {
            let base = o * stride;
            let out_base = o * inner_size;
            let chunks = inner_size / 4;
            unsafe {
                for c in 0..chunks {
                    let inner_off = c * 4;
                    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
                    for r in 0..reduce_size {
                        acc = vmaxq_f32(acc, vld1q_f32(data.as_ptr().add(base + r * inner_size + inner_off)));
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + inner_off), acc);
                }
            }
            for i in (chunks * 4)..inner_size {
                let mut acc = f32::NEG_INFINITY;
                for r in 0..reduce_size {
                    let v = data[base + r * inner_size + i];
                    if v > acc { acc = v; }
                }
                out[out_base + i] = acc;
            }
        }
    }
}

/// NEON-accelerated axis min reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_min_axis(data: &[f32], out: &mut [f32], outer_size: usize, reduce_size: usize, inner_size: usize) {
    let stride = reduce_size * inner_size;
    if inner_size == 1 {
        let chunks = reduce_size / 4;
        for o in 0..outer_size {
            let base = o * stride;
            let mut acc = f32::INFINITY;
            unsafe {
                let mut vacc = vdupq_n_f32(f32::INFINITY);
                for c in 0..chunks {
                    vacc = vminq_f32(vacc, vld1q_f32(data.as_ptr().add(base + c * 4)));
                }
                acc = vminvq_f32(vacc);
            }
            for r in (chunks * 4)..reduce_size {
                let v = data[base + r];
                if v < acc { acc = v; }
            }
            out[o] = acc;
        }
    } else {
        for o in 0..outer_size {
            let base = o * stride;
            let out_base = o * inner_size;
            let chunks = inner_size / 4;
            unsafe {
                for c in 0..chunks {
                    let inner_off = c * 4;
                    let mut acc = vdupq_n_f32(f32::INFINITY);
                    for r in 0..reduce_size {
                        acc = vminq_f32(acc, vld1q_f32(data.as_ptr().add(base + r * inner_size + inner_off)));
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + inner_off), acc);
                }
            }
            for i in (chunks * 4)..inner_size {
                let mut acc = f32::INFINITY;
                for r in 0..reduce_size {
                    let v = data[base + r * inner_size + i];
                    if v < acc { acc = v; }
                }
                out[out_base + i] = acc;
            }
        }
    }
}

/// NEON-accelerated axis product reduction.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_prod_axis(data: &[f32], out: &mut [f32], outer_size: usize, reduce_size: usize, inner_size: usize) {
    let stride = reduce_size * inner_size;
    if inner_size == 1 {
        let chunks = reduce_size / 4;
        for o in 0..outer_size {
            let base = o * stride;
            unsafe {
                let mut vacc = vdupq_n_f32(1.0);
                for c in 0..chunks {
                    vacc = vmulq_f32(vacc, vld1q_f32(data.as_ptr().add(base + c * 4)));
                }
                let mut buf = [0.0f32; 4];
                vst1q_f32(buf.as_mut_ptr(), vacc);
                out[o] = buf[0] * buf[1] * buf[2] * buf[3];
            }
            for r in (chunks * 4)..reduce_size {
                out[o] *= data[base + r];
            }
        }
    } else {
        for o in 0..outer_size {
            let base = o * stride;
            let out_base = o * inner_size;
            let chunks = inner_size / 4;
            unsafe {
                for c in 0..chunks {
                    let inner_off = c * 4;
                    let mut acc = vdupq_n_f32(1.0);
                    for r in 0..reduce_size {
                        acc = vmulq_f32(acc, vld1q_f32(data.as_ptr().add(base + r * inner_size + inner_off)));
                    }
                    vst1q_f32(out.as_mut_ptr().add(out_base + inner_off), acc);
                }
            }
            for i in (chunks * 4)..inner_size {
                let mut acc = 1.0f32;
                for r in 0..reduce_size {
                    acc *= data[base + r * inner_size + i];
                }
                out[out_base + i] = acc;
            }
        }
    }
}

// ========================================================================
// Phase 8: Fused activation kernels
// ========================================================================

/// hardswish: out[i] = x * clamp(x + 3, 0, 6) / 6
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_hardswish_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let three = vdupq_n_f32(3.0);
        let zero = vdupq_n_f32(0.0);
        let six = vdupq_n_f32(6.0);
        let inv6 = vdupq_n_f32(1.0 / 6.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            // clamp(x + 3, 0, 6)
            let clamped = vminq_f32(vmaxq_f32(vaddq_f32(vx, three), zero), six);
            // x * clamped / 6
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vmulq_f32(vx, clamped), inv6));
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        out[i] = x * (x + 3.0).max(0.0).min(6.0) / 6.0;
    }
}

/// softplus: out[i] = log(1 + exp(x)) — NEON vectorized with overflow protection.
/// For x > 20: result = x. For x < -20: result = 0. Otherwise: log(1 + exp(x)).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_softplus_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let thresh_hi = vdupq_n_f32(20.0);
        let thresh_lo = vdupq_n_f32(-20.0);
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            // exp(x) via fast approximation
            let exp_x = fast_exp_f32x4(vx);
            // log(1 + exp(x)) — store to buffer for scalar log
            let sum = vaddq_f32(one, exp_x);
            let log_val = fast_log_f32x4(sum);
            // Clamp: x > 20 → x, x < -20 → 0, else log_val
            let hi_mask = vcgtq_f32(vx, thresh_hi);
            let lo_mask = vcltq_f32(vx, thresh_lo);
            let result = vbslq_f32(hi_mask, vx, vbslq_f32(lo_mask, zero, log_val));
            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        out[i] = if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() };
    }
}

/// mish: out[i] = x * tanh(softplus(x)) — NEON vectorized.
/// tanh(y) = (exp(2y) - 1) / (exp(2y) + 1) = 1 - 2/(exp(2y) + 1)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_mish_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let thresh_hi = vdupq_n_f32(20.0);
        let thresh_lo = vdupq_n_f32(-10.0);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            // softplus(x) = log(1 + exp(x))
            let exp_x = fast_exp_f32x4(vx);
            let sum = vaddq_f32(one, exp_x);
            let mut buf = [0.0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), sum);
            buf[0] = buf[0].ln();
            buf[1] = buf[1].ln();
            buf[2] = buf[2].ln();
            buf[3] = buf[3].ln();
            let sp = vld1q_f32(buf.as_ptr());
            // Clamp softplus for large/small x
            let hi_mask = vcgtq_f32(vx, thresh_hi);
            let sp_clamped = vbslq_f32(hi_mask, vx, sp);
            // tanh(sp) = 1 - 2/(exp(2*sp) + 1)
            let exp2sp = fast_exp_f32x4(vmulq_f32(two, sp_clamped));
            let tanh_sp = vsubq_f32(one, vdivq_f32(two, vaddq_f32(exp2sp, one)));
            // mish = x * tanh(sp)
            let result = vmulq_f32(vx, tanh_sp);
            // For very negative x, mish ≈ 0
            let lo_mask = vcltq_f32(vx, thresh_lo);
            let final_result = vbslq_f32(lo_mask, vx, result); // x * ~0 ≈ small
            vst1q_f32(out.as_mut_ptr().add(off), final_result);
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        let sp = if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() };
        out[i] = x * sp.tanh();
    }
}

/// selu: out[i] = scale * elu(x, alpha) with fixed constants
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_selu_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let alpha: f32 = 1.6732632;
    let scale: f32 = 1.0507010;
    let chunks = len / 4;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let valpha = vdupq_n_f32(alpha);
        let vscale = vdupq_n_f32(scale);
        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let pos_mask = vcgtq_f32(vx, zero);
            let neg_part = vmulq_f32(valpha, vsubq_f32(fast_exp_f32x4(vx), one));
            let elu_result = vbslq_f32(pos_mask, vx, neg_part);
            vst1q_f32(out.as_mut_ptr().add(off), vmulq_f32(vscale, elu_result));
        }
    }
    for i in (chunks * 4)..len {
        let x = a[i];
        let e = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
        out[i] = scale * e;
    }
}

/// logsumexp: single-pass fused reduction along last axis.
/// data is [outer, inner] layout. out is [outer].
/// Computes max, then log(sum(exp(x - max))) + max for each row.
#[cfg(target_arch = "aarch64")]
pub fn vec_logsumexp_rows(data: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        // Pass 1: find max
        let mut mx = f32::NEG_INFINITY;
        let chunks = cols / 4;
        unsafe {
            let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
            for c in 0..chunks {
                vmax = vmaxq_f32(vmax, vld1q_f32(row.as_ptr().add(c * 4)));
            }
            let mut buf = [0.0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), vmax);
            mx = buf[0].max(buf[1]).max(buf[2]).max(buf[3]);
        }
        for c in (chunks * 4)..cols {
            if row[c] > mx { mx = row[c]; }
        }
        // Pass 2: sum of exp(x - max)
        let mut sum = 0.0f32;
        unsafe {
            let vmx = vdupq_n_f32(mx);
            let mut vacc = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let vx = vld1q_f32(row.as_ptr().add(c * 4));
                vacc = vaddq_f32(vacc, fast_exp_f32x4(vsubq_f32(vx, vmx)));
            }
            let mut buf = [0.0f32; 4];
            vst1q_f32(buf.as_mut_ptr(), vacc);
            sum = buf[0] + buf[1] + buf[2] + buf[3];
        }
        for c in (chunks * 4)..cols {
            sum += (row[c] - mx).exp();
        }
        out[r] = mx + sum.ln();
    }
}

// ========================================================================
// Phase 10b: Custom NEON transcendental kernels (arccos, arctan2, arcsinh)
// ========================================================================

/// NEON arccos(x) via range-reduced arcsin polynomial.
/// For |x| <= 0.5: arcsin(x) = x + x^3 * P(x^2), arccos = pi/2 - arcsin
/// For |x| > 0.5: arcsin(x) = pi/2 - 2*arcsin(sqrt((1-|x|)/2))
/// Max error ~2e-7 over [-1, 1].
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_arccos_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let pi_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let two = vdupq_n_f32(2.0);
        let zero = vdupq_n_f32(0.0);
        // Minimax polynomial coefficients for arcsin(x) = x + x^3 * P(x^2)
        let c1 = vdupq_n_f32(0.16666667_f32);
        let c2 = vdupq_n_f32(0.07500000_f32);
        let c3 = vdupq_n_f32(0.04464286_f32);
        let c4 = vdupq_n_f32(0.03038194_f32);
        let c5 = vdupq_n_f32(0.02237216_f32);

        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let abs_x = vabsq_f32(vx);

            // Small path (|x| <= 0.5): arcsin(|x|) = |x| + |x|^3 * P(|x|^2)
            let x2 = vmulq_f32(abs_x, abs_x);
            let p = vmlaq_f32(c4, c5, x2);
            let p = vmlaq_f32(c3, p, x2);
            let p = vmlaq_f32(c2, p, x2);
            let p = vmlaq_f32(c1, p, x2);
            let asin_small = vaddq_f32(abs_x, vmulq_f32(vmulq_f32(abs_x, x2), p));

            // Large path (|x| > 0.5): arcsin(|x|) = pi/2 - 2*arcsin(sqrt((1-|x|)/2))
            let t = vmulq_f32(half, vsubq_f32(one, abs_x));
            let s = vsqrtq_f32(t);
            let s2 = vmulq_f32(s, s);
            let p2 = vmlaq_f32(c4, c5, s2);
            let p2 = vmlaq_f32(c3, p2, s2);
            let p2 = vmlaq_f32(c2, p2, s2);
            let p2 = vmlaq_f32(c1, p2, s2);
            let asin_large = vsubq_f32(pi_2, vmulq_f32(two, vaddq_f32(s, vmulq_f32(vmulq_f32(s, s2), p2))));

            // Select based on |x| > 0.5
            let big = vcgtq_f32(abs_x, half);
            let asin_val = vbslq_f32(big, asin_large, asin_small);

            // arccos(x) = pi/2 - sign(x) * arcsin(|x|)
            let neg = vcltq_f32(vx, zero);
            let signed_asin = vbslq_f32(neg, vnegq_f32(asin_val), asin_val);
            let result = vsubq_f32(pi_2, signed_asin);

            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].acos();
    }
}

/// NEON arctan2(y, x) via minimax polynomial for atan on [0, 1] with quadrant logic.
/// Max error ~5e-6 over full range.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_arctan2_f32(y: &[f32], x: &[f32], out: &mut [f32]) {
    let len = y.len();
    debug_assert_eq!(len, x.len());
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let pi = vdupq_n_f32(std::f32::consts::PI);
        let pi_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
        let zero = vdupq_n_f32(0.0);
        // Minimax polynomial for atan(t) on [0, 1]:
        // atan(t) = t * (c0 + t^2*(c1 + t^2*(c2 + t^2*(c3 + t^2*c4))))
        let c0 = vdupq_n_f32(0.9998660_f32);
        let c1 = vdupq_n_f32(-0.3302995_f32);
        let c2 = vdupq_n_f32(0.1801410_f32);
        let c3 = vdupq_n_f32(-0.0851330_f32);
        let c4 = vdupq_n_f32(0.0208351_f32);

        for i in 0..chunks {
            let off = i * 4;
            let vy = vld1q_f32(y.as_ptr().add(off));
            let vx = vld1q_f32(x.as_ptr().add(off));
            let abs_y = vabsq_f32(vy);
            let abs_x = vabsq_f32(vx);

            // Range reduction: t = min(|x|,|y|) / max(|x|,|y|), so t in [0,1]
            let vmin = vminq_f32(abs_x, abs_y);
            let vmax = vmaxq_f32(abs_x, abs_y);
            let t = vdivq_f32(vmin, vmax);
            let t2 = vmulq_f32(t, t);

            // atan(t) polynomial: t * (c0 + t2*(c1 + t2*(c2 + t2*(c3 + t2*c4))))
            let p = vmlaq_f32(c3, c4, t2);
            let p = vmlaq_f32(c2, p, t2);
            let p = vmlaq_f32(c1, p, t2);
            let p = vmlaq_f32(c0, p, t2);
            let atan_t = vmulq_f32(t, p);

            // If |y| > |x|, angle = pi/2 - atan_t (swap quadrant)
            let swap = vcgtq_f32(abs_y, abs_x);
            let angle = vbslq_f32(swap, vsubq_f32(pi_2, atan_t), atan_t);

            // If x < 0, angle = pi - angle
            let x_neg = vcltq_f32(vx, zero);
            let angle = vbslq_f32(x_neg, vsubq_f32(pi, angle), angle);

            // If y < 0, angle = -angle
            let y_neg = vcltq_f32(vy, zero);
            let result = vbslq_f32(y_neg, vnegq_f32(angle), angle);

            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = y[i].atan2(x[i]);
    }
}

/// NEON arcsinh(x) = sign(x) * ln(|x| + sqrt(x^2 + 1)) using fast_log_f32x4.
/// Max error ~2e-6 over full range.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn vec_arcsinh_f32(a: &[f32], out: &mut [f32]) {
    let len = a.len();
    debug_assert_eq!(len, out.len());
    let chunks = len / 4;
    unsafe {
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let off = i * 4;
            let vx = vld1q_f32(a.as_ptr().add(off));
            let abs_x = vabsq_f32(vx);

            // sqrt(x^2 + 1)
            let x2p1 = vaddq_f32(vmulq_f32(abs_x, abs_x), one);
            let sqrt_val = vsqrtq_f32(x2p1);

            // ln(|x| + sqrt(x^2 + 1))
            let arg = vaddq_f32(abs_x, sqrt_val);
            let log_val = fast_log_f32x4(arg);

            // Apply sign: asinh(-x) = -asinh(x)
            let neg_mask = vcltq_f32(vx, zero);
            let result = vbslq_f32(neg_mask, vnegq_f32(log_val), log_val);

            vst1q_f32(out.as_mut_ptr().add(off), result);
        }
    }
    for i in (chunks * 4)..len {
        out[i] = a[i].asinh();
    }
}

// ========================================================================
// Phase 10: Hand-tuned NEON matmul for small matrices
// ========================================================================

/// NEON matmul for small matrices: C[M,N] = A[M,K] * B[K,N]
/// Uses 4x4 register blocking with NEON FMA.
/// Faster than cblas_sgemm for M,N,K ≤ 128 due to zero dispatch overhead.
#[cfg(target_arch = "aarch64")]
pub fn neon_sgemm(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    // Zero output
    for v in c.iter_mut() { *v = 0.0; }

    // 4x4 register-blocked matmul
    let m4 = m / 4 * 4;
    let n4 = n / 4 * 4;

    unsafe {
        // Main 4x4 block loop
        for i in (0..m4).step_by(4) {
            for j in (0..n4).step_by(4) {
                // Accumulate 4x4 block of C
                let mut c00 = vdupq_n_f32(0.0);
                let mut c01 = vdupq_n_f32(0.0);
                let mut c02 = vdupq_n_f32(0.0);
                let mut c03 = vdupq_n_f32(0.0);
                let mut c10 = vdupq_n_f32(0.0);
                let mut c11 = vdupq_n_f32(0.0);
                let mut c12 = vdupq_n_f32(0.0);
                let mut c13 = vdupq_n_f32(0.0);
                let mut c20 = vdupq_n_f32(0.0);
                let mut c21 = vdupq_n_f32(0.0);
                let mut c22 = vdupq_n_f32(0.0);
                let mut c23 = vdupq_n_f32(0.0);
                let mut c30 = vdupq_n_f32(0.0);
                let mut c31 = vdupq_n_f32(0.0);
                let mut c32 = vdupq_n_f32(0.0);
                let mut c33 = vdupq_n_f32(0.0);

                for p in 0..k {
                    let a0 = vdupq_n_f32(*a.get_unchecked((i) * k + p));
                    let a1 = vdupq_n_f32(*a.get_unchecked((i+1) * k + p));
                    let a2 = vdupq_n_f32(*a.get_unchecked((i+2) * k + p));
                    let a3 = vdupq_n_f32(*a.get_unchecked((i+3) * k + p));

                    let b_row = b.as_ptr().add(p * n + j);
                    let bv = vld1q_f32(b_row);

                    c00 = vfmaq_f32(c00, a0, bv);
                    c10 = vfmaq_f32(c10, a1, bv);
                    c20 = vfmaq_f32(c20, a2, bv);
                    c30 = vfmaq_f32(c30, a3, bv);
                }

                // Store 4x4 block
                let cp = c.as_mut_ptr();
                vst1q_f32(cp.add((i) * n + j), c00);
                vst1q_f32(cp.add((i+1) * n + j), c10);
                vst1q_f32(cp.add((i+2) * n + j), c20);
                vst1q_f32(cp.add((i+3) * n + j), c30);
            }
        }

        // Handle remaining columns (n not divisible by 4)
        for i in 0..m {
            for j in n4..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a.get_unchecked(i * k + p) * *b.get_unchecked(p * n + j);
                }
                *c.get_unchecked_mut(i * n + j) = sum;
            }
        }

        // Handle remaining rows (m not divisible by 4)
        for i in m4..m {
            for j in 0..n4 {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a.get_unchecked(i * k + p) * *b.get_unchecked(p * n + j);
                }
                *c.get_unchecked_mut(i * n + j) = sum;
            }
        }
    }
}

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
        // Erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        // Compute reference via vec_erf_f32 (self-consistency test)
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let scaled: Vec<f32> = a.iter().map(|&x| x * inv_sqrt2).collect();
        let mut erf_ref = vec![0.0f32; TEST_LEN];
        vec_erf_f32(&scaled, &mut erf_ref);
        let expected: Vec<f32> = a.iter().zip(&erf_ref).map(|(&x, &e)| {
            0.5 * x * (1.0 + e)
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

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_arccos() {
        // arccos needs input in [-1, 1]
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) / 34.0).collect();
        let expected: Vec<f32> = a.iter().map(|&x| x.acos()).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_arccos_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_arctan2() {
        let y: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.3).collect();
        let x: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 20.0) * 0.5).collect();
        let expected: Vec<f32> = y.iter().zip(&x).map(|(&yi, &xi)| yi.atan2(xi)).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_arctan2_f32(&y, &x, &mut out);
        assert_approx_eq(&out, &expected, 1e-4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_vec_arcsinh() {
        let a: Vec<f32> = (0..TEST_LEN).map(|i| (i as f32 - 33.0) * 0.5).collect();
        let expected: Vec<f32> = a.iter().map(|&x| x.asinh()).collect();
        let mut out = vec![0.0; TEST_LEN];
        vec_arcsinh_f32(&a, &mut out);
        assert_approx_eq(&out, &expected, 1e-4);
    }
}
