use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// vDSP FFI (macOS Accelerate)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[repr(C)]
struct DSPSplitComplex {
    realp: *mut f32,
    imagp: *mut f32,
}

#[cfg(target_os = "macos")]
const FFT_FORWARD: i32 = 1;
#[cfg(target_os = "macos")]
const FFT_INVERSE: i32 = -1;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vDSP_create_fftsetup(log2n: u64, radix: i32) -> *mut std::ffi::c_void;
    fn vDSP_destroy_fftsetup(setup: *mut std::ffi::c_void);
    fn vDSP_fft_zrip(
        setup: *mut std::ffi::c_void,
        c: *mut DSPSplitComplex,
        stride: i64,
        log2n: u64,
        direction: i32,
    );
}

// ---------------------------------------------------------------------------
// Cooley-Tukey radix-2 DIT FFT (portable fallback)
// ---------------------------------------------------------------------------

/// In-place radix-2 DIT FFT on parallel real/imag arrays.
/// `n` must be a power of 2.
fn cooley_tukey_inplace(real: &mut [f32], imag: &mut [f32], inverse: bool) {
    let n = real.len();
    assert_eq!(n, imag.len());
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // Butterfly operations
    let sign: f32 = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * std::f32::consts::PI / len as f32;
        for i in (0..n).step_by(len) {
            for k in 0..half {
                let w_re = (angle * k as f32).cos();
                let w_im = (angle * k as f32).sin();
                let u_re = real[i + k];
                let u_im = imag[i + k];
                let t_re = real[i + k + half] * w_re - imag[i + k + half] * w_im;
                let t_im = real[i + k + half] * w_im + imag[i + k + half] * w_re;
                real[i + k] = u_re + t_re;
                imag[i + k] = u_im + t_im;
                real[i + k + half] = u_re - t_re;
                imag[i + k + half] = u_im - t_im;
            }
        }
        len *= 2;
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for v in real.iter_mut() {
            *v *= scale;
        }
        for v in imag.iter_mut() {
            *v *= scale;
        }
    }
}

/// Next power of two >= n.
fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 1D real FFT: input `[N]` -> complex output `[N/2+1, 2]`.
///
/// `n` specifies the FFT length (zero-padded or truncated). If `None`, uses
/// the input length rounded up to the next power of two.
pub fn rfft(input: &Tensor, n: Option<usize>) -> Tensor {
    let data = input.data();
    let fft_n = next_pow2(n.unwrap_or(data.len()));

    #[cfg(target_os = "macos")]
    {
        let log2n = fft_n.trailing_zeros() as u64;

        let mut real = vec![0.0f32; fft_n / 2];
        let mut imag_buf = vec![0.0f32; fft_n / 2];

        // Pack real input into split-complex format for vDSP_fft_zrip:
        // even-indexed samples go to real, odd-indexed go to imag.
        let copy_len = data.len().min(fft_n);
        let mut padded = vec![0.0f32; fft_n];
        padded[..copy_len].copy_from_slice(&data[..copy_len]);
        for i in 0..fft_n / 2 {
            real[i] = padded[2 * i];
            imag_buf[i] = padded[2 * i + 1];
        }

        unsafe {
            let setup = vDSP_create_fftsetup(log2n, 2);
            assert!(!setup.is_null(), "vDSP_create_fftsetup failed");

            let mut sc = DSPSplitComplex {
                realp: real.as_mut_ptr(),
                imagp: imag_buf.as_mut_ptr(),
            };

            vDSP_fft_zrip(setup, &mut sc, 1, log2n, FFT_FORWARD);
            vDSP_destroy_fftsetup(setup);
        }

        // vDSP packs DC in real[0] and Nyquist in imag[0].
        let out_n = fft_n / 2 + 1;
        let mut out = vec![0.0f32; out_n * 2];
        // DC
        out[0] = real[0] * 0.5;
        out[1] = 0.0;
        // Nyquist
        out[(out_n - 1) * 2] = imag_buf[0] * 0.5;
        out[(out_n - 1) * 2 + 1] = 0.0;
        // Other bins
        for i in 1..fft_n / 2 {
            out[i * 2] = real[i] * 0.5;
            out[i * 2 + 1] = imag_buf[i] * 0.5;
        }

        return Tensor::new(out, vec![out_n, 2], false);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut real = vec![0.0f32; fft_n];
        let mut imag = vec![0.0f32; fft_n];
        let copy_len = data.len().min(fft_n);
        real[..copy_len].copy_from_slice(&data[..copy_len]);

        cooley_tukey_inplace(&mut real, &mut imag, false);

        let out_n = fft_n / 2 + 1;
        let mut out = vec![0.0f32; out_n * 2];
        for i in 0..out_n {
            out[i * 2] = real[i];
            out[i * 2 + 1] = imag[i];
        }
        Tensor::new(out, vec![out_n, 2], false)
    }
}

/// 1D inverse real FFT: complex input `[N/2+1, 2]` -> real output `[N]`.
///
/// `n` is the desired output length. If `None`, uses `2 * (input_bins - 1)`.
pub fn irfft(input: &Tensor, n: Option<usize>) -> Tensor {
    let shape = input.shape();
    assert!(
        shape.len() == 2 && shape[1] == 2,
        "irfft expects [..., 2] complex format"
    );
    let bins = shape[0];
    let fft_n = next_pow2(n.unwrap_or(2 * (bins - 1)));

    let data = input.data();

    #[cfg(target_os = "macos")]
    {
        let log2n = fft_n.trailing_zeros() as u64;

        let mut real = vec![0.0f32; fft_n / 2];
        let mut imag_buf = vec![0.0f32; fft_n / 2];

        // Pack back into vDSP split-complex format (reverse of rfft unpacking).
        // DC -> real[0], Nyquist -> imag[0]
        real[0] = data[0] * 2.0;
        imag_buf[0] = if bins > fft_n / 2 {
            data[(fft_n / 2) * 2] * 2.0
        } else {
            0.0
        };
        for i in 1..fft_n / 2 {
            if i < bins {
                real[i] = data[i * 2] * 2.0;
                imag_buf[i] = data[i * 2 + 1] * 2.0;
            }
        }

        unsafe {
            let setup = vDSP_create_fftsetup(log2n, 2);
            assert!(!setup.is_null(), "vDSP_create_fftsetup failed");

            let mut sc = DSPSplitComplex {
                realp: real.as_mut_ptr(),
                imagp: imag_buf.as_mut_ptr(),
            };

            vDSP_fft_zrip(setup, &mut sc, 1, log2n, FFT_INVERSE);
            vDSP_destroy_fftsetup(setup);
        }

        // Unpack: even from real, odd from imag
        let mut out = vec![0.0f32; fft_n];
        let scale = 1.0 / (fft_n as f32);
        for i in 0..fft_n / 2 {
            out[2 * i] = real[i] * scale;
            out[2 * i + 1] = imag_buf[i] * scale;
        }

        let actual_n = n.unwrap_or(fft_n);
        out.truncate(actual_n);
        return Tensor::new(out, vec![actual_n], false);
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Reconstruct full complex spectrum (Hermitian symmetry)
        let mut real = vec![0.0f32; fft_n];
        let mut imag = vec![0.0f32; fft_n];
        for i in 0..bins.min(fft_n / 2 + 1) {
            real[i] = data[i * 2];
            imag[i] = data[i * 2 + 1];
        }
        // Mirror
        for i in 1..fft_n / 2 {
            if fft_n - i < fft_n {
                real[fft_n - i] = real[i];
                imag[fft_n - i] = -imag[i];
            }
        }

        cooley_tukey_inplace(&mut real, &mut imag, true);

        let actual_n = n.unwrap_or(fft_n);
        real.truncate(actual_n);
        Tensor::new(real, vec![actual_n], false)
    }
}

/// Complex-to-complex 1D FFT.
///
/// Input shape: `[N, 2]` (last dimension = real + imag).
/// Output shape: `[N, 2]`.
pub fn fft(input: &Tensor, n: Option<usize>) -> Tensor {
    let shape = input.shape();
    assert!(
        shape.len() == 2 && shape[1] == 2,
        "fft expects [N, 2] complex format"
    );
    let input_n = shape[0];
    let fft_n = next_pow2(n.unwrap_or(input_n));
    let data = input.data();

    let mut real = vec![0.0f32; fft_n];
    let mut imag = vec![0.0f32; fft_n];
    for i in 0..input_n.min(fft_n) {
        real[i] = data[i * 2];
        imag[i] = data[i * 2 + 1];
    }

    cooley_tukey_inplace(&mut real, &mut imag, false);

    let mut out = vec![0.0f32; fft_n * 2];
    for i in 0..fft_n {
        out[i * 2] = real[i];
        out[i * 2 + 1] = imag[i];
    }
    Tensor::new(out, vec![fft_n, 2], false)
}

/// Complex-to-complex 1D inverse FFT.
///
/// Input shape: `[N, 2]`. Output shape: `[N, 2]`.
pub fn ifft(input: &Tensor, n: Option<usize>) -> Tensor {
    let shape = input.shape();
    assert!(
        shape.len() == 2 && shape[1] == 2,
        "ifft expects [N, 2] complex format"
    );
    let input_n = shape[0];
    let fft_n = next_pow2(n.unwrap_or(input_n));
    let data = input.data();

    let mut real = vec![0.0f32; fft_n];
    let mut imag = vec![0.0f32; fft_n];
    for i in 0..input_n.min(fft_n) {
        real[i] = data[i * 2];
        imag[i] = data[i * 2 + 1];
    }

    cooley_tukey_inplace(&mut real, &mut imag, true);

    let mut out = vec![0.0f32; fft_n * 2];
    for i in 0..fft_n {
        out[i * 2] = real[i];
        out[i * 2 + 1] = imag[i];
    }
    Tensor::new(out, vec![fft_n, 2], false)
}

/// Shift zero-frequency component to center (for 1D complex).
///
/// Input: `[N, 2]`. Rotates the first half with the second half.
pub fn fftshift(input: &Tensor) -> Tensor {
    let shape = input.shape();
    assert!(
        shape.len() == 2 && shape[1] == 2,
        "fftshift expects [N, 2]"
    );
    let n = shape[0];
    let data = input.data();
    let mid = n / 2;
    let mut out = vec![0.0f32; n * 2];
    for i in 0..n {
        let src = (i + mid) % n;
        out[i * 2] = data[src * 2];
        out[i * 2 + 1] = data[src * 2 + 1];
    }
    Tensor::new(out, vec![n, 2], false)
}

/// Inverse of `fftshift`.
pub fn ifftshift(input: &Tensor) -> Tensor {
    let shape = input.shape();
    assert!(
        shape.len() == 2 && shape[1] == 2,
        "ifftshift expects [N, 2]"
    );
    let n = shape[0];
    let data = input.data();
    let mid = (n + 1) / 2; // ceil(n/2)
    let mut out = vec![0.0f32; n * 2];
    for i in 0..n {
        let src = (i + mid) % n;
        out[i * 2] = data[src * 2];
        out[i * 2 + 1] = data[src * 2 + 1];
    }
    Tensor::new(out, vec![n, 2], false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_rfft_dc() {
        // Constant signal -> all energy at DC
        let input = Tensor::new(vec![1.0; 8], vec![8], false);
        let result = rfft(&input, None);
        let shape = result.shape();
        assert_eq!(shape, vec![5, 2]); // N/2+1 = 5, each with [re, im]
        let d = result.data();
        // DC bin should be ~8.0 + 0i
        assert!(
            approx_eq(d[0], 8.0, 0.01),
            "DC real = {} (expected 8.0)",
            d[0]
        );
        assert!(
            approx_eq(d[1], 0.0, 0.01),
            "DC imag = {} (expected 0.0)",
            d[1]
        );
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        // Complex roundtrip: ifft(fft(x)) == x
        let n = 8;
        let mut data = vec![0.0f32; n * 2];
        for i in 0..n {
            data[i * 2] = (i + 1) as f32;
            data[i * 2 + 1] = 0.0;
        }
        let input = Tensor::new(data.clone(), vec![n, 2], false);
        let freq = fft(&input, None);
        let recovered = ifft(&freq, None);
        let rd = recovered.data();
        for i in 0..n {
            assert!(
                approx_eq(rd[i * 2], (i + 1) as f32, 0.01),
                "mismatch at real[{}]: {} vs {}",
                i,
                rd[i * 2],
                (i + 1) as f32,
            );
            assert!(
                approx_eq(rd[i * 2 + 1], 0.0, 0.01),
                "nonzero imag at [{}]: {}",
                i,
                rd[i * 2 + 1],
            );
        }
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip() {
        let n = 8;
        let mut data = vec![0.0f32; n * 2];
        for i in 0..n {
            data[i * 2] = i as f32;
            data[i * 2 + 1] = 0.0;
        }
        let input = Tensor::new(data.clone(), vec![n, 2], false);
        let shifted = fftshift(&input);
        let unshifted = ifftshift(&shifted);
        let d = unshifted.data();
        for i in 0..n {
            assert!(
                approx_eq(d[i * 2], i as f32, 1e-6),
                "mismatch at [{}]",
                i,
            );
        }
    }

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(7), 8);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(9), 16);
    }
}
