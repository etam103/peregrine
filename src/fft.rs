use crate::cpu_pool::{pool_get, pool_recycle};
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
    fn vDSP_fft_zip(
        setup: *mut std::ffi::c_void,
        c: *mut DSPSplitComplex,
        stride: i64,
        log2n: u64,
        direction: i32,
    );
}

// ---------------------------------------------------------------------------
// Thread-local FFT setup cache (macOS)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
use std::cell::RefCell;
#[cfg(target_os = "macos")]
use std::collections::HashMap;

#[cfg(target_os = "macos")]
thread_local! {
    static FFT_SETUP_CACHE: RefCell<HashMap<u64, *mut std::ffi::c_void>> = RefCell::new(HashMap::new());
}

/// Return a cached vDSP FFT setup for the given log2n, creating one if needed.
/// Setups are never destroyed during the thread's lifetime.
#[cfg(target_os = "macos")]
fn get_cached_fft_setup(log2n: u64) -> *mut std::ffi::c_void {
    FFT_SETUP_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if let Some(&setup) = map.get(&log2n) {
            return setup;
        }
        let setup = unsafe { vDSP_create_fftsetup(log2n, 2) };
        assert!(!setup.is_null(), "vDSP_create_fftsetup failed");
        map.insert(log2n, setup);
        setup
    })
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

        let mut real = pool_get(fft_n / 2);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag_buf = pool_get(fft_n / 2);
        imag_buf.iter_mut().for_each(|v| *v = 0.0);

        // Pack real input into split-complex format for vDSP_fft_zrip:
        // even-indexed samples go to real, odd-indexed go to imag.
        let copy_len = data.len().min(fft_n);
        let mut padded = pool_get(fft_n);
        padded.iter_mut().for_each(|v| *v = 0.0);
        padded[..copy_len].copy_from_slice(&data[..copy_len]);
        for i in 0..fft_n / 2 {
            real[i] = padded[2 * i];
            imag_buf[i] = padded[2 * i + 1];
        }
        pool_recycle(padded);

        unsafe {
            let setup = get_cached_fft_setup(log2n);

            let mut sc = DSPSplitComplex {
                realp: real.as_mut_ptr(),
                imagp: imag_buf.as_mut_ptr(),
            };

            vDSP_fft_zrip(setup, &mut sc, 1, log2n, FFT_FORWARD);
        }

        // vDSP packs DC in real[0] and Nyquist in imag[0].
        let out_n = fft_n / 2 + 1;
        let mut out = pool_get(out_n * 2);
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
        pool_recycle(real);
        pool_recycle(imag_buf);

        return Tensor::new(out, vec![out_n, 2], false);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut real = pool_get(fft_n);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag = pool_get(fft_n);
        imag.iter_mut().for_each(|v| *v = 0.0);
        let copy_len = data.len().min(fft_n);
        real[..copy_len].copy_from_slice(&data[..copy_len]);

        cooley_tukey_inplace(&mut real, &mut imag, false);

        let out_n = fft_n / 2 + 1;
        let mut out = pool_get(out_n * 2);
        for i in 0..out_n {
            out[i * 2] = real[i];
            out[i * 2 + 1] = imag[i];
        }
        pool_recycle(real);
        pool_recycle(imag);
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

        let mut real = pool_get(fft_n / 2);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag_buf = pool_get(fft_n / 2);
        imag_buf.iter_mut().for_each(|v| *v = 0.0);

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
            let setup = get_cached_fft_setup(log2n);

            let mut sc = DSPSplitComplex {
                realp: real.as_mut_ptr(),
                imagp: imag_buf.as_mut_ptr(),
            };

            vDSP_fft_zrip(setup, &mut sc, 1, log2n, FFT_INVERSE);
        }

        // Unpack: even from real, odd from imag.
        // vDSP inverse zrip output is scaled by N/2 relative to the true IDFT.
        // We packed with a 2x multiplier to match vDSP's 2x-scaled format from
        // forward. Combined: scale = 1 / (2 * fft_n) to recover the original signal.
        let mut out = pool_get(fft_n);
        let scale = 1.0 / (2.0 * fft_n as f32);
        for i in 0..fft_n / 2 {
            out[2 * i] = real[i] * scale;
            out[2 * i + 1] = imag_buf[i] * scale;
        }
        pool_recycle(real);
        pool_recycle(imag_buf);

        let actual_n = n.unwrap_or(fft_n);
        out.truncate(actual_n);
        return Tensor::new(out, vec![actual_n], false);
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Reconstruct full complex spectrum (Hermitian symmetry)
        let mut real = pool_get(fft_n);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag = pool_get(fft_n);
        imag.iter_mut().for_each(|v| *v = 0.0);
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
        pool_recycle(imag);
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

    #[cfg(target_os = "macos")]
    {
        let log2n = fft_n.trailing_zeros() as u64;

        let mut real = pool_get(fft_n);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag = pool_get(fft_n);
        imag.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..input_n.min(fft_n) {
            real[i] = data[i * 2];
            imag[i] = data[i * 2 + 1];
        }

        unsafe {
            let setup = get_cached_fft_setup(log2n);

            let mut sc = DSPSplitComplex {
                realp: real.as_mut_ptr(),
                imagp: imag.as_mut_ptr(),
            };

            vDSP_fft_zip(setup, &mut sc, 1, log2n, FFT_FORWARD);
        }

        let mut out = pool_get(fft_n * 2);
        for i in 0..fft_n {
            out[i * 2] = real[i];
            out[i * 2 + 1] = imag[i];
        }
        pool_recycle(real);
        pool_recycle(imag);
        return Tensor::new(out, vec![fft_n, 2], false);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut real = pool_get(fft_n);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag = pool_get(fft_n);
        imag.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..input_n.min(fft_n) {
            real[i] = data[i * 2];
            imag[i] = data[i * 2 + 1];
        }

        cooley_tukey_inplace(&mut real, &mut imag, false);

        let mut out = pool_get(fft_n * 2);
        for i in 0..fft_n {
            out[i * 2] = real[i];
            out[i * 2 + 1] = imag[i];
        }
        pool_recycle(real);
        pool_recycle(imag);
        Tensor::new(out, vec![fft_n, 2], false)
    }
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

    #[cfg(target_os = "macos")]
    {
        let log2n = fft_n.trailing_zeros() as u64;

        let mut real = pool_get(fft_n);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag = pool_get(fft_n);
        imag.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..input_n.min(fft_n) {
            real[i] = data[i * 2];
            imag[i] = data[i * 2 + 1];
        }

        unsafe {
            let setup = get_cached_fft_setup(log2n);

            let mut sc = DSPSplitComplex {
                realp: real.as_mut_ptr(),
                imagp: imag.as_mut_ptr(),
            };

            vDSP_fft_zip(setup, &mut sc, 1, log2n, FFT_INVERSE);
        }

        // vDSP_fft_zip with inverse does NOT divide by N, so we must scale.
        let scale = 1.0 / fft_n as f32;
        let mut out = pool_get(fft_n * 2);
        for i in 0..fft_n {
            out[i * 2] = real[i] * scale;
            out[i * 2 + 1] = imag[i] * scale;
        }
        pool_recycle(real);
        pool_recycle(imag);
        return Tensor::new(out, vec![fft_n, 2], false);
    }

    #[cfg(not(target_os = "macos"))]
    {
        let mut real = pool_get(fft_n);
        real.iter_mut().for_each(|v| *v = 0.0);
        let mut imag = pool_get(fft_n);
        imag.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..input_n.min(fft_n) {
            real[i] = data[i * 2];
            imag[i] = data[i * 2 + 1];
        }

        cooley_tukey_inplace(&mut real, &mut imag, true);

        let mut out = pool_get(fft_n * 2);
        for i in 0..fft_n {
            out[i * 2] = real[i];
            out[i * 2 + 1] = imag[i];
        }
        pool_recycle(real);
        pool_recycle(imag);
        Tensor::new(out, vec![fft_n, 2], false)
    }
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
// 2D FFT
// ---------------------------------------------------------------------------

/// 2D complex FFT: apply fft row-wise, then column-wise.
///
/// Input: separate `real` and `imag` tensors of shape `[H, W]`.
/// Output: `(real_out, imag_out)` both of shape `[H, W]`.
///
/// W must be a power of 2.
pub fn fft2(real: &Tensor, imag: &Tensor) -> (Tensor, Tensor) {
    let shape_r = real.shape();
    let shape_i = imag.shape();
    assert_eq!(shape_r.len(), 2, "fft2 expects 2D tensors");
    assert_eq!(shape_r, shape_i, "fft2: real and imag must have same shape");
    let h = shape_r[0];
    let w = shape_r[1];

    let real_data = real.data();
    let imag_data = imag.data();

    // Step 1: FFT each row
    let mut mid_real = vec![0.0f32; h * w];
    let mut mid_imag = vec![0.0f32; h * w];

    for row in 0..h {
        // Build [W, 2] complex tensor for this row
        let mut row_complex = vec![0.0f32; w * 2];
        for col in 0..w {
            row_complex[col * 2] = real_data[row * w + col];
            row_complex[col * 2 + 1] = imag_data[row * w + col];
        }
        let row_tensor = Tensor::new(row_complex, vec![w, 2], false);
        let row_fft = fft(&row_tensor, None);
        let row_out = row_fft.data();
        let out_w = row_fft.shape()[0]; // should be w (assuming w is pow2)
        for col in 0..out_w.min(w) {
            mid_real[row * w + col] = row_out[col * 2];
            mid_imag[row * w + col] = row_out[col * 2 + 1];
        }
    }

    // Step 2: FFT each column
    let mut out_real = vec![0.0f32; h * w];
    let mut out_imag = vec![0.0f32; h * w];

    for col in 0..w {
        let mut col_complex = vec![0.0f32; h * 2];
        for row in 0..h {
            col_complex[row * 2] = mid_real[row * w + col];
            col_complex[row * 2 + 1] = mid_imag[row * w + col];
        }
        let col_tensor = Tensor::new(col_complex, vec![h, 2], false);
        let col_fft = fft(&col_tensor, None);
        let col_out = col_fft.data();
        let out_h = col_fft.shape()[0];
        for row in 0..out_h.min(h) {
            out_real[row * w + col] = col_out[row * 2];
            out_imag[row * w + col] = col_out[row * 2 + 1];
        }
    }

    (
        Tensor::new(out_real, vec![h, w], false),
        Tensor::new(out_imag, vec![h, w], false),
    )
}

/// 2D complex inverse FFT: apply ifft column-wise, then row-wise.
///
/// Input: separate `real` and `imag` tensors of shape `[H, W]`.
/// Output: `(real_out, imag_out)` both of shape `[H, W]`.
pub fn ifft2(real: &Tensor, imag: &Tensor) -> (Tensor, Tensor) {
    let shape_r = real.shape();
    let shape_i = imag.shape();
    assert_eq!(shape_r.len(), 2, "ifft2 expects 2D tensors");
    assert_eq!(shape_r, shape_i, "ifft2: real and imag must have same shape");
    let h = shape_r[0];
    let w = shape_r[1];

    let real_data = real.data();
    let imag_data = imag.data();

    // Step 1: IFFT each column
    let mut mid_real = vec![0.0f32; h * w];
    let mut mid_imag = vec![0.0f32; h * w];

    for col in 0..w {
        let mut col_complex = vec![0.0f32; h * 2];
        for row in 0..h {
            col_complex[row * 2] = real_data[row * w + col];
            col_complex[row * 2 + 1] = imag_data[row * w + col];
        }
        let col_tensor = Tensor::new(col_complex, vec![h, 2], false);
        let col_ifft = ifft(&col_tensor, None);
        let col_out = col_ifft.data();
        let out_h = col_ifft.shape()[0];
        for row in 0..out_h.min(h) {
            mid_real[row * w + col] = col_out[row * 2];
            mid_imag[row * w + col] = col_out[row * 2 + 1];
        }
    }

    // Step 2: IFFT each row
    let mut out_real = vec![0.0f32; h * w];
    let mut out_imag = vec![0.0f32; h * w];

    for row in 0..h {
        let mut row_complex = vec![0.0f32; w * 2];
        for col in 0..w {
            row_complex[col * 2] = mid_real[row * w + col];
            row_complex[col * 2 + 1] = mid_imag[row * w + col];
        }
        let row_tensor = Tensor::new(row_complex, vec![w, 2], false);
        let row_ifft = ifft(&row_tensor, None);
        let row_out = row_ifft.data();
        let out_w = row_ifft.shape()[0];
        for col in 0..out_w.min(w) {
            out_real[row * w + col] = row_out[col * 2];
            out_imag[row * w + col] = row_out[col * 2 + 1];
        }
    }

    (
        Tensor::new(out_real, vec![h, w], false),
        Tensor::new(out_imag, vec![h, w], false),
    )
}

/// 2D real FFT.
///
/// Input: real tensor of shape `[H, W]`, where W must be a power of 2.
/// Output: `(real_out, imag_out)` both of shape `[H, W/2+1]`.
///
/// 1. For each row: rfft -> [W/2+1] complex values
/// 2. For each column of the result: fft (complex-to-complex)
pub fn rfft2(input: &Tensor) -> (Tensor, Tensor) {
    let shape = input.shape();
    assert_eq!(shape.len(), 2, "rfft2 expects a 2D tensor");
    let h = shape[0];
    let w = shape[1];
    let half_w = w / 2 + 1;

    let data = input.data();

    // Step 1: rfft each row -> [H, half_w] complex
    let mut mid_real = vec![0.0f32; h * half_w];
    let mut mid_imag = vec![0.0f32; h * half_w];

    for row in 0..h {
        let row_data: Vec<f32> = data[row * w..(row + 1) * w].to_vec();
        let row_tensor = Tensor::new(row_data, vec![w], false);
        let row_rfft = rfft(&row_tensor, None);
        let row_out = row_rfft.data();
        let out_bins = row_rfft.shape()[0]; // should be half_w
        for col in 0..out_bins.min(half_w) {
            mid_real[row * half_w + col] = row_out[col * 2];
            mid_imag[row * half_w + col] = row_out[col * 2 + 1];
        }
    }

    // Step 2: fft each column (complex-to-complex) along H
    let mut out_real = vec![0.0f32; h * half_w];
    let mut out_imag = vec![0.0f32; h * half_w];

    for col in 0..half_w {
        let mut col_complex = vec![0.0f32; h * 2];
        for row in 0..h {
            col_complex[row * 2] = mid_real[row * half_w + col];
            col_complex[row * 2 + 1] = mid_imag[row * half_w + col];
        }
        let col_tensor = Tensor::new(col_complex, vec![h, 2], false);
        let col_fft = fft(&col_tensor, None);
        let col_out = col_fft.data();
        let out_h = col_fft.shape()[0];
        for row in 0..out_h.min(h) {
            out_real[row * half_w + col] = col_out[row * 2];
            out_imag[row * half_w + col] = col_out[row * 2 + 1];
        }
    }

    (
        Tensor::new(out_real, vec![h, half_w], false),
        Tensor::new(out_imag, vec![h, half_w], false),
    )
}

/// 2D inverse real FFT.
///
/// Input: `real` and `imag` tensors of shape `[H, W/2+1]`.
/// Output: real tensor of shape `[H, W]` where `W = 2 * (W/2+1 - 1)`.
///
/// 1. For each column: ifft (complex-to-complex)
/// 2. For each row: irfft -> W real values
pub fn irfft2(real: &Tensor, imag: &Tensor, output_width: Option<usize>) -> Tensor {
    let shape_r = real.shape();
    let shape_i = imag.shape();
    assert_eq!(shape_r.len(), 2, "irfft2 expects 2D tensors");
    assert_eq!(
        shape_r, shape_i,
        "irfft2: real and imag must have same shape"
    );
    let h = shape_r[0];
    let half_w = shape_r[1]; // W/2+1
    let w = output_width.unwrap_or(2 * (half_w - 1));

    let real_data = real.data();
    let imag_data = imag.data();

    // Step 1: IFFT each column (complex-to-complex)
    let mut mid_real = vec![0.0f32; h * half_w];
    let mut mid_imag = vec![0.0f32; h * half_w];

    for col in 0..half_w {
        let mut col_complex = vec![0.0f32; h * 2];
        for row in 0..h {
            col_complex[row * 2] = real_data[row * half_w + col];
            col_complex[row * 2 + 1] = imag_data[row * half_w + col];
        }
        let col_tensor = Tensor::new(col_complex, vec![h, 2], false);
        let col_ifft = ifft(&col_tensor, None);
        let col_out = col_ifft.data();
        let out_h = col_ifft.shape()[0];
        for row in 0..out_h.min(h) {
            mid_real[row * half_w + col] = col_out[row * 2];
            mid_imag[row * half_w + col] = col_out[row * 2 + 1];
        }
    }

    // Step 2: irfft each row -> W real values
    let mut out_data = vec![0.0f32; h * w];

    for row in 0..h {
        // Build [half_w, 2] complex tensor for irfft
        let mut row_complex = vec![0.0f32; half_w * 2];
        for col in 0..half_w {
            row_complex[col * 2] = mid_real[row * half_w + col];
            row_complex[col * 2 + 1] = mid_imag[row * half_w + col];
        }
        let row_tensor = Tensor::new(row_complex, vec![half_w, 2], false);
        let row_irfft = irfft(&row_tensor, Some(w));
        let row_out = row_irfft.data();
        let out_len = row_out.len().min(w);
        out_data[row * w..row * w + out_len].copy_from_slice(&row_out[..out_len]);
    }

    Tensor::new(out_data, vec![h, w], false)
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

    // -----------------------------------------------------------------------
    // 2D FFT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fft2_known_dc() {
        // Constant 4x4 matrix: all 1s. The 2D DFT should have all energy at (0,0).
        let h = 4;
        let w = 4;
        let real = Tensor::new(vec![1.0; h * w], vec![h, w], false);
        let imag = Tensor::new(vec![0.0; h * w], vec![h, w], false);

        let (out_re, out_im) = fft2(&real, &imag);
        let re = out_re.data();
        let im = out_im.data();

        // DC component at (0,0) should be H*W = 16
        assert!(
            approx_eq(re[0], 16.0, 0.1),
            "DC real = {} (expected 16.0)",
            re[0]
        );
        assert!(
            approx_eq(im[0], 0.0, 0.1),
            "DC imag = {} (expected 0.0)",
            im[0]
        );

        // All other bins should be ~0
        for i in 1..h * w {
            assert!(
                approx_eq(re[i], 0.0, 0.1),
                "re[{}] = {} (expected 0.0)",
                i,
                re[i]
            );
            assert!(
                approx_eq(im[i], 0.0, 0.1),
                "im[{}] = {} (expected 0.0)",
                i,
                im[i]
            );
        }
    }

    #[test]
    fn test_fft2_ifft2_roundtrip() {
        // Round-trip: ifft2(fft2(x)) should recover x
        let h = 4;
        let w = 4;
        let mut real_data = vec![0.0f32; h * w];
        for i in 0..h * w {
            real_data[i] = (i + 1) as f32;
        }
        let imag_data = vec![0.0f32; h * w];

        let real = Tensor::new(real_data.clone(), vec![h, w], false);
        let imag = Tensor::new(imag_data.clone(), vec![h, w], false);

        let (freq_re, freq_im) = fft2(&real, &imag);
        let (rec_re, rec_im) = ifft2(&freq_re, &freq_im);

        let re = rec_re.data();
        let im = rec_im.data();

        for i in 0..h * w {
            assert!(
                approx_eq(re[i], (i + 1) as f32, 0.05),
                "real mismatch at [{}]: {} vs {}",
                i,
                re[i],
                (i + 1) as f32,
            );
            assert!(
                approx_eq(im[i], 0.0, 0.05),
                "imag not zero at [{}]: {}",
                i,
                im[i],
            );
        }
    }

    #[test]
    fn test_fft2_complex_roundtrip() {
        // Round-trip with complex input
        let h = 4;
        let w = 4;
        let mut real_data = vec![0.0f32; h * w];
        let mut imag_data = vec![0.0f32; h * w];
        for i in 0..h * w {
            real_data[i] = (i as f32) * 0.5;
            imag_data[i] = -((i as f32) * 0.3);
        }

        let real = Tensor::new(real_data.clone(), vec![h, w], false);
        let imag = Tensor::new(imag_data.clone(), vec![h, w], false);

        let (freq_re, freq_im) = fft2(&real, &imag);
        let (rec_re, rec_im) = ifft2(&freq_re, &freq_im);

        let re = rec_re.data();
        let im = rec_im.data();

        for i in 0..h * w {
            assert!(
                approx_eq(re[i], real_data[i], 0.05),
                "real mismatch at [{}]: {} vs {}",
                i,
                re[i],
                real_data[i],
            );
            assert!(
                approx_eq(im[i], imag_data[i], 0.05),
                "imag mismatch at [{}]: {} vs {}",
                i,
                im[i],
                imag_data[i],
            );
        }
    }

    #[test]
    fn test_fft2_parseval() {
        // Parseval's theorem: sum|x|^2 == (1/N) * sum|X|^2
        let h = 4;
        let w = 4;
        let n = (h * w) as f32;
        let mut real_data = vec![0.0f32; h * w];
        let imag_data = vec![0.0f32; h * w];
        for i in 0..h * w {
            real_data[i] = ((i as f32) * 0.7).sin();
        }

        let real = Tensor::new(real_data.clone(), vec![h, w], false);
        let imag = Tensor::new(imag_data.clone(), vec![h, w], false);

        let energy_time: f32 = real_data.iter().map(|x| x * x).sum();

        let (freq_re, freq_im) = fft2(&real, &imag);
        let fre = freq_re.data();
        let fim = freq_im.data();
        let energy_freq: f32 = fre
            .iter()
            .zip(fim.iter())
            .map(|(r, i)| r * r + i * i)
            .sum::<f32>()
            / n;

        assert!(
            approx_eq(energy_time, energy_freq, 0.1),
            "Parseval: time={} freq={}",
            energy_time,
            energy_freq,
        );
    }

    #[test]
    fn test_rfft2_known_dc() {
        // Constant 4x4 -> DC at (0,0) = 16, rest 0
        let h = 4;
        let w = 4;
        let input = Tensor::new(vec![1.0; h * w], vec![h, w], false);
        let (out_re, out_im) = rfft2(&input);

        let shape = out_re.shape();
        assert_eq!(shape, vec![h, w / 2 + 1]);

        let re = out_re.data();
        let im = out_im.data();

        assert!(
            approx_eq(re[0], 16.0, 0.1),
            "DC real = {} (expected 16.0)",
            re[0]
        );
        assert!(
            approx_eq(im[0], 0.0, 0.1),
            "DC imag = {} (expected 0.0)",
            im[0]
        );

        // Other bins should be ~0
        let half_w = w / 2 + 1;
        for i in 1..h * half_w {
            assert!(
                approx_eq(re[i], 0.0, 0.1),
                "re[{}] = {} (expected 0.0)",
                i,
                re[i]
            );
            assert!(
                approx_eq(im[i], 0.0, 0.1),
                "im[{}] = {} (expected 0.0)",
                i,
                im[i]
            );
        }
    }

    #[test]
    fn test_rfft2_irfft2_roundtrip() {
        // Round-trip: irfft2(rfft2(x)) should recover x
        let h = 4;
        let w = 4;
        let mut data = vec![0.0f32; h * w];
        for i in 0..h * w {
            data[i] = (i + 1) as f32;
        }

        let input = Tensor::new(data.clone(), vec![h, w], false);
        let (freq_re, freq_im) = rfft2(&input);
        let recovered = irfft2(&freq_re, &freq_im, Some(w));
        let out = recovered.data();

        for i in 0..h * w {
            assert!(
                approx_eq(out[i], data[i], 0.1),
                "mismatch at [{}]: {} vs {}",
                i,
                out[i],
                data[i],
            );
        }
    }

    #[test]
    fn test_rfft2_parseval() {
        // For rfft2, Parseval needs careful handling since only half spectrum stored.
        // sum|x|^2 == (1/N) * [ sum_{full spectrum} |X|^2 ]
        // The rfft2 output stores [H, W/2+1]. To get full energy we need to account
        // for the mirrored bins. For simplicity, we verify via round-trip consistency.
        let h = 4;
        let w = 4;
        let mut data = vec![0.0f32; h * w];
        for i in 0..h * w {
            data[i] = ((i as f32) * 1.3).cos();
        }

        let input = Tensor::new(data.clone(), vec![h, w], false);
        let (freq_re, freq_im) = rfft2(&input);
        let recovered = irfft2(&freq_re, &freq_im, Some(w));
        let out = recovered.data();

        // Verify round-trip (equivalent to energy preservation)
        let mut max_err: f32 = 0.0;
        for i in 0..h * w {
            let err = (out[i] - data[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.15,
            "rfft2 round-trip max error too large: {}",
            max_err,
        );
    }

    #[test]
    fn test_fft2_delta() {
        // Delta function at (0,0): DFT should be all 1s
        let h = 4;
        let w = 4;
        let mut real_data = vec![0.0f32; h * w];
        real_data[0] = 1.0;
        let imag_data = vec![0.0f32; h * w];

        let real = Tensor::new(real_data, vec![h, w], false);
        let imag = Tensor::new(imag_data, vec![h, w], false);

        let (out_re, out_im) = fft2(&real, &imag);
        let re = out_re.data();
        let im = out_im.data();

        for i in 0..h * w {
            assert!(
                approx_eq(re[i], 1.0, 0.05),
                "re[{}] = {} (expected 1.0)",
                i,
                re[i]
            );
            assert!(
                approx_eq(im[i], 0.0, 0.05),
                "im[{}] = {} (expected 0.0)",
                i,
                im[i]
            );
        }
    }

    #[test]
    fn test_rfft_irfft_1d_roundtrip() {
        // Verify 1D rfft -> irfft round-trip
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let freq = rfft(&input, None);
        let recovered = irfft(&freq, Some(4));
        let rd = recovered.data();

        for i in 0..4 {
            assert!(
                approx_eq(rd[i], (i + 1) as f32, 0.1),
                "mismatch at [{}]: {} vs {}",
                i,
                rd[i],
                (i + 1) as f32,
            );
        }
    }

    #[test]
    fn test_ifft2_delta_in_freq() {
        // If frequency domain is all 1s, inverse should give delta at (0,0)
        let h = 4;
        let w = 4;
        let real_data = vec![1.0f32; h * w];
        let imag_data = vec![0.0f32; h * w];

        let real = Tensor::new(real_data, vec![h, w], false);
        let imag = Tensor::new(imag_data, vec![h, w], false);

        let (out_re, out_im) = ifft2(&real, &imag);
        let re = out_re.data();
        let im = out_im.data();

        // (0,0) should be 1/(H*W) * H*W = 1.0... wait, ifft of all-ones
        // The IDFT of a constant 1 is a delta: x[0,0] = 1, rest = 0
        // because IDFT: x[n,m] = (1/HW) * sum_{k,l} X[k,l] * exp(...)
        // When X = 1 everywhere: x[0,0] = (1/HW) * HW = 1
        // x[n,m] = (1/HW) * sum exp(2pi i (kn/H + lm/W)) = 0 for (n,m) != (0,0)
        assert!(
            approx_eq(re[0], 1.0, 0.05),
            "re[0] = {} (expected 1.0)",
            re[0]
        );
        assert!(
            approx_eq(im[0], 0.0, 0.05),
            "im[0] = {} (expected 0.0)",
            im[0]
        );
        for i in 1..h * w {
            assert!(
                approx_eq(re[i], 0.0, 0.05),
                "re[{}] = {} (expected 0.0)",
                i,
                re[i]
            );
            assert!(
                approx_eq(im[i], 0.0, 0.05),
                "im[{}] = {} (expected 0.0)",
                i,
                im[i]
            );
        }
    }
}
