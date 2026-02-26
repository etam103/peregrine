use crate::random;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Fan computation
// ---------------------------------------------------------------------------

fn compute_fans(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            // Conv layout: [out_channels, in_channels, k1, k2, ...]
            let receptive: usize = shape[2..].iter().product();
            (shape[1] * receptive, shape[0] * receptive)
        }
    }
}

// ---------------------------------------------------------------------------
// Glorot (Xavier) initialisations
// ---------------------------------------------------------------------------

/// Glorot / Xavier uniform initialisation.
/// U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out)).
pub fn glorot_uniform(shape: &[usize], requires_grad: bool) -> Tensor {
    let (fan_in, fan_out) = compute_fans(shape);
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    random::uniform(shape, -limit, limit, requires_grad)
}

/// Glorot / Xavier normal initialisation.
/// N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
pub fn glorot_normal(shape: &[usize], requires_grad: bool) -> Tensor {
    let (fan_in, fan_out) = compute_fans(shape);
    let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    random::normal(shape, 0.0, std, requires_grad)
}

// ---------------------------------------------------------------------------
// He (Kaiming) initialisations
// ---------------------------------------------------------------------------

/// He / Kaiming normal initialisation.
/// N(0, std) where std = sqrt(2 / fan_in).
pub fn he_normal(shape: &[usize], requires_grad: bool) -> Tensor {
    let (fan_in, _) = compute_fans(shape);
    let std = (2.0 / fan_in as f32).sqrt();
    random::normal(shape, 0.0, std, requires_grad)
}

/// He / Kaiming uniform initialisation.
/// U(-limit, limit) where limit = sqrt(6 / fan_in).
pub fn he_uniform(shape: &[usize], requires_grad: bool) -> Tensor {
    let (fan_in, _) = compute_fans(shape);
    let limit = (6.0 / fan_in as f32).sqrt();
    random::uniform(shape, -limit, limit, requires_grad)
}

// ---------------------------------------------------------------------------
// Other initialisations
// ---------------------------------------------------------------------------

/// Constant initialisation.
pub fn constant(shape: &[usize], value: f32, requires_grad: bool) -> Tensor {
    Tensor::full(shape, value, requires_grad)
}

/// LeCun normal initialisation.
/// N(0, std) where std = sqrt(1 / fan_in).
pub fn lecun_normal(shape: &[usize], requires_grad: bool) -> Tensor {
    let (fan_in, _) = compute_fans(shape);
    let std = (1.0 / fan_in as f32).sqrt();
    random::normal(shape, 0.0, std, requires_grad)
}

/// Orthogonal-like initialisation.
///
/// A proper orthogonal init requires QR decomposition. This implementation
/// uses a normal distribution scaled so variance equals `1 / max(fan_in, fan_out)`,
/// which is a reasonable approximation until `linalg::qr` is available.
pub fn orthogonal(shape: &[usize], requires_grad: bool) -> Tensor {
    assert!(
        shape.len() >= 2,
        "orthogonal init requires at least 2D shape"
    );
    let (fan_in, fan_out) = compute_fans(shape);
    let std = (1.0 / fan_in.max(fan_out) as f32).sqrt();
    random::normal(shape, 0.0, std, requires_grad)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glorot_uniform_shape_and_range() {
        crate::random::seed(42);
        let t = glorot_uniform(&[128, 64], true);
        assert_eq!(t.shape(), vec![128, 64]);
        let limit = (6.0 / (128.0 + 64.0) as f32).sqrt();
        let d = t.data();
        for &x in &d {
            assert!(x >= -limit && x <= limit, "value {} out of range", x);
        }
    }

    #[test]
    fn test_glorot_normal_shape() {
        crate::random::seed(42);
        let t = glorot_normal(&[32, 16], true);
        assert_eq!(t.shape(), vec![32, 16]);
        assert_eq!(t.data().len(), 32 * 16);
    }

    #[test]
    fn test_he_normal_shape() {
        crate::random::seed(42);
        let t = he_normal(&[64, 32], true);
        assert_eq!(t.shape(), vec![64, 32]);
        assert_eq!(t.data().len(), 64 * 32);
    }

    #[test]
    fn test_he_uniform_range() {
        crate::random::seed(42);
        let t = he_uniform(&[64, 32], true);
        let limit = (6.0 / 32.0f32).sqrt();
        let d = t.data();
        for &x in &d {
            assert!(x >= -limit && x <= limit, "value {} out of range", x);
        }
    }

    #[test]
    fn test_constant() {
        let t = constant(&[3, 3], 0.5, false);
        assert_eq!(t.data(), vec![0.5; 9]);
    }

    #[test]
    fn test_compute_fans_conv() {
        // [out=16, in=8, kH=3, kW=3]
        let (fi, fo) = compute_fans(&[16, 8, 3, 3]);
        assert_eq!(fi, 8 * 9); // 72
        assert_eq!(fo, 16 * 9); // 144
    }

    #[test]
    fn test_lecun_normal() {
        crate::random::seed(42);
        let t = lecun_normal(&[64, 32], true);
        assert_eq!(t.shape(), vec![64, 32]);
        assert_eq!(t.data().len(), 64 * 32);
    }

    #[test]
    fn test_orthogonal() {
        crate::random::seed(42);
        let t = orthogonal(&[64, 64], true);
        assert_eq!(t.shape(), vec![64, 64]);
    }
}
