use crate::tensor::Tensor;

/// Compute gradients of a scalar-valued function with respect to its inputs.
///
/// The closure `f` receives a slice of tensor references and must return a
/// scalar (single-element) tensor representing the loss.  After `f` completes,
/// `backward()` is called on the result and the accumulated gradients are
/// extracted from each input.
///
/// # Example
/// ```ignore
/// let x = Tensor::new(vec![2.0], vec![1], true);
/// let grads = peregrine::transforms::grad(|inputs| {
///     inputs[0].scale(3.0)  // f(x) = 3x
/// }, &[&x]);
/// // grads[0] ~= [3.0]
/// ```
pub fn grad<F>(f: F, inputs: &[&Tensor]) -> Vec<Tensor>
where
    F: Fn(&[&Tensor]) -> Tensor,
{
    // Zero any existing gradients so we get a clean accumulation.
    for t in inputs {
        t.zero_grad();
    }

    let loss = f(inputs);
    loss.backward();

    inputs
        .iter()
        .map(|t| {
            let inner = t.0.borrow();
            let g = inner
                .grad
                .clone()
                .unwrap_or_else(|| vec![0.0; inner.data.len()]);
            Tensor::new(g, inner.shape.clone(), false)
        })
        .collect()
}

/// Compute both the function value and gradients in a single forward+backward
/// pass.
///
/// Returns `(loss_tensor, vec_of_gradient_tensors)`.
pub fn value_and_grad<F>(f: F, inputs: &[&Tensor]) -> (Tensor, Vec<Tensor>)
where
    F: Fn(&[&Tensor]) -> Tensor,
{
    for t in inputs {
        t.zero_grad();
    }

    let loss = f(inputs);
    loss.backward();

    let grads: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            let inner = t.0.borrow();
            let g = inner
                .grad
                .clone()
                .unwrap_or_else(|| vec![0.0; inner.data.len()]);
            Tensor::new(g, inner.shape.clone(), false)
        })
        .collect();

    // Return a detached copy of the loss value (no grad graph).
    let loss_val = Tensor::new(loss.data(), loss.shape(), false);
    (loss_val, grads)
}

/// Gradient checkpointing stub.
///
/// A proper implementation would discard intermediate activations during the
/// forward pass and recompute them during backward to save memory. For now
/// this simply executes the forward pass normally.
pub fn checkpoint<F>(f: F, inputs: &[Tensor]) -> Tensor
where
    F: Fn(&[Tensor]) -> Tensor,
{
    f(inputs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_grad_linear() {
        // f(x) = 3*x, df/dx = 3
        let x = Tensor::new(vec![2.0], vec![1], true);
        let grads = grad(
            |inputs| inputs[0].scale(3.0),
            &[&x],
        );
        assert_eq!(grads.len(), 1);
        let gd = grads[0].data();
        assert!(
            (gd[0] - 3.0).abs() < 1e-5,
            "expected gradient 3.0, got {}",
            gd[0],
        );
    }

    #[test]
    fn test_value_and_grad() {
        // f(x) = x * x  (element-wise), sum to scalar
        let x = Tensor::new(vec![3.0], vec![1], true);
        let (val, grads) = value_and_grad(
            |inputs| {
                let sq = inputs[0].mul(inputs[0]);
                sq.sum()
            },
            &[&x],
        );
        let vd = val.data();
        assert!(
            (vd[0] - 9.0).abs() < 1e-5,
            "expected value 9.0, got {}",
            vd[0],
        );
        let gd = grads[0].data();
        // d/dx(x^2) = 2x = 6
        assert!(
            (gd[0] - 6.0).abs() < 1e-5,
            "expected gradient 6.0, got {}",
            gd[0],
        );
    }

    #[test]
    fn test_grad_multi_input() {
        // f(a, b) = sum(a * b)
        let a = Tensor::new(vec![2.0, 3.0], vec![2], true);
        let b = Tensor::new(vec![4.0, 5.0], vec![2], true);
        let grads = grad(
            |inputs| inputs[0].mul(inputs[1]).sum(),
            &[&a, &b],
        );
        assert_eq!(grads.len(), 2);
        // df/da = b, df/db = a
        let ga = grads[0].data();
        let gb = grads[1].data();
        assert!((ga[0] - 4.0).abs() < 1e-5);
        assert!((ga[1] - 5.0).abs() < 1e-5);
        assert!((gb[0] - 2.0).abs() < 1e-5);
        assert!((gb[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_checkpoint_passthrough() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
        let result = checkpoint(|inputs| inputs[0].scale(2.0), &[x.clone()]);
        let d = result.data();
        assert!((d[0] - 2.0).abs() < 1e-6);
        assert!((d[1] - 4.0).abs() < 1e-6);
        assert!((d[2] - 6.0).abs() < 1e-6);
    }
}
