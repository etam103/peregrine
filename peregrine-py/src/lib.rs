use pyo3::prelude::*;

mod py_tensor;
mod py_nn;
mod py_inference;

/// Peregrine: A from-scratch deep learning inference engine in Rust.
#[pymodule]
fn peregrine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_tensor::PyTensor>()?;
    m.add_class::<py_inference::TextGenerator>()?;
    m.add_class::<py_inference::TokenIterator>()?;
    m.add_function(wrap_pyfunction!(py_inference::load_model, m)?)?;

    // nn submodule
    let nn = PyModule::new(m.py(), "nn")?;
    nn.add_class::<py_nn::PyLinear>()?;
    nn.add_class::<py_nn::PyEmbedding>()?;
    nn.add_class::<py_nn::PyRMSNorm>()?;
    nn.add_class::<py_nn::PyLayerNorm>()?;
    m.add_submodule(&nn)?;

    Ok(())
}
