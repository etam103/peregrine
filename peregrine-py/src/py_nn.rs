use pyo3::prelude::*;
use crate::py_tensor::PyTensor;

#[pyclass(unsendable, name = "Linear")]
pub struct PyLinear {
    inner: peregrine::nn::Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize) -> Self {
        PyLinear {
            inner: peregrine::nn::Linear::new(in_features, out_features),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.forward(&x.inner),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    #[getter]
    fn bias(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.bias.clone(),
        }
    }

    fn __repr__(&self) -> String {
        let w_shape = self.inner.weight.shape();
        format!("peregrine.nn.Linear(in_features={}, out_features={})", w_shape[0], w_shape[1])
    }
}

#[pyclass(unsendable, name = "Embedding")]
pub struct PyEmbedding {
    inner: peregrine::nn::Embedding,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        PyEmbedding {
            inner: peregrine::nn::Embedding::new(num_embeddings, embedding_dim),
        }
    }

    fn forward(&self, indices: Vec<usize>) -> PyTensor {
        PyTensor {
            inner: self.inner.forward(&indices),
        }
    }

    fn __call__(&self, indices: Vec<usize>) -> PyTensor {
        self.forward(indices)
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        PyTensor {
            inner: self.inner.weight.clone(),
        }
    }

    fn __repr__(&self) -> String {
        let w_shape = self.inner.weight.shape();
        format!("peregrine.nn.Embedding(num_embeddings={}, embedding_dim={})", w_shape[0], w_shape[1])
    }
}

#[pyclass(unsendable, name = "RMSNorm")]
pub struct PyRMSNorm {
    inner: peregrine::nn::RMSNorm,
}

#[pymethods]
impl PyRMSNorm {
    #[new]
    #[pyo3(signature = (dim, eps=1e-5))]
    fn new(dim: usize, eps: f32) -> Self {
        PyRMSNorm {
            inner: peregrine::nn::RMSNorm::new(dim, eps),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.forward(&x.inner),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }

    fn __repr__(&self) -> String {
        let dim = self.inner.weight.shape()[0];
        format!("peregrine.nn.RMSNorm(dim={})", dim)
    }
}

#[pyclass(unsendable, name = "LayerNorm")]
pub struct PyLayerNorm {
    inner: peregrine::nn::LayerNorm,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    fn new(dim: usize) -> Self {
        PyLayerNorm {
            inner: peregrine::nn::LayerNorm::new(dim),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor {
            inner: self.inner.forward(&x.inner),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        self.forward(x)
    }

    fn __repr__(&self) -> String {
        let dim = self.inner.weight.shape()[0];
        format!("peregrine.nn.LayerNorm(dim={})", dim)
    }
}
