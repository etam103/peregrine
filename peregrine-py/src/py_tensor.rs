use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use peregrine::tensor::Tensor;
use pyo3::prelude::*;

#[pyclass(unsendable, name = "Tensor")]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

#[pymethods]
impl PyTensor {
    /// Create a Tensor from a flat list of values and a shape.
    #[new]
    #[pyo3(signature = (data, shape))]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(), shape, expected
            )));
        }
        Ok(PyTensor {
            inner: Tensor::new(data, shape, false),
        })
    }

    /// Create a Tensor from a NumPy ndarray (copies data).
    #[staticmethod]
    fn from_numpy(_py: Python<'_>, arr: PyReadonlyArrayDyn<'_, f32>) -> PyTensor {
        let shape: Vec<usize> = arr.shape().to_vec();
        let data: Vec<f32> = arr.as_slice().unwrap().to_vec();
        PyTensor {
            inner: Tensor::new(data, shape, false),
        }
    }

    /// Create a Tensor of zeros.
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> PyTensor {
        PyTensor {
            inner: Tensor::zeros(&shape, false),
        }
    }

    /// Create a Tensor of ones.
    #[staticmethod]
    fn ones(shape: Vec<usize>) -> PyTensor {
        let n: usize = shape.iter().product();
        PyTensor {
            inner: Tensor::new(vec![1.0f32; n], shape, false),
        }
    }

    /// Create a Tensor with random normal values.
    #[staticmethod]
    fn randn(shape: Vec<usize>) -> PyTensor {
        PyTensor {
            inner: Tensor::randn(&shape, false),
        }
    }

    /// Convert to NumPy ndarray.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let data = self.inner.data();
        let shape: Vec<usize> = self.inner.shape();
        let flat = numpy::PyArray1::from_vec(py, data);
        flat.reshape(shape)
    }

    /// NumPy __array__ protocol.
    fn __array__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        self.numpy(py)
    }

    /// Shape as a tuple.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape().len()
    }

    /// Total number of elements.
    #[getter]
    fn numel(&self) -> usize {
        self.inner.shape().iter().product()
    }

    /// Extract scalar value (only for 1-element tensors).
    fn item(&self) -> PyResult<f32> {
        let data = self.inner.data();
        if data.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "item() requires a scalar tensor, got {} elements",
                data.len()
            )));
        }
        Ok(data[0])
    }

    /// Flat list of all values.
    fn tolist(&self) -> Vec<f32> {
        self.inner.data()
    }

    fn __repr__(&self) -> String {
        let shape = self.inner.shape();
        let data = self.inner.data();
        let n = data.len();
        if n <= 10 {
            format!("peregrine.Tensor(shape={:?}, data={:?})", shape, data)
        } else {
            format!(
                "peregrine.Tensor(shape={:?}, data=[{:.4}, {:.4}, {:.4}, ..., {:.4}, {:.4}])",
                shape, data[0], data[1], data[2], data[n - 2], data[n - 1]
            )
        }
    }

    /// Rich HTML repr for Jupyter notebooks.
    fn _repr_html_(&self) -> String {
        let shape = self.inner.shape();
        let data = self.inner.data();
        let n = data.len();
        let shape_str = format!("{:?}", shape);
        let preview = if n <= 20 {
            format!("{:?}", data)
        } else {
            let first: Vec<String> = data[..5].iter().map(|v| format!("{:.4}", v)).collect();
            let last: Vec<String> = data[n - 5..].iter().map(|v| format!("{:.4}", v)).collect();
            format!("[{}, ..., {}]", first.join(", "), last.join(", "))
        };

        format!(
            "<div style='font-family:monospace;padding:4px 8px;background:#f8f8f8;border-radius:4px;display:inline-block'>\
            <b>peregrine.Tensor</b> &nbsp; shape={} &nbsp; dtype=f32 &nbsp; {} elements<br/>\
            <span style='color:#555'>{}</span></div>",
            shape_str, n, preview
        )
    }

    // --- Arithmetic ops ---

    fn __add__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.add(&other.inner) }
    }

    fn __sub__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.sub(&other.inner) }
    }

    fn __mul__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.mul(&other.inner) }
    }

    fn __matmul__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.matmul(&other.inner) }
    }

    fn __neg__(&self) -> PyTensor {
        PyTensor { inner: self.inner.neg() }
    }

    // --- Unary ops ---

    fn relu(&self) -> PyTensor { PyTensor { inner: self.inner.relu() } }
    fn gelu(&self) -> PyTensor { PyTensor { inner: self.inner.gelu() } }
    fn silu(&self) -> PyTensor { PyTensor { inner: self.inner.silu() } }
    fn sigmoid(&self) -> PyTensor { PyTensor { inner: self.inner.sigmoid() } }
    fn tanh(&self) -> PyTensor { PyTensor { inner: self.inner.tanh() } }
    fn exp(&self) -> PyTensor { PyTensor { inner: self.inner.exp() } }
    fn log(&self) -> PyTensor { PyTensor { inner: self.inner.log() } }
    fn sqrt(&self) -> PyTensor { PyTensor { inner: self.inner.sqrt() } }
    fn abs(&self) -> PyTensor { PyTensor { inner: self.inner.abs() } }
    #[pyo3(signature = (dim=-1))]
    fn softmax(&self, dim: isize) -> PyTensor { PyTensor { inner: self.inner.softmax(dim) } }

    // --- Reductions ---

    fn sum(&self) -> PyTensor { PyTensor { inner: self.inner.sum() } }
    fn mean(&self) -> PyTensor { PyTensor { inner: self.inner.mean() } }

    // --- Shape ops ---

    fn reshape(&self, shape: Vec<usize>) -> PyTensor {
        PyTensor { inner: self.inner.reshape(shape) }
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyTensor {
        PyTensor { inner: self.inner.transpose(dim0, dim1) }
    }

    fn flatten(&self) -> PyTensor {
        let n: usize = self.inner.shape().iter().product();
        PyTensor { inner: self.inner.reshape(vec![n]) }
    }
}
