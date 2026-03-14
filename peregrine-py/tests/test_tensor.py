"""Tests for peregrine.Tensor and peregrine.nn Python bindings."""

import numpy as np
import peregrine


def test_tensor_creation():
    t = peregrine.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.shape == [2, 2]
    assert t.ndim == 2
    assert t.numel == 4
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_numpy_interop():
    t = peregrine.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    arr = t.numpy()
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr, [[1, 2, 3], [4, 5, 6]])


def test_from_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = peregrine.Tensor.from_numpy(arr)
    assert t.shape == [2, 2]
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_arithmetic():
    a = peregrine.Tensor([1.0, 2.0, 3.0], [3])
    b = peregrine.Tensor([4.0, 5.0, 6.0], [3])
    assert (a + b).tolist() == [5.0, 7.0, 9.0]
    assert (a * b).tolist() == [4.0, 10.0, 18.0]
    assert (a - b).tolist() == [-3.0, -3.0, -3.0]


def test_matmul():
    m1 = peregrine.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    m2 = peregrine.Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    result = (m1 @ m2).tolist()
    assert result == [19.0, 22.0, 43.0, 50.0]


def test_unary_ops():
    t = peregrine.Tensor([-1.0, 0.0, 1.0], [3])
    assert t.relu().tolist() == [0.0, 0.0, 1.0]
    assert t.abs().tolist() == [1.0, 0.0, 1.0]

    t2 = peregrine.Tensor([0.0], [1])
    assert abs(t2.exp().item() - 1.0) < 1e-6


def test_item():
    t = peregrine.Tensor([42.0], [1])
    assert t.item() == 42.0


def test_reshape():
    t = peregrine.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    t2 = t.reshape([3, 2])
    assert t2.shape == [3, 2]
    assert t2.tolist() == t.tolist()


def test_static_constructors():
    z = peregrine.Tensor.zeros([2, 3])
    assert z.shape == [2, 3]
    assert all(v == 0.0 for v in z.tolist())

    o = peregrine.Tensor.ones([3])
    assert o.tolist() == [1.0, 1.0, 1.0]

    r = peregrine.Tensor.randn([4, 4])
    assert r.shape == [4, 4]
    assert r.numel == 16


def test_repr():
    t = peregrine.Tensor([1.0, 2.0], [2])
    r = repr(t)
    assert "peregrine.Tensor" in r
    assert "[2]" in r


def test_html_repr():
    t = peregrine.Tensor([1.0, 2.0], [2])
    html = t._repr_html_()
    assert "peregrine.Tensor" in html
    assert "f32" in html


def test_nn_linear():
    linear = peregrine.nn.Linear(4, 3)
    x = peregrine.Tensor.randn([2, 4])
    y = linear(x)
    assert y.shape == [2, 3]
    assert linear.weight.shape == [4, 3]


def test_nn_embedding():
    emb = peregrine.nn.Embedding(100, 32)
    y = emb([0, 5, 10])
    assert y.shape == [3, 32]


def test_nn_rmsnorm():
    norm = peregrine.nn.RMSNorm(4)
    x = peregrine.Tensor.randn([2, 4])
    y = norm(x)
    assert y.shape == [2, 4]


def test_nn_layernorm():
    ln = peregrine.nn.LayerNorm(4)
    x = peregrine.Tensor.randn([2, 4])
    y = ln(x)
    assert y.shape == [2, 4]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  {test.__name__} ... ok")
    print(f"\n{len(tests)} tests passed!")
