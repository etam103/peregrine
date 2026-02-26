use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// LAPACK FFI (macOS Accelerate framework)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn sgesv_(
        n: *const i32, nrhs: *const i32, a: *mut f32, lda: *const i32,
        ipiv: *mut i32, b: *mut f32, ldb: *const i32, info: *mut i32,
    );
    fn sgetrf_(
        m: *const i32, n: *const i32, a: *mut f32, lda: *const i32,
        ipiv: *mut i32, info: *mut i32,
    );
    fn sgetri_(
        n: *const i32, a: *mut f32, lda: *const i32, ipiv: *const i32,
        work: *mut f32, lwork: *const i32, info: *mut i32,
    );
    fn spotrf_(
        uplo: *const u8, n: *const i32, a: *mut f32, lda: *const i32,
        info: *mut i32,
    );
    fn sgesdd_(
        jobz: *const u8, m: *const i32, n: *const i32, a: *mut f32,
        lda: *const i32, s: *mut f32, u: *mut f32, ldu: *const i32,
        vt: *mut f32, ldvt: *const i32, work: *mut f32, lwork: *const i32,
        iwork: *mut i32, info: *mut i32,
    );
    fn sgeqrf_(
        m: *const i32, n: *const i32, a: *mut f32, lda: *const i32,
        tau: *mut f32, work: *mut f32, lwork: *const i32, info: *mut i32,
    );
    fn sorgqr_(
        m: *const i32, n: *const i32, k: *const i32, a: *mut f32,
        lda: *const i32, tau: *const f32, work: *mut f32, lwork: *const i32,
        info: *mut i32,
    );
    fn ssyev_(
        jobz: *const u8, uplo: *const u8, n: *const i32, a: *mut f32,
        lda: *const i32, w: *mut f32, work: *mut f32, lwork: *const i32,
        info: *mut i32,
    );
    fn strtrs_(
        uplo: *const u8, trans: *const u8, diag: *const u8, n: *const i32,
        nrhs: *const i32, a: *const f32, lda: *const i32, b: *mut f32,
        ldb: *const i32, info: *mut i32,
    );
}

// ---------------------------------------------------------------------------
// Row-major <-> column-major helpers
// ---------------------------------------------------------------------------

/// Transpose a row-major `[rows, cols]` buffer in-place (via temporary copy).
fn transpose_buf(data: &mut [f32], rows: usize, cols: usize) {
    let mut tmp = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            tmp[j * rows + i] = data[i * cols + j];
        }
    }
    data.copy_from_slice(&tmp);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Vector / matrix norm.
///
/// - `ord = None` or `2.0`: L2 (Frobenius for matrices)
/// - `ord = 1.0`: L1
/// - `ord = f32::INFINITY`: L-inf
/// - `ord = f32::NEG_INFINITY`: min absolute value
/// - Any other `p`: Lp norm
pub fn norm(input: &Tensor, ord: Option<f32>) -> Tensor {
    let data = input.data();
    let result = match ord {
        None | Some(2.0) => data.iter().map(|x| x * x).sum::<f32>().sqrt(),
        Some(v) if v == 1.0 => data.iter().map(|x| x.abs()).sum::<f32>(),
        Some(v) if v == f32::INFINITY => data.iter().map(|x| x.abs()).fold(0.0f32, f32::max),
        Some(v) if v == f32::NEG_INFINITY => {
            data.iter().map(|x| x.abs()).fold(f32::INFINITY, f32::min)
        }
        Some(p) => data
            .iter()
            .map(|x| x.abs().powf(p))
            .sum::<f32>()
            .powf(1.0 / p),
    };
    Tensor::new(vec![result], vec![1], false)
}

/// Solve `Ax = B` for `x`. `A`: `[n, n]`, `B`: `[n]` or `[n, nrhs]`.
#[cfg(target_os = "macos")]
pub fn solve(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert!(a_shape.len() == 2 && a_shape[0] == a_shape[1], "A must be square");
    let n = a_shape[0] as i32;
    let nrhs = if b_shape.len() > 1 { b_shape[1] as i32 } else { 1 };

    let mut a_data = a.data();
    let mut b_data = b.data();
    let mut ipiv = vec![0i32; n as usize];
    let mut info = 0i32;

    // LAPACK uses column-major order
    transpose_buf(&mut a_data, n as usize, n as usize);
    if nrhs > 1 {
        transpose_buf(&mut b_data, n as usize, nrhs as usize);
    }

    unsafe {
        sgesv_(
            &n, &nrhs, a_data.as_mut_ptr(), &n,
            ipiv.as_mut_ptr(), b_data.as_mut_ptr(), &n, &mut info,
        );
    }
    assert!(info == 0, "LAPACK sgesv failed with info={}", info);

    if nrhs > 1 {
        transpose_buf(&mut b_data, nrhs as usize, n as usize);
    }
    Tensor::new(b_data, b_shape.to_vec(), false)
}

#[cfg(not(target_os = "macos"))]
pub fn solve(_a: &Tensor, _b: &Tensor) -> Tensor {
    panic!("linalg::solve requires macOS Accelerate framework");
}

/// Matrix inverse via LU factorisation.
#[cfg(target_os = "macos")]
pub fn inv(a: &Tensor) -> Tensor {
    let shape = a.shape();
    assert!(shape.len() == 2 && shape[0] == shape[1], "inv requires square matrix");
    let n = shape[0] as i32;

    let mut data = a.data();
    transpose_buf(&mut data, n as usize, n as usize);

    let mut ipiv = vec![0i32; n as usize];
    let mut info = 0i32;

    unsafe {
        sgetrf_(&n, &n, data.as_mut_ptr(), &n, ipiv.as_mut_ptr(), &mut info);
    }
    assert!(info == 0, "LAPACK sgetrf failed with info={}", info);

    // Query optimal workspace
    let mut work_query = vec![0.0f32; 1];
    let lwork_query: i32 = -1;
    unsafe {
        sgetri_(
            &n, data.as_mut_ptr(), &n, ipiv.as_ptr(),
            work_query.as_mut_ptr(), &lwork_query, &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f32; lwork as usize];

    unsafe {
        sgetri_(
            &n, data.as_mut_ptr(), &n, ipiv.as_ptr(),
            work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert!(info == 0, "LAPACK sgetri failed with info={}", info);

    transpose_buf(&mut data, n as usize, n as usize);
    Tensor::new(data, shape.to_vec(), false)
}

#[cfg(not(target_os = "macos"))]
pub fn inv(_a: &Tensor) -> Tensor {
    panic!("linalg::inv requires macOS Accelerate framework");
}

/// Cholesky decomposition (lower triangular L such that A = L L^T).
#[cfg(target_os = "macos")]
pub fn cholesky(a: &Tensor) -> Tensor {
    let shape = a.shape();
    assert!(shape.len() == 2 && shape[0] == shape[1], "cholesky requires square matrix");
    let n = shape[0] as i32;

    let mut data = a.data();
    transpose_buf(&mut data, n as usize, n as usize);

    let uplo = b'L';
    let mut info = 0i32;

    unsafe {
        spotrf_(&uplo, &n, data.as_mut_ptr(), &n, &mut info);
    }
    assert!(info == 0, "LAPACK spotrf failed with info={} (matrix not positive definite?)", info);

    // Zero out upper triangle (LAPACK only writes the lower part)
    for i in 0..n as usize {
        for j in (i + 1)..n as usize {
            data[j * n as usize + i] = 0.0; // column-major upper
        }
    }

    transpose_buf(&mut data, n as usize, n as usize);
    Tensor::new(data, shape.to_vec(), false)
}

#[cfg(not(target_os = "macos"))]
pub fn cholesky(_a: &Tensor) -> Tensor {
    panic!("linalg::cholesky requires macOS Accelerate framework");
}

/// SVD: `A = U * diag(S) * V^T`. Returns `(U, S, Vt)`.
///
/// - `A`: `[m, n]`
/// - `U`: `[m, m]`
/// - `S`: `[min(m,n)]`
/// - `Vt`: `[n, n]`
#[cfg(target_os = "macos")]
pub fn svd(a: &Tensor) -> (Tensor, Tensor, Tensor) {
    let shape = a.shape();
    assert!(shape.len() == 2, "svd requires 2D matrix");
    let m = shape[0] as i32;
    let n = shape[1] as i32;
    let k = m.min(n);

    let mut a_data = a.data();
    transpose_buf(&mut a_data, m as usize, n as usize);

    let mut s = vec![0.0f32; k as usize];
    let mut u = vec![0.0f32; (m * m) as usize];
    let mut vt = vec![0.0f32; (n * n) as usize];
    let mut iwork = vec![0i32; 8 * k as usize];
    let mut info = 0i32;
    let jobz = b'A'; // compute all of U and Vt

    // Workspace query
    let mut work_query = vec![0.0f32; 1];
    let lwork_query: i32 = -1;
    unsafe {
        sgesdd_(
            &jobz, &m, &n, a_data.as_mut_ptr(), &m,
            s.as_mut_ptr(), u.as_mut_ptr(), &m, vt.as_mut_ptr(), &n,
            work_query.as_mut_ptr(), &lwork_query, iwork.as_mut_ptr(), &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f32; lwork as usize];

    unsafe {
        sgesdd_(
            &jobz, &m, &n, a_data.as_mut_ptr(), &m,
            s.as_mut_ptr(), u.as_mut_ptr(), &m, vt.as_mut_ptr(), &n,
            work.as_mut_ptr(), &lwork, iwork.as_mut_ptr(), &mut info,
        );
    }
    assert!(info == 0, "LAPACK sgesdd failed with info={}", info);

    // Transpose U and Vt back to row-major
    transpose_buf(&mut u, m as usize, m as usize);
    transpose_buf(&mut vt, n as usize, n as usize);

    (
        Tensor::new(u, vec![m as usize, m as usize], false),
        Tensor::new(s, vec![k as usize], false),
        Tensor::new(vt, vec![n as usize, n as usize], false),
    )
}

#[cfg(not(target_os = "macos"))]
pub fn svd(_a: &Tensor) -> (Tensor, Tensor, Tensor) {
    panic!("linalg::svd requires macOS Accelerate framework");
}

/// QR decomposition: returns `(Q, R)`.
///
/// - `A`: `[m, n]` with `m >= n`
/// - `Q`: `[m, n]`
/// - `R`: `[n, n]`
#[cfg(target_os = "macos")]
pub fn qr(a: &Tensor) -> (Tensor, Tensor) {
    let shape = a.shape();
    assert!(shape.len() == 2, "qr requires 2D matrix");
    let m = shape[0] as i32;
    let n = shape[1] as i32;
    let k = m.min(n);

    let mut a_data = a.data();
    transpose_buf(&mut a_data, m as usize, n as usize);

    let mut tau = vec![0.0f32; k as usize];
    let mut info = 0i32;

    // Workspace query for sgeqrf
    let mut work_query = vec![0.0f32; 1];
    let lwork_query: i32 = -1;
    unsafe {
        sgeqrf_(
            &m, &n, a_data.as_mut_ptr(), &m,
            tau.as_mut_ptr(), work_query.as_mut_ptr(), &lwork_query, &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f32; lwork as usize];

    unsafe {
        sgeqrf_(
            &m, &n, a_data.as_mut_ptr(), &m,
            tau.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert!(info == 0, "LAPACK sgeqrf failed with info={}", info);

    // Extract R (upper triangular) from a_data (column-major)
    let rn = n as usize;
    let mut r_data = vec![0.0f32; rn * rn];
    for j in 0..rn {
        for i in 0..=j.min(m as usize - 1) {
            r_data[i * rn + j] = a_data[j * m as usize + i]; // col-major -> row-major
        }
    }

    // Generate Q via sorgqr
    let mut work_query2 = vec![0.0f32; 1];
    unsafe {
        sorgqr_(
            &m, &k, &k, a_data.as_mut_ptr(), &m,
            tau.as_ptr(), work_query2.as_mut_ptr(), &lwork_query, &mut info,
        );
    }
    let lwork2 = work_query2[0] as i32;
    let mut work2 = vec![0.0f32; lwork2 as usize];

    unsafe {
        sorgqr_(
            &m, &k, &k, a_data.as_mut_ptr(), &m,
            tau.as_ptr(), work2.as_mut_ptr(), &lwork2, &mut info,
        );
    }
    assert!(info == 0, "LAPACK sorgqr failed with info={}", info);

    // Extract Q columns (column-major -> row-major)
    let qcols = k as usize;
    let qrows = m as usize;
    let mut q_data = vec![0.0f32; qrows * qcols];
    for j in 0..qcols {
        for i in 0..qrows {
            q_data[i * qcols + j] = a_data[j * qrows + i];
        }
    }

    (
        Tensor::new(q_data, vec![qrows, qcols], false),
        Tensor::new(r_data, vec![rn, rn], false),
    )
}

#[cfg(not(target_os = "macos"))]
pub fn qr(_a: &Tensor) -> (Tensor, Tensor) {
    panic!("linalg::qr requires macOS Accelerate framework");
}

/// Eigenvalues and eigenvectors of a symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted ascending.
/// - `eigenvalues`: `[n]`
/// - `eigenvectors`: `[n, n]` (columns are eigenvectors, returned in row-major)
#[cfg(target_os = "macos")]
pub fn eigh(a: &Tensor) -> (Tensor, Tensor) {
    let shape = a.shape();
    assert!(shape.len() == 2 && shape[0] == shape[1], "eigh requires square matrix");
    let n = shape[0] as i32;

    let mut a_data = a.data();
    transpose_buf(&mut a_data, n as usize, n as usize);

    let mut w = vec![0.0f32; n as usize];
    let mut info = 0i32;
    let jobz = b'V'; // compute eigenvalues and eigenvectors
    let uplo = b'U';

    // Workspace query
    let mut work_query = vec![0.0f32; 1];
    let lwork_query: i32 = -1;
    unsafe {
        ssyev_(
            &jobz, &uplo, &n, a_data.as_mut_ptr(), &n,
            w.as_mut_ptr(), work_query.as_mut_ptr(), &lwork_query, &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f32; lwork as usize];

    unsafe {
        ssyev_(
            &jobz, &uplo, &n, a_data.as_mut_ptr(), &n,
            w.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert!(info == 0, "LAPACK ssyev failed with info={}", info);

    // a_data now contains eigenvectors in columns (column-major)
    transpose_buf(&mut a_data, n as usize, n as usize);

    (
        Tensor::new(w, vec![n as usize], false),
        Tensor::new(a_data, vec![n as usize, n as usize], false),
    )
}

#[cfg(not(target_os = "macos"))]
pub fn eigh(_a: &Tensor) -> (Tensor, Tensor) {
    panic!("linalg::eigh requires macOS Accelerate framework");
}

/// LU factorisation: returns `(L, U, pivots)`.
///
/// `A`: `[m, n]`. `L`: `[m, k]`, `U`: `[k, n]` where `k = min(m, n)`.
#[cfg(target_os = "macos")]
pub fn lu(a: &Tensor) -> (Tensor, Tensor, Vec<i32>) {
    let shape = a.shape();
    assert!(shape.len() == 2, "lu requires 2D matrix");
    let m = shape[0] as i32;
    let n = shape[1] as i32;
    let k = m.min(n) as usize;

    let mut data = a.data();
    transpose_buf(&mut data, m as usize, n as usize);

    let mut ipiv = vec![0i32; k];
    let mut info = 0i32;

    unsafe {
        sgetrf_(&m, &n, data.as_mut_ptr(), &m, ipiv.as_mut_ptr(), &mut info);
    }
    assert!(info >= 0, "LAPACK sgetrf failed with info={}", info);

    let mu = m as usize;
    let nu = n as usize;

    // Extract L (lower triangular with unit diagonal) [m, k]
    let mut l_data = vec![0.0f32; mu * k];
    for j in 0..k {
        for i in 0..mu {
            if i == j {
                l_data[i * k + j] = 1.0;
            } else if i > j {
                l_data[i * k + j] = data[j * mu + i]; // column-major
            }
        }
    }

    // Extract U (upper triangular) [k, n]
    let mut u_data = vec![0.0f32; k * nu];
    for j in 0..nu {
        for i in 0..k {
            if i <= j {
                u_data[i * nu + j] = data[j * mu + i]; // column-major
            }
        }
    }

    (
        Tensor::new(l_data, vec![mu, k], false),
        Tensor::new(u_data, vec![k, nu], false),
        ipiv,
    )
}

#[cfg(not(target_os = "macos"))]
pub fn lu(_a: &Tensor) -> (Tensor, Tensor, Vec<i32>) {
    panic!("linalg::lu requires macOS Accelerate framework");
}

/// Determinant via LU factorisation.
pub fn det(a: &Tensor) -> Tensor {
    let shape = a.shape();
    assert!(shape.len() == 2 && shape[0] == shape[1], "det requires square matrix");

    #[cfg(target_os = "macos")]
    {
        let n = shape[0] as i32;
        let mut data = a.data();
        transpose_buf(&mut data, n as usize, n as usize);

        let mut ipiv = vec![0i32; n as usize];
        let mut info = 0i32;

        unsafe {
            sgetrf_(&n, &n, data.as_mut_ptr(), &n, ipiv.as_mut_ptr(), &mut info);
        }
        assert!(info >= 0, "LAPACK sgetrf failed with info={}", info);

        if info > 0 {
            // Singular matrix -> det = 0
            return Tensor::new(vec![0.0], vec![1], false);
        }

        // det = product of diagonal * sign from pivots
        let mut d = 1.0f32;
        let mut swaps = 0;
        for i in 0..n as usize {
            d *= data[i * n as usize + i]; // diagonal of U in column-major
            if ipiv[i] != (i as i32 + 1) {
                swaps += 1;
            }
        }
        if swaps % 2 == 1 {
            d = -d;
        }
        Tensor::new(vec![d], vec![1], false)
    }

    #[cfg(not(target_os = "macos"))]
    {
        panic!("linalg::det requires macOS Accelerate framework");
    }
}

/// Pseudoinverse via SVD: `A^+ = V * diag(1/s) * U^T`.
pub fn pinv(a: &Tensor) -> Tensor {
    let (u, s, vt) = svd(a);
    let s_data = s.data();
    let u_data = u.data();
    let vt_data = vt.data();
    let u_shape = u.shape();
    let vt_shape = vt.shape();
    let m = u_shape[0];
    let n = vt_shape[1];
    let k = s_data.len();

    // Threshold for near-zero singular values
    let max_s = s_data.iter().cloned().fold(0.0f32, f32::max);
    let eps = max_s * 1e-6 * (m.max(n) as f32);

    // Compute V * diag(1/s) * U^T  (row-major)
    // pinv shape: [n, m]
    let mut result = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0f32;
            for l in 0..k {
                if s_data[l] > eps {
                    // V[i,l] = Vt[l,i] transposed
                    let v_il = vt_data[l * n + i];
                    // U^T[l,j] = U[j,l]
                    let ut_lj = u_data[j * m + l]; // U is [m,m], but we want U[j,l]
                    sum += v_il * (1.0 / s_data[l]) * ut_lj;
                }
            }
            result[i * m + j] = sum;
        }
    }

    Tensor::new(result, vec![n, m], false)
}

/// Cross product of two 3D vectors.
pub fn cross(a: &Tensor, b: &Tensor) -> Tensor {
    let ad = a.data();
    let bd = b.data();
    assert!(ad.len() == 3 && bd.len() == 3, "cross product requires 3-element vectors");
    let result = vec![
        ad[1] * bd[2] - ad[2] * bd[1],
        ad[2] * bd[0] - ad[0] * bd[2],
        ad[0] * bd[1] - ad[1] * bd[0],
    ];
    Tensor::new(result, vec![3], false)
}

/// Triangular solve: solve `Lx = b` or `Ux = b` for `x`.
///
/// - `upper`: if true, solve upper-triangular; otherwise lower-triangular.
/// - `a`: `[n, n]` triangular matrix.
/// - `b`: `[n]` or `[n, nrhs]`.
#[cfg(target_os = "macos")]
pub fn triangular_solve(a: &Tensor, b: &Tensor, upper: bool) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert!(a_shape.len() == 2 && a_shape[0] == a_shape[1], "triangular matrix must be square");
    let n = a_shape[0] as i32;
    let nrhs = if b_shape.len() > 1 { b_shape[1] as i32 } else { 1 };

    let mut a_data = a.data();
    let mut b_data = b.data();
    transpose_buf(&mut a_data, n as usize, n as usize);
    if nrhs > 1 {
        transpose_buf(&mut b_data, n as usize, nrhs as usize);
    }

    let uplo: u8 = if upper { b'U' } else { b'L' };
    let trans = b'N';
    let diag = b'N';
    let mut info = 0i32;

    unsafe {
        strtrs_(
            &uplo, &trans, &diag, &n, &nrhs,
            a_data.as_ptr(), &n, b_data.as_mut_ptr(), &n, &mut info,
        );
    }
    assert!(info == 0, "LAPACK strtrs failed with info={}", info);

    if nrhs > 1 {
        transpose_buf(&mut b_data, nrhs as usize, n as usize);
    }
    Tensor::new(b_data, b_shape.to_vec(), false)
}

#[cfg(not(target_os = "macos"))]
pub fn triangular_solve(_a: &Tensor, _b: &Tensor, _upper: bool) -> Tensor {
    panic!("linalg::triangular_solve requires macOS Accelerate framework");
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
    fn test_norm_l2() {
        let t = Tensor::new(vec![3.0, 4.0], vec![2], false);
        let n = norm(&t, None);
        assert!(approx_eq(n.data()[0], 5.0, 1e-5));
    }

    #[test]
    fn test_norm_l1() {
        let t = Tensor::new(vec![-3.0, 4.0], vec![2], false);
        let n = norm(&t, Some(1.0));
        assert!(approx_eq(n.data()[0], 7.0, 1e-5));
    }

    #[test]
    fn test_norm_linf() {
        let t = Tensor::new(vec![-3.0, 4.0, 2.0], vec![3], false);
        let n = norm(&t, Some(f32::INFINITY));
        assert!(approx_eq(n.data()[0], 4.0, 1e-5));
    }

    #[test]
    fn test_cross_product() {
        let a = Tensor::new(vec![1.0, 0.0, 0.0], vec![3], false);
        let b = Tensor::new(vec![0.0, 1.0, 0.0], vec![3], false);
        let c = cross(&a, &b);
        let d = c.data();
        assert!(approx_eq(d[0], 0.0, 1e-6));
        assert!(approx_eq(d[1], 0.0, 1e-6));
        assert!(approx_eq(d[2], 1.0, 1e-6));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_solve_identity() {
        // Solve Ix = b => x = b
        let a = Tensor::eye(3, false);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
        let x = solve(&a, &b);
        let d = x.data();
        assert!(approx_eq(d[0], 1.0, 1e-4));
        assert!(approx_eq(d[1], 2.0, 1e-4));
        assert!(approx_eq(d[2], 3.0, 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_inv_identity() {
        let a = Tensor::eye(3, false);
        let a_inv = inv(&a);
        let d = a_inv.data();
        // Should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(d[i * 3 + j], expected, 1e-4),
                    "inv[{},{}] = {} (expected {})", i, j, d[i * 3 + j], expected,
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_inv_2x2() {
        // A = [[1, 2], [3, 4]], A^-1 = [[-2, 1], [1.5, -0.5]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let a_inv = inv(&a);
        let d = a_inv.data();
        assert!(approx_eq(d[0], -2.0, 1e-4));
        assert!(approx_eq(d[1], 1.0, 1e-4));
        assert!(approx_eq(d[2], 1.5, 1e-4));
        assert!(approx_eq(d[3], -0.5, 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cholesky() {
        // A = [[4, 2], [2, 3]]  (positive definite)
        // L = [[2, 0], [1, sqrt(2)]]
        let a = Tensor::new(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], false);
        let l = cholesky(&a);
        let d = l.data();
        assert!(approx_eq(d[0], 2.0, 1e-4));
        assert!(approx_eq(d[1], 0.0, 1e-4));
        assert!(approx_eq(d[2], 1.0, 1e-4));
        assert!(approx_eq(d[3], (2.0f32).sqrt(), 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_svd_identity() {
        let a = Tensor::eye(3, false);
        let (u, s, vt) = svd(&a);
        let sd = s.data();
        // Singular values of identity are all 1
        for &sv in &sd {
            assert!(approx_eq(sv, 1.0, 1e-4));
        }
        assert_eq!(u.shape(), vec![3, 3]);
        assert_eq!(vt.shape(), vec![3, 3]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_qr_shapes() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false);
        let (q, r) = qr(&a);
        assert_eq!(q.shape(), vec![3, 2]);
        assert_eq!(r.shape(), vec![2, 2]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_eigh_identity() {
        let a = Tensor::eye(3, false);
        let (vals, _vecs) = eigh(&a);
        let vd = vals.data();
        for &v in &vd {
            assert!(approx_eq(v, 1.0, 1e-4));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_det_2x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let d = det(&a);
        // det = 1*4 - 2*3 = -2
        assert!(approx_eq(d.data()[0].abs(), 2.0, 1e-3));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_lu_shapes() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false);
        let (l, u, pivots) = lu(&a);
        assert_eq!(l.shape(), vec![3, 2]); // [m, k]
        assert_eq!(u.shape(), vec![2, 2]); // [k, n]
        assert_eq!(pivots.len(), 2);
    }
}
