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
    fn ssyevd_(
        jobz: *const u8, uplo: *const u8, n: *const i32, a: *mut f32,
        lda: *const i32, w: *mut f32, work: *mut f32, lwork: *const i32,
        iwork: *mut i32, liwork: *const i32, info: *mut i32,
    );
    fn strtrs_(
        uplo: *const u8, trans: *const u8, diag: *const u8, n: *const i32,
        nrhs: *const i32, a: *const f32, lda: *const i32, b: *mut f32,
        ldb: *const i32, info: *mut i32,
    );
    fn sgels_(
        trans: *const u8, m: *const i32, n: *const i32, nrhs: *const i32,
        a: *mut f32, lda: *const i32, b: *mut f32, ldb: *const i32,
        work: *mut f32, lwork: *const i32, info: *mut i32,
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
    let nu = n as usize;

    let mut data = a.data();
    // A is symmetric so row-major data = column-major data (A^T = A).
    // Use uplo='U': LAPACK computes U in column-major upper triangle.
    // Column-major upper triangle positions (j*n+i, i<=j) correspond to
    // row-major lower triangle (row=j, col=i, col<=row), and the values
    // are U[i][j] read as (row=j,col=i) = L[j][i] since L = U^T.
    // Result: L already sits in the row-major lower triangle — no transpose needed.

    let uplo = b'U';
    let mut info = 0i32;

    unsafe {
        spotrf_(&uplo, &n, data.as_mut_ptr(), &n, &mut info);
    }
    assert!(info == 0, "LAPACK spotrf failed with info={} (matrix not positive definite?)", info);

    // Zero upper triangle in row-major (positions i*n+j where j > i)
    for i in 0..nu {
        for j in (i + 1)..nu {
            data[i * nu + j] = 0.0;
        }
    }

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
    // Skip input transpose: symmetric matrix A = A^T, so row-major = column-major

    let mut w = vec![0.0f32; n as usize];
    let mut info = 0i32;
    let jobz = b'V'; // compute eigenvalues and eigenvectors
    let uplo = b'U';

    // Workspace query (ssyevd: divide-and-conquer, much faster for n>64)
    let mut work_query = vec![0.0f32; 1];
    let mut iwork_query = vec![0i32; 1];
    let lwork_query: i32 = -1;
    let liwork_query: i32 = -1;
    unsafe {
        ssyevd_(
            &jobz, &uplo, &n, a_data.as_mut_ptr(), &n,
            w.as_mut_ptr(), work_query.as_mut_ptr(), &lwork_query,
            iwork_query.as_mut_ptr(), &liwork_query, &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let liwork = iwork_query[0];
    let mut work = vec![0.0f32; lwork as usize];
    let mut iwork = vec![0i32; liwork as usize];

    unsafe {
        ssyevd_(
            &jobz, &uplo, &n, a_data.as_mut_ptr(), &n,
            w.as_mut_ptr(), work.as_mut_ptr(), &lwork,
            iwork.as_mut_ptr(), &liwork, &mut info,
        );
    }
    assert!(info == 0, "LAPACK ssyevd failed with info={}", info);

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
        // Skip transpose: det(A) = det(A^T), so row-major = column-major for det

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

/// Matrix rank via SVD: count singular values > tolerance.
///
/// Default tolerance: `max_sv * max(m, n) * f32::EPSILON`.
pub fn matrix_rank(a: &Tensor, tol: Option<f32>) -> usize {
    let shape = a.shape();
    assert!(shape.len() == 2, "matrix_rank requires 2D matrix");
    let m = shape[0];
    let n = shape[1];
    let (_u, s, _vt) = svd(a);
    let s_data = s.data();
    let max_sv = s_data.iter().cloned().fold(0.0f32, f32::max);
    let threshold = tol.unwrap_or(max_sv * (m.max(n) as f32) * f32::EPSILON);
    s_data.iter().filter(|&&sv| sv > threshold).count()
}

/// Condition number: ratio of largest to smallest singular value (2-norm condition number).
pub fn cond(a: &Tensor) -> f32 {
    let shape = a.shape();
    assert!(shape.len() == 2, "cond requires 2D matrix");
    let (_u, s, _vt) = svd(a);
    let s_data = s.data();
    let max_sv = s_data.iter().cloned().fold(0.0f32, f32::max);
    let min_sv = s_data.iter().cloned().fold(f32::INFINITY, f32::min);
    if min_sv == 0.0 {
        f32::INFINITY
    } else {
        max_sv / min_sv
    }
}

/// Least-squares solution via LAPACK sgels_: minimise ||Ax - b||.
///
/// - `a`: `[m, n]`
/// - `b`: `[m]` or `[m, nrhs]`
/// - Returns the least-squares solution `x` of shape `[n]` or `[n, nrhs]`.
#[cfg(target_os = "macos")]
pub fn lstsq(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert!(a_shape.len() == 2, "lstsq: A must be 2D");
    let m = a_shape[0] as i32;
    let n = a_shape[1] as i32;
    let nrhs = if b_shape.len() > 1 { b_shape[1] as i32 } else { 1 };
    assert!(
        b_shape[0] == m as usize,
        "lstsq: first dimension of b must match rows of A"
    );

    let mut a_data = a.data();
    transpose_buf(&mut a_data, m as usize, n as usize);

    // b needs to be in a buffer of size max(m,n) * nrhs for sgels_
    let ldb = m.max(n);
    let mut b_col = vec![0.0f32; (ldb * nrhs) as usize];
    if nrhs == 1 {
        // b is a vector [m]
        for i in 0..m as usize {
            b_col[i] = b.data()[i];
        }
    } else {
        // b is [m, nrhs] row-major -> column-major in ldb rows
        let bd = b.data();
        for j in 0..nrhs as usize {
            for i in 0..m as usize {
                b_col[j * ldb as usize + i] = bd[i * nrhs as usize + j];
            }
        }
    }

    let trans = b'N';
    let mut info = 0i32;

    // Workspace query
    let mut work_query = vec![0.0f32; 1];
    let lwork_query: i32 = -1;
    unsafe {
        sgels_(
            &trans, &m, &n, &nrhs,
            a_data.as_mut_ptr(), &m,
            b_col.as_mut_ptr(), &ldb,
            work_query.as_mut_ptr(), &lwork_query, &mut info,
        );
    }
    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f32; lwork as usize];

    unsafe {
        sgels_(
            &trans, &m, &n, &nrhs,
            a_data.as_mut_ptr(), &m,
            b_col.as_mut_ptr(), &ldb,
            work.as_mut_ptr(), &lwork, &mut info,
        );
    }
    assert!(info == 0, "LAPACK sgels failed with info={}", info);

    // Extract solution x from first n rows of b_col
    if nrhs == 1 {
        let x_data: Vec<f32> = b_col[..n as usize].to_vec();
        Tensor::new(x_data, vec![n as usize], false)
    } else {
        // Column-major [ldb, nrhs] -> row-major [n, nrhs]
        let nu = n as usize;
        let nrhs_u = nrhs as usize;
        let mut x_data = vec![0.0f32; nu * nrhs_u];
        for j in 0..nrhs_u {
            for i in 0..nu {
                x_data[i * nrhs_u + j] = b_col[j * ldb as usize + i];
            }
        }
        Tensor::new(x_data, vec![nu, nrhs_u], false)
    }
}

#[cfg(not(target_os = "macos"))]
pub fn lstsq(_a: &Tensor, _b: &Tensor) -> Tensor {
    panic!("linalg::lstsq requires macOS Accelerate framework");
}

/// Matrix power: raise a square matrix to an integer power n.
///
/// For n >= 0: repeated squaring. For n < 0: invert first, then raise to |n|.
pub fn matrix_power(a: &Tensor, n: i32) -> Tensor {
    let shape = a.shape();
    assert!(
        shape.len() == 2 && shape[0] == shape[1],
        "matrix_power requires a square matrix"
    );
    let sz = shape[0];

    if n == 0 {
        return Tensor::eye(sz, false);
    }

    let (base, exp) = if n < 0 {
        (inv(a), (-n) as u32)
    } else {
        (a.clone(), n as u32)
    };

    // Exponentiation by squaring
    let mut result = Tensor::eye(sz, false);
    let mut base_pow = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.matmul(&base_pow);
        }
        e >>= 1;
        if e > 0 {
            let tmp = base_pow.clone();
            base_pow = tmp.matmul(&base_pow);
        }
    }

    // Return without autograd
    Tensor::new(result.data(), result.shape().to_vec(), false)
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

    // -----------------------------------------------------------------------
    // matrix_rank tests
    // -----------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_rank_identity() {
        let a = Tensor::eye(4, false);
        assert_eq!(matrix_rank(&a, None), 4);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_rank_zero() {
        let a = Tensor::zeros(&[3, 3], false);
        assert_eq!(matrix_rank(&a, None), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_rank_rank_deficient() {
        // Row 2 = 2 * Row 1 -> rank 1
        let a = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], false);
        assert_eq!(matrix_rank(&a, None), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_rank_rectangular() {
        // [3, 2] full-rank matrix
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2], false);
        assert_eq!(matrix_rank(&a, None), 2);
    }

    // -----------------------------------------------------------------------
    // cond tests
    // -----------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cond_identity() {
        let a = Tensor::eye(3, false);
        let c = cond(&a);
        assert!(approx_eq(c, 1.0, 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cond_diagonal() {
        // diag(1, 2, 4) -> cond = 4/1 = 4
        let a = Tensor::new(
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0],
            vec![3, 3],
            false,
        );
        let c = cond(&a);
        assert!(approx_eq(c, 4.0, 1e-3));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_cond_singular() {
        // Nearly singular matrix -> very large condition number
        let a = Tensor::new(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], false);
        let c = cond(&a);
        // SVD may produce a tiny (not exactly zero) smallest singular value,
        // so the condition number is extremely large but possibly not infinite.
        assert!(c > 1e6 || c.is_infinite(), "cond = {} should be very large", c);
    }

    // -----------------------------------------------------------------------
    // lstsq tests
    // -----------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn test_lstsq_exact() {
        // Exact solution: A x = b where A is square and invertible
        // A = [[1, 0], [0, 1]], b = [3, 4] -> x = [3, 4]
        let a = Tensor::eye(2, false);
        let b = Tensor::new(vec![3.0, 4.0], vec![2], false);
        let x = lstsq(&a, &b);
        let xd = x.data();
        assert!(approx_eq(xd[0], 3.0, 1e-4));
        assert!(approx_eq(xd[1], 4.0, 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_lstsq_overdetermined() {
        // Overdetermined: 3 equations, 2 unknowns
        // A = [[1, 0], [0, 1], [1, 1]], b = [1, 2, 4]
        // Least squares: minimise ||Ax - b||^2
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2], false);
        let b = Tensor::new(vec![1.0, 2.0, 4.0], vec![3], false);
        let x = lstsq(&a, &b);
        assert_eq!(x.shape(), vec![2]);
        // Verify Ax is close to b in least-squares sense
        let xd = x.data();
        // Residual should be small relative to original
        let r0 = xd[0] - 1.0;
        let r1 = xd[1] - 2.0;
        let r2 = xd[0] + xd[1] - 4.0;
        let res_norm = (r0 * r0 + r1 * r1 + r2 * r2).sqrt();
        assert!(res_norm < 2.0, "residual too large: {}", res_norm);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_lstsq_multi_rhs() {
        // A = I, B = [[1, 2], [3, 4]] -> X = B
        let a = Tensor::eye(2, false);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let x = lstsq(&a, &b);
        assert_eq!(x.shape(), vec![2, 2]);
        let xd = x.data();
        assert!(approx_eq(xd[0], 1.0, 1e-4));
        assert!(approx_eq(xd[1], 2.0, 1e-4));
        assert!(approx_eq(xd[2], 3.0, 1e-4));
        assert!(approx_eq(xd[3], 4.0, 1e-4));
    }

    // -----------------------------------------------------------------------
    // matrix_power tests
    // -----------------------------------------------------------------------

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_zero() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let r = matrix_power(&a, 0);
        let d = r.data();
        // A^0 = I
        assert!(approx_eq(d[0], 1.0, 1e-6));
        assert!(approx_eq(d[1], 0.0, 1e-6));
        assert!(approx_eq(d[2], 0.0, 1e-6));
        assert!(approx_eq(d[3], 1.0, 1e-6));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_one() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let r = matrix_power(&a, 1);
        let d = r.data();
        assert!(approx_eq(d[0], 1.0, 1e-5));
        assert!(approx_eq(d[1], 2.0, 1e-5));
        assert!(approx_eq(d[2], 3.0, 1e-5));
        assert!(approx_eq(d[3], 4.0, 1e-5));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_two() {
        // A = [[1, 2], [3, 4]]
        // A^2 = [[7, 10], [15, 22]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let r = matrix_power(&a, 2);
        let d = r.data();
        assert!(approx_eq(d[0], 7.0, 1e-4));
        assert!(approx_eq(d[1], 10.0, 1e-4));
        assert!(approx_eq(d[2], 15.0, 1e-4));
        assert!(approx_eq(d[3], 22.0, 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_three() {
        // A^3 = A^2 * A = [[7,10],[15,22]] * [[1,2],[3,4]] = [[37,54],[81,118]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let r = matrix_power(&a, 3);
        let d = r.data();
        assert!(approx_eq(d[0], 37.0, 1e-3));
        assert!(approx_eq(d[1], 54.0, 1e-3));
        assert!(approx_eq(d[2], 81.0, 1e-3));
        assert!(approx_eq(d[3], 118.0, 1e-3));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_negative_one() {
        // A^-1 for A = [[1, 2], [3, 4]]
        // A^-1 = [[-2, 1], [1.5, -0.5]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let r = matrix_power(&a, -1);
        let d = r.data();
        assert!(approx_eq(d[0], -2.0, 1e-4));
        assert!(approx_eq(d[1], 1.0, 1e-4));
        assert!(approx_eq(d[2], 1.5, 1e-4));
        assert!(approx_eq(d[3], -0.5, 1e-4));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_negative_two() {
        // A^-2 = (A^-1)^2
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let r = matrix_power(&a, -2);
        // (A^-1)^2 = [[-2,1],[1.5,-0.5]] * [[-2,1],[1.5,-0.5]] = [[5.5, -2.5], [-3.75, 1.75]]
        let d = r.data();
        assert!(approx_eq(d[0], 5.5, 1e-3));
        assert!(approx_eq(d[1], -2.5, 1e-3));
        assert!(approx_eq(d[2], -3.75, 1e-3));
        assert!(approx_eq(d[3], 1.75, 1e-3));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_matrix_power_identity() {
        let a = Tensor::eye(3, false);
        let r = matrix_power(&a, 5);
        let d = r.data();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(d[i * 3 + j], expected, 1e-5));
            }
        }
    }
}
