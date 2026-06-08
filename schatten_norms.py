"""
Schatten p-Norm Estimation via Krylov Subspace Methods
======================================================

Methods implemented:
  1. Stochastic Lanczos Quadrature (SLQ) — from Benzi, Rinelli, Simunec (2026) "Gaps" paper
     Runs Lanczos on A†A, applies f(λ)=λ^{p/2} to tridiagonal eigenvalues,
     averages over Hutchinson random vectors.

  2. Golub–Kahan Bidiagonalization (GKB) Quadrature — from Arrigo, Benzi, Fenu (2016)
     Builds bidiagonal B_ℓ via GKB, computes SVD of B_ℓ, evaluates Gauss
     quadrature with f(σ)=σ^p to estimate v† (A†A)^{p/2} v.

  3. Brute-force SVD reference for validation.

Tested on Kronecker powers of random 3×3 complex density matrices
(Hermitian, positive semidefinite, trace 1).
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import LinearOperator, svds
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import sys
import io
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================
# 1. Random Density Matrix Generation
# ============================================================

def random_density_matrix(d=3):
    """
    Generate a random d×d complex density matrix (Hermitian, PSD, trace 1).
    Method: draw a random complex Gaussian matrix G, form ρ = G G† / tr(G G†).
    """
    G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = G @ G.conj().T
    rho = rho / np.trace(rho)
    return rho


# ============================================================
# 2. Matrix-Free Tensor (Kronecker) Operations
# ============================================================

def apply_tensor_power(M_base, v, n):
    """
    Compute (M_base ⊗ ... ⊗ M_base) @ v without building the full matrix.
    Reshapes v into an n-dimensional tensor and applies M_base along each axis.
    Works for complex matrices.
    """
    d = M_base.shape[0]
    X = v.reshape((d,) * n)
    for i in range(n):
        X = np.tensordot(M_base, X, axes=([1], [i]))
        X = np.moveaxis(X, 0, i)
    return X.flatten()


def make_kronecker_operator(rho, n_kron):
    """
    Build a LinearOperator for ρ^{⊗n} (n-fold Kronecker power of rho).
    Returns the operator and the full dimension N = d^n.
    """
    d = rho.shape[0]
    N = d ** n_kron

    def matvec(v):
        return apply_tensor_power(rho, v, n_kron)

    def rmatvec(v):
        return apply_tensor_power(rho.conj().T, v, n_kron)

    op = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec, dtype=complex)
    return op, N


# ============================================================
# 3. Brute-Force Schatten Norm (Full SVD)
# ============================================================

def brute_force_schatten(A_dense, p):
    """
    Compute ||A||_p exactly using full SVD.
    For p=np.inf, returns the largest singular value (operator norm).
    """
    S = np.linalg.svd(A_dense, compute_uv=False)
    S = S[S > 1e-15]  # drop numerical zeros
    if p == np.inf:
        return S[0]  # largest
    return np.sum(S ** p) ** (1.0 / p)


def build_dense_kronecker(rho, n_kron):
    """Build the full dense Kronecker power matrix ρ^{⊗n}."""
    M = rho
    for _ in range(1, n_kron):
        M = np.kron(M, rho)
    return M


# ============================================================
# 4. Lanczos Tridiagonalization on A†A
# ============================================================

def lanczos_hermitian(matvec_ata, v0, m):
    """
    Run m steps of Lanczos on the Hermitian operator H = A†A
    with starting vector v0.

    Returns:
        T_m: m×m real symmetric tridiagonal matrix (as dense)
        beta_hist: off-diagonal entries for a-posteriori bounds
    """
    n = len(v0)
    v_norm = np.linalg.norm(v0)
    q = v0 / v_norm

    alpha = np.zeros(m)
    beta = np.zeros(m)
    q_prev = np.zeros(n, dtype=v0.dtype)
    beta_prev = 0.0

    for j in range(m):
        w = matvec_ata(q)
        w = w - beta_prev * q_prev
        alpha[j] = np.real(np.vdot(q, w))  # must be real for Hermitian
        w = w - alpha[j] * q
        beta_val = np.linalg.norm(w)
        if beta_val < 1e-14:
            # Lucky breakdown
            m_actual = j + 1
            T = np.diag(alpha[:m_actual])
            if m_actual > 1:
                T += np.diag(beta[1:m_actual], k=1) + np.diag(beta[1:m_actual], k=-1)
            return T, v_norm, m_actual
        if j < m - 1:
            beta[j + 1] = beta_val
            q_prev = q
            beta_prev = beta_val
            q = w / beta_val

    T = np.diag(alpha)
    if m > 1:
        T += np.diag(beta[1:m], k=1) + np.diag(beta[1:m], k=-1)
    return T, v_norm, m


# ============================================================
# 5. Stochastic Lanczos Quadrature (SLQ) for Schatten p-norms
#    From Benzi, Rinelli, Simunec "Gaps" (Algorithm 1 / SLQ)
# ============================================================

def slq_schatten_p_norm(matvec, rmatvec, N, p, s=10, m=30):
    """
    Estimate ||A||_p using Stochastic Lanczos Quadrature.

    ||A||_p^p = tr((A†A)^{p/2})

    For each of s random Gaussian vectors x_i:
      - Run m-step Lanczos on A†A with x_i
      - Get tridiagonal T_m, eigendecompose T_m = U D U^T
      - Estimate: q_i = ||x_i||^2 · e_1^T · D^{p/2} · e_1

    Returns ||A||_p = (mean(q_i))^{1/p}

    For p=inf, returns max singular value via svds instead.
    """
    if p == np.inf:
        # Use ARPACK for largest singular value
        try:
            op = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec, dtype=complex)
            _, S_max, _ = svds(op, k=1, which="LM")
            return S_max[0]
        except Exception:
            return np.nan

    def matvec_ata(v):
        return rmatvec(matvec(v))

    trace_est = 0.0
    for _ in range(s):
        # Gaussian random vector (complex)
        x = np.random.randn(N) + 1j * np.random.randn(N)
        x /= np.sqrt(2)  # so E[|x_i|^2] = 1

        T, v_norm, m_actual = lanczos_hermitian(matvec_ata, x, m)

        # Eigendecompose the small tridiagonal matrix
        eigvals, eigvecs = np.linalg.eigh(T)
        eigvals = np.maximum(eigvals, 0)  # clip negative eigenvalues from roundoff

        # w = ||x||_2 * U^T e_1
        w = v_norm * eigvecs[0, :]

        # Quadrature estimate: w^T D^{p/2} w
        q = np.sum(np.abs(w) ** 2 * eigvals ** (p / 2))
        trace_est += np.real(q)

    schatten_p_to_p = trace_est / s
    if schatten_p_to_p < 0:
        schatten_p_to_p = 0
    return schatten_p_to_p ** (1.0 / p)


# ============================================================
# 6. Golub-Kahan Bidiagonalization (GKB)
#    From Arrigo, Benzi, Fenu (2016) — adapted for complex matrices
# ============================================================

def golub_kahan_bidiag(matvec, rmatvec, v, l_steps):
    """
    Run l_steps of Golub–Kahan bidiagonalization.

    Given A accessible via matvec/rmatvec and starting vector v:
      A Q_ℓ = P_ℓ B_ℓ
      A† P_ℓ = Q_ℓ B_ℓ† + γ_ℓ q_ℓ e_ℓ^T

    Returns:
      omega: main diagonal of B_ℓ (length l_actual)
      gamma: superdiagonal of B_ℓ (length l_actual - 1)
      v_norm: ||v||
    """
    N = len(v)
    v_norm = np.linalg.norm(v)
    q = v / v_norm

    omega = np.zeros(l_steps)
    gamma = np.zeros(l_steps)

    p_prev = np.zeros(N, dtype=v.dtype)
    gamma_prev = 0.0

    for j in range(l_steps):
        # u-step
        u_unnorm = matvec(q) - gamma_prev * p_prev
        omega[j] = np.linalg.norm(u_unnorm)

        if omega[j] < 1e-14:
            l_actual = j + 1
            return omega[:l_actual], gamma[:l_actual - 1], v_norm

        p = u_unnorm / omega[j]

        # v-step
        if j < l_steps - 1:
            v_unnorm = rmatvec(p) - omega[j] * q
            gamma[j] = np.linalg.norm(v_unnorm)
            if gamma[j] < 1e-14:
                l_actual = j + 1
                return omega[:l_actual], gamma[:l_actual - 1], v_norm
            q = v_unnorm / gamma[j]
            gamma_prev = gamma[j]
            p_prev = p

    l_actual = l_steps
    return omega[:l_actual], gamma[:l_actual - 1], v_norm


def gkb_quadrature_schatten(matvec, rmatvec, v, l_steps, p):
    """
    Estimate v† (A†A)^{p/2} v using GKB + Gauss quadrature.

    For p=1: this is v† |A| v  (existing code's approach, using sqrtm).
    For general p: evaluates f(σ) = σ^p at the singular values of B_ℓ.

    The Gauss quadrature rule has:
      - nodes = singular values θ_j of B_ℓ
      - weights = (e_1^T ν_j)^2 / θ_j  (right singular vectors)

    The estimate is: ||v||^2 · Σ_j w_j · f(θ_j) = ||v||^2 · Σ_j (e_1^T ν_j)^2 θ_j^{p-1}
    """
    omega, gamma, v_norm = golub_kahan_bidiag(matvec, rmatvec, v, l_steps)
    l_actual = len(omega)

    # Build the upper bidiagonal matrix B_ℓ
    B_l = np.diag(omega)
    if l_actual > 1 and len(gamma) > 0:
        B_l[:len(gamma), 1:len(gamma) + 1] += np.diag(gamma)

    if p == 1:
        # Original approach: use sqrtm on T = B†B
        T_l = B_l.conj().T @ B_l
        sqrt_T = sqrtm(T_l)
        return (v_norm ** 2) * np.real(sqrt_T[0, 0])
    else:
        # General p: SVD of B_ℓ, then Gauss quadrature with f(σ)=σ^p
        try:
            U_b, theta, Vt_b = np.linalg.svd(B_l, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.nan

        theta = np.maximum(theta, 0)
        # Right singular vectors: columns of V_b = Vt_b.T
        # weights = (e_1^T v_j)^2 · θ_j^{p-1}
        e1 = np.zeros(Vt_b.shape[1])
        e1[0] = 1.0
        weights_v = Vt_b @ e1  # V^T e_1, so entries are v_j^T e_1

        estimate = 0.0
        for j in range(len(theta)):
            if theta[j] > 1e-15:
                estimate += np.abs(weights_v[j]) ** 2 * theta[j] ** (p - 1)

        return (v_norm ** 2) * np.real(estimate)


def hutchinson_gkb_schatten(matvec, rmatvec, N, p, K_samples=20, l_steps=30):
    """
    Estimate ||A||_p using Hutchinson's estimator + GKB quadrature.

    ||A||_p^p = tr((A†A)^{p/2}) ≈ (1/K) Σ_k  v_k† (A†A)^{p/2} v_k

    For p=inf, uses svds for the largest singular value.
    """
    if p == np.inf:
        try:
            op = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec, dtype=complex)
            _, S_max, _ = svds(op, k=1, which="LM")
            return S_max[0]
        except Exception:
            return np.nan

    trace_est = 0.0
    for _ in range(K_samples):
        # Rademacher-like vector (complex: random signs on real and imag)
        v = np.random.choice([-1.0, 1.0], size=N) + 1j * np.random.choice([-1.0, 1.0], size=N)
        v /= np.sqrt(2)
        trace_est += gkb_quadrature_schatten(matvec, rmatvec, v, l_steps, p)

    schatten_p_to_p = np.real(trace_est) / K_samples
    if schatten_p_to_p < 0:
        schatten_p_to_p = 0
    return schatten_p_to_p ** (1.0 / p)


# ============================================================
# 7. Benchmarking Functions
# ============================================================

def run_accuracy_experiment(rho, n_kron_values, p_values, m_values, s_values,
                            l_values=None, K_values=None, n_trials=3):
    """
    Run accuracy experiments comparing SLQ and GKB vs brute force.
    Uses Kronecker powers of rho.

    Returns dict with all results.
    """
    if l_values is None:
        l_values = m_values  # same range for GKB steps
    if K_values is None:
        K_values = s_values  # same range for GKB samples

    results = {}

    for n_kron in n_kron_values:
        d = rho.shape[0]
        N = d ** n_kron
        print(f"\n{'='*60}")
        print(f"  Kronecker power n={n_kron}, matrix size N={N}×{N}")
        print(f"{'='*60}")

        # Build dense matrix for ground truth (only if feasible)
        if N <= 2000:
            M_dense = build_dense_kronecker(rho, n_kron)
        else:
            M_dense = None
            print("  [Skipping brute force — matrix too large]")

        op, _ = make_kronecker_operator(rho, n_kron)
        matvec = op.matvec
        rmatvec = op.rmatvec

        for p in p_values:
            p_label = "inf" if p == np.inf else str(p)
            print(f"\n  --- p = {p_label} ---")

            # Ground truth
            if M_dense is not None:
                exact_val = brute_force_schatten(M_dense, p)
                print(f"  Exact ||A||_{p_label} = {exact_val:.8f}")
            else:
                exact_val = None

            # --- SLQ: vary m with fixed s ---
            key_slq_m = (n_kron, p, "slq_vs_m")
            results[key_slq_m] = {"m": [], "error": [], "time": []}
            s_fixed = max(s_values)
            for m in m_values:
                errors = []
                times = []
                for _ in range(n_trials):
                    t0 = time.perf_counter()
                    est = slq_schatten_p_norm(matvec, rmatvec, N, p, s=s_fixed, m=m)
                    dt = time.perf_counter() - t0
                    times.append(dt)
                    if exact_val is not None and exact_val > 0:
                        errors.append(abs(est - exact_val) / exact_val)
                    else:
                        errors.append(np.nan)
                results[key_slq_m]["m"].append(m)
                results[key_slq_m]["error"].append(np.median(errors))
                results[key_slq_m]["time"].append(np.median(times))
            print(f"  SLQ vs m: best rel error = {min(results[key_slq_m]['error']):.2e}")

            # --- SLQ: vary s with fixed m ---
            key_slq_s = (n_kron, p, "slq_vs_s")
            results[key_slq_s] = {"s": [], "error": [], "time": []}
            m_fixed = max(m_values)
            for s in s_values:
                errors = []
                times = []
                for _ in range(n_trials):
                    t0 = time.perf_counter()
                    est = slq_schatten_p_norm(matvec, rmatvec, N, p, s=s, m=m_fixed)
                    dt = time.perf_counter() - t0
                    times.append(dt)
                    if exact_val is not None and exact_val > 0:
                        errors.append(abs(est - exact_val) / exact_val)
                    else:
                        errors.append(np.nan)
                results[key_slq_s]["s"].append(s)
                results[key_slq_s]["error"].append(np.median(errors))
                results[key_slq_s]["time"].append(np.median(times))
            print(f"  SLQ vs s: best rel error = {min(results[key_slq_s]['error']):.2e}")

            # --- GKB: vary l with fixed K ---
            key_gkb_l = (n_kron, p, "gkb_vs_l")
            results[key_gkb_l] = {"l": [], "error": [], "time": []}
            K_fixed = max(K_values)
            for l in l_values:
                errors = []
                times = []
                for _ in range(n_trials):
                    t0 = time.perf_counter()
                    est = hutchinson_gkb_schatten(matvec, rmatvec, N, p,
                                                  K_samples=K_fixed, l_steps=l)
                    dt = time.perf_counter() - t0
                    times.append(dt)
                    if exact_val is not None and exact_val > 0:
                        errors.append(abs(est - exact_val) / exact_val)
                    else:
                        errors.append(np.nan)
                results[key_gkb_l]["l"].append(l)
                results[key_gkb_l]["error"].append(np.median(errors))
                results[key_gkb_l]["time"].append(np.median(times))
            print(f"  GKB vs l: best rel error = {min(results[key_gkb_l]['error']):.2e}")

            # --- GKB: vary K with fixed l ---
            key_gkb_K = (n_kron, p, "gkb_vs_K")
            results[key_gkb_K] = {"K": [], "error": [], "time": []}
            l_fixed = max(l_values)
            for K in K_values:
                errors = []
                times = []
                for _ in range(n_trials):
                    t0 = time.perf_counter()
                    est = hutchinson_gkb_schatten(matvec, rmatvec, N, p,
                                                  K_samples=K, l_steps=l_fixed)
                    dt = time.perf_counter() - t0
                    times.append(dt)
                    if exact_val is not None and exact_val > 0:
                        errors.append(abs(est - exact_val) / exact_val)
                    else:
                        errors.append(np.nan)
                results[key_gkb_K]["K"].append(K)
                results[key_gkb_K]["error"].append(np.median(errors))
                results[key_gkb_K]["time"].append(np.median(times))
            print(f"  GKB vs K: best rel error = {min(results[key_gkb_K]['error']):.2e}")

    return results


def run_speed_experiment(rho, p_values, m_fixed=30, s_fixed=10,
                         l_fixed=30, K_fixed=10, n_trials=3, limit_brute_time=True):
    """
    Time comparison: brute force SVD vs SLQ vs GKB across matrix sizes.
    """
    speed_results = {"n_kron": [], "N": [], "p": [],
                     "t_brute": [], "t_slq": [], "t_gkb": [],
                     "mem_brute": [], "mem_slq": [], "mem_gkb": []}
    
    # Store the n at which each method failed due to memory
    failed_n = {"brute": None, "slq": None, "gkb": None}
    
    n_kron = 2
    while True:
        # Stop if all methods have failed
        if failed_n["brute"] is not None and failed_n["slq"] is not None and failed_n["gkb"] is not None:
            print("  All methods have reached their limits.")
            break

        # Temporary boolean check to limit brute force SVD from hanging the system
        if limit_brute_time and n_kron >= 8 and failed_n["brute"] is None:
            failed_n["brute"] = n_kron

        d = rho.shape[0]
        N = d ** n_kron
        op, _ = make_kronecker_operator(rho, n_kron)
        matvec = op.matvec
        rmatvec = op.rmatvec

        for p in p_values:
            p_label = "inf" if p == np.inf else str(p)
            print(f"  Speed test: n={n_kron}, N={N}, p={p_label} ... ", end="", flush=True)

            t_brute = np.nan
            if failed_n["brute"] is None:
                try:
                    M_dense = build_dense_kronecker(rho, n_kron)
                    times_brute = []
                    for _ in range(n_trials):
                        t0 = time.perf_counter()
                        _ = brute_force_schatten(M_dense, p)
                        times_brute.append(time.perf_counter() - t0)
                    t_brute = np.median(times_brute)
                except (MemoryError, np.core._exceptions._ArrayMemoryError, OSError):
                    failed_n["brute"] = n_kron

            t_slq = np.nan
            if failed_n["slq"] is None:
                try:
                    times_slq = []
                    for _ in range(n_trials):
                        t0 = time.perf_counter()
                        _ = slq_schatten_p_norm(matvec, rmatvec, N, p, s=s_fixed, m=m_fixed)
                        times_slq.append(time.perf_counter() - t0)
                    t_slq = np.median(times_slq)
                except (MemoryError, np.core._exceptions._ArrayMemoryError, OSError):
                    failed_n["slq"] = n_kron

            t_gkb = np.nan
            if failed_n["gkb"] is None:
                try:
                    times_gkb = []
                    for _ in range(n_trials):
                        t0 = time.perf_counter()
                        _ = hutchinson_gkb_schatten(matvec, rmatvec, N, p,
                                                    K_samples=K_fixed, l_steps=l_fixed)
                        times_gkb.append(time.perf_counter() - t0)
                    t_gkb = np.median(times_gkb)
                except (MemoryError, np.core._exceptions._ArrayMemoryError, OSError):
                    failed_n["gkb"] = n_kron


            # Theoretical memory usage (number of floats)
            # Brute force: N x N complex matrix = 2 * N^2 floats
            mem_brute = 2 * (N ** 2) if not np.isnan(t_brute) else np.nan
            # SLQ: ~4 complex vectors of size N in memory at a time = 8 * N floats
            mem_slq = 8 * N
            # GKB: ~5 complex vectors of size N in memory at a time = 10 * N floats
            mem_gkb = 10 * N

            speed_results["n_kron"].append(n_kron)
            speed_results["N"].append(N)
            speed_results["p"].append(p)
            speed_results["t_brute"].append(t_brute)
            speed_results["t_slq"].append(t_slq)
            speed_results["t_gkb"].append(t_gkb)
            speed_results["mem_brute"].append(mem_brute)
            speed_results["mem_slq"].append(mem_slq)
            speed_results["mem_gkb"].append(mem_gkb)

            bf_str = f"{t_brute:.4f}s" if not np.isnan(t_brute) else ("OOM" if failed_n["brute"] is not None else "N/A")
            slq_str = f"{t_slq:.4f}s" if not np.isnan(t_slq) else "OOM"
            gkb_str = f"{t_gkb:.4f}s" if not np.isnan(t_gkb) else "OOM"
            print(f"brute={bf_str}, SLQ={slq_str}, GKB={gkb_str}")

        n_kron += 1

    return speed_results, failed_n


# ============================================================
# 8. Plotting
# ============================================================

def plot_accuracy_results(results, rho, save_dir="."):
    """Create accuracy-vs-parameter plots."""

    # Collect all (n_kron, p) combos
    combos = set()
    for key in results:
        n_kron, p, method = key
        combos.add((n_kron, p))
    combos = sorted(combos)

    for n_kron, p in combos:
        p_label = "inf" if p == np.inf else str(p)
        N = rho.shape[0] ** n_kron

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Schatten-{p_label} Norm Accuracy  |  "
            f"rho^(x){n_kron} ({N}x{N} density matrix)",
            fontsize=14, fontweight="bold"
        )

        # SLQ vs m
        ax = axes[0, 0]
        key = (n_kron, p, "slq_vs_m")
        if key in results:
            r = results[key]
            ax.semilogy(r["m"], r["error"], "o-", color="#2196F3", linewidth=2, markersize=5)
        ax.set_xlabel("Lanczos steps m")
        ax.set_ylabel("Relative error")
        ax.set_title("SLQ: error vs Lanczos steps m (s fixed)")
        ax.grid(True, alpha=0.3)

        # SLQ vs s
        ax = axes[0, 1]
        key = (n_kron, p, "slq_vs_s")
        if key in results:
            r = results[key]
            ax.semilogy(r["s"], r["error"], "s-", color="#4CAF50", linewidth=2, markersize=5)
        ax.set_xlabel("Hutchinson samples s")
        ax.set_ylabel("Relative error")
        ax.set_title("SLQ: error vs samples s (m fixed)")
        ax.grid(True, alpha=0.3)

        # GKB vs l
        ax = axes[1, 0]
        key = (n_kron, p, "gkb_vs_l")
        if key in results:
            r = results[key]
            ax.semilogy(r["l"], r["error"], "^-", color="#FF9800", linewidth=2, markersize=5)
        ax.set_xlabel("GKB steps l")
        ax.set_ylabel("Relative error")
        ax.set_title("GKB: error vs bidiag steps l (K fixed)")
        ax.grid(True, alpha=0.3)

        # GKB vs K
        ax = axes[1, 1]
        key = (n_kron, p, "gkb_vs_K")
        if key in results:
            r = results[key]
            ax.semilogy(r["K"], r["error"], "D-", color="#E91E63", linewidth=2, markersize=5)
        ax.set_xlabel("Hutchinson samples K")
        ax.set_ylabel("Relative error")
        ax.set_title("GKB: error vs samples K (l fixed)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        p_file = "inf" if p == np.inf else str(p)
        fname = f"{save_dir}/accuracy_n{n_kron}_p{p_file}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_speed_results(speed_results, failed_n, save_dir="."):
    """Create speed comparison plots."""
    p_values = sorted(set(speed_results["p"]), key=lambda x: (x != np.inf, x))

    fig, axes = plt.subplots(1, len(p_values), figsize=(6 * len(p_values), 5))
    if len(p_values) == 1:
        axes = [axes]

    for ax, p in zip(axes, p_values):
        p_label = "inf" if p == np.inf else str(int(p))

        mask = [i for i, pp in enumerate(speed_results["p"]) if pp == p]
        Ns = [speed_results["N"][i] for i in mask]
        ns = [speed_results["n_kron"][i] for i in mask]
        t_brute = [speed_results["t_brute"][i] for i in mask]
        t_slq = [speed_results["t_slq"][i] for i in mask]
        t_gkb = [speed_results["t_gkb"][i] for i in mask]

        x_labels = [str(n) for n in ns]
        x_pos = np.arange(len(ns))

        # Only plot brute force where available
        brute_valid = [(i, t) for i, t in enumerate(t_brute) if not np.isnan(t)]
        if brute_valid:
            bi, bt = zip(*brute_valid)
            ax.semilogy([x_pos[i] for i in bi], bt, "s-",
                       color="#F44336", linewidth=2, markersize=7, label="Brute Force SVD")

        ax.semilogy(x_pos, t_slq, "o-",
                   color="#2196F3", linewidth=2, markersize=7, label="SLQ")
        ax.semilogy(x_pos, t_gkb, "^-",
                   color="#FF9800", linewidth=2, markersize=7, label="GKB")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Tensor power n")
        ax.set_ylabel("Time (seconds)")
        ax.set_title(f"Schatten-{p_label} Norm")
        
        # Add dashed lines where methods failed
        if failed_n["brute"] is not None and failed_n["brute"] in ns:
            idx = ns.index(failed_n["brute"])
            ax.axvline(x=idx, color="#F44336", linestyle="--", alpha=0.7)
            ax.text(idx, ax.get_ylim()[0] * 2, 'Brute Limit/OOM', color="#F44336", rotation=90, va='bottom', ha='right')
            
        if failed_n["slq"] is not None and failed_n["slq"] in ns:
            idx = ns.index(failed_n["slq"])
            ax.axvline(x=idx, color="#2196F3", linestyle="--", alpha=0.7)
            ax.text(idx, ax.get_ylim()[0] * 2, 'SLQ OOM', color="#2196F3", rotation=90, va='bottom', ha='right')
            
        if failed_n["gkb"] is not None and failed_n["gkb"] in ns:
            idx = ns.index(failed_n["gkb"])
            ax.axvline(x=idx, color="#FF9800", linestyle="--", alpha=0.7)
            ax.text(idx, ax.get_ylim()[0] * 2, 'GKB OOM', color="#FF9800", rotation=90, va='bottom', ha='right')

        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Speed Comparison: Krylov Methods vs Brute Force SVD",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = f"{save_dir}/speed_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_memory_results(speed_results, failed_n, save_dir="."):
    """Create theoretical memory comparison plots."""
    # We only need one plot since memory usage doesn't depend on p
    p_ref = speed_results["p"][0]
    mask = [i for i, p in enumerate(speed_results["p"]) if p == p_ref]
    
    Ns = [speed_results["N"][i] for i in mask]
    ns = [speed_results["n_kron"][i] for i in mask]
    mem_brute = [speed_results["mem_brute"][i] for i in mask]
    mem_slq = [speed_results["mem_slq"][i] for i in mask]
    mem_gkb = [speed_results["mem_gkb"][i] for i in mask]

    x_labels = [str(n) for n in ns]
    x_pos = np.arange(len(ns))

    fig, ax = plt.subplots(figsize=(8, 6))

    brute_valid = [(i, m) for i, m in enumerate(mem_brute) if not np.isnan(m)]
    if brute_valid:
        bi, bm = zip(*brute_valid)
        ax.semilogy([x_pos[i] for i in bi], bm, "s-",
                   color="#F44336", linewidth=2, markersize=7, label="Brute Force SVD (O(N^2))")

    ax.semilogy(x_pos, mem_slq, "o-",
               color="#2196F3", linewidth=2, markersize=7, label="SLQ (O(N))")
    ax.semilogy(x_pos, mem_gkb, "^-",
               color="#FF9800", linewidth=2, markersize=7, label="GKB (O(N))")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Tensor power n")
    ax.set_ylabel("Peak floats in memory (Analytical)")
    ax.set_title("Memory Usage Estimation (Log Scale)")
    
    # Add dashed lines where methods failed
    if failed_n["brute"] is not None and failed_n["brute"] in ns:
        idx = ns.index(failed_n["brute"])
        ax.axvline(x=idx, color="#F44336", linestyle="--", alpha=0.7)
        ax.text(idx, ax.get_ylim()[0] * 2, 'Brute Limit/OOM', color="#F44336", rotation=90, va='bottom', ha='right')
        
    if failed_n["slq"] is not None and failed_n["slq"] in ns:
        idx = ns.index(failed_n["slq"])
        ax.axvline(x=idx, color="#2196F3", linestyle="--", alpha=0.7)
        ax.text(idx, ax.get_ylim()[0] * 2, 'SLQ OOM', color="#2196F3", rotation=90, va='bottom', ha='right')
        
    if failed_n["gkb"] is not None and failed_n["gkb"] in ns:
        idx = ns.index(failed_n["gkb"])
        ax.axvline(x=idx, color="#FF9800", linestyle="--", alpha=0.7)
        ax.text(idx, ax.get_ylim()[0] * 2, 'GKB OOM', color="#FF9800", rotation=90, va='bottom', ha='right')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{save_dir}/memory_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_combined_accuracy(results, rho, p_values, n_kron, save_dir="."):
    """
    Combined plot: all p values side by side, SLQ vs GKB.
    """
    N = rho.shape[0] ** n_kron
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Accuracy Comparison: SLQ vs GKB  |  rho^(x){n_kron} ({N}x{N})",
        fontsize=14, fontweight="bold"
    )

    n_p = len([pp for pp in p_values if pp != np.inf])
    gs = GridSpec(2, max(n_p, 1), figure=fig)

    col = 0
    for p in p_values:
        if p == np.inf:
            continue  # skip inf for accuracy plots (svds is exact)
        p_label = str(int(p))

        # Error vs Krylov steps
        ax = fig.add_subplot(gs[0, col])
        key_slq = (n_kron, p, "slq_vs_m")
        key_gkb = (n_kron, p, "gkb_vs_l")
        if key_slq in results:
            r = results[key_slq]
            ax.semilogy(r["m"], r["error"], "o-", color="#2196F3",
                       linewidth=2, markersize=5, label="SLQ")
        if key_gkb in results:
            r = results[key_gkb]
            ax.semilogy(r["l"], r["error"], "^-", color="#FF9800",
                       linewidth=2, markersize=5, label="GKB")
        ax.set_xlabel("Krylov steps")
        ax.set_ylabel("Relative error")
        ax.set_title(f"p = {p_label}: error vs steps")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error vs samples
        ax = fig.add_subplot(gs[1, col])
        key_slq = (n_kron, p, "slq_vs_s")
        key_gkb = (n_kron, p, "gkb_vs_K")
        if key_slq in results:
            r = results[key_slq]
            ax.semilogy(r["s"], r["error"], "o-", color="#2196F3",
                       linewidth=2, markersize=5, label="SLQ")
        if key_gkb in results:
            r = results[key_gkb]
            ax.semilogy(r["K"], r["error"], "^-", color="#FF9800",
                       linewidth=2, markersize=5, label="GKB")
        ax.set_xlabel("Hutchinson samples")
        ax.set_ylabel("Relative error")
        ax.set_title(f"p = {p_label}: error vs samples")
        ax.legend()
        ax.grid(True, alpha=0.3)

        col += 1

    plt.tight_layout()
    fname = f"{save_dir}/combined_accuracy_n{n_kron}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ============================================================
# 9. Main Experiment
# ============================================================

def main():
    np.random.seed(42)
    save_dir = "."
    
    print("=" * 60)
    print("  SCHATTEN p-NORM ESTIMATION VIA KRYLOV METHODS")
    print("  Using 3x3 complex random density matrices (rho^(x)n)")
    print("=" * 60)

    # Generate a random 3x3 density matrix
    rho = random_density_matrix(d=3)
    print(f"\nDensity matrix rho (3x3):")
    print(f"  Hermitian check: ||rho - rho^dag||_F = {np.linalg.norm(rho - rho.conj().T):.2e}")
    print(f"  Trace: {np.trace(rho).real:.10f}")
    print(f"  Eigenvalues: {np.linalg.eigvalsh(rho)}")
    eigvals = np.linalg.eigvalsh(rho)
    print(f"  PSD check: min eigenvalue = {eigvals.min():.2e}")

    # Parameters
    p_values = [1, 2, np.inf]

    # ── Accuracy experiments ──
    print("\n" + "=" * 60)
    print("  ACCURACY EXPERIMENTS")
    print("=" * 60)

    # Use n_kron=4 (N=81) and n_kron=5 (N=243) for accuracy (small enough for brute force)
    accuracy_n_krons = [4, 5]
    m_values = [3, 5, 8, 10, 15, 20, 30, 40]
    s_values = [1, 2, 5, 10, 20, 50]

    acc_results = run_accuracy_experiment(
        rho,
        n_kron_values=accuracy_n_krons,
        p_values=p_values,
        m_values=m_values,
        s_values=s_values,
        n_trials=5,
    )

    print("\n\nGenerating accuracy plots...")
    plot_accuracy_results(acc_results, rho, save_dir=save_dir)
    for n_kron in accuracy_n_krons:
        plot_combined_accuracy(acc_results, rho, p_values, n_kron, save_dir=save_dir)

    # ── Speed experiments ──
    print("\n" + "=" * 60)
    print("  SPEED EXPERIMENTS")
    print("=" * 60)

    # Test from small to large: n=2..8 => N = 9, 27, 81, 243, 729, 2187, 6561
    speed_results, failed_n = run_speed_experiment(
        rho,
        p_values=p_values,
        m_fixed=20,
        s_fixed=10,
        l_fixed=20,
        K_fixed=10,
        n_trials=3,
    )

    print("\nGenerating speed and memory plots...")
    plot_speed_results(speed_results, failed_n, save_dir=save_dir)
    plot_memory_results(speed_results, failed_n, save_dir=save_dir)

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'n':>3} {'N':>6} {'p':>4} | {'Brute(s)':>10} {'SLQ(s)':>10} {'GKB(s)':>10} | {'Speedup SLQ':>12} {'Speedup GKB':>12}")
    print("-" * 85)
    for i in range(len(speed_results["n_kron"])):
        n = speed_results["n_kron"][i]
        N = speed_results["N"][i]
        p = speed_results["p"][i]
        tb = speed_results["t_brute"][i]
        ts = speed_results["t_slq"][i]
        tg = speed_results["t_gkb"][i]
        p_label = "inf" if p == np.inf else str(int(p))
        tb_str = f"{tb:.4f}" if not np.isnan(tb) else "N/A"
        su_slq = f"{tb/ts:.1f}x" if not np.isnan(tb) else "N/A"
        su_gkb = f"{tb/tg:.1f}x" if not np.isnan(tb) else "N/A"
        print(f"{n:>3} {N:>6} {p_label:>4} | {tb_str:>10} {ts:>10.4f} {tg:>10.4f} | {su_slq:>12} {su_gkb:>12}")

    print("\nDone! Check the generated PNG files for plots.")


if __name__ == "__main__":
    main()
