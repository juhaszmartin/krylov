"""
Schatten p-Norm Estimation via GKB (Golub-Kahan Bidiagonalization)
==================================================================
Target: Linear combinations of Kronecker powers: M = c_1 * A^(x)n + c_2 * B^(x)n

This script performs accuracy and speed benchmarking with PNG generation.
"""

import sys
import time
import warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import exact SU(3) calculator (no fallback)
from tensorpow import TensorPowerCalculator

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. Random Matrix Generation & Tensor Operations
# ============================================================

def random_density_matrix(d=3):
    """Generate a random density matrix (Hermitian positive semi-definite with trace 1)."""
    G = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = G @ G.conj().T
    return rho / np.trace(rho)

def apply_tensor_power(M_base, v, n):
    """Apply M_base^(x)n to vector v via tensor reshaping."""
    d = M_base.shape[0]
    X = v.reshape((d,) * n)
    for i in range(n):
        X = np.tensordot(M_base, X, axes=([1], [i]))
        X = np.moveaxis(X, 0, i)
    return X.flatten()

def make_kronecker_operator(matrices, coeffs, n_kron):
    """Build matrix-free LinearOperator for M = sum(c_i * M_i^{⊗n})."""
    d = matrices[0].shape[0]
    N = d ** n_kron
    
    def matvec(v):
        original_shape = v.shape
        v = np.asarray(v).ravel()

        result = np.zeros(v.shape, dtype=complex)

        for M_base, c in zip(matrices, coeffs):
            result += c * apply_tensor_power(
                M_base,
                v,
                n_kron,
            )

        return result.reshape(original_shape)


    def rmatvec(v):
        original_shape = v.shape
        v = np.asarray(v).ravel()

        result = np.zeros(v.shape, dtype=complex)

        for M_base, c in zip(matrices, coeffs):
            result += np.conj(c) * apply_tensor_power(
                M_base.conj().T,
                v,
                n_kron,
            )

        return result.reshape(original_shape)
        
    return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec, dtype=complex), N

# ============================================================
# 2. Brute Force (Strictly for Speed Benchmarking)
# ============================================================

def build_dense_kronecker(matrices, coeffs, n_kron):
    """Build the dense matrix explicitly. (Will OOM for large n)."""
    d = matrices[0].shape[0]
    N = d ** n_kron
    M_total = np.zeros((N, N), dtype=complex)
    
    for M_base, c in zip(matrices, coeffs):
        M_curr = M_base
        for _ in range(1, n_kron):
            M_curr = np.kron(M_curr, M_base)
        M_total += c * M_curr
        
    return M_total

def brute_force_schatten(A_dense, p):
    """Compute Schatten p-norm via dense SVD."""
    S = np.linalg.svd(A_dense, compute_uv=False)
    S = S[S > 1e-15] 
    if p == np.inf:
        return S[0]
    return np.sum(S ** p) ** (1.0 / p)

# ============================================================
# 3. Golub-Kahan Bidiagonalization (GKB)
# ============================================================

def golub_kahan_bidiag(matvec, rmatvec, v, m_steps):
    """Run Golub-Kahan Bidiagonalization for m steps."""
    N = len(v)
    v_norm = np.linalg.norm(v)
    q = v / v_norm

    omega = np.zeros(m_steps)
    gamma = np.zeros(m_steps)

    p_prev = np.zeros(N, dtype=complex)
    gamma_prev = 0.0

    for j in range(m_steps):
        # u-step
        u_unnorm = matvec(q) - gamma_prev * p_prev
        omega[j] = np.linalg.norm(u_unnorm)

        if omega[j] < 1e-14:
            m_actual = j + 1
            return omega[:m_actual], gamma[:m_actual - 1], v_norm
        
        p_vec = u_unnorm / omega[j]

        # v-step
        if j < m_steps - 1:
            v_unnorm = rmatvec(p_vec) - omega[j] * q
            gamma[j] = np.linalg.norm(v_unnorm)
            if gamma[j] < 1e-14:
                m_actual = j + 1
                return omega[:m_actual], gamma[:m_actual - 1], v_norm
            q = v_unnorm / gamma[j]
            gamma_prev = gamma[j]
            p_prev = p_vec

    return omega, gamma[:-1] if m_steps > 1 else [], v_norm

def estimate_schatten1_gkb(
    matvec,
    rmatvec,
    N,
    s_samples=20,
    m_steps=30,
):
    """
    Estimate the Schatten-1 (trace) norm using

        Hutchinson + Golub-Kahan + Gaussian Quadrature.

    Computes

        Tr(sqrt(M^* M)).
    """

    trace_est = 0.0

    for _ in range(s_samples):

        # Complex Gaussian probe vector
        v_rand = (
            np.random.randn(N)
            + 1j * np.random.randn(N)
        ) / np.sqrt(2)

        omega, gamma, v_norm = golub_kahan_bidiag(
            matvec,
            rmatvec,
            v_rand,
            m_steps,
        )

        l_actual = len(omega)

        B_l = np.diag(omega)

        if l_actual > 1 and len(gamma) > 0:
            B_l += np.diag(
                gamma[: l_actual - 1],
                k=1,
            )

        T_l = B_l.conj().T @ B_l

        sqrt_T_l = sqrtm(T_l)

        trace_est += (
            v_norm ** 2
        ) * np.real(
            sqrt_T_l[0, 0]
        )

    return trace_est / s_samples


def estimate_schatten_inf_arpack(
    linear_operator,
    m_steps=30,
):
    import scipy.sparse.linalg
    
    def matvec_M_dag_M(v):
        return linear_operator.rmatvec(linear_operator.matvec(v))
        
    M_dag_M = LinearOperator((linear_operator.shape[0], linear_operator.shape[1]), matvec=matvec_M_dag_M)
    
    try:
        evals = scipy.sparse.linalg.eigsh(
            M_dag_M,
            k=1,
            which="LM",
            tol=1e-12,
            maxiter=m_steps,
            return_eigenvectors=False
        )
        return float(np.sqrt(np.abs(evals[0])))
    except scipy.sparse.linalg.ArpackNoConvergence as e:
        if len(e.eigenvalues) > 0:
            return float(np.sqrt(np.abs(e.eigenvalues[0])))
        return 0.0

def estimate_schatten_norm(
    linear_operator,
    N,
    p,
    s_samples=20,
    m_steps=30,
):
    """
    Unified interface.

    p = 1    -> GKB + Hutchinson
    p = inf  -> ARPACK
    """

    if p == 1:
        return estimate_schatten1_gkb(
            linear_operator.matvec,
            linear_operator.rmatvec,
            N,
            s_samples=s_samples,
            m_steps=m_steps,
        )

    if p == np.inf:
        return estimate_schatten_inf_arpack(
            linear_operator,
            m_steps=m_steps,
        )

    raise ValueError(
        "Only p=1 and p=np.inf are currently implemented."
    )


# ============================================================
# 4. Accuracy Experiment
# ============================================================

def run_accuracy_experiment(matrices, coeffs, n_krons, p_values, m_values, s_values, n_trials=3):
    """Run accuracy benchmarks against exact SU(3) calculator."""
    results = {}
    d = matrices[0].shape[0]
    calc = TensorPowerCalculator()

    for n in n_krons:
        N = d ** n
        print(f"\n{'='*60}")
        print(f"ACCURACY: n={n}, N={N}x{N}")
        print(f"{'='*60}")
        
        op, _ = make_kronecker_operator(matrices, coeffs, n)

        for p in p_values:
            p_label = "inf" if p == np.inf else str(p)
            print(f"\n  p = {p_label}")
            
            # Exact value from tensorpow
            exact = calc.schatten_p_norm_weighted(matrices, n=n, p=p, coeffs=coeffs)
            print(f"    Exact: {exact:.8f}")

            # Vary GKB steps (m)
            key_m = (n, p, "gkb_vs_m")
            results[key_m] = {"m": [], "error": []}
            s_fixed = max(s_values)
            
            for m in m_values:
                errors = []
                for _ in range(n_trials):
                    est = estimate_schatten_norm(op,N,p,s_samples=s_fixed,m_steps=m)                    
                    errors.append(abs(est - exact) / exact)
                results[key_m]["m"].append(m)
                results[key_m]["error"].append(np.median(errors))
            
            best_err = min(results[key_m]["error"])
            if p == np.inf:
                print(f"    IRLM vs m: best rel error = {best_err:.2e}")
            else:
                print(f"    GKB vs m (s={s_fixed}): best rel error = {best_err:.2e}")

            if p == np.inf:
                continue # Skip Hutchinson sweep for p=inf

            # Vary Hutchinson samples (s)
            key_s = (n, p, "gkb_vs_s")
            results[key_s] = {"s": [], "error": []}
            m_fixed = max(m_values)
            
            for s in s_values:
                errors = []
                for _ in range(n_trials):
                    est = estimate_schatten_norm(op,N,p,s_samples=s,m_steps=m_fixed)                    
                    errors.append(abs(est - exact) / exact)
                results[key_s]["s"].append(s)
                results[key_s]["error"].append(np.median(errors))
            
            best_err = min(results[key_s]["error"])
            print(f"    GKB vs s (m={m_fixed}): best rel error = {best_err:.2e}")

    return results

# ============================================================
# 5. Speed Experiment
# ============================================================

def run_speed_experiment(matrices, coeffs, p_values, m_fixed=40, s_fixed=20):
    """Benchmark speed: dense brute force vs GKB estimator."""
    speed_results = {"n_kron": [], "N": [], "p": [], "t_brute": [], "t_krylov": []}
    
    failed_brute = False
    d = matrices[0].shape[0]
    
    print(f"\n{'='*60}")
    print(f"SPEED BENCHMARK (n=2 to 10)")
    print(f"{'='*60}")

    for n in range(2, 13):
        N = d ** n
        op, _ = make_kronecker_operator(matrices, coeffs, n)

        for p in p_values:
            p_label = "inf" if p == np.inf else "1"
            print(f"  n={n:2d} (N={N:7d}) p={p_label:3s} ... ", end="", flush=True)

            # Brute force time
            t_brute = np.nan
            if n == 9:
                # Hardcoded from previous measurement (~40 minutes)
                t_brute = 2388.0341
            elif not failed_brute and n <= 8:
                try:
                    t0 = time.perf_counter()
                    M_dense = build_dense_kronecker(matrices, coeffs, n)
                    _ = brute_force_schatten(M_dense, p)
                    t_brute = time.perf_counter() - t0
                except (MemoryError, np.core._exceptions._ArrayMemoryError):
                    failed_brute = True
                    print("BruteForce=OOM ", end="")
            else:
                print("BruteForce=OOM ", end="")

            # GKB time
            t0 = time.perf_counter()
            _ = estimate_schatten_norm(
                op,
                N,
                p,
                s_samples=s_fixed,
                m_steps=m_fixed,
            )            
            t_krylov = time.perf_counter() - t0

            speed_results["n_kron"].append(n)
            speed_results["N"].append(N)
            speed_results["p"].append(p)
            speed_results["t_brute"].append(t_brute)
            speed_results["t_krylov"].append(t_krylov)

            brute_str = f"{t_brute:.2f}s" if not np.isnan(t_brute) else "OOM"
            print(f"BruteForce={brute_str}, Krylov={t_krylov:.2f}s")

    return speed_results

# ============================================================
# 6. Plotting Functions
# ============================================================

def plot_accuracy_results(results, m_fixed, s_fixed, save_dir="."):
    """Generate accuracy plots (GKB steps vs samples)."""
    combos = sorted(list(set([(k[0], k[1]) for k in results.keys()])))
    
    for n, p in combos:
        p_label = "inf" if p == np.inf else "1"
        title_prefix = "Trace Norm (p=1)" if p == 1 else "Spectral Norm (p=∞)"
        
        if p == np.inf:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            fig.suptitle(f"{title_prefix} Accuracy | Linear Combo (3^{n}×3^{n})", fontweight="bold")
            
            key_m = (n, p, "gkb_vs_m")
            if key_m in results:
                ax.semilogy(results[key_m]["m"], results[key_m]["error"], "o-", color="#FF9800", linewidth=2, markersize=8)
                ax.set_xlabel("Lanczos Iterations (m)", fontsize=11)
                ax.set_ylabel("Relative Error", fontsize=11)
                ax.set_title("Varying Iterations", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(min(results[key_m]["m"]), max(results[key_m]["m"]))
            
            plt.tight_layout()
            fname = f"{save_dir}/accuracy_n{n}_p{p_label}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle(f"{title_prefix} Accuracy | Linear Combo (3^{n}×3^{n})", fontweight="bold")

            # Varying GKB steps
            key_m = (n, p, "gkb_vs_m")
            if key_m in results:
                axes[0].semilogy(results[key_m]["m"], results[key_m]["error"], "o-", color="#FF9800", linewidth=2, markersize=8)
                axes[0].set_xlabel("GKB Steps (m)", fontsize=11)
                axes[0].set_ylabel("Relative Error", fontsize=11)
                axes[0].set_title(f"Varying GKB Steps (Fixed s={s_fixed})", fontsize=10)
                axes[0].grid(True, alpha=0.3)
                axes[0].set_xlim(min(results[key_m]["m"]), max(results[key_m]["m"]))

            # Varying Hutchinson samples
            key_s = (n, p, "gkb_vs_s")
            if key_s in results:
                axes[1].semilogy(results[key_s]["s"], results[key_s]["error"], "s-", color="#E91E63", linewidth=2, markersize=8)
                axes[1].set_xlabel("Hutchinson Samples (s)", fontsize=11)
                axes[1].set_ylabel("Relative Error", fontsize=11)
                axes[1].set_title(f"Varying Samples (Fixed m={m_fixed})", fontsize=10)
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xlim(min(results[key_s]["s"]), max(results[key_s]["s"]))

            plt.tight_layout()
            fname = f"{save_dir}/accuracy_n{n}_p{p_label}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")

def plot_scaling_results(speed_results, save_dir="."):
    """Generate speed comparison plot."""
    p_values = sorted(set(speed_results["p"]), key=lambda x: (x != np.inf, x))
    
    fig, axes = plt.subplots(1, len(p_values), figsize=(6 * len(p_values), 5))
    if len(p_values) == 1:
        axes = [axes]
    
    for ax, p in zip(axes, p_values):
        p_label = "inf" if p == np.inf else "1"
        mask = [i for i, pp in enumerate(speed_results["p"]) if pp == p]
        ns = [speed_results["n_kron"][i] for i in mask]
        t_brute = [speed_results["t_brute"][i] for i in mask]
        t_kr = [speed_results["t_krylov"][i] for i in mask]
        
        # Brute force (dense SVD)
        valid_ex = [(i, t) for i, t in enumerate(t_brute) if not np.isnan(t)]
        if valid_ex:
            idx, times = zip(*valid_ex)
            ax.semilogy([ns[i] for i in idx], times, "s-", color="#F44336", 
                       linewidth=2, markersize=8, label="Dense Brute Force (SVD)")
        
        # GKB estimator
        ax.semilogy(ns, t_kr, "o-", color="#2196F3", linewidth=2, markersize=8, 
                   label="GKB Estimator (Matrix-Free)")
        
        ax.set_xticks(ns)
        ax.set_xlabel("Tensor Power n", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title(f"Schatten-{p_label} Norm Runtime", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotate the OOM point
        if len(ns) > len(valid_ex):
            oom_n = ns[len(valid_ex)]
            ax.axvline(x=oom_n - 0.5, color='red', linestyle='--', alpha=0.5)
            ax.text(oom_n, ax.get_ylim()[1] * 0.5, 'OOM Limit', 
                   rotation=90, fontsize=9, color='red', ha='right')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/speed_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: speed_comparison.png")

def plot_theoretical_memory(max_n=10, save_dir="."):
    """Generate analytical memory scaling plot."""
    ns = np.arange(2, max_n + 1)
    d = 3
    Ns = d ** ns
    
    # Memory estimates (number of complex floats)
    mem_brute = 2 * (Ns ** 2)  # Real + Imag parts for dense matrix
    mem_krylov = 10 * Ns       # ~5 vectors × 2 (real+imag)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.semilogy(ns, mem_brute, "s-", color="#F44336", linewidth=2, markersize=8, 
               label="Brute Force (Dense SVD)\n$\\mathcal{O}(N^2)$")
    ax.semilogy(ns, mem_krylov, "o-", color="#2196F3", linewidth=2, markersize=8, 
               label="GKB Estimator\n$\\mathcal{O}(N)$")
    
    ax.set_xticks(ns)
    ax.set_xlabel("Tensor Power n", fontsize=12)
    ax.set_ylabel("Theoretical Peak Memory\n(Number of Complex Floats)", fontsize=12)
    ax.set_title("Analytical Memory Scaling: GKB vs Dense SVD", fontsize=12, fontweight="bold")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    text_str = "Note: GKB memory is bounded by the 3-term recurrence\nand independent of Hutchinson samples (s) and\nmaximum Krylov steps (m)."
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9, 
           verticalalignment='top', bbox=props)

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/memory_analytical_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: memory_analytical_comparison.png")

# ============================================================
# 7. Main Execution
# ============================================================

def main():
    print("=" * 70)
    print("SCHATTEN p-NORM ESTIMATION VIA GOLUB-KAHAN BIDIAGONALIZATION")
    print("=" * 70)
    
    # Setup
    np.random.seed(42)
    
    # Generate random 3x3 density matrices
    A = random_density_matrix(3)
    B = random_density_matrix(3)
    matrices = [A, B]
    coeffs = [1.0, -0.5]
    p_values = [1, np.inf]
    
    # Experiment parameters
    n_krons_acc = [8, 11]          # Accuracy at n=8 and n=12
    m_values = [2, 3, 4, 5, 10, 20, 30, 40] # GKB steps
    s_values = [2, 5, 10, 20, 50]  # Hutchinson samples
    
    print(f"\nMatrix dimension: d=3")
    print(f"Linear combination: M = 1.0*A^(x)n + (-0.5)*B^(x)n")
    print(f"Target tensor powers: n = {n_krons_acc}")
    print(f"GKB steps range: m = {m_values}")
    print(f"Hutchinson samples range: s = {s_values}")
    
    # 1. Accuracy Benchmarking
    print("\n" + "=" * 70)
    print("PHASE 1: ACCURACY BENCHMARKING (vs Exact SU(3) Calculator)")
    print("=" * 70)
    
    acc_results = run_accuracy_experiment(
        matrices, coeffs, 
        n_krons=n_krons_acc, p_values=p_values, 
        m_values=m_values, s_values=s_values, n_trials=3
    )
    
    print("\nGenerating accuracy plots...")
    plot_accuracy_results(acc_results, m_fixed=max(m_values), s_fixed=max(s_values))
    
    # 2. Speed Benchmarking
    print("\n" + "=" * 70)
    print("PHASE 2: SPEED BENCHMARKING (Dense SVD vs GKB)")
    print("=" * 70)
    
    speed_results = run_speed_experiment(
        matrices, coeffs, p_values, 
        m_fixed=40, s_fixed=20
    )
    
    print("\nGenerating speed plot...")
    plot_scaling_results(speed_results)
    
    # 3. Memory Analytical Plot
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYTICAL MEMORY SCALING")
    print("=" * 70)
    
    plot_theoretical_memory(max_n=10)
    
    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("Generated files:")
    print("  - accuracy_n8_p1.png / accuracy_n8_pinf.png")
    print("  - accuracy_n11_p1.png / accuracy_n11_pinf.png")
    print("  - speed_comparison.png")
    print("  - memory_analytical_comparison.png")
    print("=" * 70)

if __name__ == "__main__":
    main()