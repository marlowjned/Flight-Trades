# eof.py
# EOF decomposition and Monte Carlo ensemble sampling of wind profiles

import numpy as np


class EOFModel:
    """
    Empirical Orthogonal Function decomposition of ERA5 wind profiles.

    Decomposes the joint (u, v) anomaly field into EOF modes, then draws
    physically correlated Monte Carlo perturbation profiles.
    """

    def __init__(
        self,
        u_profiles: np.ndarray,
        v_profiles: np.ndarray,
        alt_grid: np.ndarray,
        n_modes: int = None,
    ):
        """
        Parameters
        ----------
        u_profiles : (N, K) array
            N time samples of u-component wind at K altitude levels (m/s).
        v_profiles : (N, K) array
            N time samples of v-component wind at K altitude levels (m/s).
        alt_grid : (K,) array
            Altitude levels (m AGL).
        n_modes : int, optional
            Number of EOF modes to retain. If None, retains enough to explain
            95% of total variance.
        """
        u_profiles = np.asarray(u_profiles, dtype=np.float64)
        v_profiles = np.asarray(v_profiles, dtype=np.float64)
        N, K = u_profiles.shape

        self.alt_grid = np.asarray(alt_grid, dtype=np.float64)
        self.u_mean = u_profiles.mean(axis=0)  # (K,)
        self.v_mean = v_profiles.mean(axis=0)  # (K,)

        # Joint anomaly matrix
        X = np.hstack([u_profiles - self.u_mean,
                       v_profiles - self.v_mean])  # (N, 2K)

        # Need at least 2 profiles for a meaningful decomposition
        if N < 2:
            self.n_modes = 0
            self.eigenvalues = np.array([])
            self.EOFs = np.empty((0, 2 * K))
            self._cumvar = np.array([1.0])
            self._X = X
            return

        # SVD on the scaled anomaly matrix
        _, S, Vt = np.linalg.svd(X / np.sqrt(N - 1), full_matrices=False)
        eigenvalues = S ** 2          # variance explained by each mode
        EOFs = Vt                     # (min(N, 2K), 2K)

        # Flip sign: ensure max-abs component of each EOF is positive
        for i in range(EOFs.shape[0]):
            if EOFs[i, np.argmax(np.abs(EOFs[i]))] < 0:
                EOFs[i] *= -1.0

        # Select number of modes
        total_var = eigenvalues.sum()
        cumvar = np.cumsum(eigenvalues) / total_var
        if n_modes is None:
            n_modes = int(np.searchsorted(cumvar, 0.95)) + 1

        self.n_modes = n_modes
        self.eigenvalues = eigenvalues[:n_modes]
        self.EOFs = EOFs[:n_modes]      # (n_modes, 2K)
        self._cumvar = cumvar
        self._X = X                    # stored for diagnostics

    def sample(
        self,
        n_draws: int = 1,
        rng: np.random.Generator = None,
        scale: float = 1.0,
    ):
        """
        Draw n_draws random wind profiles from the EOF ensemble.

        Parameters
        ----------
        n_draws : int
            Number of profiles to draw.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        scale : float
            Multiplicative factor on perturbation amplitude (default 1.0).

        Returns
        -------
        u_samples : (n_draws, K) array
        v_samples : (n_draws, K) array
        """
        if rng is None:
            rng = np.random.default_rng()

        K = len(self.alt_grid)

        if self.n_modes == 0:
            # Single-profile case — no ensemble spread, just return the mean
            u_samples = np.tile(self.u_mean, (n_draws, 1))
            v_samples = np.tile(self.v_mean, (n_draws, 1))
            return u_samples, v_samples

        scores = rng.standard_normal((n_draws, self.n_modes))   # (n_draws, n_modes)
        sqrt_eig = np.sqrt(self.eigenvalues)                     # (n_modes,)
        dX = (scores * sqrt_eig) @ self.EOFs                    # (n_draws, 2K)
        dX *= scale

        u_samples = self.u_mean + dX[:, :K]   # (n_draws, K)
        v_samples = self.v_mean + dX[:, K:]   # (n_draws, K)

        return u_samples, v_samples

    def variance_explained(self) -> np.ndarray:
        """Cumulative variance explained fraction for modes 1..n_modes."""
        return self._cumvar[:self.n_modes].copy()

    def plot_eofs(self, n: int = 4):
        """
        Plot the first n EOF spatial patterns (u and v vs altitude).
        Saves to eof_modes.png.
        """
        import matplotlib.pyplot as plt

        n = min(n, self.n_modes)
        K = len(self.alt_grid)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n), sharey=True)
        if n == 1:
            axes = axes[np.newaxis, :]

        cumvar = self.variance_explained()

        for i in range(n):
            eof_u = self.EOFs[i, :K]
            eof_v = self.EOFs[i, K:]
            var_pct = cumvar[i] * 100 if i == 0 else (cumvar[i] - cumvar[i - 1]) * 100

            axes[i, 0].plot(eof_u, self.alt_grid / 1000, color="steelblue")
            axes[i, 0].axvline(0, color="k", linewidth=0.5, linestyle="--")
            axes[i, 0].set_title(f"EOF {i+1} — u  ({var_pct:.1f}% var)")
            axes[i, 0].set_xlabel("EOF coefficient")
            axes[i, 0].set_ylabel("Altitude (km)")

            axes[i, 1].plot(eof_v, self.alt_grid / 1000, color="darkorange")
            axes[i, 1].axvline(0, color="k", linewidth=0.5, linestyle="--")
            axes[i, 1].set_title(f"EOF {i+1} — v  ({var_pct:.1f}% var)")
            axes[i, 1].set_xlabel("EOF coefficient")

        fig.tight_layout()
        fig.savefig("eof_modes.png", dpi=150)
        plt.close(fig)
