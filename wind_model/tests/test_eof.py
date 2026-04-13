"""Tests for EOF decomposition and ensemble sampling."""

import numpy as np
import pytest

from wind_model.eof import EOFModel


def _synthetic_data(N=200, K=40, n_true_modes=3, seed=0):
    """Generate synthetic wind profiles with known structure."""
    rng = np.random.default_rng(seed)
    alt_grid = np.linspace(0, 10000, K)

    u_mean = np.sin(np.linspace(0, np.pi, K)) * 5.0
    v_mean = np.cos(np.linspace(0, np.pi, K)) * 3.0

    # True EOF patterns (sinusoidal modes)
    true_eofs = np.array([
        np.sin(np.linspace(0, (i + 1) * np.pi, K))
        for i in range(n_true_modes)
    ])
    amplitudes = np.array([3.0, 2.0, 1.0])[:n_true_modes]

    scores = rng.standard_normal((N, n_true_modes)) * amplitudes
    anomaly = scores @ true_eofs

    u_profiles = u_mean + anomaly
    v_profiles = v_mean + rng.standard_normal((N, K)) * 0.1  # tiny noise

    return u_profiles, v_profiles, alt_grid, u_mean, v_mean


class TestEOFModel:

    def test_mean_recovery(self):
        """Fitted u_mean and v_mean should match the sample mean exactly."""
        u, v, alt, u_mean_true, v_mean_true = _synthetic_data()
        model = EOFModel(u, v, alt)
        np.testing.assert_allclose(model.u_mean, u.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(model.v_mean, v.mean(axis=0), atol=1e-10)

    def test_variance_explained_threshold(self):
        """With n_modes=None, cumulative variance explained must reach 0.95."""
        u, v, alt, _, _ = _synthetic_data()
        model = EOFModel(u, v, alt)
        cumvar = model.variance_explained()
        assert cumvar[-1] >= 0.95, (
            f"Cumulative variance only {cumvar[-1]:.4f} < 0.95"
        )

    def test_sample_shape(self):
        """sample(n_draws=100) should return arrays of shape (100, K)."""
        u, v, alt, _, _ = _synthetic_data(K=30)
        model = EOFModel(u, v, alt)
        rng = np.random.default_rng(1)
        u_s, v_s = model.sample(n_draws=100, rng=rng)
        assert u_s.shape == (100, 30)
        assert v_s.shape == (100, 30)

    def test_sample_statistics(self):
        """
        Draw many samples; sample covariance of u-anomalies should approximate
        the theoretical EOF covariance to within rtol=0.10.
        """
        u, v, alt, _, _ = _synthetic_data(N=500, K=20)
        model = EOFModel(u, v, alt)
        rng = np.random.default_rng(42)
        n_draws = 5000
        u_s, _ = model.sample(n_draws=n_draws, rng=rng)
        u_anom = u_s - model.u_mean

        # Sample covariance
        sample_cov = (u_anom.T @ u_anom) / (n_draws - 1)  # (K, K)

        # Theoretical covariance from retained modes
        K = len(alt)
        eofs_u = model.EOFs[:, :K]  # u-half of EOFs
        theory_cov = (eofs_u.T * model.eigenvalues) @ eofs_u

        # Check Frobenius-norm relative difference
        rel_diff = np.linalg.norm(sample_cov - theory_cov) / np.linalg.norm(theory_cov)
        assert rel_diff < 0.10, f"Sample covariance deviates by {rel_diff:.4f} > 0.10"

    def test_scale_parameter(self):
        """Samples with scale=0.5 should have ~half the std of scale=1.0."""
        u, v, alt, _, _ = _synthetic_data(N=300, K=20)
        model = EOFModel(u, v, alt)

        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)  # same seed

        u_full, _ = model.sample(n_draws=2000, rng=rng1, scale=1.0)
        u_half, _ = model.sample(n_draws=2000, rng=rng2, scale=0.5)

        std_full = (u_full - model.u_mean).std()
        std_half = (u_half - model.u_mean).std()

        ratio = std_half / std_full
        assert 0.45 < ratio < 0.55, (
            f"scale=0.5 std ratio is {ratio:.3f}, expected ~0.5"
        )
