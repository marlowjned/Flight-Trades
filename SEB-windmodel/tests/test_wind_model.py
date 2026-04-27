"""Tests for the top-level WindModel class."""

import numpy as np
import pytest

from seb_wind_model.eof import EOFModel
from seb_wind_model.wind_model import WindModel


def _make_model(N=100, K=20, dt=0.05, seed=0):
    """Build a small WindModel for testing."""
    rng = np.random.default_rng(seed)
    alt_grid = np.linspace(0, 5000, K)
    u_profiles = rng.standard_normal((N, K)) * 5.0 + 3.0
    v_profiles = rng.standard_normal((N, K)) * 3.0 - 1.0
    eof = EOFModel(u_profiles, v_profiles, alt_grid)
    return WindModel(eof, dt=dt), eof


class TestWindModel:

    def test_realisation_reproducible(self):
        """Two realisations with the same seed must produce identical output."""
        model, _ = _make_model()

        w1 = model.realisation(seed=42)
        w2 = model.realisation(seed=42)

        for i in range(100):
            t = i * model.dt
            z = 10.0 + i * 5.0
            V = 50.0 + i * 0.5
            r1 = w1(t, z, V)
            r2 = w2(t, z, V)
            assert r1 == r2, f"Step {i}: {r1} != {r2}"

    def test_realisation_different_seeds(self):
        """Two realisations with different seeds must not produce identical output."""
        model, _ = _make_model()
        w1 = model.realisation(seed=0)
        w2 = model.realisation(seed=1)

        results_same = True
        for i in range(20):
            t = i * model.dt
            z = 100.0
            V = 100.0
            if w1(t, z, V) != w2(t, z, V):
                results_same = False
                break

        assert not results_same, "Different seeds produced identical realisations"

    def test_ensemble_length(self):
        """ensemble(n=20) should return exactly 20 callables."""
        model, _ = _make_model()
        members = model.ensemble(n=20)
        assert len(members) == 20

    def test_ensemble_independence(self):
        """Ensemble members must be independent (different seeds)."""
        model, _ = _make_model()
        members = model.ensemble(n=5)

        # All members should produce different output at the same call
        outputs = [m(0.0, 100.0, 50.0) for m in members]
        unique_u = len(set(o[0] for o in outputs))
        assert unique_u > 1, "All ensemble members produced identical u output"

    def test_base_wind_mean(self):
        """
        Mean of base_wind(z) across 1000 ensemble members should match the
        EOF mean to within 2 standard errors.
        """
        N, K = 300, 15
        rng = np.random.default_rng(99)
        alt_grid = np.linspace(0, 3000, K)
        u_profiles = rng.standard_normal((N, K)) * 4.0 + 6.0
        v_profiles = rng.standard_normal((N, K)) * 2.0 - 2.0
        eof = EOFModel(u_profiles, v_profiles, alt_grid)
        model = WindModel(eof, dt=0.05)

        n_members = 1000
        members = model.ensemble(n=n_members)

        test_altitudes = [500.0, 1000.0, 2000.0]
        for z in test_altitudes:
            u_samples = np.array([m.base_wind(z)[0] for m in members])
            v_samples = np.array([m.base_wind(z)[1] for m in members])

            # Compare sample mean to EOF mean (interpolated)
            from scipy.interpolate import PchipInterpolator
            u_mean_at_z = float(PchipInterpolator(alt_grid, eof.u_mean)(z))
            v_mean_at_z = float(PchipInterpolator(alt_grid, eof.v_mean)(z))

            u_se = u_samples.std() / np.sqrt(n_members)
            v_se = v_samples.std() / np.sqrt(n_members)

            assert abs(u_samples.mean() - u_mean_at_z) < 2 * u_se, (
                f"u mean mismatch at z={z}: sample={u_samples.mean():.4f}, "
                f"eof_mean={u_mean_at_z:.4f}, 2*SE={2*u_se:.4f}"
            )
            assert abs(v_samples.mean() - v_mean_at_z) < 2 * v_se, (
                f"v mean mismatch at z={z}: sample={v_samples.mean():.4f}, "
                f"eof_mean={v_mean_at_z:.4f}, 2*SE={2*v_se:.4f}"
            )

    def test_wind_returns_floats(self):
        """wind() should return plain Python floats, not numpy scalars."""
        model, _ = _make_model()
        w = model.realisation(seed=0)
        u, v, ww = w(0.0, 500.0, 100.0)
        assert isinstance(u, float), f"u is {type(u)}, expected float"
        assert isinstance(v, float), f"v is {type(v)}, expected float"
        assert isinstance(ww, float), f"w is {type(ww)}, expected float"

    def test_scale_factor(self):
        """
        Ensemble with scale=0.0 should return only the mean wind (no perturbation).
        """
        N, K = 100, 10
        rng = np.random.default_rng(5)
        alt_grid = np.linspace(0, 2000, K)
        u_profiles = rng.standard_normal((N, K)) * 3.0 + 5.0
        v_profiles = rng.standard_normal((N, K)) * 2.0
        eof = EOFModel(u_profiles, v_profiles, alt_grid)
        model_full = WindModel(eof, dt=0.05, scale=1.0)
        model_zero = WindModel(eof, dt=0.05, scale=0.0)

        w_full = model_full.realisation(seed=0)
        w_zero = model_zero.realisation(seed=0)

        # With scale=0, base_wind should equal eof mean exactly
        from scipy.interpolate import PchipInterpolator
        u_mean_interp = PchipInterpolator(alt_grid, eof.u_mean)
        z_test = 1000.0
        u_base_zero, _ = w_zero.base_wind(z_test)
        assert abs(u_base_zero - float(u_mean_interp(z_test))) < 1e-10
