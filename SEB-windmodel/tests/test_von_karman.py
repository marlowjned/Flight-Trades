"""Tests for Von Kármán turbulence generator."""

import numpy as np
import pytest
from scipy.signal import welch

from seb_wind_model.von_karman import sigma_u, scale_length, VonKarmanFilter


class TestSigmaU:

    def test_sigma_continuity_at_305m(self):
        """sigma_u should be continuous across the 305 m boundary."""
        below = sigma_u(304.9)
        above = sigma_u(305.1)
        assert abs(below - above) < 1e-3, (
            f"Discontinuity at 305 m: sigma({304.9}) = {below:.4f}, "
            f"sigma({305.1}) = {above:.4f}"
        )

    def test_sigma_at_zero(self):
        """sigma_u at z=0 should be finite and positive."""
        s = sigma_u(0.0)
        assert s > 0
        assert np.isfinite(s)

    def test_sigma_above_surface(self):
        """sigma_u values above 305 m should match MILSPEC table."""
        from seb_wind_model.von_karman import MILSPEC_SIGMA_TABLE
        for row in MILSPEC_SIGMA_TABLE[1:]:
            z_mid = (row[0] + row[1]) / 2
            assert abs(sigma_u(z_mid) - row[2]) < 1e-9


class TestScaleLength:

    def test_scale_length_boundary(self):
        """scale_length should not jump by more than 10% at 305 m."""
        below = scale_length(304.9)
        above = scale_length(305.1)
        assert abs(below - above) / above < 0.10, (
            f"scale_length discontinuity: L({304.9}) = {below:.1f}, "
            f"L({305.1}) = {above:.1f}"
        )

    def test_scale_length_minimum(self):
        """scale_length should never return less than 1.0 m."""
        assert scale_length(0.0) >= 1.0
        assert scale_length(1.0) >= 1.0

    def test_scale_length_free_atm(self):
        """Above 305 m, scale_length should return L_FREE_ATM = 533 m."""
        from seb_wind_model.von_karman import L_FREE_ATM
        assert scale_length(1000.0) == L_FREE_ATM
        assert scale_length(5000.0) == L_FREE_ATM


class TestVonKarmanFilter:

    def test_filter_output_psd(self):
        """
        PSD of u_t output should have spectral slope between -1.5 and -1.8
        in the inertial subrange (1–20 Hz), consistent with the -5/3 law.
        """
        dt = 0.01
        V = 100.0
        z = 1000.0
        n_steps = 10000

        filt = VonKarmanFilter(dt=dt, airspeed=V, z_m=z)
        rng = np.random.default_rng(0)

        u_ts = np.array([filt.step(rng)[0] for _ in range(n_steps)])

        freqs, psd = welch(u_ts, fs=1.0 / dt, nperseg=512)

        # Restrict to inertial subrange 1–20 Hz
        mask = (freqs >= 1.0) & (freqs <= 20.0)
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-30)

        slope, _ = np.polyfit(log_f, log_p, 1)
        assert -1.8 <= slope <= -1.5, (
            f"Spectral slope {slope:.3f} outside expected range [-1.8, -1.5]"
        )

    def test_zero_airspeed_guard(self):
        """VonKarmanFilter should not raise at very low airspeed."""
        filt = VonKarmanFilter(dt=0.01, airspeed=0.01, z_m=100.0)
        rng = np.random.default_rng(0)
        u, v, w = filt.step(rng)
        assert np.isfinite(u)
        assert np.isfinite(v)
        assert np.isfinite(w)

    def test_output_finite(self):
        """Filter output must be finite for 1000 steps."""
        filt = VonKarmanFilter(dt=0.05, airspeed=50.0, z_m=500.0)
        rng = np.random.default_rng(1)
        for _ in range(1000):
            u, v, w = filt.step(rng)
            assert np.isfinite(u) and np.isfinite(v) and np.isfinite(w)

    def test_update_altitude_no_state_reset(self):
        """
        update_altitude should not zero the filter state — output should be
        continuous (not jump to near-zero) after an altitude update.
        """
        dt = 0.05
        filt = VonKarmanFilter(dt=dt, airspeed=100.0, z_m=500.0)
        rng = np.random.default_rng(2)

        # Warm up
        for _ in range(200):
            filt.step(rng)

        u_before = filt.step(rng)[0]
        filt.update_altitude(600.0, 105.0)
        u_after = filt.step(rng)[0]

        # After a small altitude change, output should not collapse to zero
        # (if state were reset, it would be near zero — much smaller than warm u)
        assert abs(u_after) > 0.001 * abs(u_before) or abs(u_before) < 0.01
