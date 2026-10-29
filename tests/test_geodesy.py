"""Unit tests for Geodesy and Coordinate Transformations."""

import numpy as np
import pytest
from src.geodesy import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
    geodetic_to_enu,
    check_trajectory_continuity,
)


def test_origin_maps_to_zero_enu():
    """Origin coordinate (lat0, lon0, alt0) must map exactly to (0, 0, 0) in Local ENU."""
    lat0, lon0, alt0 = 52.40166, -1.50529, 147.5
    e, n, u = geodetic_to_enu(
        np.array([lat0]), np.array([lon0]), np.array([alt0]),
        lat0, lon0, alt0
    )
    assert np.isclose(e[0], 0.0, atol=1e-6)
    assert np.isclose(n[0], 0.0, atol=1e-6)
    assert np.isclose(u[0], 0.0, atol=1e-6)


def test_roundtrip_geodetic_ecef():
    """Geodetic -> ECEF -> Geodetic must roundtrip to sub-millimeter precision."""
    lats = np.array([52.40166, 0.0, -33.8688, 45.0])
    lons = np.array([-1.50529, 0.0, 151.2093, 9.0])
    alts = np.array([147.5, 0.0, 50.0, 1200.0])
    
    x, y, z = geodetic_to_ecef(lats, lons, alts)
    lat_rt, lon_rt, alt_rt = ecef_to_geodetic(x, y, z)
    
    np.testing.assert_allclose(lats, lat_rt, atol=1e-8)
    np.testing.assert_allclose(lons, lon_rt, atol=1e-8)
    np.testing.assert_allclose(alts, alt_rt, atol=1e-4)


def test_roundtrip_enu_ecef():
    """ENU -> ECEF -> ENU must roundtrip exactly."""
    lat0, lon0, alt0 = 52.40166, -1.50529, 147.5
    east = np.array([100.0, -500.0, 1250.0])
    north = np.array([200.0, -300.0, 800.0])
    up = np.array([5.0, -10.0, 25.0])
    
    x, y, z = enu_to_ecef(east, north, up, lat0, lon0, alt0)
    e_rt, n_rt, u_rt = ecef_to_enu(x, y, z, lat0, lon0, alt0)
    
    np.testing.assert_allclose(east, e_rt, atol=1e-6)
    np.testing.assert_allclose(north, n_rt, atol=1e-6)
    np.testing.assert_allclose(up, u_rt, atol=1e-6)


def test_trajectory_continuity_detector():
    """Continuity checker must catch impossible velocity jumps."""
    time_s = np.array([0.0, 1.0, 2.0, 3.0])
    # Step from t=1 to t=2 has impossible jump of 1000m in 1s (1000 m/s >> 60 m/s threshold)
    east = np.array([0.0, 10.0, 1010.0, 1020.0])
    north = np.array([0.0, 0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 0.0, 0.0])
    
    metrics = check_trajectory_continuity(east, north, up, time_s, max_speed_mps=60.0)
    assert metrics["speed_jump_count"] >= 1
    assert 1 in metrics["speed_jump_indices"]
