"""
Geodesy and Coordinate Transformations.

Implements rigorous WGS84 ellipsoidal transformations between Geodetic (Lat, Lon, Alt),
ECEF (Earth-Centered, Earth-Fixed), and Local ENU (East, North, Up) tangent plane.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


# WGS84 Ellipsoid standard constants
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # Semi-minor axis (~6356752.3142 m)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # First eccentricity squared (~0.00669437999014)


def geodetic_to_ecef(
    lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic coordinates (lat, lon, alt) to ECEF coordinates (X, Y, Z).
    
    Args:
        lat_deg: Latitude in decimal degrees [-90, 90]
        lon_deg: Longitude in decimal degrees [-180, 180]
        alt_m: Ellipsoidal altitude in meters
        
    Returns:
        (X, Y, Z) in meters in ECEF frame.
    """
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    # Prime vertical radius of curvature
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat_rad) ** 2)
    
    x = (n + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1.0 - WGS84_E2) + alt_m) * np.sin(lat_rad)
    
    return x, y, z


def ecef_to_geodetic(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF coordinates (X, Y, Z) back to geodetic coordinates (lat, lon, alt)
    using Bowring's closed-form method (accurate to sub-millimeter).
    
    Args:
        x: ECEF X in meters
        y: ECEF Y in meters
        z: ECEF Z in meters
        
    Returns:
        (lat_deg, lon_deg, alt_m)
    """
    p = np.sqrt(x**2 + y**2)
    e_prime2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_B**2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    
    lat_rad = np.arctan2(
        z + e_prime2 * WGS84_B * np.sin(theta)**3,
        p - WGS84_E2 * WGS84_A * np.cos(theta)**3
    )
    lon_rad = np.arctan2(y, x)
    
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat_rad)**2)
    alt_m = p / np.cos(lat_rad) - n
    
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    
    return lat_deg, lon_deg, alt_m


def ecef_to_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF coordinates to local ENU tangent plane relative to origin (lat0, lon0, alt0).
    
    Args:
        x, y, z: ECEF coordinates in meters
        lat0_deg, lon0_deg, alt0_m: Origin coordinates
        
    Returns:
        (east, north, up) in meters.
    """
    x0, y0, z0 = geodetic_to_ecef(np.array([lat0_deg]), np.array([lon0_deg]), np.array([alt0_m]))
    dx = x - x0[0]
    dy = y - y0[0]
    dz = z - z0[0]
    
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)
    
    # Standard transformation matrix R_ecef_to_enu:
    # [ -sin(lon)          cos(lon)          0       ]
    # [ -sin(lat)*cos(lon) -sin(lat)*sin(lon) cos(lat)]
    # [  cos(lat)*cos(lon)  cos(lat)*sin(lon) sin(lat)]
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return east, north, up


def enu_to_ecef(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert local ENU coordinates back to ECEF coordinates."""
    x0, y0, z0 = geodetic_to_ecef(np.array([lat0_deg]), np.array([lon0_deg]), np.array([alt0_m]))
    
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    
    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)
    
    # Inverse rotation matrix (transpose of orthogonal matrix)
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up
    
    return x0[0] + dx, y0[0] + dy, z0[0] + dz


def geodetic_to_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience direct conversion from Geodetic to Local ENU."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m)


def check_trajectory_continuity(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    time_s: np.ndarray,
    max_speed_mps: float = 60.0,
    max_accel_mps2: float = 15.0,
) -> Dict[str, Any]:
    """Check trajectory continuity, impossible jumps, stationary periods, and bounds.
    
    Args:
        east, north, up: Local metric trajectory in meters.
        time_s: Monotonic timestamps in seconds.
        max_speed_mps: Physical vehicle speed threshold (default 60 m/s = 216 km/h).
        max_accel_mps2: Physical vehicle acceleration threshold.
        
    Returns:
        Dict with metrics and detected jump indices.
    """
    dt = np.diff(time_s)
    # Avoid zero-division if duplicated timestamps exist
    dt = np.where(dt <= 0, 1e-6, dt)
    
    de = np.diff(east)
    dn = np.diff(north)
    du = np.diff(up)
    
    displacement_2d = np.sqrt(de**2 + dn**2)
    displacement_3d = np.sqrt(de**2 + dn**2 + du**2)
    speed_2d = displacement_2d / dt
    
    dv = np.diff(speed_2d)
    dt_mid = 0.5 * (dt[:-1] + dt[1:])
    accel_2d = np.abs(dv / dt_mid)
    
    speed_jumps = np.where(speed_2d > max_speed_mps)[0]
    accel_jumps = np.where(accel_2d > max_accel_mps2)[0]
    
    stationary_mask = speed_2d < 0.5  # < 0.5 m/s (~1.8 km/h)
    stationary_ratio = float(np.mean(stationary_mask))
    
    return {
        "total_path_length_m": float(np.sum(displacement_2d)),
        "max_speed_mps": float(np.max(speed_2d)) if len(speed_2d) > 0 else 0.0,
        "mean_speed_mps": float(np.mean(speed_2d)) if len(speed_2d) > 0 else 0.0,
        "speed_jump_count": int(len(speed_jumps)),
        "speed_jump_indices": speed_jumps.tolist(),
        "accel_jump_count": int(len(accel_jumps)),
        "stationary_ratio": stationary_ratio,
        "east_span_m": float(np.max(east) - np.min(east)),
        "north_span_m": float(np.max(north) - np.min(north)),
        "up_span_m": float(np.max(up) - np.min(up)),
    }
