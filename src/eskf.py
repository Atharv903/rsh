"""
Error-State Kalman Filter (ESKF) for Vehicle Dead Reckoning with Smartphone IMU.

State Vector (6D):
- p_e: East position (m)
- p_n: North position (m)
- v_e: East velocity (m/s)
- v_n: North velocity (m/s)
- psi: Vehicle geographic azimuth (rad, clockwise from North: 0 = North, pi/2 = East)
- b_g: Gyroscope pitch (yaw-rate) bias (rad/s)

Error State Vector:
delta_x = [delta_p_e, delta_p_n, delta_v_e, delta_v_n, delta_psi, delta_b_g]^T

Nominal Kinematics:
- dpsi/dt = -(gyro_pitch - b_g)
- dp_e/dt = v_e
- dp_n/dt = v_n

Supported Measurement Updates:
1. Learned forward speed: z_v = v_pred (with heteroscedastic uncertainty)
   h(x) = v_e * sin(psi) + v_n * cos(psi)
2. Non-Holonomic Constraint (NHC): z_nhc = 0 (lateral velocity in vehicle frame is ~0)
   h(x) = v_e * cos(psi) - v_n * sin(psi)
3. Zero-Velocity Update (ZUPT): z_zupt = [0, 0]^T during verified vehicle stops
4. Zero-Angular-Rate Update (ZARU): z_zaru = 0 during verified vehicle stops
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

@dataclass
class ESKFState:
    p_e: float
    p_n: float
    v_e: float
    v_n: float
    psi: float  # radians, [0, 2*pi)
    b_g: float  # rad/s
    P: np.ndarray  # 6x6 covariance matrix

    def to_dict(self) -> Dict[str, float]:
        return {
            "p_e": self.p_e,
            "p_n": self.p_n,
            "v_e": self.v_e,
            "v_n": self.v_n,
            "psi_deg": float(np.rad2deg(self.psi)),
            "b_g_rads": self.b_g,
            "speed_fwd_mps": float(self.v_e * np.sin(self.psi) + self.v_n * np.cos(self.psi)),
            "speed_lat_mps": float(self.v_e * np.cos(self.psi) - self.v_n * np.sin(self.psi)),
            "sigma_pos_m": float(np.sqrt(self.P[0, 0] + self.P[1, 1])),
            "sigma_vel_mps": float(np.sqrt(self.P[2, 2] + self.P[3, 3])),
            "sigma_psi_deg": float(np.rad2deg(np.sqrt(max(0.0, self.P[4, 4])))),
            "sigma_bg_rads": float(np.sqrt(max(0.0, self.P[5, 5]))),
        }

class VehicleESKF:
    def __init__(
        self,
        init_p_e: float,
        init_p_n: float,
        init_speed_mps: float,
        init_heading_rad: float,
        init_gyro_bias_rads: float = 0.0,
        sigma_pos_init: float = 1.0,
        sigma_vel_init: float = 0.5,
        sigma_psi_init: float = np.deg2rad(2.0),
        sigma_bg_init: float = 0.005,
        # Process noise spectral densities
        q_pos: float = 0.01,         # m^2/s
        q_vel: float = 0.20,         # m^2/s^3
        q_psi: float = 0.005,        # rad^2/s (gyro angular random walk)
        q_bg: float = 1e-6,          # rad^2/s^3 (gyro bias instability)
        h_sign: float = -1.0,        # Heading integration sign: dpsi/dt = h_sign * (gyro_pitch - b_g)
    ):
        self.h_sign = h_sign
        
        # Initial state
        psi0 = init_heading_rad % (2.0 * np.pi)
        ve0 = init_speed_mps * np.sin(psi0)
        vn0 = init_speed_mps * np.cos(psi0)
        
        P0 = np.diag([
            sigma_pos_init**2,
            sigma_pos_init**2,
            sigma_vel_init**2,
            sigma_vel_init**2,
            sigma_psi_init**2,
            sigma_bg_init**2,
        ]).astype(np.float64)
        
        self.state = ESKFState(
            p_e=float(init_p_e),
            p_n=float(init_p_n),
            v_e=float(ve0),
            v_n=float(vn0),
            psi=float(psi0),
            b_g=float(init_gyro_bias_rads),
            P=P0,
        )
        
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_psi = q_psi
        self.q_bg = q_bg
        
        # Diagnostic counters and logs
        self.diagnostics = {
            "speed_updates_accepted": 0,
            "speed_updates_rejected": 0,
            "nhc_updates_accepted": 0,
            "nhc_updates_rejected": 0,
            "zupt_updates_accepted": 0,
            "innovations_speed": [],
            "innovations_nhc": [],
            "covariance_trace": [],
        }

    def predict(
        self,
        dt: float,
        gyro_pitch_rads: float,
        accel_x_mps2: Optional[float] = None,
        accel_bias_mps2: float = 0.0,
        use_accel_propagation: bool = False,
    ):
        """
        Nominal state propagation and error covariance propagation.
        """
        if dt <= 0.0:
            return
            
        # 1. Propagate heading
        # Gyro yaw rate around vehicle vertical axis
        omega = self.h_sign * (gyro_pitch_rads - self.state.b_g)
        psi_new = (self.state.psi + omega * dt) % (2.0 * np.pi)
        
        # 2. Propagate velocity and position
        if use_accel_propagation and accel_x_mps2 is not None:
            a_fwd = accel_x_mps2 - accel_bias_mps2
            # Average heading over step
            psi_avg = (self.state.psi + 0.5 * omega * dt) % (2.0 * np.pi)
            dv_e = a_fwd * np.sin(psi_avg) * dt
            dv_n = a_fwd * np.cos(psi_avg) * dt
            ve_new = self.state.v_e + dv_e
            vn_new = self.state.v_n + dv_n
        else:
            # Kinematic velocity propagation (velocity is driven by measurement updates or drag model)
            ve_new = self.state.v_e
            vn_new = self.state.v_n
            
        pe_new = self.state.p_e + 0.5 * (self.state.v_e + ve_new) * dt
        pn_new = self.state.p_n + 0.5 * (self.state.v_n + vn_new) * dt
        
        # 3. Error-State Transition Matrix F (6x6)
        # delta_p_k = delta_p_{k-1} + delta_v_{k-1} * dt
        # delta_v_k = delta_v_{k-1} (or with accel coupling if active)
        # delta_psi_k = delta_psi_{k-1} - h_sign * dt * delta_b_g
        # delta_b_g_k = delta_b_g_{k-1}
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = -self.h_sign * dt  # Coupling gyro bias error to heading error
        
        if use_accel_propagation and accel_x_mps2 is not None:
            a_fwd = accel_x_mps2 - accel_bias_mps2
            F[2, 4] = a_fwd * np.cos(self.state.psi) * dt
            F[3, 4] = -a_fwd * np.sin(self.state.psi) * dt
            
        # 4. Discrete Process Noise Covariance Q_d
        Q_d = np.diag([
            self.q_pos * dt,
            self.q_pos * dt,
            self.q_vel * dt,
            self.q_vel * dt,
            self.q_psi * dt,
            self.q_bg * dt,
        ])
        
        # Propagate covariance
        P_pred = F @ self.state.P @ F.T + Q_d
        # Ensure symmetry
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        # Update nominal state
        self.state.p_e = float(pe_new)
        self.state.p_n = float(pn_new)
        self.state.v_e = float(ve_new)
        self.state.v_n = float(vn_new)
        self.state.psi = float(psi_new)
        self.state.P = P_pred
        
        self.diagnostics["covariance_trace"].append(float(np.trace(P_pred)))

    def update_speed(
        self,
        v_pred_mps: float,
        var_v: float,
        gate_threshold: float = 16.0,  # 4-sigma gate
    ) -> bool:
        """
        Fuses scalar forward speed measurement z = v_pred.
        Observation model: h(x) = v_e * sin(psi) + v_n * cos(psi)
        """
        psi = self.state.psi
        v_e = self.state.v_e
        v_n = self.state.v_n
        
        # Estimated forward speed
        h_val = v_e * np.sin(psi) + v_n * np.cos(psi)
        y = v_pred_mps - h_val  # Innovation
        
        # Measurement Jacobian H (1x6)
        # dh/dve = sin(psi), dh/dvn = cos(psi)
        # dh/dpsi = v_e * cos(psi) - v_n * sin(psi) = v_lat
        v_lat = v_e * np.cos(psi) - v_n * np.sin(psi)
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 2] = np.sin(psi)
        H[0, 3] = np.cos(psi)
        H[0, 4] = v_lat
        
        # Innovation covariance
        S = float((H @ self.state.P @ H.T)[0, 0] + var_v)
        if S <= 0.0 or np.isnan(S):
            return False
            
        nis = (y * y) / S
        if nis > gate_threshold:
            self.diagnostics["speed_updates_rejected"] += 1
            return False
            
        # Kalman Gain (6x1)
        K = (self.state.P @ H.T) / S  # shape (6, 1)
        
        # Error state correction
        delta_x = (K * y).flatten()
        
        # Nominal state update
        self.state.p_e += delta_x[0]
        self.state.p_n += delta_x[1]
        self.state.v_e += delta_x[2]
        self.state.v_n += delta_x[3]
        self.state.psi = (self.state.psi + delta_x[4]) % (2.0 * np.pi)
        self.state.b_g += delta_x[5]
        
        # Joseph Form Covariance Update
        I_KH = np.eye(6) - K @ H
        R_mat = np.array([[var_v]], dtype=np.float64)
        P_upd = I_KH @ self.state.P @ I_KH.T + K @ R_mat @ K.T
        self.state.P = 0.5 * (P_upd + P_upd.T)
        
        self.diagnostics["speed_updates_accepted"] += 1
        self.diagnostics["innovations_speed"].append(float(y))
        return True

    def update_nhc(
        self,
        var_nhc: float = 0.10,        # (m/s)^2, corresponding to sigma ~ 0.31 m/s
        gate_threshold: float = 9.0,   # 3-sigma gate
    ) -> bool:
        """
        Non-Holonomic Constraint (NHC) measurement update:
        Virtual observation: z_nhc = 0.
        Observation model: h(x) = v_e * cos(psi) - v_n * sin(psi) = v_lat
        """
        psi = self.state.psi
        v_e = self.state.v_e
        v_n = self.state.v_n
        
        v_lat = v_e * np.cos(psi) - v_n * np.sin(psi)
        y = 0.0 - v_lat  # Innovation
        
        # Jacobian H_nhc (1x6)
        # dh/dve = cos(psi), dh/dvn = -sin(psi)
        # dh/dpsi = -v_e * sin(psi) - v_n * cos(psi) = -v_fwd
        v_fwd = v_e * np.sin(psi) + v_n * np.cos(psi)
        H = np.zeros((1, 6), dtype=np.float64)
        H[0, 2] = np.cos(psi)
        H[0, 3] = -np.sin(psi)
        H[0, 4] = -v_fwd
        
        S = float((H @ self.state.P @ H.T)[0, 0] + var_nhc)
        if S <= 0.0 or np.isnan(S):
            return False
            
        nis = (y * y) / S
        if nis > gate_threshold:
            self.diagnostics["nhc_updates_rejected"] += 1
            return False
            
        K = (self.state.P @ H.T) / S
        delta_x = (K * y).flatten()
        
        self.state.p_e += delta_x[0]
        self.state.p_n += delta_x[1]
        self.state.v_e += delta_x[2]
        self.state.v_n += delta_x[3]
        self.state.psi = (self.state.psi + delta_x[4]) % (2.0 * np.pi)
        self.state.b_g += delta_x[5]
        
        I_KH = np.eye(6) - K @ H
        R_mat = np.array([[var_nhc]], dtype=np.float64)
        P_upd = I_KH @ self.state.P @ I_KH.T + K @ R_mat @ K.T
        self.state.P = 0.5 * (P_upd + P_upd.T)
        
        self.diagnostics["nhc_updates_accepted"] += 1
        self.diagnostics["innovations_nhc"].append(float(y))
        return True

    def update_zupt(
        self,
        var_zupt: float = 0.0025,  # (0.05 m/s)^2
    ) -> bool:
        """
        Zero-Velocity Update (ZUPT):
        Observation: z = [0, 0]^T
        h(x) = [v_e, v_n]^T
        """
        y = np.array([-self.state.v_e, -self.state.v_n], dtype=np.float64)
        H = np.zeros((2, 6), dtype=np.float64)
        H[0, 2] = 1.0
        H[1, 3] = 1.0
        
        R = np.diag([var_zupt, var_zupt])
        S = H @ self.state.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
            
        K = self.state.P @ H.T @ S_inv
        delta_x = K @ y
        
        self.state.p_e += delta_x[0]
        self.state.p_n += delta_x[1]
        self.state.v_e += delta_x[2]
        self.state.v_n += delta_x[3]
        self.state.psi = (self.state.psi + delta_x[4]) % (2.0 * np.pi)
        self.state.b_g += delta_x[5]
        
        I_KH = np.eye(6) - K @ H
        P_upd = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T
        self.state.P = 0.5 * (P_upd + P_upd.T)
        
        self.diagnostics["zupt_updates_accepted"] += 1
        return True
