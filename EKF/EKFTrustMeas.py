import numpy as np

class EKF_BallTracker:
    """
    Extended Kalman Filter for tracking a free-falling ball.
    Assumes y-coordinate increases downwards.
    Velocity (vy) is positive downwards.
    """
    def __init__(self,
                 initial_y: float, initial_vy: float,
                 initial_P_yy: float, initial_P_vv: float,
                 process_noise_y: float, process_noise_vy: float,
                 measurement_noise_y: float,
                 dt: float, g_pix: float, k_drag_pix: float):
        # State vector [y, vy]^T
        self.x_hat = np.array([initial_y, initial_vy], dtype=float)

        # State covariance matrix P_k|k
        self.P = np.diag([initial_P_yy, initial_P_vv]).astype(float)

        # Process noise covariance Q
        self.Q = np.diag([process_noise_y, process_noise_vy]).astype(float)

        # Measurement noise covariance R (scalar for y measurement)
        self.R = float(measurement_noise_y)

        # Time step
        self.dt = float(dt)

        # Physical parameters (in pixel units)
        self.g_pix = float(g_pix)
        self.k_drag_pix = float(k_drag_pix)  # Units: 1/pixel

        # Measurement matrix H (constant)
        self.H = np.array([[1, 0]], dtype=float)
        self.H_T = self.H.T

        # Identity matrix
        self._I = np.eye(2)

    def predict(self):
        """
        EKF prediction step.
        """
        y_prev, vy_prev = self.x_hat
        a_net = self.g_pix - self.k_drag_pix * vy_prev**2

        # Second-order kinematic update
        y_pred = y_prev + vy_prev * self.dt + 0.5 * a_net * self.dt**2
        vy_pred = vy_prev + a_net * self.dt
        self.x_hat = np.array([y_pred, vy_pred])

        # Jacobian F
        F = np.zeros((2, 2), dtype=float)
        F[0, 0] = 1.0
        F[0, 1] = self.dt - self.k_drag_pix * abs(vy_prev) * self.dt**2
        F[1, 0] = 0.0
        F[1, 1] = 1.0 - 2.0 * self.k_drag_pix * abs(vy_prev) * self.dt

        # Covariance update
        self.P = F @ self.P @ F.T + self.Q
        return self.x_hat

    def update(self, measurement_y: float):
        """
        EKF update step using a new measurement.
        """
        # Innovation
        y_tilde = measurement_y - self.x_hat[0]

        # Innovation covariance
        S = self.P[0, 0] + self.R
        if S == 0:
            S = 1e-9

        # Kalman gain
        K = self.P @ self.H_T / S

        # State update
        self.x_hat = self.x_hat + (K * y_tilde).flatten()

        # Covariance update and symmetrize
        self.P = (self._I - K @ self.H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat

    def get_current_position(self) -> float:
        return self.x_hat[0]

    def get_current_velocity(self) -> float:
        return self.x_hat[1]

    def get_predicted_position_for_pid(self) -> float:
        """
        Predict next-step position for PID use.
        """
        y_u, vy_u = self.x_hat
        a_net_u = self.g_pix - self.k_drag_pix * vy_u**2
        return y_u + vy_u * self.dt + 0.5 * a_net_u * self.dt**2

# Example usage in __main__ omitted for brevity; residual clamping removed entirely.

if __name__ == '__main__':
    # These parameters MUST be determined for your specific setup and ball
    s_p2m_example = 0.00064505  # m/pixel (from previous discussion)
    g_m_s2 = 9.81
    # CRITICAL: Ensure g_pix_example is float when passed to EKF
    g_pix_example = float(g_m_s2 / s_p2m_example)


    m_ball_dry = 0.0042 # kg - YOU MUST MEASURE YOUR ACTUAL DRY BALL MASS
    rho_air = 1.225     # kg/m^3
    Cd = 0.47           # Drag coefficient for a sphere
    radius_m = 0.025    # m (for a 5cm diameter ball)
    A_m2 = np.pi * radius_m**2 # Cross-sectional area m^2
    k_drag_pix_dry_example = (0.5 * rho_air * Cd * A_m2 * s_p2m_example) / m_ball_dry

    print(f"Example s_p2m: {s_p2m_example:.6f} m/pixel")
    print(f"Example g_pix: {g_pix_example:.2f} pixels/s^2") # Will use the float value
    print(f"Example k_drag_pix (using m_ball_dry={m_ball_dry} kg): {k_drag_pix_dry_example:.6e} 1/pixel")

    #measurements_y_data = [6.5, 27.5, 36.5, 110.0, 199.5, 308.0, 368.5, 433.5, 534.5, 577.0]
    measurements_y_data = [9.5, 24.0, 40.0, 53.0, 116.5, 140.0, 232.0, 260.5, 342.5, 436.0, 502.0, 574.5]
    #measurements_y_data = [6.5, 26.5, 69.0, 92.5, 167.5, 219.0, 346.0, 484.0, 519.5]

    dt_example = 1/50.0

    if len(measurements_y_data) < 2:
        print("Not enough measurements to initialize velocity. Need at least 2.")
    else:
        z0 = float(measurements_y_data[0]) # Ensure float
        z1 = float(measurements_y_data[1]) # Ensure float

        initial_y_val = z0
        initial_vy_val = (z1 - z0) / dt_example

        print(f"\nInitializing EKF with: y0 = {initial_y_val:.1f}, vy0 = {initial_vy_val:.1f} (from first two measurements)")

        # --- Suggested New Tuning Parameters ---
        # Rationale for changes from your last attempt:
        # - Corrected dtypes in EKF class to float.
        # - measurement_noise_y (R): Suggested a common baseline (std dev ~3px).
        # - process_noise_vy (Q[1,1]): Significantly increased again. If innovations are large,
        #   the model needs to be allowed to adapt its velocity (and thus acceleration) faster.
        #   Your previous 80.0 was much lower than my earlier suggestion of 500.0.
        # - process_noise_y (Q[0,0]): Adjusted to be consistent.
        ekf = EKF_BallTracker(
            initial_y=initial_y_val, initial_vy=initial_vy_val,
            initial_P_yy=30.0,        # Initial uncertainty for y (variance) - Tune
            initial_P_vv=10.0,     # High initial uncertainty for vy (very uncertain from 2 pts) - Tune
            process_noise_y=800,      # Process noise for y - Tune (was 2.0 in your last code)
            process_noise_vy=300.0,   # Significantly Increased: Process noise for vy - Tune (was 80.0)
            measurement_noise_y=20.0,  # Measurement noise for y (e.g., (3px)^2) - VERIFY & Tune (was 12.0)
            dt=dt_example,
            g_pix=g_pix_example,      # Pass the float value
            k_drag_pix= k_drag_pix_dry_example
        )

        print("\nSimulating EKF steps with new suggested tuning (floats used, no residual clamping):")
        header = "Time (s) | Meas Zk | Pred Y (k|k-1) | Upd Y (k|k) | Upd Vy (k|k) | PID Y (k+1|k)"
        print(header)
        print("-" * len(header))

        time_k0 = 0.0
        # Initial state print: x_hat_0|0 and y_hat_1|0 (prediction for next step based on initial state)
        pid_y_for_t1 = ekf.get_predicted_position_for_pid()
        print(f"{time_k0:8.2f} | {z0:6.1f} | {'N/A (init)':>14} | {ekf.get_current_position():11.1f} | {ekf.get_current_velocity():12.1f} | {pid_y_for_t1:13.1f}")

        for i in range(1, len(measurements_y_data)):
            zk = float(measurements_y_data[i]) # Ensure measurement is float
            current_time_s = i * dt_example

            predicted_state_k_km1 = ekf.predict()
            y_pred_k_km1 = predicted_state_k_km1[0]

            updated_state_k_k = ekf.update(zk)
            y_updated_k_k = updated_state_k_k[0]
            vy_updated_k_k = updated_state_k_k[1]
            
            pid_target_y_kp1_k = ekf.get_predicted_position_for_pid()

            print(f"{current_time_s:8.2f} | {zk:6.1f} | {y_pred_k_km1:14.1f} | {y_updated_k_k:11.1f} | {vy_updated_k_k:12.1f} | {pid_target_y_kp1_k:13.1f}")
