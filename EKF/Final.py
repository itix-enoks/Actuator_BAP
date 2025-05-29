import numpy as np
import math

class EKF_BallTracker:
    def __init__(self,
                 initial_y, initial_vy,
                 initial_P_yy, initial_P_vv,
                 process_noise_y, process_noise_vy,
                 measurement_noise_y,
                 dt, g_pix, k_drag_pix, px_per_m):
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
        y_prev, vy_prev = self.x_hat
        a_net = self.g_pix - self.k_drag_pix * vy_prev**2 # Simple quadratic drag model

        # State prediction (kinematic update)
        y_pred = y_prev + vy_prev * self.dt + 0.5 * a_net * self.dt**2
        vy_pred = vy_prev + a_net * self.dt
        self.x_hat = np.array([y_pred, vy_pred])

        F = np.zeros((2, 2), dtype=float)
        F[0, 0] = 1.0
        F[0, 1] = self.dt - self.k_drag_pix * vy_prev * self.dt**2 # Use vy_prev from start of interval
        F[1, 0] = 0.0
        F[1, 1] = 1.0 - 2.0 * self.k_drag_pix * vy_prev * self.dt # Use vy_prev from start of interval

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        return self.x_hat

    def update(self, measurement_y):
        # Innovation (measurement residual)
        y_tilde = measurement_y - self.x_hat[0] # x_hat[0] is y_pred_k|k-1

        # Innovation covariance
        # S = H P_k|k-1 H^T + R
        # Since H = [1, 0], H P H^T = P[0,0]
        S = self.P[0, 0] + self.R
        if S == 0: # Avoid division by zero
            S = 1e-9 # Add a tiny epsilon

        # Kalman gain K
        # K = P_k|k-1 H^T S^-1
        # P_k|k-1 H^T = [P[0,0], P[1,0]]^T
        K_column_vector = self.P @ self.H_T
        K = K_column_vector / S

        # State update
        self.x_hat = self.x_hat + (K * y_tilde).flatten()

        # Covariance update (Joseph form recommended for stability, but (I-KH)P is common)
        self.P = (self._I - K @ self.H) @ self.P
        # Ensure P remains symmetric
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat

    def get_current_position(self):
        return self.x_hat[0]

    def get_current_velocity(self):
        return self.x_hat[1] / px_per_m

    def get_predicted(self):
        """
        Predicts next-step position based on the current updated state x_hat (k|k).
        This is y_hat (k+1|k).
        """
        y_curr_updated, vy_curr_updated = self.x_hat # These are x_k|k
        a_net_curr_updated = self.g_pix - self.k_drag_pix * vy_curr_updated**2
        return y_curr_updated + vy_curr_updated * self.dt + 0.5 * a_net_curr_updated * self.dt**2

if __name__ == '__main__':
    FRAME_HEIGHT      = 640   # image height in pixels
    SENSOR_HEIGHT_MM  = 4.712   # Pi HQ camera sensor height in mm
    FOCAL_LENGTH_MM   = 25     # lens focal length in mm
    DROP_DIST_M       = 2.5     # approximate distance from camera to drop in meters
    
    # 1) vertical field-of-view (radians)
    vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))

    # 2) physical height (m) of that FOV at the drop distance
    world_h_m = 2 * DROP_DIST_M * np.tan(vfov_rad / 2)

    # 3) the amount of pixels needed to represent one meter in the image
    px_per_m = FRAME_HEIGHT / world_h_m
    g = 9.81               # Standard gravity m/s^2
    g_pix_example = (g * px_per_m) # pixels/s^2

    # Ball properties (example values - YOU MUST MEASURE/DETERMINE YOURS)
    mass = 0.0042     # kg (example mass of a dry ping pong ball)
    rho_air = 1.2         # kg/m^3 (air density at sea level, 15°C)
    Cd = 0.47               # Drag coefficient for a sphere (typical value)
    radius_m = 0.025        # m (for a 4cm diameter ball, e.g., ping pong ball)
    A_m2 = np.pi * radius_m**2 # Cross-sectional area m^2
    # k_drag_world = 0.5 * rho_air * Cd * A_m2 (units: kg/m)
    # k_drag_pix = k_drag_world / (px_per_m * mass) (units: 1/pixel)
    k_drag_pix = (0.5 * rho_air * Cd * A_m2 / px_per_m) / mass

    measurements_y_data = [6.5, 27.5, 36.5, 110.0, 199.5, 308.0, 368.5, 433.5, 534.5, 577.0]
    # measurements_y_data = [9.5, 24.0, 40.0, 53.0, 116.5, 140.0, 232.0, 260.5, 342.5, 436.0, 502.0, 574.5]
    # measurements_y_data = [6.5, 26.5, 69.0, 92.5, 167.5, 219.0, 346.0, 484.0, 519.5]
    # measurements_y_data = [22.5, 78.5, 107.0, 163.5, 194.5, 263.0, 370.0, 478.0, 620.0]


    dt = 1/50.0 

    
    z0 = float(measurements_y_data[0])
    z1 = float(measurements_y_data[1])

    initial_y_val = z0
    initial_vy_val = (z1 - z0) / dt

    print(f"\nInitializing EKF with: y0 = {initial_y_val:.1f}, vy0 = {initial_vy_val:.1f} px/s (from first two measurements)")
    
    # R: Measurement noise variance. Std dev of measurement error ~50px => R = 50^2 = 2500
    R_val = 2500.0
    # Q_y: Process noise variance for y. Std dev of unmodeled position change ~30px => Q_y = 30^2 = 900
    Q_y_val = 900.0
    # Q_vy: Process noise variance for vy. Std dev of unmodeled velocity change ~30.4px/s => Q_vy ~ 925
    Q_vy_val = 925.0 # (Can also use 900.0 for simplicity)

    print(f"Steady-state R: {R_val:.1f}")
    print(f"Steady-state Q_y: {Q_y_val:.1f}, Q_vy: {Q_vy_val:.1f}")

    # --- Calculate Initial State Covariance P0: LARGE to trust initial measurements more ---
    # Reflects uncertainty of initial_y_val and initial_vy_val derived from noisy measurements
    P_yy_initial = R_val  # Variance of initial y (from z0)
    P_vv_initial = (2 * R_val) / (dt**2) # Variance of initial vy (from (z1-z0)/dt)

    print(f"Calculated Initial P_yy: {P_yy_initial:.1f}")
    print(f"Calculated Initial P_vv: {P_vv_initial:.1e}") # Use scientific notation

    ekf = EKF_BallTracker(
        initial_y=initial_y_val, initial_vy=initial_vy_val,
        initial_P_yy=P_yy_initial,      # Using calculated large initial uncertainty for y
        initial_P_vv=P_vv_initial,      # Using calculated very large initial uncertainty for vy
        process_noise_y=Q_y_val,        # Steady-state process noise for y
        process_noise_vy=Q_vy_val,      # Steady-state process noise for vy
        measurement_noise_y=R_val,      # Steady-state measurement noise for y
        dt=dt,
        g_pix=g_pix_example,
        k_drag_pix=k_drag_pix,
        px_per_m=px_per_m
        )

    print("\nSimulating EKF steps:")
    header = "Time (s) | Meas Zk | Pred Y (k|k-1) | Upd Y (k|k) | Upd Vy (k|k) | PID Y (k+1|k)"
    print(header)
    print("-" * len(header))

    # Initial state print (k=0): x_hat_0|0
    # The 'PID Y (k+1|k)' here is y_hat_1|0 (prediction for next step based on initial state x_0|0)
    pid_y_for_t1 = ekf.get_predicted()
    print(f"{0.00:8.2f} | {z0:6.1f} | {'N/A (init)':>14} | {ekf.get_current_position():11.1f} | {ekf.get_current_velocity():12.1f} | {pid_y_for_t1:13.1f}")

    # Loop for k=1 to N-1 (using measurements z_1 to z_{N-1})
    for i in range(1, len(measurements_y_data)):
        zk = float(measurements_y_data[i]) # Current measurement z_k
        current_time_s = i * dt

        # Predict step: Generates x_hat_k|k-1 and P_k|k-1
        predicted_state_k_km1 = ekf.predict()
        y_pred_k_km1 = predicted_state_k_km1[0]

        # Update step: Generates x_hat_k|k and P_k|k using z_k
        updated_state_k_k = ekf.update(zk)
        y_updated_k_k = updated_state_k_k[0]
        vy_updated_k_k = ekf.get_current_velocity()
            
        # Get prediction for PID for the *next* time step (k+1), based on current updated state (k|k)
        pid_target_y_kp1_k = ekf.get_predicted()

        print(f"{current_time_s:8.2f} | {zk:6.1f} | {y_pred_k_km1:14.1f} | {y_updated_k_k:11.1f} | {vy_updated_k_k:12.1f} | {pid_target_y_kp1_k:13.1f}")
