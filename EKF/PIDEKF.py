import time
import cv2
import numpy as np
import pantilthat as pth

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


class PD:
    def __init__(self, kp, kd, setpoint, deg_per_px,
                 tau=0.02, max_dt=0.1):
        # pd gains
        self.kp = kp
        self.kd = kd

        # Desired setpoint in pixels (vertical center of the image)
        self.setpoint = setpoint

        # Conversion factor (degrees per pixel)
        self.deg_per_px = deg_per_px

        # Derivative smoothing and dt cap
        self.tau = tau
        self.max_dt = max_dt


        # Internal state
        self.prev_time = time.monotonic()
        self.deriv = 0.0
        self.prev_error = None
        self.last_error = 0.0
        self.last_dt = 1e-6

    def update(self, measurement):
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))

        # Compute error in pixels (use predicted y here)
        error = self.setpoint - measurement

        # Derivative on error, low-pass filtered
        if self.prev_error is None:
            self.deriv = 0.0
        else:
            raw_deriv = (error - self.prev_error) / dt
            alpha = dt / (self.tau + dt)
            self.deriv = alpha * raw_deriv + (1 - alpha) * self.deriv

        # PD terms (in pixel-space)
        P = self.kp * error
        D = self.kd * self.deriv
        output_px = P + D

        # Save state
        self.prev_time = now
        self.prev_error = error
        self.last_error = error
        self.last_dt = dt

        # Convert pixel-output to degrees and invert sign if needed
        return -output_px * self.deg_per_px

    def reset(self):
        self.deriv = 0.0
        self.prev_error = None
        self.prev_time = time.monotonic()
        self.last_error = 0.0
        self.last_dt = 1e-6


def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))

# Placeholder—for your actual detection; should return a pixel‐coordinate y or None
def get_measurement_from_camera(frame):
    # e.g. your contour‐based or ML‐based detection
    return None

if __name__ == '__main__':
    # 1) OpenCV camera init
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera capture failed.")
        cap.release()
        exit()

    FRAME_HEIGHT = 640
    FRAME_RATE = 50.0  
    dt = 1.0 / FRAME_RATE  

    # 2) Compute vertical FoV in pixels 
    SENSOR_HEIGHT_MM = 4.712
    FOCAL_LENGTH_MM = 25
    DROP_DIST_M = 2.5  # your estimated drop distance

    vfov_rad = 2 * np.arctan(SENSOR_HEIGHT_MM / (2.0 * FOCAL_LENGTH_MM))
    vfov_deg = np.degrees(vfov_rad)
    deg_per_px = vfov_deg / FRAME_HEIGHT

    # 3) Compute px_per_m (so we can convert gravity into “pixels/s^2”)
    world_h_m = 2.0 * DROP_DIST_M * np.tan(vfov_rad / 2.0)
    px_per_m = FRAME_HEIGHT / world_h_m

    # 4) Convert g to pixel units
    g = 9.81  # m/s^2
    g_pix = g * px_per_m  # pixels/s^2

    # 5) Drag coefficient etc. (same as your code)
    mass = 0.0042       # kg
    rho_air = 1.2       # kg/m^3
    Cd = 0.47           # sphere
    radius_m = 0.025    # m
    A_m2 = np.pi * radius_m**2
    # world‐drag 	 = 0.5 * rho * Cd * A_m2  (units: kg/m)
    # pixel drag: divide by (px_per_m * mass)
    k_drag_pix = (0.5 * rho_air * Cd * A_m2 / px_per_m) / mass

    # 6) EKF initialization: use first two measurements to get y0, vy0
    # Suppose you buffer the first two frames to get an initial velocity estimate.
    # For simplicity here, we’ll just wait until we have two valid measurements.
    init_y = None
    init_vy = None

    # We’ll store the last valid measurement and timestamp to estimate initial vy:
    last_valid_y = None
    last_valid_t = None

    ekf = None
    ekf_ready = False

    pd_setpoint = FRAME_HEIGHT / 2.0

    pd = PD(
        kp=0.9,
        kd=0.1,
        setpoint=pd_setpoint,
        deg_per_px=deg_per_px,
        tau=0.02,
        max_dt=0.1
    )

    # 8) Pan‐tilt servo init
    try:
        pth.servo_enable(2, True)
        current_tilt = 0.0
        pth.tilt(int(round(current_tilt)))
        print("[INFO] Tilt servo initialized.")
    except Exception as e:
        print(f"[ERROR] Could not initialize PanTilt HAT: {e}")
        cap.release()
        exit()

    PIXEL_DEADBAND = 31    # px
    MISS_LIMIT = 120       # frames
    miss_count = 0
    pd_active = False

    print("[INFO] Starting tracking loop. Press Ctrl+C to exit.")
    try:
        while True:
            loop_start = time.monotonic()

            # a) grab a frame
            ret, frame = cap.read()
            if not ret:
                continue

            # b) detect the object (y‐pixel) or None
            measurement_y = get_measurement_from_camera(frame)

            # c) Miss‐count logic
            if measurement_y is None:
                miss_count += 1
            else:
                miss_count = 0

            # d) State‐machine on misses
            if miss_count >= MISS_LIMIT and pd_active:
                pd_active = False
            elif miss_count == 0 and not pd_active and measurement_y is not None:
                # We just regained lock; use this moment to initialize EKF if needed
                pd.reset()

                # If EKF is not ready, we need two consecutive valid measurements to set initial y & vy
                if not ekf_ready:
                    if last_valid_y is None:
                        last_valid_y = measurement_y
                        last_valid_t = loop_start
                    else:
                        # We have two valid measurements separated in time
                        dt_init = loop_start - last_valid_t
                        if dt_init <= 0:
                            dt_init = dt  # fallback

                        init_y = last_valid_y
                        init_vy = (measurement_y - last_valid_y) / dt_init  # px/s
                        print(f"[INFO] Initializing EKF: y0={init_y:.1f} px, vy0={init_vy:.1f} px/s")

                        # Build EKF now (use large covariances)
                        R_val = 2500.0             # measurement noise var (50 px std → 2500)
                        Qy = 900.0                 # process noise var for y (~30 px std)
                        Qvy = 925.0                # process noise var for vy
                        Pyy0 = R_val               # initial var on y
                        Pvv0 = (2 * R_val) / (dt_init * dt_init)  # initial var on vy

                        ekf = EKF_BallTracker(
                            initial_y=init_y,
                            initial_vy=init_vy,
                            initial_P_yy=Pyy0,
                            initial_P_vv=Pvv0,
                            process_noise_y=Qy,
                            process_noise_vy=Qvy,
                            measurement_noise_y=R_val,
                            dt=dt,
                            g_pix=g_pix,
                            k_drag_pix=k_drag_pix,
                            px_per_m=px_per_m
                        )
                        ekf_ready = True
                pd_active = True

            # e) pd update & servo command when active & EKF is ready
            if pd_active and ekf_ready and measurement_y is not None:
                # 1) EKF prediction step
                ekf.predict()

                # 2) EKF update with raw measurement (pixel y)
                ekf.update(measurement_y)

                # 3) Fetch EKF’s current state:
                y_hat = ekf.x_hat[0]    # px
                vy_hat = ekf.x_hat[1]   # px/s

                # 4) Compute total drag‐corrected acceleration in pixel units
                a_net_pix = g_pix - k_drag_pix * (vy_hat ** 2)

                # 5) **PREDICT 9 FRAMES AHEAD** (9 * dt seconds)
                steps_ahead = 9
                lead_time = steps_ahead * dt  # e.g. 9 * 0.02 = 0.18 s
                y_pred_9 = (y_hat
                            + vy_hat * lead_time
                            + 0.5 * a_net_pix * (lead_time ** 2))
                # (If you also need predicted velocity: 
                # v_pred_9 = vy_hat + a_net_pix * lead_time )

                # 6) Feed that predicted pixel Y into pd.update()
                #    so error = (setpoint - y_pred_9)
                pd_output_deg = pd.update(y_pred_9)

                # 7) Apply the tilt to PIM183 (PanTilt HAT)
                #    Ensure we do not exceed servo limits
                desired_tilt = current_tilt + pd_output_deg
                current_tilt = clamp(desired_tilt, -90, 90)
                pth.tilt(int(round(current_tilt)))

            # f) Maintain loop timing for ≈50 fps
            elapsed = time.monotonic() - loop_start
            sleep_time = (1.0 / FRAME_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting, disabling tilt servo.")
    finally:
        pth.servo_enable(2, False)
        cap.release()
        cv2.destroyAllWindows()
