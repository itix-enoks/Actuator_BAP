# --- set up your camera / vision parameters exactly as you already do in __main__ ---
FRAME_HEIGHT      = 640
SENSOR_HEIGHT_MM  = 4.712
FOCAL_LENGTH_MM   = 25
DROP_DIST_M       = 2.5

vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))
world_h_m = 2 * DROP_DIST_M * math.tan(vfov_rad / 2)
px_per_m = FRAME_HEIGHT / world_h_m
g_pix = 9.81 * px_per_m
# compute k_drag_pix the same way you already do…

dt = 1/50.0

# initialize with the first measurement (we’ll overwrite vy once we get the 2nd sample)
initial_y   = 0.0
initial_vy  = 0.0
P_y0        = 2500.0         # e.g. measurement variance
P_vy0       = (2*P_y0)/(dt**2)
Qy, Qvy     = 900.0, 925.0
Ry          = 2500.0

ekf = EKF_BallTracker(
    initial_y, initial_vy,
    P_y0, P_vy0,
    Qy, Qvy,
    Ry,
    dt, g_pix, k_drag_pix, px_per_m
)
ekf_initialized = False
prev_meas = None


    measurement_y, camera_preview_output, _ = proc_color.process_frames(…)
    if measurement_y is not None:
        if not ekf_initialized:
            # bootstrap velocity from first two samples
            if prev_meas is None:
                prev_meas = measurement_y
            else:
                initial_vy = (measurement_y - prev_meas) / dt
                ekf.x_hat = np.array([measurement_y, initial_vy])
                ekf_initialized = True
        else:
            # 1) EKF prediction
            ekf.predict()

            # 2) EKF update with the new pixel measurement
            ekf.update(measurement_y)

            # 3) Get the filtered position & velocity
            y_filt  = ekf.get_current_position()
            vy_filt = ekf.get_current_velocity()

            # 4) (Optional) Predict next-step y for your PID set-point
            y_next_pred = ekf.get_predicted()

            # debug
            print(f"EKF filt: y={y_filt:.1f}px, vy={vy_filt:.1f}px/s, y_next={y_next_pred:.1f}px")
