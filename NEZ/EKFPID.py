import numpy as np
import math
import time
import cv2
import pantilthat as pth
from EKF import EKFTracker  # ← Your EKF class

# ----------------------------
# PID controller definition
# ----------------------------
class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px, tau=0.02, max_dt=0.1, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.deg_per_px = deg_per_px
        self.tau = tau
        self.max_dt = max_dt
        self.integral_limit = integral_limit

        self.prev_time = time.monotonic()
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_error = None
        self.last_error = 0.0
        self.last_dt = 1e-6

    def update(self, measurement):
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))

        error = self.setpoint - measurement
        integral_candidate = self.integral + error * dt
        if self.integral_limit is not None:
            integral_candidate = max(-self.integral_limit,
                                     min(self.integral_limit, integral_candidate))
        self.integral = integral_candidate

        if self.prev_error is None:
            self.deriv = 0.0
        else:
            raw_deriv = (error - self.prev_error) / dt
            alpha = dt / (self.tau + dt)
            self.deriv = alpha * raw_deriv + (1 - alpha) * self.deriv

        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * self.deriv
        output_px = P + I + D

        self.prev_time = now
        self.prev_error = error
        self.last_error = error
        self.last_dt = dt

        return -output_px * self.deg_per_px

    def reset(self):
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_error = None
        self.prev_time = time.monotonic()
        self.last_error = 0.0
        self.last_dt = 1e-6

def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))

def get_measurement_from_camera(frame):
    # Replace with your actual object detection logic
    return None  # ← Placeholder

# ----------------------------
# Main tracking loop
# ----------------------------
if __name__ == '__main__':
    # camera startup
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera capture failed.")
        cap.release()
        exit()

    # geometry & timing
    FRAME_HEIGHT   = frame.shape[0]
    FRAME_RATE     = 120.0
    LOOP_DT_TARGET = 1.0 / FRAME_RATE

    SENSOR_HEIGHT_MM = 4.712
    FOCAL_LENGTH_MM  = 16
    DROP_DIST_M      = 2.0

    vfov_rad  = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))
    vfov_deg  = math.degrees(vfov_rad)
    deg_per_px= vfov_deg / FRAME_HEIGHT
    px_per_m  = FRAME_HEIGHT / (2 * DROP_DIST_M * math.tan(vfov_rad / 2))

    g_pix        = 9.81 * px_per_m
    c_over_m_pix = 0.1  * px_per_m

    pid_setpoint = FRAME_HEIGHT / 2
    SERVO_MIN, SERVO_MAX = -90, 90
    I_LIMIT               = SERVO_MAX / 0.10

    # instantiate controllers
    pid = PID(
        kp=0.11, ki=0.10, kd=0.02,
        setpoint=pid_setpoint,
        deg_per_px=deg_per_px,
        tau=0.02, max_dt=0.1,
        integral_limit=I_LIMIT
    )

    ekf = EKFTracker(
        dt=1.0 / FRAME_RATE,
        c_over_m_pix=c_over_m_pix,
        g_pix=g_pix,
        process_var=50.0,
        meas_var=16.0,
        deg_per_px=deg_per_px
    )

    filter_initialized = False    

    try:
        pth.servo_enable(2, True)
        current_tilt = 0.0
        initial_tilt = current_tilt
        pth.tilt(int(current_tilt))
        print("[INFO] Tilt servo initialized.")
    except Exception as e:
        print(f"[ERROR] Could not initialize PanTilt HAT: {e}")
        cap.release()
        exit()

    PIXEL_DEADBAND = 31
    MISS_LIMIT     = 120
    miss_count     = 0
    pid_active     = False

    print("[INFO] Starting tracking loop. Press Ctrl+C to exit.")
    try:
        while True:
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            measurement_y = get_measurement_from_camera(frame)

            if measurement_y is None:
                miss_count += 1
            else:
                miss_count = 0

            # On first valid detection, initialize EKF state to measurement
            if measurement_y is not None and not filter_initialized:
                ekf.x = np.array([[measurement_y], [0.0]])       
                ekf.P = np.diag([1.0, 100.0])                     
                filter_initialized = True                           
                # Skip PID & motion for this frame if desired:
                continue                                        

            if miss_count >= MISS_LIMIT and pid_active:
                pid_active = False
            elif miss_count == 0 and not pid_active and filter_initialized:  
                pid.reset()
                ekf.reset()
                pid_active = True

            if pid_active and measurement_y is not None:
                # EKF predict→update as before
                predicted_state = ekf.predict()
                predicted_y     = float(predicted_state[0])

                ekf.update(measurement_y, camera_angle_deg=current_tilt - initial_tilt)

                if abs(pid.setpoint - predicted_y) > PIXEL_DEADBAND:
                    delta_deg = pid.update(predicted_y)
                else:
                    delta_deg = 0.0

                desired = current_tilt + delta_deg
                current_tilt = clamp(desired, SERVO_MIN, SERVO_MAX)
                pth.tilt(int(round(current_tilt)))

            # maintain loop rate
            elapsed    = time.monotonic() - loop_start
            sleep_time = LOOP_DT_TARGET - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting, disabling tilt servo.")
    finally:
        pth.servo_enable(2, False)
        cap.release()
        cv2.destroyAllWindows()
