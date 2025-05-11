import time
import math
import cv2
import pantilthat as pth

class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px,
                 tau=0.02, max_dt=0.1, integral_limit=None):
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Desired setpoint in pixels
        self.setpoint = setpoint

        # Conversion factor 
        self.deg_per_px = deg_per_px

        # Derivative smoothing and dt cap
        self.tau = tau
        self.max_dt = max_dt

        # Anti-windup limit for integral 
        self.integral_limit = integral_limit

        # Internal state
        self.prev_time = time.monotonic()
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_error = None
        self.last_error = 0.0
        self.last_dt = 1e-6

    def update(self, measurement):
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))

        # Compute error 
        error = self.setpoint - measurement

        # Integrate with clamping
        integral_candidate = self.integral + error * dt
        if self.integral_limit is not None:
            integral_candidate = max(-self.integral_limit,
                                     min(self.integral_limit,
                                         integral_candidate))
        self.integral = integral_candidate

        # Derivative on error, low-pass filtered
        if self.prev_error is None:
            self.deriv = 0.0
        else:
            raw_deriv = (error - self.prev_error) / dt
            alpha = dt / (self.tau + dt)
            self.deriv = alpha * raw_deriv + (1 - alpha) * self.deriv

        # PID terms
        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * self.deriv
        output_px = P + I + D

        # Save state
        self.prev_time = now
        self.prev_error = error
        self.last_error = error
        self.last_dt = dt

        # Output in degrees
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



if __name__ == '__main__':
    # Initialize camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera capture failed.")
        cap.release()
        exit()

    # Frame dimensions and timing
    FRAME_HEIGHT = frame.shape[0]
    FRAME_RATE = 120.0                     # Camera runs at ~120 FPS
    LOOP_DT_TARGET = 1.0 / FRAME_RATE      # Loop interval tied to frame rate

    # Compute vertical FoV
    PIXEL_SIZE_M = 1.55e-6                 # meters per pixel
    SENSOR_HEIGHT_M = FRAME_HEIGHT * PIXEL_SIZE_M
    FOCAL_LENGTH_M = 35e-3                 # 35 mm lens in meters
    vfov_rad = 2 * math.atan(SENSOR_HEIGHT_M / (2 * FOCAL_LENGTH_M))
    vfov_deg = math.degrees(vfov_rad)
    deg_per_px = vfov_deg / FRAME_HEIGHT

    # PID and servo settings
    pid_setpoint = FRAME_HEIGHT / 2
    SERVO_MIN, SERVO_MAX = -90, 90
    I_LIMIT = SERVO_MAX / 0.10             # based on ki

    pid = PID(
        kp=0.11,
        ki=0.10,
        kd=0.02,
        setpoint=pid_setpoint,
        deg_per_px=deg_per_px,
        tau=0.02,
        max_dt=0.1,
        integral_limit=I_LIMIT
    )

    # Initialize PIM183
    try:
        pth.servo_enable(2, True)
        current_tilt = 0.0
        pth.tilt(int(current_tilt))
        print("[INFO] Tilt servo initialized.")
    except Exception as e:
        print(f"[ERROR] Could not initialize PanTilt HAT: {e}")
        cap.release()
        exit()

    PIXEL_DEADBAND = 10    # pixels
    pid_active = False

    print("[INFO] Starting tracking loop. Press Ctrl+C to exit.")
    try:
        while True:
            loop_start = time.monotonic()

            # Read frame; skip if the camera failed
            ret, frame = cap.read()
            if not ret:
                continue

            # Get object measurement once per frame
            measurement_y = get_measurement_from_camera(frame)

            # PID activation state machine
            if measurement_y is not None and not pid_active:
                pid.reset()
                pid_active = True
            elif measurement_y is None and pid_active:
                pid_active = False

            # Only update and write to servo when active
            if pid_active:
                if abs(pid.last_error) > PIXEL_DEADBAND:
                    delta_deg = pid.update(measurement_y)
                else:
                    delta_deg = 0.0

                # Compute and clamp new tilt angle
                desired = current_tilt + delta_deg
                current_tilt = clamp(desired, SERVO_MIN, SERVO_MAX)
                pth.tilt(int(round(current_tilt)))

            # Sleep to maintain camera-driven loop rate
            elapsed = time.monotonic() - loop_start
            sleep_time = LOOP_DT_TARGET - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting, disabling tilt servo.")
    finally:
        pth.servo_enable(2, False)
        cap.release()
        cv2.destroyAllWindows()
