import time
import math
import cv2
import pantilthat as pth

# ─────────────── PID CONTROLLER ───────────────
class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px,
                 tau=0.02, max_dt=0.1):
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Target in pixels
        self.setpoint = setpoint

        # Conversion factor (degrees per pixel)
        self.deg_per_px = deg_per_px

        # Smoothing and time-step limits
        self.tau = tau
        self.max_dt = max_dt

        # Internal state
        self.prev_time = time.monotonic()
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_measurement = None
        # for anti-windup in main loop
        self.last_error = 0.0
        self.last_dt = 1e-6

    def update(self, measurement):
        # 1) Compute elapsed time, clamped
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))

        # 2) Proportional term
        error = self.setpoint - measurement

        # 3) Integral accumulator (always accumulate here)
        self.integral += error * dt

        # 4) Derivative on measurement, low-pass
        if self.prev_measurement is None:
            self.deriv = 0.0
        else:
            raw_deriv = (measurement - self.prev_measurement) / dt
            alpha = dt / (self.tau + dt)
            self.deriv = alpha * raw_deriv + (1 - alpha) * self.deriv

        # 5) PID output in pixel units
        P = self.kp * error
        I = self.ki * self.integral
        D = -self.kd * self.deriv
        output_px = P + I + D

        # 6) Save state for next iteration and for anti-windup
        self.prev_time = now
        self.prev_measurement = measurement
        self.last_error = error
        self.last_dt = dt

        # 7) Convert to degrees and return
        return output_px * self.deg_per_px

# Stub for measurement retrieval (replace with real vision code)
def get_measurement():
    ret, frame = cap.read()
    if not ret:
        return pid.setpoint
    # TODO: implement detection, return x-pixel
    return pid.setpoint

if __name__ == "__main__":
    # Camera setup
    cap = cv2.VideoCapture(0)

    # Frame and optics parameters
    FRAME = 500
    PIXEL_SIZE_MM = 0.0025
    FOCAL_LENGTH_MM = 0.05
    vfov_rad = 2 * math.atan((FRAME * PIXEL_SIZE_MM) / (2 * FOCAL_LENGTH_MM))
    vfov_deg = math.degrees(vfov_rad)
    deg_per_px = vfov_deg / FRAME

    # Initialize PID
    pid = PID(kp=0.11, ki=0.10, kd=0.02,
              setpoint=FRAME/2,
              deg_per_px=deg_per_px,
              tau=0.02, max_dt=0.1)

    # Tilt state and limits
    current_tilt = 0.0
    SERVO_MIN = -90
    SERVO_MAX = 90

    # Enable servo
    pth.servo_enable(channel=1, enable=True)
    loop_dt = 0.005  # 200 Hz

    try:
        while True:
            # 1) Acquire measurement
            measurement = get_measurement()

            # 2) Compute raw delta angle (float degrees)
            delta_deg = pid.update(measurement)

            # 3) Apply anti-windup: if next tilt would saturate, remove last integral increment
            prospective_tilt = current_tilt + delta_deg
            if prospective_tilt > SERVO_MAX or prospective_tilt < SERVO_MIN:
                # back-calc windup: subtract the last integrated error
                pid.integral -= pid.last_error * pid.last_dt

            # 4) Update and clamp current_tilt
            current_tilt = max(SERVO_MIN, min(SERVO_MAX, prospective_tilt))

            # 5) Send command to servo (integer degrees)
            pth.tilt(int(round(current_tilt)))

            # 6) Wait for next loop
            time.sleep(loop_dt)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting, disabling tilt servo")

    finally:
        pth.servo_enable(channel=1, enable=False)
        cap.release()

