import math
import time

# PID class (as in your original script)
class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px,
                 tau=0.02, max_dt=0.1, integral_limit=None):
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

    def update(self, measurement):
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))
        self.prev_time = now

        error = self.setpoint - measurement

        # integral with clamping
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(-self.integral_limit,
                                min(self.integral_limit,
                                    self.integral))

        # derivative (low-pass)
        if self.prev_error is None:
            self.deriv = 0.0
        else:
            raw_deriv = (error - self.prev_error) / dt
            alpha = dt / (self.tau + dt)
            self.deriv = alpha * raw_deriv + (1 - alpha) * self.deriv

        self.prev_error = error

        # PID terms
        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * self.deriv

        output_px = P + I + D
        return -output_px * self.deg_per_px

# --- Simulation parameters ---
FRAME_HEIGHT = 500               # pixels
SENSOR_HEIGHT_MM = 4.712
FOCAL_LENGTH_MM = 16
vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))
vfov_deg = math.degrees(vfov_rad)
deg_per_px = vfov_deg / FRAME_HEIGHT
print(f"Conversion: {deg_per_px}°")

pid = PID(
    kp=0.9,
    ki=0.0,
    kd=0.0,
    setpoint= FRAME_HEIGHT / 2,
    deg_per_px=deg_per_px,
    tau=0.02,
    max_dt=0.1,
    integral_limit=(90 / 0.10)
)

# Simulate measurements from 0 to 500 px in ~30 px steps
for measurement in range(0, 501, 20):
    output_deg = pid.update(measurement)
    print(f"Measurement: {measurement:>3} Error: {250 - measurement} px → Output: {output_deg:>7.4f}°")
    # Optional: mimic a fixed loop time
    time.sleep(1/120.0)
