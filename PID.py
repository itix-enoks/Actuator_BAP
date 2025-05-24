import time
import math
import cv2
import pantilthat as pth

from processor.camera import CameraStream, SharedObject
from processor.algorithms.frame_difference import process_frames  # TODO: implement motion compensation


class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px,
                 tau=0.02, max_dt=0.1, integral_limit=None):
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Desired setpoint in pixels (center of image height in camera frame)
        self.setpoint = setpoint

        # Conversion factor (amount of degrees a pixel represents)
        self.deg_per_px = deg_per_px

        # Derivative smoothing and dt cap
        self.tau = tau
        self.max_dt = max_dt

        # Anti-windup limit for integral (in pixel-seconds)
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

        # Compute error in pixels
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

        # Convert pixel-output to degrees
        return -output_px * self.deg_per_px

    def reset(self):
        # Clear integral and derivative history
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_error = None
        self.prev_time = time.monotonic()
        self.last_error = 0.0
        self.last_dt = 1e-6


def clamp(value, vmin, vmax):
    return max(vmin, min(vmax, value))

# Placeholder for detection logic
def get_measurement_from_camera(frame):
    # Replace with your actual detection returning y-coordinate or None
    return None

# This will run process and tilt separately, such that blocking does not occur anymore.
def run_tasks_in_parallel(tasks):
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            running_task.result()


if __name__ == '__main__':
    # TODO: storing frames

    # Create shared-memory for capturing, processing and tilting
    shared_obj = SharedObject()

    # Initialize camera
    camera = CameraStream(shared_obj)
    camera.start()

    # Frame dimensions and timing
    FRAME_HEIGHT = 1332
    FRAME_RATE = 120.0
    LOOP_DT_TARGET = 1.0 / FRAME_RATE

    # Compute vertical FoV
    SENSOR_HEIGHT_MM = 4.712
    FOCAL_LENGTH_MM = 16
    vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))
    vfov_deg = math.degrees(vfov_rad)
    deg_per_px = vfov_deg / FRAME_HEIGHT

    # PID and servo settings
    pid_setpoint = FRAME_HEIGHT / 2
    SERVO_MIN, SERVO_MAX = -90, 90
    I_LIMIT = SERVO_MAX / 0.10

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

    # Initialize PanTilt HAT
    try:
        pth.servo_enable(2, True)
        current_tilt = 0.0
        pth.tilt(int(current_tilt))
        print("[INFO] Tilt servo initialized.")
    except Exception as e:
        print(f"[ERROR] Could not initialize PanTilt HAT: {e}")
        camera.stop()
        exit()

    # Tracking control variables
    PIXEL_DEADBAND = 31    # px
    MISS_LIMIT = 120        # frames of missed detection before deactivating
    miss_count = 0
    pid_active = False

    # Camera variables
    # Measurement
    camera_preview_output = None
    camera_prev_gray = None

    # FPS Overlay
    camera_prev_time = time.time_ns()
    camera_diff_time = 0
    camera_frame_per_sec = 0
    camera_frame_cnt_in_sec = 0
    camera_is_one_sec_passed = False

    # Storing frames
    camera_frame_buffer = []

    print("[INFO] Starting tracking loop. Press Ctrl+C to exit.")
    try:
        while True:
            loop_start = time.monotonic()

            # 1) Read frame
            current_frame = shared_obj.frame
            current_gray_frame = cv.cvtColor(current_frame, cv.COLOR_RGB2GRAY) if current_frame is not None else None

            # 2) Detect object
            if current_frame is None or camera_prev_gray is None:
                camera_preview_output, measurement_y = None, None
            else:
                camera_preview_output, measurement_y = process_frames(camera_prev_gray, current_gray_frame, current_frame)

            camera_prev_gray = current_gray_frame

            # 2.1) FPS overlay
            camera_frame_cnt_in_sec += 1
            camera_curr_time = time.time_ns()
            camera_diff_time += (camera_curr_time - camera_prev_time) / 1e6

            if int(camera_diff_time) >= 1000:
                camera_frame_per_sec = camera_frame_cnt_in_sec
                camera_frame_cnt_in_sec = 0
                camera_diff_time = 0
                camera_is_one_sec_passed = True

            if camera_is_one_sec_passed:
                cv.putText(camera_preview_output, f"FPS: {camera_frame_per_sec}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv.putText(camera_preview_output, f"FPS: (WAITING...)", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            camera_prev_time = camera_curr_time

            # 2.2) Preview frame
            if camera_preview_output is not None:
                cv.imshow(f'[{RECORDING_ID}] [Live] Processed Frame', camera_preview_output)

            # 3) Miss-count logic instead of immediate reset on single-frame dropout
            if measurement_y is None:
                miss_count += 1
            else:
                miss_count = 0

            # 4) State-machine with MISS_LIMIT
            if miss_count >= MISS_LIMIT and pid_active:
                pid_active = False
            elif miss_count == 0 and not pid_active and measurement_y is not None:
                pid.reset()
                pid_active = True

            # 5) PID update and servo write when active
            if pid_active and measurement_y is not None:
                if abs(pid.setpoint - measurement_y) > PIXEL_DEADBAND:
                    delta_deg = pid.update(measurement_y)
                else:
                    delta_deg = 0.0

                desired = current_tilt + delta_deg
                current_tilt = clamp(desired, SERVO_MIN, SERVO_MAX)
                pth.tilt(int(round(current_tilt)))

            # 6) Maintain loop timing
            elapsed = time.monotonic() - loop_start
            sleep_time = LOOP_DT_TARGET - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # 7) Exit & Store frames
            if cv.waitKey(1) & 0xFF == ord('q'):
                shared_obj.is_exit = True
                # TODO: store frames


    except KeyboardInterrupt:
        print("\n[INFO] Exiting, disabling tilt servo.")
    finally:
        pth.servo_enable(2, False)
        camera.stop()
        cv2.destroyAllWindows()
