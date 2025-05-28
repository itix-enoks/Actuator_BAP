import os
import sys
import time
import math
import cv2 as cv
import numpy as np
from threading import Thread

# --- Placeholder/Optional Imports ---
try:
    from processor.camera import CameraStream, SharedObject
except ImportError:
    print("WARNING: processor.camera module not found. Using placeholder SharedObject and CameraStream.")
    class SharedObject:
        def __init__(self):
            self.frame = None
            self.current_tilt = 0
            self.LOOP_DT_TARGET = 1.0 / 70.0
            self.is_exit = False
            self.frame_buffer = [] # List to store frames for saving

    class CameraStream:
        def __init__(self, shared_obj):
            self.shared_obj = shared_obj
            self.running = False
            self.cap = None 
            self.thread = None
            self._stopped = True
            print("[INFO] Placeholder CameraStream initialized.")

        def start(self):
            self.running = True
            self._stopped = False
            print("[INFO] Placeholder CameraStream started.")
            if self.shared_obj.frame is None:
                 self.shared_obj.frame = np.zeros((1332, 1920, 3), dtype=np.uint8)

        def _capture_loop(self):
            while self.running: time.sleep(0.01)
            self._stopped = True

        def stop(self):
            self.running = False
            print("[INFO] Placeholder CameraStream stopped.")

try:
    import processor.algorithms.colored_frame_difference as proc_color
except ImportError:
    print("WARNING: processor.algorithms.colored_frame_difference not found. Detection will use placeholder.")
    def placeholder_process_frames(prev_hsv, curr_hsv, display_frame_to_draw_on, hue, hue_tolerance):
        return None, display_frame_to_draw_on, None 
    proc_color = type('obj', (object,), {'process_frames' : placeholder_process_frames})

try:
    import pantilthat as pth
except ImportError:
    print("WARNING: pantilthat module not found. Servo control will use placeholder.")
    class PlaceholderPanTiltHat:
        def __init__(self): self._tilt_angle = 0
        def servo_enable(self, servo, enable): print(f"PTH_Placeholder: Servo {servo} Enable: {enable}")
        def pan(self, angle): pass # Assuming pan is not used or fixed
        def tilt(self, angle): self._tilt_angle = angle # Store for completeness if needed
    pth = PlaceholderPanTiltHat()
# --- End Placeholder/Optional Imports ---


# --- SimpleKalmanFilter Class Definition ---
class SimpleKalmanFilter:
    def __init__(self, dt, g_pix_acceleration, process_var, measurement_var):
        self.dt = dt
        self.g_pix = g_pix_acceleration
        self.F = np.array([[1, self.dt], [0, 1]], dtype=float)
        self.B = np.array([[0.5 * self.dt**2], [self.dt]], dtype=float)
        self.H = np.array([[1, 0]], dtype=float)
        dt2, dt3, dt4 = self.dt**2, self.dt**3, self.dt**4
        self.Q = process_var * np.array([[dt4/4, dt3/2], [dt3/2, dt2]], dtype=float)
        self.R = np.array([[measurement_var]], dtype=float)
        self.x_hat = np.zeros((2, 1), dtype=float)
        self.P = np.eye(2, dtype=float) * 500.0

    def predict_step(self):
        self.x_hat = self.F @ self.x_hat + self.B * self.g_pix
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x_hat.copy()

    def update_step(self, meas_true_y):
        y_tilde = meas_true_y - self.H @ self.x_hat
        S_inv = self.H @ self.P @ self.H.T + self.R
        if np.abs(S_inv[0,0]) < 1e-9: 
             print("[WARNING] KF S matrix singular, skipping update.")
             return
        K = self.P @ self.H.T @ np.linalg.inv(S_inv)
        self.x_hat = self.x_hat + K * y_tilde
        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def predict_future_state(self, num_frames_ahead):
        x_future = self.x_hat.copy()
        for _ in range(num_frames_ahead):
            x_future = self.F @ x_future + self.B * self.g_pix
        return x_future[0, 0], x_future[1, 0]

    def reset(self):
        self.x_hat = np.zeros((2, 1), dtype=float)
        self.P = np.eye(2, dtype=float) * 500.0


# --- PID Class Definition ---
class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px,
                 tau=0.05, max_dt=0.1, integral_limit=None): # Using tau=0.05 from user's code
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.deg_per_px = deg_per_px if deg_per_px != 0 else 1e-9
        self.tau, self.max_dt = tau, max_dt
        self.integral_limit = integral_limit
        self.prev_time = time.monotonic()
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_error = None

    def update(self, measurement):
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))
        error = self.setpoint - measurement
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)
        current_deriv = 0.0
        if self.prev_error is not None and dt > 1e-5:
            raw_deriv = (error - self.prev_error) / dt
            alpha = dt / (self.tau + dt) if (self.tau + dt) > 1e-9 else 1.0
            current_deriv = alpha * raw_deriv + (1 - alpha) * self.deriv
        self.deriv = current_deriv
        P_term, I_term, D_term = self.kp*error, self.ki*self.integral, self.kd*self.deriv
        output_px = P_term + I_term + D_term
        self.prev_time, self.prev_error = now, error
        return -output_px * self.deg_per_px

    def reset(self):
        self.integral, self.deriv, self.prev_error = 0.0, 0.0, None
        self.prev_time = time.monotonic()

def clamp(value, vmin, vmax): return max(vmin, min(vmax, value))

def tilt_thread_func(shared_obj_local):
    print("[INFO] Tilt thread started.")
    while not shared_obj_local.is_exit:
        loop_start_tilt = time.monotonic()
        try: pth.tilt(shared_obj_local.current_tilt)
        except Exception: pass 
        elapsed_tilt = time.monotonic() - loop_start_tilt
        sleep_time_tilt = shared_obj_local.LOOP_DT_TARGET - elapsed_tilt
        if sleep_time_tilt > 0: time.sleep(sleep_time_tilt)
    print("[INFO] Tilt thread exiting.")

# --- Main Execution ---
if __name__ == '__main__':
    shared_obj = SharedObject()
    camera = CameraStream(shared_obj)
    camera.start()

    FRAME_HEIGHT = 640
    FRAME_RATE = 70.0 
    LOOP_DT = 1.0 / FRAME_RATE
    shared_obj.LOOP_DT_TARGET = LOOP_DT
    NUM_FRAMES_PREDICT_AHEAD = 7 

    tilt_th = Thread(target=tilt_thread_func, args=(shared_obj,), daemon=True)
    tilt_th.start()

    SENSOR_HEIGHT_MM, FOCAL_LENGTH_MM, DROP_DIST_M = 4.712, 25.0, 2.5
    vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))
    vfov_deg = math.degrees(vfov_rad)
    if FRAME_HEIGHT == 0: raise ValueError("FRAME_HEIGHT error.")
    deg_per_px = vfov_deg / FRAME_HEIGHT
    world_h_m = 2 * DROP_DIST_M * math.tan(vfov_rad / 2)
    if world_h_m == 0: raise ValueError("world_h_m error.")
    px_per_m = FRAME_HEIGHT / world_h_m
    g_pix_accel = 9.81 * px_per_m * 1.2

    # KF and PID parameters from user's provided code
    kf_process_var, kf_measurement_var = 200.0, 25.0 
    kf = SimpleKalmanFilter(LOOP_DT, g_pix_accel, kf_process_var, kf_measurement_var)
    filter_initialized = False

    pid_target_center = FRAME_HEIGHT / 2.0 # User's setpoint
    SERVO_MIN, SERVO_MAX = -90, 90
    I_LIMIT_DENOM = 0.10
    pid_i_limit = SERVO_MAX / I_LIMIT_DENOM if I_LIMIT_DENOM != 0 else SERVO_MAX * 10
    pid = PID(0.75, 0, 0.04, pid_target_center, deg_per_px, tau=0.05, integral_limit=pid_i_limit) # User's PID gains and tau

    main_loop_current_tilt = -30.0 
    INITIAL_CAMERA_REFERENCE_TILT = -30.0
    shared_obj.current_tilt = int(round(main_loop_current_tilt))

    try:
        pth.servo_enable(2, True)
        pth.tilt(shared_obj.current_tilt)
        print(f"[INFO] Tilt servo initialized. Target: {shared_obj.current_tilt} deg (Negative=UP).")
    except Exception as e:
        print(f"[ERROR] PanTilt HAT init: {e}")
        shared_obj.is_exit=True; camera.stop()
        if tilt_th.is_alive(): tilt_th.join()
        sys.exit(1)

    PIXEL_DEADBAND, MISS_LIMIT = 31, int(FRAME_RATE * 0.5)
    miss_count, pid_active = 0, False
    camera_prev_hsv, recording_id = None, time.strftime('%y%m%d%H%M%S-%f')
    color_hues = {"Rose": 165}

    print(f"[INFO] Tracking. PID Setpoint: {pid.setpoint:.1f}px. Activation Threshold: {FRAME_HEIGHT/2.0:.1f}px.") 
    print(f"[INFO] Predicting {NUM_FRAMES_PREDICT_AHEAD} frames ahead. Initial Tilt: {main_loop_current_tilt} (Neg=UP). Ref Servo Cmd: {INITIAL_CAMERA_REFERENCE_TILT} (for 30deg UP phys).")

    try:
        while not shared_obj.is_exit:
            loop_start_main = time.monotonic()
            current_frame = shared_obj.frame
            if current_frame is None: time.sleep(0.001); continue

            display_frame = current_frame.copy()
            current_hsv_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2HSV)
            measurement_y = None

            if camera_prev_hsv is not None:
                try:
                    res = proc_color.process_frames(camera_prev_hsv, current_hsv_frame, display_frame, color_hues["Rose"], 10)
                    if isinstance(res, tuple) and len(res) > 0 and res[0] is not None:
                        measurement_y = float(res[0])
                        if len(res) > 1 and res[1] is not None : display_frame = res[1]
                    elif isinstance(res, (int, float)): measurement_y = float(res)
                except Exception as e: print(f"[ERROR] Detection: {e}")
            camera_prev_hsv = current_hsv_frame

            future_pred_y_for_pid = None
            if measurement_y is not None:
                miss_count = 0
                angle_for_compensation = INITIAL_CAMERA_REFERENCE_TILT - main_loop_current_tilt
                meas_true_y = measurement_y + (angle_for_compensation / deg_per_px if deg_per_px != 0 else 0)

                if not filter_initialized:
                    kf.x_hat[0,0], kf.x_hat[1,0] = meas_true_y, 0.0
                    kf.P = np.diag([kf_measurement_var, 10000.0])
                    filter_initialized = True
                    print(f"[INFO] KF Init. MeasY={measurement_y:.1f}, TrueY={meas_true_y:.1f}, Tilt={main_loop_current_tilt:.1f}")
                
                if filter_initialized:
                    kf.predict_step()
                    kf.update_step(meas_true_y)
            else:
                miss_count += 1
                if filter_initialized: kf.predict_step()
                
                if miss_count >= MISS_LIMIT and filter_initialized:
                    print(f"[INFO] Object lost ({miss_count} misses). PID deactivated, KF reset. Tilt remains at last position.")
                    pid_active = False
                    filter_initialized = False
                    kf.reset()
                    pid.reset() 
            
            if filter_initialized:
                future_pos_y, _ = kf.predict_future_state(NUM_FRAMES_PREDICT_AHEAD)
                future_pred_y_for_pid = future_pos_y
                
                activation_threshold = FRAME_HEIGHT / 2.0 
                if not pid_active and future_pred_y_for_pid is not None and future_pred_y_for_pid >= activation_threshold:
                    print(f"[INFO] PredY Fut ({future_pred_y_for_pid:.1f}) >= Threshold ({activation_threshold:.1f}). PID ON.")
                    pid_active, _ = True, pid.reset()

            if pid_active and future_pred_y_for_pid is not None:
                delta_deg = 0.0 
                if not np.isfinite(future_pred_y_for_pid):
                    print(f"[WARNING] future_pred_y_for_pid non-finite: {future_pred_y_for_pid}. KF: {kf.x_hat.flatten()}. Skipping PID.")
                elif abs(pid.setpoint - future_pred_y_for_pid) > PIXEL_DEADBAND:
                    delta_deg = pid.update(future_pred_y_for_pid)
                    if not np.isfinite(delta_deg):
                        print(f"[WARNING] delta_deg non-finite: {delta_deg}. Err: {pid.setpoint - future_pred_y_for_pid}. Reset PID.")
                        pid.reset(); delta_deg = 0.0
                
                main_loop_current_tilt = clamp(main_loop_current_tilt + delta_deg, SERVO_MIN, SERVO_MAX)
                shared_obj.current_tilt = int(round(main_loop_current_tilt))
            
            # --- Drawing ---
            cv.line(display_frame,(0,int(pid.setpoint)),(display_frame.shape[1]-1,int(pid.setpoint)),(255,0,255),1)
            cv.line(display_frame,(0,int(FRAME_HEIGHT/2.0)),(display_frame.shape[1]-1,int(FRAME_HEIGHT/2.0)),(128,128,128),1,cv.LINE_AA)
            cv.putText(display_frame,"Act Thrshld",(10,int(FRAME_HEIGHT/2.0)-5),cv.FONT_HERSHEY_SIMPLEX,0.4,(128,128,128),1)

            if filter_initialized:
                cv.circle(display_frame,(display_frame.shape[1]//2,int(kf.x_hat[0,0])),7,(0,255,0),-1)
                cv.putText(display_frame,f"KFNow:{kf.x_hat[0,0]:.0f}",(display_frame.shape[1]//2+15,int(kf.x_hat[0,0])),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
                if future_pred_y_for_pid is not None:
                    cv.circle(display_frame,(display_frame.shape[1]//2+30,int(future_pred_y_for_pid)),5,(255,100,0),-1)
                    cv.putText(display_frame,f"KFFut{NUM_FRAMES_PREDICT_AHEAD}:{future_pred_y_for_pid:.0f}",(display_frame.shape[1]//2+48,int(future_pred_y_for_pid)),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,100,0),1)
            
            if measurement_y is not None: # Corrected: Check if measurement_y is not None before drawing
                 cv.circle(display_frame,(display_frame.shape[1]//2-30,int(measurement_y)),5,(0,0,255),-1)
                 cv.putText(display_frame,f"Raw:{measurement_y:.0f}",(display_frame.shape[1]//2-100,int(measurement_y)),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            
            cv.putText(display_frame,f"PID:{'ACT' if pid_active else 'INA'}",(10,display_frame.shape[0]-10),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255) if pid_active else (255,0,0),2)
            
            cv.imshow(f'[{recording_id}] Tracking Output', display_frame)

            # --- Add frame to buffer for saving --- MOVED TO AFTER ALL DRAWING AND IMSHOW ---
            if display_frame is not None:
                 shared_obj.frame_buffer.append(display_frame.copy())

            key = cv.waitKey(1)&0xFF
            if key==ord('q'):shared_obj.is_exit=True;break
            elif key==ord('r'):print("[RST]");filter_initialized,pid_active=False,False;kf.reset();pid.reset()
            
            elapsed_main = time.monotonic()-loop_start_main
            if (sleep_main := LOOP_DT-elapsed_main) > 0:time.sleep(sleep_main) # Python 3.8+
            # For older Python:
            # sleep_main = LOOP_DT - elapsed_main
            # if sleep_main > 0: time.sleep(sleep_main)
            
    except KeyboardInterrupt: print("\n[INFO] KI: Exiting...")
    finally:
        print("[INFO] Cleaning up..."); shared_obj.is_exit=True
        camera.stop()
        if tilt_th.is_alive(): print("[INFO] Waiting for tilt thread..."); tilt_th.join(timeout=1.0)
        try:
            if 'pth' in sys.modules and not isinstance(pth, PlaceholderPanTiltHat): pth.servo_enable(2,False)
            print("[INFO] Tilt servo disabled.")
        except Exception as e: print(f"[ERROR] Servo disable: {e}")
        cv.destroyAllWindows(); print("[INFO] CV windows closed.")

        if len(shared_obj.frame_buffer) > 0:
            output_dir = os.path.join("output_frames_kf", recording_id)
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Saving {len(shared_obj.frame_buffer)} frames to {output_dir}...")
            for i, frame_to_save in enumerate(shared_obj.frame_buffer):
                if frame_to_save is not None:
                    filename = os.path.join(output_dir, f"frame_{i:06d}.png")
                    try: cv.imwrite(filename, frame_to_save)
                    except Exception as e_imwrite: print(f"[ERROR] Write frame {filename}: {e_imwrite}")
                else: print(f"[WARNING] Frame {i} in buffer None, skipping.")
            print(f"[INFO] Frames saved.")
        else: print("[INFO] No frames in buffer to save.")
        print("[INFO] Application exited.")
