import numpy as np
import math


class EKFTracker:
    """
    Extended Kalman Filter for vertical drop tracking with nonlinear drag.
    State: [position; velocity] in pixels and px/s.
    Process:
      p_{k+1} = p_k + v_k*dt
      v_{k+1} = v_k + (g_pix - c_over_m_pix * v_k * |v_k|)*dt
    Measurement:
      y_k = pixel position corrected for camera tilt.
    """
    def __init__(self, dt, c_over_m_pix, g_pix, process_var, meas_var, deg_per_px, px_per_m):
        self.dt         = dt
        self.c_over_m   = c_over_m_pix
        self.g          = g_pix
        self.deg_per_px = deg_per_px
        self.px_per_m   = px_per_m

        # state & covariance
        self.x = np.zeros((2,1))           # [p; v]
        self.P = np.eye(2) * 100.0         # initial uncertainty

        # process noise covariance Q (approximation)
        dt2 = dt*dt
        dt3 = dt2*dt
        dt4 = dt2*dt2
        self.Q = process_var * np.array([[dt4/4, dt3/2],
                                         [dt3/2,    dt2 ]])

        # measurement noise covariance
        self.R = np.array([[meas_var]])

    def predict(self):
        p, v = self.x.flatten()
        # nonlinear accel with drag in pixel domain
        a = self.g - self.c_over_m * v * abs(v)

        # state prediction
        p_pred = p + v * self.dt
        v_pred = v + a * self.dt
        self.x = np.array([[p_pred], [v_pred]])

        # Jacobian of process model
        F_j = np.array([
            [1,             self.dt],
            [0, 1 - 2*self.c_over_m*abs(v)*self.dt]
        ])

        # covariance prediction
        self.P = F_j.dot(self.P).dot(F_j.T) + self.Q
        return self.x.copy()

    def update(self, meas_y, camera_angle_deg):
        # compensate measurement for camera tilt (deg → px)
        meas_true = meas_y + (camera_angle_deg / self.deg_per_px)

        # measurement model
        H = np.array([[1, 0]])
        z_pred = H.dot(self.x)

        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        # state update
        y_resid = meas_true - z_pred
        self.x  = self.x + K.dot(y_resid)
        self.P  = (np.eye(2) - K.dot(H)).dot(self.P)
        return self.x.copy()

    def reset(self):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 100.0

    def get_state(self):
        # return position and velocity
        p_px = float(self.x[0])
        v_px_s = float(self.x[1])
        v_m_s = v_px_s / self.px_per_m
        return p_px, v_m_s

if __name__ == '__main__':
    # Camera & scene parameters
    FRAME_HEIGHT      = 1332   # image height in pixels
    SENSOR_HEIGHT_MM  = 4.712   # Pi HQ camera sensor height in mm
    FOCAL_LENGTH_MM   = 32      # lens focal length in mm
    DROP_DIST_M       = 2.5     # approximate distance from camera to drop in meters

    # Physical parameters
    G_M_S2    = 9.81           # gravity in m/s²
    C_OVER_M  = 0.005            # drag coefficient over mass in 1/m

    # 1) vertical field-of-view (radians)
    vfov_rad = 2 * math.atan(SENSOR_HEIGHT_MM / (2 * FOCAL_LENGTH_MM))

    # 2) physical height (m) of that FOV at your drop distance
    world_h_m = 2 * DROP_DIST_M * np.tan(vfov_rad / 2)
  
    # 3) pixels per meter in the image
    px_per_m = FRAME_HEIGHT / world_h_m

    #  4) convert into pixel units
    g_pix         = G_M_S2    * px_per_m      # px/s²
    c_over_m_pix  = C_OVER_M  

    # 5) degrees per pixel (for tilt‐compensation & display)
    vfov_deg   = np.degrees(vfov_rad)
    deg_per_px = vfov_deg / FRAME_HEIGHT

    FRAME_RATE = 70.0
    dt = 1.0 / FRAME_RATE

    # instantiate EKF with converted pixel‐domain parameters
    ekf = EKFTracker(
        dt=dt,
        c_over_m_pix=c_over_m_pix,
        g_pix=g_pix,
        process_var=50.0,     # tune this
        meas_var=16.0,        # tune this
        deg_per_px=deg_per_px,
        px_per_m=px_per_m
    )

