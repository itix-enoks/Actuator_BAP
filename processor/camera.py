import time

import cv2 as cv

from threading import Thread
from picamera2 import Picamera2


class SharedObject:
    current_tilt = -30.0

    frame = None
    frame_buffer = []

    is_exit: bool = False


class CameraStream:
    def __init__(self, shared_obj, width=1332, height=990):
        self.shared = shared_obj
        self.width = width
        self.height = height
        self.picam2 = Picamera2()
        self._configure()
        self.thread = Thread(target=self._update_frames, daemon=True)  # Note that we do not use concurrent.futures here
        self.frame_count = 0

    def _configure(self):
        fastmode = self.picam2.sensor_modes[0]
        config = self.picam2.create_preview_configuration(
            sensor={'output_size': fastmode['size'], 'bit_depth': fastmode['bit_depth']},
            controls={"FrameDurationLimits": (8333, 8333)}
        )
        self.picam2.configure(config)

    def start(self):
        self.picam2.start()
        time.sleep(1)
        self.thread.start()  # As noted above; initiate this thread here instead of `run_tasks_in_parallel`

    def _update_frames(self):
        while not self.shared.is_exit:
            frame = cv.cvtColor(self.picam2.capture_array("main"), cv.COLOR_BGR2RGB)
            rotated = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            self.shared.frame = rotated
            self.frame_count += 1
            self.shared.frame_buffer.append(rotated)

    def stop(self):
        self.picam2.stop()
