import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from ctypes import c_uint8
import serial
import threading

class ArduinoHandler:
    def __init__(self):
        self.lock = threading.Lock()
        self.serial_port = 'COM3' 
        self.baud_rate = 2000000
        self.serial = serial.Serial(self.serial_port, self.baud_rate)
        self.pee_initial_speed = 0.0

    def get_pee_initial_speed(self):
        return self.pee_initial_speed

    def start_reading(self):
        threading.Thread(target=self.read_serial_data, daemon=True).start()

    def read_serial_data(self):
        while True:
            with self.lock:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    if "X," in line:
                        parts = line.split(",")
                        self.delta_x = float(parts[1]) * 0.9
                        self.delta_y = float(parts[3])

                    if "V:" in line:
                        self.pee_initial_speed = float(line.split(" ")[1])

class PPTracker:
    def __init__(self, arduino_handler, depth_camera_controller):
        self.arduino_handler = arduino_handler
        self.pee_vertical_angle_history_size = 10
        self.pee_vertical_angle_history = deque(maxlen=self.pee_vertical_angle_history_size)
        self.depth_camera_controller = depth_camera_controller
        self.origin = None
        self.pp_diff_tip = None
        self.pp_diff_bottom = None
        self.vertical_angle = None
        self.horizontal_angle = None

    def get_diff_tip_position(self):
        return self.pp_diff_tip
    
    def get_pixel_positions(self, white_pixel_centers):
        if len(white_pixel_centers) < 2:
            return None
        
        white_pixel_centers.sort(key=lambda x: x[1])

        pp_pixel_bottom = white_pixel_centers[0]  # Assume the second centroid is the bottom point
        pp_pixel_tip = white_pixel_centers[1]  # Assume the first centroid is the to` p point

        return pp_pixel_bottom, pp_pixel_tip
    
    def get_pixel_tip_position(self, white_pixel_centers):
        _ , pp_pixel_tip = self.get_pixel_positions(white_pixel_centers)
        return pp_pixel_tip

    def get_smoothed_vertical_angle(self, latest_vertical_angle):
        self.pee_vertical_angle_history.append(latest_vertical_angle)
        return sum(self.pee_vertical_angle_history) / len(self.pee_vertical_angle_history)
    
    def get_white_pixels(self, color_image):
        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Mask the bottom half of the image
        height, _ = gray_image.shape
        mask = np.zeros_like(gray_image)
        mask[:height // 2, :] = 255
        masked_gray_image = cv2.bitwise_and(gray_image, mask)

        # Thresholding to find bright spots (retroreflective tape)
        _, thresh = cv2.threshold(masked_gray_image, 252, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), c_uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh

    def get_centers_of_white_pixels(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = [cv2.moments(contour) for contour in contours if cv2.moments(contour)['m00'] != 0]
        centroids = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in centroids]
        return centroids
    
    def pixel_to_real_position(self, pixel_x, pixel_y, depth_data):
        real_z = depth_data.get_distance(pixel_x, pixel_y)
        if real_z > 0:  # Avoid division by zero and invalid depth values
            real_x = (pixel_x - depth_camera_controller.cx_d) *  real_z / depth_camera_controller.fx_d
            real_y = (pixel_y - depth_camera_controller.cy_d) *  real_z / depth_camera_controller.fy_d
            real_position = (real_x, real_y, real_z)
            return real_position
        return None
    
    
    def calculate_pp_diff_positions(self, white_pixel_centers, depth_data):
        if self.get_pixel_positions(white_pixel_centers) == None:
            return
        
        pp_pixel_bottom, pp_pixel_tip = self.get_pixel_positions(white_pixel_centers)

        self.pp_diff_tip = self.pixel_to_diff(pp_pixel_tip[0], pp_pixel_tip[1], depth_data)
        self.pp_diff_bottom = self.pixel_to_diff(pp_pixel_bottom[0], pp_pixel_bottom[1], depth_data)

    def diff_from_origin(self, real_x, real_y, real_z):
        diff_x = real_x - self.origin[0]
        diff_y = real_y - self.origin[1]
        diff_z = real_z - self.origin[2]  # Keep the Z coordinate positive
        if diff_z < 0:  # Ensure positive Z value
            diff_z = -diff_z
        return (diff_x, diff_y, diff_z)
    
    def pixel_to_diff(self, pixel_x, pixel_y, depth_data):
        if self.pixel_to_real_position(pixel_x, pixel_y, depth_data):
            real_x, real_y, real_z = self.pixel_to_real_position(pixel_x, pixel_y, depth_data)
            return self.diff_from_origin(real_x, real_y, real_z)
        return None
    
    def calculate_pp_orientation(self):

        if self.pp_diff_tip and self.pp_diff_bottom:

            pp_orientation = np.array(self.pp_diff_tip) - np.array(self.pp_diff_bottom)
            diff_x = pp_orientation[0]
            diff_y = pp_orientation[1]
            diff_z = pp_orientation[2]

            # Calculate the angles
            raw_vertical_angle = np.degrees(np.arctan2(diff_z, np.sqrt(diff_x**2 + diff_y**2)))

            self.horizontal_angle = np.degrees(np.arctan2(diff_y,diff_x))
            self.vertical_angle = self.get_smoothed_vertical_angle(raw_vertical_angle)

    def set_origin(self, real_x, real_y, real_z):
        self.origin = (real_x, real_y, real_z)

    def angle_text(self):
        if self.vertical_angle == None or self.horizontal_angle == None:
            return f"Vertical Angle: N/A deg\nHorizontal Angle: N/A deg"
        return f"Vertical Angle: {self.vertical_angle:.2f} deg\nHorizontal Angle: {self.horizontal_angle:.2f} deg"
    
    def coord_text(self):
        if self.pp_diff_tip == None:
            return f"Bottle Tip: (N/A, N/A, N/A)"
        return f"Bottle Tip: ({self.pp_diff_tip[0]:.2f}, {self.pp_diff_tip[1]:.2f}, {self.pp_diff_tip[2]:.2f})"
    
    def get_initial_speed(self):
        return self.arduino_handler.get_pee_initial_speed()
    
    def velocity_text(self):
        return f"Velocity: {self.get_initial_speed():.2f} m/s"


class DepthCameraController:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = None
        self.depth_intrinsics = None
        self.fx_d, self.fy_d = None, None
        self.cx_d, self.cy_d = None, None
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
    
    def set_intrinsics(self):
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.fx_d, self.fy_d = self.depth_intrinsics.fx, self.depth_intrinsics.fy
        self.cx_d, self.cy_d = self.depth_intrinsics.ppx, self.depth_intrinsics.ppy

    def start_camera(self):
        self.profile = self.pipeline.start(self.config)
        self.set_intrinsics()

    def get_depth_and_color_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_data = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_data or not color_frame:
            return None
        depth_image = np.asanyarray(depth_data.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_data, depth_image, color_image
    
class ModeTracker:
    def __init__(self):
        self.mode_list = ['origin', 'track']
        self.mode = self.mode_list[0]

    def set_to_origin(self):
        self.mode = self.mode_list[0]

    def set_to_track(self):
        self.mode = self.mode_list[1]
    
    def is_tracking(self):
        return self.mode == self.mode_list[1]
    
class UIHandler:
    def __init__(self, pp_tracker, mode_tracker):
        self.pp_tracker = pp_tracker
        self.mode_tracker = mode_tracker
        cv2.namedWindow('Human Vision')
        cv2.namedWindow('Reflective Tape')


    def display_feed(self, color_image, thresh):
        cv2.imshow('Human Vision', color_image)
        cv2.imshow('Reflective Tape', thresh)
    
    def draw_origin_text(self, color_image):
        cv2.putText(color_image, "Valid origin! Press 'o' to set origin point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def draw_invalid_origin_text(self, color_image, count):
        cv2.putText(color_image, f'Invalid number of points detected: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def draw_tracking_ui(self, color_image, white_pixel_centers):

        angle_text = self.pp_tracker.angle_text()
        y0, dy = 15, 15
        for i, line in enumerate(angle_text.split('\n')):
            y = y0 + i*dy 
            cv2.putText(color_image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the 3D coordinates on the frame
        coord_text = self.pp_tracker.coord_text()
        cv2.putText(color_image, coord_text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the velocity on the frame
        velocity_text = self.pp_tracker.velocity_text()
        cv2.putText(color_image, velocity_text, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.pp_tracker.pp_diff_tip and self.pp_tracker.pp_diff_bottom:
            cv2.circle(color_image, self.pp_tracker.get_pixel_tip_position(white_pixel_centers), 5, (0, 255, 0), -1)


try:
    arduino_handler = ArduinoHandler()
    arduino_handler.start_reading()
    depth_camera_controller = DepthCameraController()
    depth_camera_controller.start_camera()
    pp_tracker = PPTracker(arduino_handler, depth_camera_controller)
    mode_tracker = ModeTracker()
    ui_handler = UIHandler(pp_tracker, mode_tracker)

    while True:
        depth_data, depth_image, color_image = depth_camera_controller.get_depth_and_color_frames()
        white_pixels = pp_tracker.get_white_pixels(color_image)
        white_pixel_centers = pp_tracker.get_centers_of_white_pixels(white_pixels)
        white_pixel_center_count = len(white_pixel_centers)

        if not mode_tracker.is_tracking():
            if white_pixel_center_count == 1:
                origin_pixel_x, origin_pixel_y = white_pixel_centers[0]
                origin_point = pp_tracker.pixel_to_real_position(origin_pixel_x, origin_pixel_y, depth_data)

                ui_handler.draw_origin_text(color_image)

                if cv2.waitKey(1) & 0xFF == ord('o'):
                        pp_tracker.set_origin(origin_point[0], origin_point[1], origin_point[2])
                        mode_tracker.set_to_track()
            else:
                ui_handler.draw_invalid_origin_text(color_image, white_pixel_center_count)

        elif white_pixel_center_count == 2:
            pp_tracker.calculate_pp_diff_positions(white_pixel_centers, depth_data)
            pp_tracker.calculate_pp_orientation()
            ui_handler.draw_tracking_ui(color_image, white_pixel_centers)

        ui_handler.display_feed(color_image, white_pixels)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
