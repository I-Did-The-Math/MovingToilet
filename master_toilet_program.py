import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from ctypes import c_uint8
import serial
import threading
import time
import math

class ToiletController:
    def __init__(self, arduino_handler):
        self.arduino_handler = arduino_handler
        self.real_pos = [0.0,0.0]
        self.init_pos = [0.0,0.0]
        self.last_pixel_pos = [0,0]
        self.gopro_velocity = 0.0
        self.pixel_pos_history = []
        self.last_velocity_check_time = time.time()
        self.just_stopped = False
        self.just_stopped_time = None
        self.pos_synced = True

        self.target_x = 0
        self.target_y = 0
        self.speedAdjustmentFactor = 0

        self.movementInProgress = False
        

        # Coordinates in mm
        self.real_coords = np.array([
            [690, 840],
            [250, 840],
            [250, 360],
            [690, 360]
        ])

        # Coordinates in pixels
        self.pixel_coords = np.array([
            [795, 492],
            [505, 492],
            [492, 141],
            [798, 142]
        ])

        self.movement_bounds = np.array(self.real_coords, np.int32).reshape((-1, 1, 2))
        self.H, _ = cv2.findHomography(self.pixel_coords, self.real_coords)
        self.H_inv = np.linalg.inv(self.H)  #Inverse homography matrix for real world to pixel conversion

    def real_point_inside_bounds(self, x, y):
        return cv2.pointPolygonTest(self.movement_bounds, (x,y), False) >= 0

    def get_encoder_velocity(self):
        self.arduino_handler

    def update_velocity(self, gopro_image):
        current_time = time.time()
        time_diff = current_time - self.last_velocity_check_time
        pixel_x, pixel_y, _ = self.get_pixel_position(gopro_image)
        if time_diff >= 0.1:
            vel_x = (pixel_x - self.last_pixel_pos[0]) / time_diff
            vel_y = (pixel_y - self.last_pixel_pos[1]) / time_diff

            self.gopro_velocity = math.sqrt(vel_x**2 + vel_y**2)

            self.last_pixel_pos[0] = pixel_x
            self.last_pixel_pos[1] = pixel_y
            self.last_velocity_check_time = current_time

    def resync_position(self, gopro_image):
        print('synced position')
        self.set_initial_position(gopro_image)
        self.arduino_handler.reset_encoders()
        self.update_real_position()

    def set_initial_position(self, gopro_image):
        pixel_x, pixel_y, _ = self.get_pixel_position(gopro_image)
        real_x, real_y = self.pixel_to_real(pixel_x, pixel_y)
        self.init_pos[0] = real_x
        self.init_pos[1] = real_y

    def update_real_position(self):
        delta_x, delta_y = self.arduino_handler.get_deltas()
        self.real_pos[0] = self.init_pos[0] + delta_x
        self.real_pos[1] = self.init_pos[1] + delta_y

    def get_real_position(self):
        self.update_real_position()
        return self.real_pos[0], self.real_pos[1]

    def center_of_white_pixels(self,gopro_image):
        gray_image = cv2.cvtColor(gopro_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh[:50, :] = 0
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, thresh  # No contours found
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        white_pixels = np.argwhere(mask == 255)
        center = np.mean(white_pixels, axis=0)
        center = (int(center[1]), int(center[0]))  # (x, y) format
        return center, mask
    
    def get_pixel_position(self,gopro_image):
        center_toilet_pixel, thresh = self.center_of_white_pixels(gopro_image)
        if center_toilet_pixel is not None:
            return center_toilet_pixel[0] + 10, center_toilet_pixel[1] - 140, thresh
        
    def pixel_to_real(self, pixel_x, pixel_y):
        pixel_point = np.array([pixel_x, pixel_y, 1]).reshape(-1, 1)
        real_world_point = np.dot(self.H, pixel_point)
        real_world_point /= real_world_point[2]  # Normalize by the third (homogeneous) coordinate
        return real_world_point[0, 0], real_world_point[1, 0]
    
    def real_to_pixel(self, real_x, real_y):
        real_world_point = np.array([real_x, real_y, 1]).reshape(-1, 1)
        pixel_point = np.dot(self.H_inv, real_world_point)
        pixel_point /= pixel_point[2]  # Normalize by the third (homogeneous) coordinate
        return int(pixel_point[0, 0]), int(pixel_point[1, 0])
        
    def move_relative(self, delta_x, delta_y):
        self.arduino_handler.move(delta_x, delta_y)

    def convert_m_to_mm(self, x):
            return x * 1000

    def move_to_position(self, target_x, target_y):
        current_x, current_y = self.get_real_position()
        delta_x = target_x - current_x
        delta_y = target_y - current_y

        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

        if distance_to_target > 15: 
            self.move_relative(delta_x, delta_y)
            return True
        return False

    def is_still(self):
        still_threshold = 15
        still_duration_threshold = 0.3

        if self.gopro_velocity < still_threshold:
            if not self.just_stopped:
                self.just_stopped = True
                self.just_stopped_time = time.time()
            elif time.time() - self.just_stopped_time > still_duration_threshold:
                    return True
        else:
            self.just_stopped = False
            self.just_stopped_time = None

        return False

    def move_to_position_relative(self,move_x, move_y):
        current_delta_x, current_delta_y = self.arduino_handler.get_deltas()
        target_x = current_delta_x + move_x
        target_y = current_delta_y + move_y

        angle = math.atan2(abs(move_y), abs(move_x))
        normalizedAngle = abs(math.sin(2 * angle))
        speedAdjustmentFactor = 0.8 + 0.2 * normalizedAngle

        self.target_x = target_x
        self.target_y = target_y
        self.speedAdjustmentFactor = speedAdjustmentFactor

        print("New movement started: deltaX=")
        print(move_x)
        print(", deltaY=")
        print(move_y)

        self.movementInProgress = True

    def initialize_PID_thread(self):
        threading.Thread(target=self.executePIDControl, daemon=True).start()

    def executePIDControl(self):
        slowDown = False
        Kp = 0.01
        initialMinPower = 50
        minPower = initialMinPower
        maxPower = 60
        closeEnoughThreshold = 2
        slowEnoughThreshold = 50
        movement_loop_interval = 0.01
        last_iteration_time = time.time()

        while True: 
            if self.movementInProgress and time.time() - last_iteration_time >= movement_loop_interval:
                last_iteration_time = time.time()
                current_delta_x, current_delta_y = self.arduino_handler.get_deltas()
                velocity = self.arduino_handler.get_velocity()

                errorX = self.target_x - current_delta_x
                errorY = self.target_y - current_delta_y
                distance_left_to_travel = math.sqrt(errorX * errorX + errorY * errorY)

                output = Kp * distance_left_to_travel * self.speedAdjustmentFactor

                # Normalize the direction vector
                directionX = errorX / distance_left_to_travel
                directionY = errorY / distance_left_to_travel
                
                #determine breaking
                slowDownDistance = velocity * velocity / 7500
                
                if (distance_left_to_travel < slowDownDistance) and slowDown == False:
                    slowDown = True

                speedMultiplier = 1

                if slowDown:
                    self.speedAdjustmentFactor = 1
                    minPower = 15
                    if velocity > 300:
                        speedMultiplier = 0

                # cos will be 1 for orthogonal, 0 for diagonal
                speed = 0
                output += minPower

                def constrain(value, min_value, max_value):
                    return max(min_value, min(value, max_value))
                
                speed =  speedMultiplier * constrain(output, minPower, maxPower) * self.speedAdjustmentFactor
                minPower = initialMinPower
                
                self.arduino_handler.move_direction(directionX, directionY, speed)
                
                #Check if the target position is reached
                if distance_left_to_travel < closeEnoughThreshold and velocity < slowEnoughThreshold: # Adjust the threshold as needed
                    slowDown = False
                    self.arduino_handler.move_direction(0, 0, 0) # Stop the motors
                    self.movementInProgress = False
                    print("done")
            time.sleep(0.01)
        
class ArduinoHandler:
    def __init__(self):
        self.lock = threading.Lock()

        self.serial_port = 'COM3' 
        self.baud_rate = 2000000
        self.serial = serial.Serial(self.serial_port, self.baud_rate)


        #in seconds
        self.last_velocity_check = time.time()
        self.velocity_update_interval = 0.001

        #in mm
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.last_delta_x = 0.0
        self.last_delta_y = 0.0
        self.total_distance = 0.0
        

        #in mm/s
        self.velocity = 0.0
        
        #in m/s
        self.pee_initial_speed = 0.0


    def get_deltas(self):
        return self.delta_x, self.delta_y

    def get_pee_initial_speed(self):
        return self.pee_initial_speed

    def get_velocity(self):
        return self.velocity

    def start_reading(self):
        threading.Thread(target=self.read_serial_data, daemon=True).start()

    def start_checking_velocity(self):
        threading.Thread(target=self.update_velocity, daemon=True).start()

    def update_velocity(self):
        while True:
            if time.time() - self.last_velocity_check >= self.velocity_update_interval:
                time_delta = time.time() - self.last_velocity_check

                current_delta_x, current_delta_y = self.get_deltas()

                diff_x = current_delta_x - self.last_delta_x
                diff_y = current_delta_y - self.last_delta_y

                delta_increment_distance = math.sqrt(math.pow(abs(diff_x),2) + math.pow(abs(diff_y),2))
            
                self.velocity = delta_increment_distance / time_delta

                self.last_delta_x = current_delta_x
                self.last_delta_y = current_delta_y
            time.sleep(0.001)

    def read_serial_data(self):
        while True:
            with self.lock:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    #format from arduino: X,[deltaX],Y,[deltaY]
                    if "X," in line:
                        parts = line.split(",")

                        last_delta_x = self.delta_x
                        last_delta_y = self.delta_y
                        
                        self.delta_x = float(parts[1]) * 0.9
                        self.delta_y = float(parts[3])

                        delta_x_increment = self.delta_x - last_delta_x
                        delta_y_increment = self.delta_y - last_delta_y
                        delta_x_abs_increment = abs(delta_x_increment)
                        delta_y_abs_increment = abs(delta_y_increment)

                        total_movement_increment = delta_x_abs_increment + delta_y_abs_increment
                        self.total_distance += total_movement_increment

                        print(f'deltaX: {self.delta_x} deltaY: {self.delta_y} totalDistance: {self.total_distance}')
                    if "V:" in line:
                        self.pee_initial_speed = float(line.split(" ")[1])
            time.sleep(0.001)
                    
    def reset_encoders(self):
        with self.lock:
            self.serial.write(b"RESET_ENCODERS\n")
            print("Encoders Set To Zero")
            time.sleep(0.1)
            self.delta_x = 0
            self.delta_y = 0
            self.total_distance = 0

    def move(self,delta_x, delta_y):
        with self.lock:
            command = f"{delta_x},{delta_y}\n"
            self.serial.write(command.encode())
            print(f"Sent: {command}")

    def move_direction(self, direction_x, direction_y, speed):
        with self.lock:
            command = f'{direction_x},{direction_y},{speed}'
            self.serial.write(command.encode())
            print(f"Sent: {command}")

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
        self.toilet_height_m = 0.33
        self.pee_speed_history_size = 10
        self.pee_speed_history = deque(maxlen=self.pee_speed_history_size)

    def get_horizontal_angle(self):
        return self.horizontal_angle

    def get_vertical_angle(self):
        return self.vertical_angle
    
    def get_smoothed_vertical_angle(self, latest_vertical_angle):
        self.pee_vertical_angle_history.append(latest_vertical_angle)
        return sum(self.pee_vertical_angle_history) / len(self.pee_vertical_angle_history)
    
    def get_initial_speed(self):
        return self.arduino_handler.get_pee_initial_speed()
    
    def get_smoothed_speed(self, latest_speed):
        self.pee_speed_history.append(latest_speed)
        return sum(self.pee_speed_history) / len(self.pee_speed_history)

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
    
    def get_white_pixels(self, color_image):
        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Mask the bottom half of the image
        height, _ = gray_image.shape
        mask = np.zeros_like(gray_image) 
        mask[:height // 2, :] = 255
        masked_gray_image = cv2.bitwise_and(gray_image, mask)

        # Thresholding to find bright spots (retroreflective tape)
        _, thresh = cv2.threshold(masked_gray_image, 251, 255, cv2.THRESH_BINARY)
        
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

    def get_depth_of_pixel(self, pixel_x, pixel_y, depth_data):
        return depth_data.get_distance(pixel_x, pixel_y)
    
    def real_to_pixel_position(self, diff_x, diff_y, diff_z):
        if self.origin is None:
            print("Error: Origin is not set. Cannot convert real-world coordinates to pixel coordinates.")
            return None

        # Add the origin back to get the real-world coordinates
        real_x = diff_x + self.origin[0]
        real_y = diff_y + self.origin[1]
        real_z = diff_z + self.origin[2]

        # Check if the depth (z) is valid
        if real_z <= 0:
            print(f"Invalid depth (real_z): {real_z}. Cannot compute pixel coordinates.")
            return None

        # Convert real-world coordinates to pixel coordinates using camera intrinsics
        pixel_x = int((real_x * self.depth_camera_controller.fx_d) / real_z + self.depth_camera_controller.cx_d)
        pixel_y = int((real_y * self.depth_camera_controller.fy_d) / real_z + self.depth_camera_controller.cy_d)

        return pixel_x, pixel_y

    def is_any_point_missing(self):
        return self.pp_diff_tip == None or self.pp_diff_bottom == None
    
    def pixel_to_real_position(self, pixel_x, pixel_y, depth_data):
        real_z = self.get_depth_of_pixel(pixel_x, pixel_y, depth_data)

        if real_z > 0:  # Avoid division by zero and invalid depth values
            real_x = (pixel_x - depth_camera_controller.cx_d) *  real_z / depth_camera_controller.fx_d
            real_y = (pixel_y - depth_camera_controller.cy_d) *  real_z / depth_camera_controller.fy_d
            real_position = (real_x, real_y, real_z)
            return real_position
        return None
    
    def calculate_pp_diff_positions(self, white_pixel_centers, depth_data):
        
        pp_pixel_positions = self.get_pixel_positions(white_pixel_centers)
        if pp_pixel_positions == None:
            return None
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
        return f"Bottle Tip: ({self.pp_diff_tip[0]:.2f}m, {self.pp_diff_tip[1]:.2f}m, {self.pp_diff_tip[2]:.2f}m)"
    
    def velocity_text(self):
        return f"Velocity: {self.get_initial_speed():.2f} m/s"
    
    def calculate_landing_location_at_toilet_height(self, diff_x, diff_y, diff_z, initial_speed, horizontal_angle, vertical_angle):
        g = 9.81

        # Calculate initial height above the ground
        start_height_from_toilet = diff_z - self.toilet_height_m

        #adjust initial speed
        initial_speed *= 0.35

        # Decompose initial velocity into components
        v_x = initial_speed * math.cos(math.radians(vertical_angle)) * math.cos(math.radians(horizontal_angle))
        v_y = initial_speed * math.cos(math.radians(vertical_angle)) * math.sin(math.radians(horizontal_angle))
        v_z = initial_speed * math.sin(math.radians(vertical_angle))
        #ensure v_z is oriented correctly, and negative v_z is equal to going down

        # Solve for time of flight using quadratic formula
        # z(t) = z0 + v_z * t - 0.5 * g * t^2

        #arranged position formula to a quadratic equation with format x(t) = at^2 + bt + c
        a = -0.5 * g
        b = v_z
        c = start_height_from_toilet

        # Quadratic formula: t = (-b ± sqrt(b^2 - 4ac)) / 2a
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            print("No real solution: the object will not hit the ground.")
            return None

        elapsed_time = (-b + math.sqrt(discriminant)) / (2 * a)  # Take the positive root
        current_time = time.time()
        landing_time = current_time + elapsed_time

        # Calculate landing position
        x_land = diff_x - v_x * elapsed_time
        y_land = diff_y - v_y * elapsed_time

        # Return the landing position and time
        return x_land, y_land, landing_time

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

    def is_setting_origin(self):
        return self.mode == self.mode_list[0]
    
class UIHandler:
    def __init__(self, pp_tracker, mode_tracker, toilet_controller):
        self.pp_tracker = pp_tracker
        self.mode_tracker = mode_tracker
        cv2.namedWindow('PP: Human Vision')
        cv2.namedWindow('PP: Computer Vision')
        self.cursor_pixel_pos = [0,0]
        self.toilet_controller = toilet_controller
        self.move_requested = False
        cv2.namedWindow('Toilet: Human Vision')
        cv2.setMouseCallback('Toilet: Human Vision', self.mouse_callback)
        self.cursor_inside_bounds = False
        self.polygon = np.array(toilet_controller.pixel_coords, np.int32).reshape((-1, 1, 2))
        self.direction = [0,0]

    def display_pp_feed(self, color_image, thresh):
        cv2.imshow('PP: Human Vision', color_image)
        cv2.imshow('PP: Computer Vision', thresh)
    
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

    def update_cursor_pos(self, x, y):
        self.cursor_pixel_pos[0] = x * 2
        self.cursor_pixel_pos[1] = y * 2

        real_x, real_y = toilet_controller.pixel_to_real(self.cursor_pixel_pos[0], self.cursor_pixel_pos[1])
        self.cursor_inside_bounds = cv2.pointPolygonTest(toilet_controller.movement_bounds, (real_x, real_y ), False) >= 0

    def get_real_cursor_pos(self):
        return self.toilet_controller.pixel_to_real(self.cursor_pixel_pos[0], self.cursor_pixel_pos[1])

    def get_toilet_cursor_real_deltas(self):
        cursor_real_pos_x, cursor_real_pos_y = self.get_real_cursor_pos()
        toilet_real_pos_x, toilet_real_pos_y = self.toilet_controller.get_real_position()
        cursor_real_delta_x = cursor_real_pos_x - toilet_real_pos_x
        cursor_real_delta_y = cursor_real_pos_y - toilet_real_pos_y
        return cursor_real_delta_x, cursor_real_delta_y
    
    def move_toilet_to_mouse(self):
        cursor_real_delta_x, cursor_real_delta_y = self.get_toilet_cursor_real_deltas()
        if self.cursor_inside_bounds:
            self.toilet_controller.move_to_position_relative(cursor_real_delta_x, cursor_real_delta_y)    

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.update_cursor_pos(x,y)
        if event == cv2.EVENT_LBUTTONDOWN:  
            self.move_toilet_to_mouse()

    def display_toilet_feed(self, frame, thresh):
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Toilet: Human Vision', frame_resized)
        # Display the thresholded image
        cv2.imshow('Toilet: Computer Vision', thresh_resized)

    def render_outdated_toilet_pos(self, gopro_image, render_data):
        cv2.circle(gopro_image, (render_data['toilet_gopro_pixel_x'], render_data['toilet_gopro_pixel_y']), 5, (0, 255, 0), -1)
        text_camera = f"Outdated Position: ({render_data['toilet_gopro_real_x']:.2f} mm, {render_data['toilet_gopro_real_y']:.2f} mm)"
        cv2.putText(gopro_image, text_camera, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        text_camera_pixel = f"Pixel Position: ({render_data['toilet_gopro_pixel_x']:.2f} px, {render_data['toilet_gopro_pixel_y']:.2f} px)"
        cv2.putText(gopro_image, text_camera_pixel, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
    
    def render_correct_toilet_pos(self, gopro_image, render_data):
        cv2.circle(gopro_image, (render_data['toilet_pixel_x'], render_data['toilet_pixel_y']), 5, (255, 0, 0), -1)
        text_real = f"Toilet In Real Life: ({render_data['toilet_real_x']:.2f} mm, {render_data['toilet_real_y']:.2f} mm)"
        cv2.putText(gopro_image, text_real, (render_data['toilet_pixel_x'] + 20, render_data['toilet_pixel_y'] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

    def render_arrow(self, gopro_image, render_data):
        arrow_color = (255, 0 , 0) if self.cursor_inside_bounds else (0, 0, 255)
        cursor_x = self.cursor_pixel_pos[0]
        cursor_y = self.cursor_pixel_pos[1]
        cv2.arrowedLine(gopro_image, (render_data['toilet_pixel_x'], render_data['toilet_pixel_y']), (cursor_x, cursor_y), arrow_color, 2)

    def compute_toilet_render_data(self, gopro_image, landing_real_x, landing_real_y):
        toilet_gopro_pixel_x, toilet_gopro_pixel_y, thresh = self.toilet_controller.get_pixel_position(gopro_image)
        toilet_gopro_real_x, toilet_gopro_real_y = self.toilet_controller.pixel_to_real(toilet_gopro_pixel_x, toilet_gopro_pixel_y)
        
        toilet_real_x, toilet_real_y = self.toilet_controller.get_real_position()
        toilet_pixel_x, toilet_pixel_y = self.toilet_controller.real_to_pixel(toilet_real_x, toilet_real_y)

        if landing_real_x == None or landing_real_y == None:
            landing_pixel_x, landing_pixel_y = None, None
        else:
            #ensure landing_real units are in mm
            landing_real_x *= 1000
            landing_real_y *= 1000
            #because toilet_controller can convert real position to pixel position if the real position is at the toilet height, use the toilet real to pixel function
            landing_pixel_x, landing_pixel_y = self.toilet_controller.real_to_pixel(landing_real_x, landing_real_y)

        render_data = {
            'toilet_gopro_pixel_x': toilet_gopro_pixel_x,
            'toilet_gopro_pixel_y':toilet_gopro_pixel_y,
            'toilet_gopro_real_x': toilet_gopro_real_x,
            'toilet_gopro_real_y': toilet_gopro_real_y,
            'toilet_real_x': toilet_real_x,
            'toilet_real_y': toilet_real_y,
            'toilet_pixel_x': toilet_pixel_x,
            'toilet_pixel_y': toilet_pixel_y,
            'thresh': thresh,
            'landing_pixel_x': landing_pixel_x,
            'landing_pixel_y': landing_pixel_y,
            'landing_real_x': landing_real_x,
            'landing_real_y': landing_real_y
        }

        return render_data

    def render_toilet_bounds(self, gopro_image):
        cv2.polylines(gopro_image, [self.polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    def render_landing_location(self, gopro_image, render_data):
        if render_data['landing_real_x'] == None or render_data['landing_real_y'] == None:
            return
        
        cv2.circle(gopro_image, (render_data['landing_pixel_x'], render_data['landing_pixel_y']), 5, (0, 255, 255), -1)
        text_real = f"Predicted Landing Pos: ({render_data['landing_real_x']:.2f} m, {render_data['landing_real_y']:.2f} m at )"
        cv2.putText(gopro_image, text_real, (render_data['landing_pixel_x'] + 20, render_data['landing_pixel_y'] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

    def draw_toilet_ui(self, gopro_image, landing_real_x, landing_real_y):
        render_data = self.compute_toilet_render_data(gopro_image, landing_real_x, landing_real_y)

        self.render_outdated_toilet_pos(gopro_image, render_data)
        self.render_correct_toilet_pos(gopro_image, render_data)
        self.render_arrow(gopro_image, render_data)
        self.render_toilet_bounds(gopro_image)
        self.render_landing_location(gopro_image, render_data)
        self.display_toilet_feed(gopro_image, render_data['thresh'])

try:
    obs_virtual_camera_index = 3 
    gopro_feed = cv2.VideoCapture(obs_virtual_camera_index, cv2.CAP_DSHOW)
    if not gopro_feed.isOpened():
        print(f"Error: Could not open virtual camera at index {obs_virtual_camera_index}.")
        exit()
    print(f"Successfully opened virtual camera at index {obs_virtual_camera_index}.")

    gopro_feed_width = 1280
    gopro_feed_height = 720
    gopro_feed.set(cv2.CAP_PROP_FRAME_WIDTH, gopro_feed_width)
    gopro_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, gopro_feed_height)
    ret, gopro_image = gopro_feed.read()

    depth_camera_controller = DepthCameraController()
    depth_camera_controller.start_camera()
    
    arduino_handler = ArduinoHandler()
    arduino_handler.start_reading()
    arduino_handler.start_checking_velocity()

    toilet_controller = ToiletController(arduino_handler)
    toilet_controller.initialize_PID_thread()
    pp_tracker = PPTracker(arduino_handler, depth_camera_controller)
    
    toilet_controller.resync_position(gopro_image)
    recently_stopped = False
    toilet_awaiting_movement_command = False
    
    mode_tracker = ModeTracker()

    ui_handler = UIHandler(pp_tracker, mode_tracker, toilet_controller)

    while True:
        #------------------------------------------------------capture PP data

        depth_data, depth_image, color_image = depth_camera_controller.get_depth_and_color_frames()
        white_pixels = pp_tracker.get_white_pixels(color_image)
        white_pixel_centers = pp_tracker.get_centers_of_white_pixels(white_pixels)
        white_pixel_center_count = len(white_pixel_centers)

        landing_x, landing_y, landing_time = None, None, None

        if mode_tracker.is_setting_origin():
            if white_pixel_center_count == 1:
                origin_pixel_x, origin_pixel_y = white_pixel_centers[0]
                origin_point = pp_tracker.pixel_to_real_position(origin_pixel_x, origin_pixel_y, depth_data)

                ui_handler.draw_origin_text(color_image)

                if cv2.waitKey(1) & 0xFF == ord('o'):
                        pp_tracker.set_origin(origin_point[0], origin_point[1], origin_point[2])
                        mode_tracker.set_to_track()
            else:
                ui_handler.draw_invalid_origin_text(color_image, white_pixel_center_count)

        elif mode_tracker.is_tracking():
            if white_pixel_center_count == 2:
                pp_tracker.calculate_pp_diff_positions(white_pixel_centers, depth_data)
                if pp_tracker.is_any_point_missing():
                    print("ERROR: missing at least one point even though white_pixel_center_count is 2")
                    continue

                #grab pp variables needed to calculate pp trajectory
                pp_tracker.calculate_pp_orientation()
                pp_diff_x, pp_diff_y, pp_diff_z = pp_tracker.get_diff_tip_position()
                pp_initial_speed = pp_tracker.get_initial_speed()
                pp_smoothed_speed = pp_tracker.get_smoothed_speed(pp_initial_speed)
                pp_vertical_angle = pp_tracker.get_vertical_angle()
                pp_horizontal_angle = pp_tracker.get_horizontal_angle()

                #predict pp trajectory
                pp_landing_info = pp_tracker.calculate_landing_location_at_toilet_height(pp_diff_x, pp_diff_y, pp_diff_z, pp_smoothed_speed, pp_horizontal_angle, pp_vertical_angle)

                if pp_landing_info:
                    landing_x, landing_y, landing_time = pp_landing_info
                else:
                    print("ERROR: error while calculating PP trajectory")

                ui_handler.draw_tracking_ui(color_image, white_pixel_centers)
            else:
                print("ERROR: expected 2 points but found a different amount, unable to track pp location")

        ui_handler.display_pp_feed(color_image, white_pixels)

        #----------------------------------------------move toilet

        ret, gopro_image = gopro_feed.read()

        if not ret:
            print("SKIPPING FRAME: Failed to capture gopro frame") 
            continue

        toilet_controller.update_velocity(gopro_image)
        
        if toilet_controller.is_still():
            if not recently_stopped:
                print("toilet just stopped moving")
                toilet_controller.resync_position(gopro_image)
                recently_stopped = True
                toilet_awaiting_movement_command  = True
            if toilet_awaiting_movement_command:
                if landing_x and landing_y:
                    landing_x_mm = toilet_controller.convert_m_to_mm(landing_x)
                    landing_y_mm = toilet_controller.convert_m_to_mm(landing_y)
                    if toilet_controller.real_point_inside_bounds(landing_x_mm, landing_y_mm):
                        if toilet_controller.move_to_position(landing_x_mm, landing_y_mm):
                            toilet_awaiting_movement_command = False
        else:
            recently_stopped = False
            toilet_awaiting_movement_command = False

        ui_handler.draw_toilet_ui(gopro_image, landing_x, landing_y)

        #quit button / render cv2
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
