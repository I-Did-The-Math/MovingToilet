import cv2
import numpy as np
import serial
import threading
import time
import math
import keyboard

paused = True
real_world_height = 330

class UIHandler:
    def __init__(self, toilet_controller):
        self.cursor_pixel_pos = [0,0]
        self.cursor_real_pos = [0,0]
        self.toilet_controller = toilet_controller
        self.move_requested = False
        cv2.namedWindow('Video Stream')
        cv2.setMouseCallback('Video Stream', self.mouse_callback)
        self.cursor_inside_bounds = False
        self.polygon = np.array(toilet_controller.pixel_coords, np.int32).reshape((-1, 1, 2))
        self.direction = [0,0]

    def normalize_vector(x, y):
        magnitude = math.sqrt(x**2 + y**2)
        if magnitude == 0:
            return 0, 0
        return x / magnitude, y / magnitude

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pixel_pos[0] = x * 2
            self.cursor_pixel_pos[1] = y * 2
            
            self.cursor_real_pos[0], self.cursor_real_pos[1] = self.toilet_controller.pixel_to_real(self.cursor_pixel_pos, self.toilet_controller.H)
            self.cursor_inside_bounds = cv2.pointPolygonTest(polygon, self.cursor_pixel_pos, False) >= 0

        if event == cv2.EVENT_LBUTTONDOWN:
            self.toilet_controller.move()

    def display_feed(frame, thresh):
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Video Stream', frame_resized)
        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresh_resized)

    def render_text(self):
        cv2.circle(frame, toilet_gopro_pixel_pos, 5, (0, 255, 0), -1)
        text_camera = f"Toilet Position In Camera: ({toilet_gopro_real_pos[0]:.2f} m, {toilet_gopro_real_pos[1]:.2f} m)"
        cv2.putText(frame, text_camera, (toilet_gopro_pixel_pos[0] + 20, toilet_gopro_pixel_pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        toilet_pixel_pos = real_world_to_pixel((toilet_real_pos[0], toilet_real_pos[1]), H_inv)
        cv2.circle(frame, (toilet_pixel_pos[0], toilet_pixel_pos[1]), 5, (255, 0, 0), -1)
        text_real = f"Toilet In Real Life: ({toilet_real_pos[0]:.2f} m, {toilet_real_pos[1]:.2f} m)"
        cv2.putText(frame, text_real, (toilet_pixel_pos[0] + 20, toilet_pixel_pos[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        arrow_color = (255, 0 , 0) if cursor_inside_bounds else (0, 0, 255)
        cv2.arrowedLine(frame, (toilet_pixel_pos[0], toilet_pixel_pos[1]), (cursor_x, cursor_y), arrow_color, 2)
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)
        predictedPixelLocation = real_world_to_pixel((toilet_real_pos[0] + delta_x, toilet_real_pos[1] + delta_y), H_inv)
        cv2.circle(frame, predictedPixelLocation, 5, (0, 255, 0), -1)
        future_toilet_inside_bounds[0] = cv2.pointPolygonTest(polygon, predictedPixelLocation, False) >= 0

class ToiletController:
    def __init__(self, arduino_handler, gopro_feed):
        self.gopro_feed = gopro_feed
        self.arduino_handler = arduino_handler
        self.real_pos = [0.0,0.0]
        self.init_pos = [0.0,0.0]
        self.last_pixel_pos = [0,0]
        self.velocity = 0.0
        self.pixel_pos_history = []
        self.last_velocity_check_time = time.time()
        self.just_stopped = False
        self.just_stopped_time = None
        self.pos_synced = True

        # Real life coordinates in meters (excluding height)
        self.real_coords = np.array([
            [180, 260],
            [795, 260],
            [795, 880],
            [180, 880]
        ])

        # Pixel coordinates from the camera
        self.pixel_coords = np.array([
            [420, 97],
            [848, 92],
            [848, 498],
            [423, 502]
        ])

        self.H, _ = cv2.findHomography(self.pixel_coords, self.real_coords)
        self.H_inv = np.linalg.inv(H)  # Inverse homography matrix for real world to pixel conversion

    def update_velocity(self):
        current_time = time.time()
        time_diff = current_time - self.last_velocity_check_time
        pixel_x, pixel_y, _ = self.get_pixel_position()
        if time_diff >= 0.1:
            vel_x = (pixel_x - self.last_pixel_pos[0]) / time_diff
            vel_y = (pixel_y - self.last_pixel_pos[1]) / time_diff

            self.velocity = math.sqrt(vel_x**2 + vel_y**2)

            self.last_pixel_pos[0] = pixel_x
            self.last_pixel_pos[1] = pixel_y
            self.last_velocity_check_time = current_time

    def resync_position(self, image):
        self.set_initial_position(image)
        self.arduino_handler.reset_encoders()
        self.update_real_position()

    def set_initial_position(self, image):
        pixel_x, pixel_y, _ = self.get_pixel_position(image)
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

    def center_of_white_pixels(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    
    def get_pixel_position(self,image):
        center_toilet_pixel, thresh = self.center_of_white_pixels(image)
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

    def is_still(self):
        
        still_threshold = 5
        still_duration_threshold = 0.05

        if self.velocity < still_threshold:
            if not self.just_stopped:
                self.just_stopped = True
                self.just_stopped_time = time.time()
            elif time.time() - self.just_stopped_time > still_duration_threshold:
                    return True
        else:
            self.just_stopped = False
            self.just_stopped_time = None

        return False
    
            

class ArduinoHandler:
    def __init__(self):
        self.lock = threading.Lock()

        self.serial_port = 'COM3' 
        self.baud_rate = 2000000
        self.serial = serial.Serial(self.serial_port, self.baud_rate)

        self.delta_x = 0.0
        self.delta_y = 0.0
        self.pee_initial_speed = 0.0

    def get_deltas(self):
        return self.delta_x, self.delta_y

    def get_pee_initial_speed(self):
        return self.pee_initial_speed

    def start_reading(self):
        threading.Thread(target=self.read_loop, daemon=True).start()

    def read_serial_data(self):
        while True:
            with self.lock:
                line = self.serial.readline().decode('utf-8').strip()
                print(f"Received line: {line}")
                if "X," in line:
                    parts = line.split(",")
                    self.delta_x = float(parts[1]) * 0.9
                    self.delta_y = float(parts[3])

                if "V:" in line:
                    self.pee_initial_speed = float(line.split(" ")[1])
                    
    def reset_encoders(self):
        with self.lock:
            self.serial.write(b"RESET_ENCODERS\n")
            print("Encoders Set To Zero")
            time.sleep(0.1)

    def move(self,delta_x, delta_y):
        with self.lock:
            command = f"{delta_x},{delta_y}\n"
            self.serial.write(command.encode())
            print(f"Sent: {command}")

try:
    #Connect To GoPro
    obs_virtual_camera_index = 3 
    gopro_feed = cv2.VideoCapture(obs_virtual_camera_index, cv2.CAP_DSHOW)
    if not gopro_feed.isOpened():
        print(f"Error: Could not open virtual camera at index {obs_virtual_camera_index}.")
        exit()
    print(f"Successfully opened virtual camera at index {obs_virtual_camera_index}.")

    #Setup GoPro Feed Window
    gopro_feed_width = 1280
    gopro_feed_height = 720
    gopro_feed.set(cv2.CAP_PROP_FRAME_WIDTH, gopro_feed_width)
    gopro_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, gopro_feed_height)

    while True:
        ret, gopro_frame = gopro_feed.read()
        if not ret:
            print("Error: Failed to capture image")

        get_toilet_position(gopro_frame)

        delta_x = cursor_real_x - toilet_real_pos[0]
        delta_y = cursor_real_y - toilet_real_pos[1]

        future_toilet_inside_bounds = [False]
        render_text(gopro_frame, toilet_gopro_real_pos, toilet_gopro_pixel_pos, delta_x, delta_y, future_toilet_inside_bounds)

        if move_requested:
            print("movement requested")
            if future_toilet_inside_bounds[0]:
                print("movement sent")
                move_delta(delta_x, delta_y)
                print("movement completed")
                move_requested = False

        display_feed(gopro_frame, thresh)

        # Check if 'Q' is pressed to quit
        if keyboard.is_pressed('q'):
            break

        #necessary for cv2 graphics to update, including each frame
        key = cv2.waitKey(1)
finally:
    stop_threads = True
    arduino.close()  # Ensure the serial connection is closed before joining the thread
    #serial_thread.join()
    gopro_feed.release()
    cv2.destroyAllWindows()
