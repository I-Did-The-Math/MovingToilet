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
        self.toilet_controller = toilet_controller
        self.move_requested = False
        cv2.namedWindow('Video Stream')
        cv2.setMouseCallback('Video Stream', self.mouse_callback)
        self.cursor_inside_bounds = False
        self.polygon = np.array(toilet_controller.pixel_coords, np.int32).reshape((-1, 1, 2))
        self.direction = [0,0]
    
    def update_cursor_pos(self, x, y):
        self.cursor_pixel_pos[0] = x * 2
        self.cursor_pixel_pos[1] = y * 2
        self.cursor_inside_bounds = cv2.pointPolygonTest(self.polygon, self.cursor_pixel_pos, False) >= 0

    def get_real_cursor_pos(self):
        return self.toilet_controller.pixel_to_real(self.cursor_pixel_pos[0], self.cursor_pixel_pos[1])

    def get_toilet_cursor_real_deltas(self)
        cursor_real_pos_x, cursor_real_pos_y = self.get_real_cursor_pos()
        toilet_real_pos_x, toilet_real_pos_y = self.toilet_controller.get_real_pos()
        cursor_real_delta_x = toilet_real_pos_x - cursor_real_pos_x
        cursor_real_delta_y = toilet_real_pos_y - cursor_real_pos_y
        return cursor_real_delta_x, cursor_real_delta_y
    
    def move_toilet_to_mouse(self):
        cursor_real_delta_x, cursor_real_delta_y = self.get_toilet_cursor_real_deltas()
        if self.cursor_inside_bounds:
            self.toilet_controller.move(cursor_real_delta_x, cursor_real_delta_y)    

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.update_cursor_pos(x,y)
        if event == cv2.EVENT_LBUTTONDOWN:  
            self.move_toilet_to_mouse()

    def display_feed(frame, thresh):
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Video Stream', frame_resized)
        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresh_resized)

    def render_outdated_toilet_pos(self, gopro_image, render_data):
        cv2.circle(gopro_image, (render_data['toilet_gopro_pixel_x'], render_data['toilet_gopro_pixel_y']), 5, (0, 255, 0), -1)
        text_camera = f"Outdated Position: ({render_data['toilet_gopro_real_x']:.2f} m, {render_data['toilet_gopro_real_y']:.2f} m)"
        cv2.putText(gopro_image, text_camera, (render_data['toilet_gopro_pixel_x'] + 20, render_data['toilet_gopro_pixel_y'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
    
    def render_correct_toilet_pos(self, gopro_image, render_data):
        cv2.circle(gopro_image, (render_data['toilet_pixel_x'], render_data['toilet_pixel_y']), 5, (255, 0, 0), -1)
        text_real = f"Toilet In Real Life: ({render_data['toilet_real_x']:.2f} m, {render_data['toilet_real_y']:.2f} m)"
        cv2.putText(gopro_image, text_real, (render_data['toilet_pixel_x'] + 20, render_data['toilet_pixel_y'] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

    def render_arrow(self, gopro_image, render_data):
        arrow_color = (255, 0 , 0) if self.cursor_inside_bounds else (0, 0, 255)
        cursor_x = self.cursor_pixel_pos[0]
        cursor_y = self.cursor_pixel_pos[1]
        cv2.arrowedLine(gopro_image, (render_data['toilet_pixel_x'], render_data['toilet_pixel_y']), (self.cursor_x, cursor_y), arrow_color, 2)

    def compute_render_data(self, gopro_image):
        toilet_gopro_pixel_x, toilet_gopro_pixel_y, thresh = self.toilet_controller.get_pixel_position(gopro_image)
        toilet_gopro_real_x, toilet_gopro_real_y = self.toilet_controller.pixel_to_real(toilet_gopro_pixel_x, toilet_gopro_pixel_y)
        
        toilet_real_x, toilet_real_y = self.toilet_controller.get_real_position()
        toilet_pixel_x, toilet_pixel_y = self.toilet_controller.real_to_pixel(toilet_real_x, toilet_real_y)

        render_data = {
            'toilet_gopro_pixel_x': toilet_gopro_pixel_x,
            'toilet_gopro_pixel_y':toilet_gopro_pixel_y,
            'toilet_gopro_real_x': toilet_gopro_real_x,
            'toilet_gopro_real_y': toilet_gopro_real_y,
            'toilet_real_x': toilet_real_x,
            'toilet_real_y': toilet_real_y,
            'toilet_pixel_x': toilet_pixel_x,
            'toilet_pixel_y': toilet_pixel_y,
            'thresh': thresh
        }

        return render_data

    def render_bounds(self, gopro_image):
        cv2.polylines(gopro_image, [self.polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    def draw_ui(self, gopro_image):
        render_data = self.compute_render_data(gopro_image)

        self.render_outdated_toilet_pos(gopro_image, render_data)
        self.render_correct_toilet_pos(gopro_image, render_data)
        self.render_arrow(gopro_image, render_data)
        self.render_bounds(gopro_image)
        self.display_feed(gopro_image, render_data['thresh'])

class ToiletController:
    def __init__(self, arduino_handler):
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

    def update_velocity(self, gopro_image):
        current_time = time.time()
        time_diff = current_time - self.last_velocity_check_time
        pixel_x, pixel_y, _ = self.get_pixel_position(gopro_image)
        if time_diff >= 0.1:
            vel_x = (pixel_x - self.last_pixel_pos[0]) / time_diff
            vel_y = (pixel_y - self.last_pixel_pos[1]) / time_diff

            self.velocity = math.sqrt(vel_x**2 + vel_y**2)

            self.last_pixel_pos[0] = pixel_x
            self.last_pixel_pos[1] = pixel_y
            self.last_velocity_check_time = current_time

    def resync_position(self, gopro_image):
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

    def is_still(self):
        still_threshold = 5
        still_duration_threshold = 0.1

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
        threading.Thread(target=self.read_serial_data, daemon=True).start()

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

    arduino_handler = ArduinoHandler()
    toilet_controller = ToiletController(arduino_handler)
    ui_handler = UIHandler(toilet_controller)

    ret, gopro_image = gopro_feed.read()

    arduino_handler.start_reading()
    toilet_controller.resync_position(gopro_image)

    while True:
        ret, gopro_image = gopro_feed.read()
        if not ret:
            print("Error: Failed to capture image")

        toilet_controller.update_velocity(gopro_image)

        if toilet_controller.is_still():
            toilet_controller.resync_position(gopro_image)

        toilet_controller.update_real_position()

        ui_handler.draw_ui(gopro_image)

        if move_requested:
            print("movement requested")
            if future_toilet_inside_bounds[0]:
                print("movement sent")
                move_delta(delta_x, delta_y)
                print("movement completed")
                move_requested = False

        display_feed(gopro_image, thresh)

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
