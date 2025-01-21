import cv2
import numpy as np
import serial
import threading
import time
import math
import keyboard

#Connect To Arduino
arduino_serial_port = 'COM3' 
arduino_baud_rate = 2000000
arduino = serial.Serial(arduino_serial_port, arduino_baud_rate)

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

#Initialize Global Variables
toilet_real_pos = [0.0,0.0]
initial_position_x, initial_position_y = 0.0, 0.0
stop_threads = False
reset_position = False
velocity_x, velocity_y = 0.0, 0.0
last_position_x, last_position_y = 0.0, 0.0
last_velocity_check_time = time.time()
reset_done = False
direction_x, direction_y = 0, 0
paused = True
cursor_inside_bounds = True
position_history = []
velocity_magnitude_history = []
move_requested = False

pause_serial_reading = threading.Event()
pause_serial_reading.set()

toilet_gopro_pixel_pos = [0,0]
toilet_gopro_real_pos = [0,0]
thresh = None

lock = threading.Lock()

def send_movement_command(ser, deltaX, deltaY):
    command = f"{deltaX},{deltaY}\n"
    ser.write(command.encode())
    print(f"Sent: {command}")

def wait_for_completion(ser):
    while True:
        response = ser.readline().decode().strip()
        if response:
            print(response)
            if "done" in response:
                break

def move_delta(deltaX, deltaY):
    pause_serial_reading.clear()
    send_movement_command(arduino, deltaX, deltaY)
    wait_for_completion(arduino)
    pause_serial_reading.set()

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y, cursor_real_x, cursor_real_y, paused, cursor_inside_bounds, move_requested

    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x * 2, y * 2  # Double the coordinates because the display is resized
        cursor_real_x, cursor_real_y = pixel_to_real_world((cursor_x, cursor_y), H)
        cursor_inside_bounds = cv2.pointPolygonTest(polygon, (cursor_x, cursor_y), False) >= 0

    if event == cv2.EVENT_LBUTTONDOWN:
        move_requested = True

def pixel_to_real_world(pixel_point, H):
    pixel_point = np.array([pixel_point[0], pixel_point[1], 1]).reshape(-1, 1)
    real_world_point = np.dot(H, pixel_point)
    real_world_point /= real_world_point[2]  # Normalize by the third (homogeneous) coordinate
    return real_world_point[0, 0], real_world_point[1, 0]

def real_world_to_pixel(real_world_point, H_inv):
    real_world_point = np.array([real_world_point[0], real_world_point[1], 1]).reshape(-1, 1)
    pixel_point = np.dot(H_inv, real_world_point)
    pixel_point /= pixel_point[2]  # Normalize by the third (homogeneous) coordinate
    return int(pixel_point[0, 0]), int(pixel_point[1, 0])


# Real life coordinates in meters (excluding height)
real_life_coords = np.array([
    [180, 260],
    [795, 260],
    [795, 880],
    [180, 880]
])

# Pixel coordinates from the camera
pixel_coords = np.array([
    [420, 97],
    [848, 92],
    [848, 498],
    [423, 502]
])

# Convert pixel coordinates to np.int32 for cv2.pointPolygonTest
polygon = np.array(pixel_coords, np.int32).reshape((-1, 1, 2))

# Real-world height in meters (constant for all points)
real_world_height = 330

# Calculate the homography matrix
H, _ = cv2.findHomography(pixel_coords, real_life_coords)
H_inv = np.linalg.inv(H)  # Inverse homography matrix for real world to pixel conversion

def get_center_of_white_pixels(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to find bright spots (retroreflective tape)
    _, thresh = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    
    # Clean up the image using morphological operations
    kernel_size = 7
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Exclude the top 50 pixels
    thresh[:50, :] = 0

    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, thresh  # No contours found

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask with only the largest contour
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Get the coordinates of all white pixels in the mask
    white_pixels = np.argwhere(mask == 255)

    # Calculate the average location of the white pixels
    center = np.mean(white_pixels, axis=0)
    center = (int(center[1]), int(center[0]))  # (x, y) format

    return center, mask

def get_gopro_toilet_pixel_pos(toilet_frame):
    center_toilet_pixel, thresh = get_center_of_white_pixels(toilet_frame)
    if center_toilet_pixel is not None:
        center_toilet_pixel = (center_toilet_pixel[0] + 10, center_toilet_pixel[1] - 140)
    return center_toilet_pixel, thresh

def get_gopro_toilet_real_pos(toilet_frame):
    center_toilet_pixel, thresh = get_gopro_toilet_pixel_pos(toilet_frame)
    return pixel_to_real_world(center_toilet_pixel, H), thresh

def get_gopro_toilet_pos(toilet_frame):
    center_gopro_toilet_pixel, thresh = get_center_of_white_pixels(toilet_frame)
    real_gopro_toilet_pos = None
    if center_gopro_toilet_pixel is not None:
        center_gopro_toilet_pixel = (center_gopro_toilet_pixel[0] + 10, center_gopro_toilet_pixel[1] - 140)
        real_gopro_toilet_pos = pixel_to_real_world(center_gopro_toilet_pixel, H)
    return center_gopro_toilet_pixel, real_gopro_toilet_pos, thresh

def get_toilet_velocity():
    global velocity_x, velocity_y, last_position_x, last_position_y
    current_time = time.time()
    time_diff = current_time - last_velocity_check_time
    if time_diff >= 0.1:
        velocity_x = (toilet_real_pos[0] - last_position_x) / time_diff
        velocity_y = (toilet_real_pos[1] - last_position_y) / time_diff
        last_position_x = toilet_real_pos[0]
        last_position_y = toilet_real_pos[1]
        last_velocity_check_time = current_time

#def update_gopro_toilet_velocity():
     # Update the position history
   # global position_history, toilet_real_pos,
   # current_time = time.time()
   # position_history.append((toilet_real_pos[0], toilet_real_pos[1], current_time))
   # if len(position_history) > 3:
    #    position_history.pop(0)

    # Calculate the velocity based on the change in position over the last three frames
   # if len(position_history) == 3:
    #    dt = position_history[-1][2] - position_history[0][2]
    #    dx = position_history[-1][0] - position_history[0][0]
    #    dy = position_history[-1][1] - position_history[0][1]
    #    velocity_x = dx / dt
    #    velocity_y = dy / dt
    #else:
    #    velocity_x, velocity_y = 0, 0


def add_encoder_deltas_to_initial_position(delta_x,delta_y):
    toilet_real_pos[0] = initial_position_x + delta_x
    toilet_real_pos[1] = initial_position_y + delta_y



def reset_encoders():
    arduino.write(b"RESET_ENCODERS\n")
    time.sleep(0.1)  # Wait for the Arduino to process the command

def is_Toilet_Still():
    return abs(velocity_x) < 0.01 and abs(velocity_y) < 0.01

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0
move_requested = False

# Start the serial reading thread
#serial_thread = threading.Thread(target=read_serial)
#serial_thread.daemon = True
#serial_thread.start()

cv2.namedWindow('Video Stream')
cv2.setMouseCallback('Video Stream', mouse_callback)

# Set initial position based on GoPro
initial_position_set = False

def set_initial_position_to_gopro():
    global initial_position_set, initial_position_x, initial_position_y
    initial_position_x = toilet_gopro_real_pos[0]
    initial_position_y = toilet_gopro_real_pos[1]
    initial_position_set = True

def sync_arduino_to_gopro_position(toilet_gopro_real_pos):
    last_position_x, last_position_y, reset_done, toilet_real_pos, last_velocity_check_time

    set_initial_position_to_gopro()
    
    reset_encoders()  # Reset encoders when setting the initial position
    toilet_real_pos[0] = initial_position_x
    toilet_real_pos[1] = initial_position_y
    last_position_x = toilet_real_pos[0]
    last_position_y = toilet_real_pos[1]
    reset_done = True
    print(f"Position Resynced Using GoPro: X: {toilet_real_pos[0]:.2f} m, Y: {toilet_real_pos[1]:.2f} m")

    if not is_Toilet_Still():
        reset_done = False



def render_text(frame, toilet_gopro_real_pos, toilet_gopro_pixel_pos, delta_x, delta_y, future_toilet_inside_bounds):
    global toilet_real_pos, cursor_inside_bounds, cursor_x, cursor_y, polygon, cursor_real_x, cursor_real_y
     # Draw a circle at the offset center of the white pixels
    cv2.circle(frame, toilet_gopro_pixel_pos, 5, (0, 255, 0), -1)

    # Write the real-life coordinates on the frame for the GoPro position
    text_camera = f"Toilet Position In Camera: ({toilet_gopro_real_pos[0]:.2f} m, {toilet_gopro_real_pos[1]:.2f} m)"
    cv2.putText(frame, text_camera, (toilet_gopro_pixel_pos[0] + 20, toilet_gopro_pixel_pos[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

    # Calculate the encoder position in pixel coordinates
    toilet_pixel_pos = real_world_to_pixel((toilet_real_pos[0], toilet_real_pos[1]), H_inv)

    # Draw a circle at the encoder position
    cv2.circle(frame, (toilet_pixel_pos[0], toilet_pixel_pos[1]), 5, (255, 0, 0), -1)

    # Write the real-life coordinates on the frame for the encoder position
    text_real = f"Toilet In Real Life: ({toilet_real_pos[0]:.2f} m, {toilet_real_pos[1]:.2f} m)"
    cv2.putText(frame, text_real, (toilet_pixel_pos[0] + 20, toilet_pixel_pos[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

    # Check if the cursor is inside the polygon
    arrow_color = (255, 0 , 0) if cursor_inside_bounds else (0, 0, 255)

    # Draw an arrow from the toilet to the cursor
    cv2.arrowedLine(frame, (toilet_pixel_pos[0], toilet_pixel_pos[1]), (cursor_x, cursor_y), arrow_color, 2)

    # Draw the polygon
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

    predictedPixelLocation = real_world_to_pixel((toilet_real_pos[0] + delta_x, toilet_real_pos[1] + delta_y), H_inv)
    #check if toilet would be moved outside bounds, if not then MOVE
    cv2.circle(frame, predictedPixelLocation, 5, (0, 255, 0), -1)

    future_toilet_inside_bounds[0] = cv2.pointPolygonTest(polygon, predictedPixelLocation, False) >= 0

    
def display_feed(frame, thresh):
    frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

    # Display the original frame with the detected white pixels
    cv2.imshow('Video Stream', frame_resized)
    # Display the thresholded image
    cv2.imshow('Thresholded Image', thresh_resized)

def is_gopro_toilet_position_accurate():
    return not initial_position_set or (is_Toilet_Still() and time.time() - last_velocity_check_time >= 0.1)


def get_toilet_position(gopro_frame):
    global toilet_gopro_pixel_pos, toilet_gopro_real_pos, thresh

    toilet_gopro_pixel_pos, toilet_gopro_real_pos, thresh = get_gopro_toilet_pos(gopro_frame)

    #delta_x, delta_y = read_arduino_encoder_data()
    add_encoder_deltas_to_initial_position(delta_x, delta_y)

    if keyboard.is_pressed('r') or is_gopro_toilet_position_accurate():
        sync_arduino_to_gopro_position(toilet_gopro_real_pos)


def read_arduino_encoder_data():
    global delta_x, delta_y
    while True:
        with lock:
            line = arduino.readline().decode('utf-8').strip()
            print(f"Received line: {line}")
            if "X," in line:
                parts = line.split(",")
                delta_x = float(parts[1]) 
                delta_x *= 0.9
                delta_y = float(parts[3])  


def read_velocity():
    global pee_initial_speed
    while True:
        try:
            # Read the serial data from Arduino
            line = arduino.readline().decode('utf-8').strip()
            print(f"Received line: {line}")  # Debug print to verify serial data
            if "V:" in line:
                pee_initial_speed = float(line.split(" ")[1])
        except Exception as e:
            print(f"Error: {e}")

# Start the thread to read velocity
#thread = threading.Thread(target = read_velocity)
#thread.daemon = True
#thread.start()

thread = threading.Thread(target = read_arduino_encoder_data)
thread.daemon = True
thread.start()

try:
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
