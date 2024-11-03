<<<<<<< HEAD
import cv2
import numpy as np
import serial
import threading
import time
import math

# Adjust these variables to match your setup
serial_port = 'COM3'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 2000000  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
encoder_position_x, encoder_position_y = 0.0, 0.0
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
breaking = False
initial_breaking_direction = (0, 0)

# Adjust these variables to match your setup
frame_width = 1280
frame_height = 720

def send_command(directionX, directionY, speed):
    """Send command to the Arduino."""
    command = f"{directionX},{directionY}\n"
    #ser.write(command.encode())

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y, cursor_real_x, cursor_real_y, paused, cursor_inside_bounds

    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x * 2, y * 2  # Double the coordinates because the display is resized
        cursor_real_x, cursor_real_y = pixel_to_real_world((cursor_x, cursor_y), H)
        cursor_inside_bounds = cv2.pointPolygonTest(polygon, (cursor_x, cursor_y), False) >= 0

    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused

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

# Replace with the correct index for the OBS virtual camera
virtual_camera_index = 3  # Adjust this index if needed

# Open the virtual camera
cap = cv2.VideoCapture(virtual_camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open virtual camera at index {virtual_camera_index}.")
    exit()

print(f"Successfully opened virtual camera at index {virtual_camera_index}.")

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
real_world_height = 0.33

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

# Function to read serial data from Arduino
def read_serial():
    global encoder_position_x, encoder_position_y, initial_position_x, initial_position_y, reset_position, velocity_x, velocity_y, last_position_x, last_position_y, last_velocity_check_time, reset_done
    while not stop_threads:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("X,"):
                try:
                    parts = line.split(",")
                    delta_x = float(parts[1]) 
                    delta_x *= 0.9
                    delta_y = float(parts[3])  

                    encoder_position_x = initial_position_x + delta_x
                    encoder_position_y = initial_position_y + delta_y

                    current_time = time.time()
                    time_diff = current_time - last_velocity_check_time
                    if time_diff >= 0.1:
                        velocity_x = (encoder_position_x - last_position_x) / time_diff
                        velocity_y = (encoder_position_y - last_position_y) / time_diff
                        last_position_x = encoder_position_x
                        last_position_y = encoder_position_y
                        last_velocity_check_time = current_time

                    print(f"Encoder Position Updated: X: {encoder_position_x:.2f} m, Y: {encoder_position_y:.2f} m")
                    print(f"Velocity: X: {velocity_x:.2f} m/s, Y: {velocity_y:.2f} m/s")
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
        except UnicodeDecodeError:
            continue  # Skip any lines that cause a UnicodeDecodeError

def reset_encoders():
    ser.write(b"RESET_ENCODERS\n")
    time.sleep(0.1)  # Wait for the Arduino to process the command

def check_velocity():
    global velocity_x, velocity_y
    return abs(velocity_x) < 0.01 and abs(velocity_y) < 0.01

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0

# Start the serial reading thread
serial_thread = threading.Thread(target=read_serial)
serial_thread.daemon = True
serial_thread.start()

cv2.namedWindow('Video Stream')
cv2.setMouseCallback('Video Stream', mouse_callback)

# Set initial position based on GoPro
initial_position_set = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Get the center of the white pixels and the thresholded image
        center, thresh = get_center_of_white_pixels(frame)

        if center is not None:
            # Offset the center by -70 pixels in the y direction
            offset_center = (center[0] + 10, center[1] - 140)

            # Convert the offset pixel coordinates to real-life coordinates
            real_x, real_y = pixel_to_real_world(offset_center, H)

            # Check if 'R' is pressed to reset initial position
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or not initial_position_set or (check_velocity() and not reset_done and time.time() - last_velocity_check_time >= 0.1):
                initial_position_x = real_x
                initial_position_y = real_y
                initial_position_set = True
                reset_encoders()  # Reset encoders when setting the initial position
                encoder_position_x = initial_position_x 
                encoder_position_y = initial_position_y
                last_position_x = encoder_position_x
                last_position_y = encoder_position_y
                reset_done = True
                print("Initial position set/reset to GoPro position")
                print(f"Encoder Position Updated: X: {encoder_position_x:.2f} m, Y: {encoder_position_y:.2f} m")

            if not check_velocity():
                reset_done = False

            # Calculate the normalized vector from the toilet to the cursor using encoder position
            vector_x = cursor_real_x - encoder_position_x
            vector_y = cursor_real_y - encoder_position_y
            magnitude = np.sqrt(vector_x**2 + vector_y**2)
            if magnitude != 0:
                direction_x = vector_x / magnitude
                direction_y = vector_y / magnitude
            else:
                direction_x, direction_y = 0, 0

            # Update the position history
            current_time = time.time()
            position_history.append((encoder_position_x, encoder_position_y, current_time))
            if len(position_history) > 3:
                position_history.pop(0)

            # Calculate the velocity based on the change in position over the last three frames
            if len(position_history) >= 3:
                dt = position_history[-1][2] - position_history[0][2]
                dx = position_history[-1][0] - position_history[0][0]
                dy = position_history[-1][1] - position_history[0][1]
                velocity_x = dx / dt
                velocity_y = dy / dt
            else:
                velocity_x, velocity_y = 0, 0

            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

            # Update the velocity magnitude history
            velocity_magnitude_history.append(velocity_magnitude)
            if len(velocity_magnitude_history) > 3:
                velocity_magnitude_history.pop(0)

            # Calculate the moving average of the last three velocity magnitudes
            if len(velocity_magnitude_history) > 0:
                moving_average_velocity = sum(velocity_magnitude_history) / len(velocity_magnitude_history)
            else:
                moving_average_velocity = 0

            # Display the moving average velocity
            moving_average_text = f"Moving Average Velocity: {moving_average_velocity:.2f} m/s"
            cv2.putText(frame, moving_average_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

            # Draw a circle at the offset center of the white pixels
            cv2.circle(frame, offset_center, 5, (0, 255, 0), -1)

            # Write the real-life coordinates on the frame for the current position
            text_real_life = f"Real Life: ({encoder_position_x:.2f} m, {encoder_position_y:.2f} m, {real_world_height:.2f} m)"
            cv2.putText(frame, text_real_life, (offset_center[0] + 20, offset_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            
            # Write the pixel coordinates on the frame for the current position
            text_pixel = f"Pixel: [{offset_center[0]}, {offset_center[1]}]"
            cv2.putText(frame, text_pixel, (offset_center[0] + 20, offset_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Display the vector and magnitude
            vector_text = f"Direction: ({direction_x:.2f}, {direction_y:.2f}) Magnitude: {magnitude:.2f} m"
            cv2.putText(frame, vector_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            # Display the real-life coordinates of the cursor
            cursor_text = f"Cursor Real Life: ({cursor_real_x:.2f} m, {cursor_real_y:.2f} m)"
            cv2.putText(frame, cursor_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

            # Check if the cursor is inside the polygon
            arrow_color = (0, 255, 0) if cursor_inside_bounds else (0, 0, 255)

            # Draw an arrow from the toilet to the cursor
            cv2.arrowedLine(frame, offset_center, (cursor_x, cursor_y), arrow_color, 2)

            # Draw the polygon
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

            startArrow, endArrow = offset_center, (cursor_x, cursor_y)

            if breaking:
                direction_x, direction_y = initial_breaking_direction
            else:
                direction_x = -direction_x
                initial_breaking_direction = (-direction_x, -direction_y)
                
            # Function to map ranges
            def map_range(x, a, b, c, d):
                return c + (x - a) * (d - c) / (b - a)

            # Thresholds and parameters
            threshold = 0.03
            minPower = 11
            maxPower = 30
            minVelocity = 0.6
            maxVelocity = 0.629 * maxPower - 10.9
            minBreakingDistance = 0
            maxBreakingDistance = 0.8

            # Calculate the speed based on the magnitude
            speed = map_range(magnitude, threshold, 0.84, minPower, maxPower)
            breakingDistance = 0.1

            closeEnough = False

            # Determine if the toilet is close enough to the target
            if magnitude < threshold:
                closeEnough = True

            # Handle different movement scenarios
            if closeEnough or not cursor_inside_bounds or paused:
                speed = 0
                breaking = False
            elif moving_average_velocity > minVelocity and magnitude < breakingDistance:
                breaking = True
                direction_x, direction_y = initial_breaking_direction
                scalar = map_range(moving_average_velocity, 0, 7, 1, 5)
                speed = speed * scalar
                startArrow, endArrow = (cursor_x, cursor_y), offset_center

            # Stop breaking if the velocity is below a certain threshold
            if moving_average_velocity < minVelocity:
                breaking = False

            # Draw breaking information
            breakingColor = (0, 255, 255)
            breakingDistanceText = f"Breaking Distance: {breakingDistance:.2f} m Currently Breaking: {breaking}"
            if breaking:
                breakingColor = (0, 0, 255)
            cv2.putText(frame, breakingDistanceText, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, breakingColor, 4)

            # Draw an additional arrow to show the direction it's currently trying to travel
            travel_arrow_end = (int(offset_center[0] - direction_x * speed * 10), int(offset_center[1] + direction_y * speed * 10))
            cv2.arrowedLine(frame, offset_center, travel_arrow_end, (255, 0, 0), 2)

            cv2.arrowedLine(frame, startArrow, endArrow, arrow_color, 2)
            send_command(-direction_x, direction_y, speed)

        # Resize the frame and thresholded image by half for display
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Video Stream', frame_resized)
        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresh_resized)

        # Check if 'Q' is pressed to quit
        if key == ord('q'):
            break
finally:
    stop_threads = True
    ser.close()  # Ensure the serial connection is closed before joining the thread
    serial_thread.join()
    cap.release()
    cv2.destroyAllWindows()
=======
import cv2
import numpy as np
import serial
import threading
import time
import math

# Adjust these variables to match your setup
serial_port = 'COM3'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 2000000  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
encoder_position_x, encoder_position_y = 0.0, 0.0
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
breaking = False
initial_breaking_direction = (0, 0)

# Adjust these variables to match your setup
frame_width = 1280
frame_height = 720

def send_command(directionX, directionY, speed):
    """Send command to the Arduino."""
    command = f"{directionX},{directionY}\n"
    #ser.write(command.encode())

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y, cursor_real_x, cursor_real_y, paused, cursor_inside_bounds

    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x * 2, y * 2  # Double the coordinates because the display is resized
        cursor_real_x, cursor_real_y = pixel_to_real_world((cursor_x, cursor_y), H)
        cursor_inside_bounds = cv2.pointPolygonTest(polygon, (cursor_x, cursor_y), False) >= 0

    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused

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

# Replace with the correct index for the OBS virtual camera
virtual_camera_index = 3  # Adjust this index if needed

# Open the virtual camera
cap = cv2.VideoCapture(virtual_camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open virtual camera at index {virtual_camera_index}.")
    exit()

print(f"Successfully opened virtual camera at index {virtual_camera_index}.")

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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
real_world_height = 0.33

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

# Function to read serial data from Arduino
def read_serial():
    global encoder_position_x, encoder_position_y, initial_position_x, initial_position_y, reset_position, velocity_x, velocity_y, last_position_x, last_position_y, last_velocity_check_time, reset_done
    while not stop_threads:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("X,"):
                try:
                    parts = line.split(",")
                    delta_x = float(parts[1]) 
                    delta_x *= 0.9
                    delta_y = float(parts[3])  

                    encoder_position_x = initial_position_x + delta_x
                    encoder_position_y = initial_position_y + delta_y

                    current_time = time.time()
                    time_diff = current_time - last_velocity_check_time
                    if time_diff >= 0.1:
                        velocity_x = (encoder_position_x - last_position_x) / time_diff
                        velocity_y = (encoder_position_y - last_position_y) / time_diff
                        last_position_x = encoder_position_x
                        last_position_y = encoder_position_y
                        last_velocity_check_time = current_time

                    print(f"Encoder Position Updated: X: {encoder_position_x:.2f} m, Y: {encoder_position_y:.2f} m")
                    print(f"Velocity: X: {velocity_x:.2f} m/s, Y: {velocity_y:.2f} m/s")
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
        except UnicodeDecodeError:
            continue  # Skip any lines that cause a UnicodeDecodeError

def reset_encoders():
    ser.write(b"RESET_ENCODERS\n")
    time.sleep(0.1)  # Wait for the Arduino to process the command

def check_velocity():
    global velocity_x, velocity_y
    return abs(velocity_x) < 0.01 and abs(velocity_y) < 0.01

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0

# Start the serial reading thread
serial_thread = threading.Thread(target=read_serial)
serial_thread.daemon = True
serial_thread.start()

cv2.namedWindow('Video Stream')
cv2.setMouseCallback('Video Stream', mouse_callback)

# Set initial position based on GoPro
initial_position_set = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Get the center of the white pixels and the thresholded image
        center, thresh = get_center_of_white_pixels(frame)

        if center is not None:
            # Offset the center by -70 pixels in the y direction
            offset_center = (center[0] + 10, center[1] - 140)

            # Convert the offset pixel coordinates to real-life coordinates
            real_x, real_y = pixel_to_real_world(offset_center, H)

            # Check if 'R' is pressed to reset initial position
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r') or not initial_position_set or (check_velocity() and not reset_done and time.time() - last_velocity_check_time >= 0.1):
                initial_position_x = real_x
                initial_position_y = real_y
                initial_position_set = True
                reset_encoders()  # Reset encoders when setting the initial position
                encoder_position_x = initial_position_x 
                encoder_position_y = initial_position_y
                last_position_x = encoder_position_x
                last_position_y = encoder_position_y
                reset_done = True
                print("Initial position set/reset to GoPro position")
                print(f"Encoder Position Updated: X: {encoder_position_x:.2f} m, Y: {encoder_position_y:.2f} m")

            if not check_velocity():
                reset_done = False

            # Calculate the normalized vector from the toilet to the cursor using encoder position
            vector_x = cursor_real_x - encoder_position_x
            vector_y = cursor_real_y - encoder_position_y
            magnitude = np.sqrt(vector_x**2 + vector_y**2)
            if magnitude != 0:
                direction_x = vector_x / magnitude
                direction_y = vector_y / magnitude
            else:
                direction_x, direction_y = 0, 0

            # Update the position history
            current_time = time.time()
            position_history.append((encoder_position_x, encoder_position_y, current_time))
            if len(position_history) > 3:
                position_history.pop(0)

            # Calculate the velocity based on the change in position over the last three frames
            if len(position_history) >= 3:
                dt = position_history[-1][2] - position_history[0][2]
                dx = position_history[-1][0] - position_history[0][0]
                dy = position_history[-1][1] - position_history[0][1]
                velocity_x = dx / dt
                velocity_y = dy / dt
            else:
                velocity_x, velocity_y = 0, 0

            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

            # Update the velocity magnitude history
            velocity_magnitude_history.append(velocity_magnitude)
            if len(velocity_magnitude_history) > 3:
                velocity_magnitude_history.pop(0)

            # Calculate the moving average of the last three velocity magnitudes
            if len(velocity_magnitude_history) > 0:
                moving_average_velocity = sum(velocity_magnitude_history) / len(velocity_magnitude_history)
            else:
                moving_average_velocity = 0

            # Display the moving average velocity
            moving_average_text = f"Moving Average Velocity: {moving_average_velocity:.2f} m/s"
            cv2.putText(frame, moving_average_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)

            # Draw a circle at the offset center of the white pixels
            cv2.circle(frame, offset_center, 5, (0, 255, 0), -1)

            # Write the real-life coordinates on the frame for the current position
            text_real_life = f"Real Life: ({encoder_position_x:.2f} m, {encoder_position_y:.2f} m, {real_world_height:.2f} m)"
            cv2.putText(frame, text_real_life, (offset_center[0] + 20, offset_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            
            # Write the pixel coordinates on the frame for the current position
            text_pixel = f"Pixel: [{offset_center[0]}, {offset_center[1]}]"
            cv2.putText(frame, text_pixel, (offset_center[0] + 20, offset_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Display the vector and magnitude
            vector_text = f"Direction: ({direction_x:.2f}, {direction_y:.2f}) Magnitude: {magnitude:.2f} m"
            cv2.putText(frame, vector_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            # Display the real-life coordinates of the cursor
            cursor_text = f"Cursor Real Life: ({cursor_real_x:.2f} m, {cursor_real_y:.2f} m)"
            cv2.putText(frame, cursor_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

            # Check if the cursor is inside the polygon
            arrow_color = (0, 255, 0) if cursor_inside_bounds else (0, 0, 255)

            # Draw an arrow from the toilet to the cursor
            cv2.arrowedLine(frame, offset_center, (cursor_x, cursor_y), arrow_color, 2)

            # Draw the polygon
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)

            startArrow, endArrow = offset_center, (cursor_x, cursor_y)

            if breaking:
                direction_x, direction_y = initial_breaking_direction
            else:
                direction_x = -direction_x
                initial_breaking_direction = (-direction_x, -direction_y)
                
            # Function to map ranges
            def map_range(x, a, b, c, d):
                return c + (x - a) * (d - c) / (b - a)

            # Thresholds and parameters
            threshold = 0.03
            minPower = 11
            maxPower = 30
            minVelocity = 0.6
            maxVelocity = 0.629 * maxPower - 10.9
            minBreakingDistance = 0
            maxBreakingDistance = 0.8

            # Calculate the speed based on the magnitude
            speed = map_range(magnitude, threshold, 0.84, minPower, maxPower)
            breakingDistance = 0.1

            closeEnough = False

            # Determine if the toilet is close enough to the target
            if magnitude < threshold:
                closeEnough = True

            # Handle different movement scenarios
            if closeEnough or not cursor_inside_bounds or paused:
                speed = 0
                breaking = False
            elif moving_average_velocity > minVelocity and magnitude < breakingDistance:
                breaking = True
                direction_x, direction_y = initial_breaking_direction
                scalar = map_range(moving_average_velocity, 0, 7, 1, 5)
                speed = speed * scalar
                startArrow, endArrow = (cursor_x, cursor_y), offset_center

            # Stop breaking if the velocity is below a certain threshold
            if moving_average_velocity < minVelocity:
                breaking = False

            # Draw breaking information
            breakingColor = (0, 255, 255)
            breakingDistanceText = f"Breaking Distance: {breakingDistance:.2f} m Currently Breaking: {breaking}"
            if breaking:
                breakingColor = (0, 0, 255)
            cv2.putText(frame, breakingDistanceText, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, breakingColor, 4)

            # Draw an additional arrow to show the direction it's currently trying to travel
            travel_arrow_end = (int(offset_center[0] - direction_x * speed * 10), int(offset_center[1] + direction_y * speed * 10))
            cv2.arrowedLine(frame, offset_center, travel_arrow_end, (255, 0, 0), 2)

            cv2.arrowedLine(frame, startArrow, endArrow, arrow_color, 2)
            send_command(-direction_x, direction_y, speed)

        # Resize the frame and thresholded image by half for display
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Video Stream', frame_resized)
        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresh_resized)

        # Check if 'Q' is pressed to quit
        if key == ord('q'):
            break
finally:
    stop_threads = True
    ser.close()  # Ensure the serial connection is closed before joining the thread
    serial_thread.join()
    cap.release()
    cv2.destroyAllWindows()
>>>>>>> c328291 (Added Project Files)
