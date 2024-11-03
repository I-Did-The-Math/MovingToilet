<<<<<<< HEAD
import cv2
import numpy as np
import math
import serial
import time

# Adjust these variables to match your setup
serial_port = 'COM5'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 9600  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
directionX, directionY = 0, 0
paused = True
cursor_inside_bounds = True
position_history = []
velocity_magnitude_history = []
mode = 'mouse'


def send_command(directionX, directionY, speed):
    """Send command to the Arduino."""
    command = f"{directionX},{directionY},{speed}\n"
    ser.write(command.encode())

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitudes

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y, cursor_real_x, cursor_real_y, paused, cursor_inside_bounds

    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x * 2, y * 2  # Double the coordinates because the display is resized
        cursor_real_x, cursor_real_y = pixel_to_real_world((cursor_x, cursor_y), H)
        cursor_inside_bounds = cv2.pointPolygonTest(polygon, (cursor_x, cursor_y), False) >= 0

    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused


# Replace with the correct index found from the previous script
virtual_camera_index = 2

# Desired resolution
frame_width = 1280
frame_height = 720

# Use the DirectShow backend
backend = cv2.CAP_DSHOW

cap = cv2.VideoCapture(virtual_camera_index, backend)

if not cap.isOpened():
    print(f"Error: Could not open virtual camera at index {virtual_camera_index} with backend {backend}.")
    exit()

print(f"Successfully opened virtual camera at index {virtual_camera_index} with backend {backend}.")

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Real life coordinates in meters (excluding height)
real_life_coords = np.array([
    [0.18, 0.26],
    [0.795, 0.26],
    [0.795, 0.88],
    [0.18, 0.88]
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

def pixel_to_real_world(pixel_point, H):
    pixel_point = np.array([pixel_point[0], pixel_point[1], 1]).reshape(-1, 1)
    real_world_point = np.dot(H, pixel_point)
    real_world_point /= real_world_point[2]  # Normalize by the third (homogeneous) coordinate
    return real_world_point[0, 0], real_world_point[1, 0]

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0

cv2.namedWindow('Video Stream')
cv2.setMouseCallback('Video Stream', mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the center of the white pixels and the thresholded image
        center, thresh = get_center_of_white_pixels(frame)

        if center is not None:
            # Offset the center by -70 pixels in the y direction
            offset_center = (center[0] + 10, center[1] - 140)

            # Convert the offset pixel coordinates to real-life coordinates
            real_x, real_y = pixel_to_real_world(offset_center, H)

           # Update the position history
            current_time = time.time()
            position_history.append((real_x, real_y, current_time))
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

            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2) * 10

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
            text_real_life = f"Real Life: ({real_x:.2f} m, {real_y:.2f} m, {real_world_height:.2f} m)"
            cv2.putText(frame, text_real_life, (offset_center[0] + 20, offset_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            
            # Write the pixel coordinates on the frame for the current position
            text_pixel = f"Pixel: [{offset_center[0]}, {offset_center[1]}]"
            cv2.putText(frame, text_pixel, (offset_center[0] + 20, offset_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            if mode == 'mouse':
                target_x, target_y = cursor_real_x, cursor_real_y
            else:  # 'piss' mode
                target_x, target_y = get_piss_target()

            # Calculate the normalized vector from the toilet to the target
            vector_x = target_x - real_x
            vector_y = target_y - real_y
            magnitude = np.sqrt(vector_x**2 + vector_y**2)
            if magnitude != 0:
                direction_x = vector_x / magnitude
                direction_y = vector_y / magnitude
            else:
                direction_x, direction_y = 0, 0

            # Display the vector and magnitude
            vector_text = f"Direction: ({direction_x:.2f}, {direction_y:.2f}) Magnitude: {magnitude:.2f} m"
            cv2.putText(frame, vector_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            # Display the real-life coordinates of the cursor
            if mode == 'mouse':
                cursor_text = f"Cursor Real Life: ({cursor_real_x:.2f} m, {cursor_real_y:.2f} m)"
                cv2.putText(frame, cursor_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

            # Check if the cursor is inside the polygon
            arrow_color = (0, 255, 0) if target else (0, 0, 255)

            # Draw an arrow from the toilet to the cursor
            cv2.arrowedLine(frame, offset_center, (cursor_x, cursor_y), arrow_color, 2)

            # Draw the polygon
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)


            direction_x = -direction_x
            direction_y = direction_y
            # Send the command to the Arduino if moving and cursor is inside bounds
            def map_range(x, a, b, c, d):
                return c + (x - a) * (d - c) / (b - a)
            
            threshold = 0.03
            minimumPower = 14
            
            speed = map_range(magnitude, threshold, 0.84, minimumPower, 50)
            
            closeEnough = False
            
            
            if magnitude < threshold:
                closeEnough = True
            
            if closeEnough or not cursor_inside_bounds or paused:
                speed = 0
            elif moving_average_velocity > 0.5 and magnitude < map_range(moving_average_velocity,0,255,0.1,0.7):
                direction_x = -velocity_x
                direction_y = -velocity_y
                speed = map_range(moving_average_velocity, 0, 150, minimumPower, 255)
                #40m/s = 80
                #9m/s = 40
                #7m/s = 30
                #3.5m/s = 20
                

            print(velocity_magnitude)
            
            send_command(direction_x, direction_y, speed)


        # Resize the frame and thresholded image by half for display
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Video Stream', frame_resized)
        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresh_resized)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
=======
import cv2
import numpy as np
import math
import serial
import time

# Adjust these variables to match your setup
serial_port = 'COM5'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 9600  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
directionX, directionY = 0, 0
paused = True
cursor_inside_bounds = True
position_history = []
velocity_magnitude_history = []
mode = 'mouse'


def send_command(directionX, directionY, speed):
    """Send command to the Arduino."""
    command = f"{directionX},{directionY},{speed}\n"
    ser.write(command.encode())

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitudes

def mouse_callback(event, x, y, flags, param):
    global cursor_x, cursor_y, cursor_real_x, cursor_real_y, paused, cursor_inside_bounds

    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x * 2, y * 2  # Double the coordinates because the display is resized
        cursor_real_x, cursor_real_y = pixel_to_real_world((cursor_x, cursor_y), H)
        cursor_inside_bounds = cv2.pointPolygonTest(polygon, (cursor_x, cursor_y), False) >= 0

    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused


# Replace with the correct index found from the previous script
virtual_camera_index = 2

# Desired resolution
frame_width = 1280
frame_height = 720

# Use the DirectShow backend
backend = cv2.CAP_DSHOW

cap = cv2.VideoCapture(virtual_camera_index, backend)

if not cap.isOpened():
    print(f"Error: Could not open virtual camera at index {virtual_camera_index} with backend {backend}.")
    exit()

print(f"Successfully opened virtual camera at index {virtual_camera_index} with backend {backend}.")

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Real life coordinates in meters (excluding height)
real_life_coords = np.array([
    [0.18, 0.26],
    [0.795, 0.26],
    [0.795, 0.88],
    [0.18, 0.88]
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

def pixel_to_real_world(pixel_point, H):
    pixel_point = np.array([pixel_point[0], pixel_point[1], 1]).reshape(-1, 1)
    real_world_point = np.dot(H, pixel_point)
    real_world_point /= real_world_point[2]  # Normalize by the third (homogeneous) coordinate
    return real_world_point[0, 0], real_world_point[1, 0]

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0

cv2.namedWindow('Video Stream')
cv2.setMouseCallback('Video Stream', mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the center of the white pixels and the thresholded image
        center, thresh = get_center_of_white_pixels(frame)

        if center is not None:
            # Offset the center by -70 pixels in the y direction
            offset_center = (center[0] + 10, center[1] - 140)

            # Convert the offset pixel coordinates to real-life coordinates
            real_x, real_y = pixel_to_real_world(offset_center, H)

           # Update the position history
            current_time = time.time()
            position_history.append((real_x, real_y, current_time))
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

            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2) * 10

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
            text_real_life = f"Real Life: ({real_x:.2f} m, {real_y:.2f} m, {real_world_height:.2f} m)"
            cv2.putText(frame, text_real_life, (offset_center[0] + 20, offset_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            
            # Write the pixel coordinates on the frame for the current position
            text_pixel = f"Pixel: [{offset_center[0]}, {offset_center[1]}]"
            cv2.putText(frame, text_pixel, (offset_center[0] + 20, offset_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            if mode == 'mouse':
                target_x, target_y = cursor_real_x, cursor_real_y
            else:  # 'piss' mode
                target_x, target_y = get_piss_target()

            # Calculate the normalized vector from the toilet to the target
            vector_x = target_x - real_x
            vector_y = target_y - real_y
            magnitude = np.sqrt(vector_x**2 + vector_y**2)
            if magnitude != 0:
                direction_x = vector_x / magnitude
                direction_y = vector_y / magnitude
            else:
                direction_x, direction_y = 0, 0

            # Display the vector and magnitude
            vector_text = f"Direction: ({direction_x:.2f}, {direction_y:.2f}) Magnitude: {magnitude:.2f} m"
            cv2.putText(frame, vector_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            # Display the real-life coordinates of the cursor
            if mode == 'mouse':
                cursor_text = f"Cursor Real Life: ({cursor_real_x:.2f} m, {cursor_real_y:.2f} m)"
                cv2.putText(frame, cursor_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

            # Check if the cursor is inside the polygon
            arrow_color = (0, 255, 0) if target else (0, 0, 255)

            # Draw an arrow from the toilet to the cursor
            cv2.arrowedLine(frame, offset_center, (cursor_x, cursor_y), arrow_color, 2)

            # Draw the polygon
            cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 0), thickness=2)


            direction_x = -direction_x
            direction_y = direction_y
            # Send the command to the Arduino if moving and cursor is inside bounds
            def map_range(x, a, b, c, d):
                return c + (x - a) * (d - c) / (b - a)
            
            threshold = 0.03
            minimumPower = 14
            
            speed = map_range(magnitude, threshold, 0.84, minimumPower, 50)
            
            closeEnough = False
            
            
            if magnitude < threshold:
                closeEnough = True
            
            if closeEnough or not cursor_inside_bounds or paused:
                speed = 0
            elif moving_average_velocity > 0.5 and magnitude < map_range(moving_average_velocity,0,255,0.1,0.7):
                direction_x = -velocity_x
                direction_y = -velocity_y
                speed = map_range(moving_average_velocity, 0, 150, minimumPower, 255)
                #40m/s = 80
                #9m/s = 40
                #7m/s = 30
                #3.5m/s = 20
                

            print(velocity_magnitude)
            
            send_command(direction_x, direction_y, speed)


        # Resize the frame and thresholded image by half for display
        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        thresh_resized = cv2.resize(thresh, (thresh.shape[1] // 2, thresh.shape[0] // 2))

        # Display the original frame with the detected white pixels
        cv2.imshow('Video Stream', frame_resized)
        # Display the thresholded image
        cv2.imshow('Thresholded Image', thresh_resized)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
>>>>>>> c328291 (Added Project Files)
