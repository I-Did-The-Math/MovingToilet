<<<<<<< HEAD
import cv2
import numpy as np
import math
import serial
import threading
import time

# Adjust these variables to match your setup
serial_port = 'COM3'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 2000000  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
encoder_position_x, encoder_position_y = 0.0, 0.0
initial_position_x, initial_position_y = 0.0, 0.0
stop_threads = False

# Adjust these variables to match your setup
frame_width = 1280
frame_height = 720

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
    global encoder_position_x, encoder_position_y
    while not stop_threads:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("X,"):
                try:
                    parts = line.split(",")
                    delta_x = float(parts[1]) / 1000  # Convert mm to meters
                    delta_y = float(parts[3]) / 1000  # Convert mm to meters
                    encoder_position_x = initial_position_x + delta_x
                    encoder_position_y = initial_position_y + delta_y

                    print(f"Encoder Position Updated: X: {encoder_position_x:.2f} m, Y: {encoder_position_y:.2f} m")
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
        except UnicodeDecodeError:
            continue  # Skip any lines that cause a UnicodeDecodeError

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0

# Start the serial reading thread
serial_thread = threading.Thread(target=read_serial)
serial_thread.daemon = True
serial_thread.start()

cv2.namedWindow('Video Stream')

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

            if not initial_position_set:
                initial_position_x = real_x
                initial_position_y = real_y
                initial_position_set = True

            # Draw a circle at the offset center of the white pixels
            cv2.circle(frame, offset_center, 5, (0, 255, 0), -1)

            # Write the real-life coordinates on the frame for the GoPro position
            text_real_life = f"GoPro Real Life: ({real_x:.2f} m, {real_y:.2f} m)"
            cv2.putText(frame, text_real_life, (offset_center[0] + 20, offset_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Write the pixel coordinates on the frame for the GoPro position
            text_pixel = f"GoPro Pixel: [{offset_center[0]}, {offset_center[1]}]"
            cv2.putText(frame, text_pixel, (offset_center[0] + 20, offset_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Calculate the encoder position in pixel coordinates
            encoder_pixel_x, encoder_pixel_y = real_world_to_pixel((encoder_position_x, encoder_position_y), H_inv)

            # Draw a circle at the encoder position
            cv2.circle(frame, (encoder_pixel_x, encoder_pixel_y), 5, (255, 0, 0), -1)

            # Write the real-life coordinates on the frame for the encoder position
            text_encoder_life = f"Encoder Real Life: ({encoder_position_x:.2f} m, {encoder_position_y:.2f} m)"
            cv2.putText(frame, text_encoder_life, (encoder_pixel_x + 20, encoder_pixel_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

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
    stop_threads = True
    ser.close()  # Ensure the serial connection is closed before joining the thread
    serial_thread.join()
    cap.release()
    cv2.destroyAllWindows()
=======
import cv2
import numpy as np
import math
import serial
import threading
import time

# Adjust these variables to match your setup
serial_port = 'COM3'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 2000000  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
encoder_position_x, encoder_position_y = 0.0, 0.0
initial_position_x, initial_position_y = 0.0, 0.0
stop_threads = False

# Adjust these variables to match your setup
frame_width = 1280
frame_height = 720

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
    global encoder_position_x, encoder_position_y
    while not stop_threads:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("X,"):
                try:
                    parts = line.split(",")
                    delta_x = float(parts[1]) / 1000  # Convert mm to meters
                    delta_y = float(parts[3]) / 1000  # Convert mm to meters
                    encoder_position_x = initial_position_x + delta_x
                    encoder_position_y = initial_position_y + delta_y

                    print(f"Encoder Position Updated: X: {encoder_position_x:.2f} m, Y: {encoder_position_y:.2f} m")
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
        except UnicodeDecodeError:
            continue  # Skip any lines that cause a UnicodeDecodeError

# Initialize variables
cursor_x, cursor_y = 0, 0
cursor_real_x, cursor_real_y = 0, 0

# Start the serial reading thread
serial_thread = threading.Thread(target=read_serial)
serial_thread.daemon = True
serial_thread.start()

cv2.namedWindow('Video Stream')

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

            if not initial_position_set:
                initial_position_x = real_x
                initial_position_y = real_y
                initial_position_set = True

            # Draw a circle at the offset center of the white pixels
            cv2.circle(frame, offset_center, 5, (0, 255, 0), -1)

            # Write the real-life coordinates on the frame for the GoPro position
            text_real_life = f"GoPro Real Life: ({real_x:.2f} m, {real_y:.2f} m)"
            cv2.putText(frame, text_real_life, (offset_center[0] + 20, offset_center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Write the pixel coordinates on the frame for the GoPro position
            text_pixel = f"GoPro Pixel: [{offset_center[0]}, {offset_center[1]}]"
            cv2.putText(frame, text_pixel, (offset_center[0] + 20, offset_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Calculate the encoder position in pixel coordinates
            encoder_pixel_x, encoder_pixel_y = real_world_to_pixel((encoder_position_x, encoder_position_y), H_inv)

            # Draw a circle at the encoder position
            cv2.circle(frame, (encoder_pixel_x, encoder_pixel_y), 5, (255, 0, 0), -1)

            # Write the real-life coordinates on the frame for the encoder position
            text_encoder_life = f"Encoder Real Life: ({encoder_position_x:.2f} m, {encoder_position_y:.2f} m)"
            cv2.putText(frame, text_encoder_life, (encoder_pixel_x + 20, encoder_pixel_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

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
    stop_threads = True
    ser.close()  # Ensure the serial connection is closed before joining the thread
    serial_thread.join()
    cap.release()
    cv2.destroyAllWindows()
>>>>>>> c328291 (Added Project Files)
