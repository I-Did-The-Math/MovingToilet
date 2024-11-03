<<<<<<< HEAD
import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from ctypes import c_uint8
import serial
import threading
import tkinter as tk

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get depth camera intrinsics
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx_d, fy_d = depth_intrinsics.fx, depth_intrinsics.fy
cx_d, cy_d = depth_intrinsics.ppx, depth_intrinsics.ppy

# Create an align object to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize variables to store the origin point
origin_point_3D = None
mode = 'detect_origin'  # Start in origin detection mode

# Deque to store the last few vertical angles for smoothing
angle_window_size = 10
vertical_angles = deque(maxlen=angle_window_size)

# Initialize the serial connection (change the COM port as necessary)
ser = serial.Serial('COM3', 2000000)  # Replace 'COM5' with your actual COM port
velocity = 0.0

def calculate_smoothed_angle(new_angle, angle_deque):
    angle_deque.append(new_angle)
    return sum(angle_deque) / len(angle_deque)

def read_velocity():
    global velocity
    while True:
        try:
            # Read the serial data from Arduino
            line = ser.readline().decode('utf-8').strip()
            print(f"Received line: {line}")  # Debug print to verify serial data
            if "Velocity:" in line:
                velocity = float(line.split(" ")[1])
        except Exception as e:
            print(f"Error: {e}")

# Start the thread to read velocity
thread = threading.Thread(target=read_velocity)
thread.daemon = True
thread.start()

def update_velocity_label():
    velocity_label.config(text=f"Velocity: {velocity:.2f} m/s")
    root.after(100, update_velocity_label)

# Initialize Tkinter window
root = tk.Tk()
root.title("Velocity Display")
velocity_label = tk.Label(root, text="Velocity: 0.00 m/s", font=("Helvetica", 24))
velocity_label.pack(pady=20)

# Start updating the velocity label
root.after(100, update_velocity_label)

def predict_landing(real_x, real_y, real_z, v, vertical_angle, horizontal_angle):
    g = 9.81  # Acceleration due to gravity (m/s²)
    target_height = 0.33  # Height of the floor (m)

    # Decompose the velocity into its components
    v_x = v * np.cos(np.radians(vertical_angle)) * np.cos(np.radians(horizontal_angle))
    v_y = v * np.cos(np.radians(vertical_angle)) * np.sin(np.radians(horizontal_angle))
    v_z = v * np.sin(np.radians(vertical_angle))

    # Calculate the time of flight until landing
    t = (-v_z + np.sqrt(v_z**2 + 2 * g * (real_z - target_height))) / g

    # Calculate the landing position
    x_land = real_x + v_x * t
    y_land = real_y + v_y * t

    return t, x_land, y_land

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Mask the bottom half of the image
        height, width = gray_image.shape
        mask = np.zeros_like(gray_image)
        mask[:height // 2, :] = 255
        masked_gray_image = cv2.bitwise_and(gray_image, mask)

        # Thresholding to find bright spots (retroreflective tape)
        _, thresh = cv2.threshold(masked_gray_image, 252, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), c_uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours and their centroids
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = [cv2.moments(contour) for contour in contours if cv2.moments(contour)['m00'] != 0]
        centroids = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in centroids]

        if mode == 'detect_origin':
            # Ensure there is exactly one bright spot
            if len(centroids) == 1:
                x_c, y_c = centroids[0]
                z = depth_frame.get_distance(x_c, y_c)
                if z > 0:  # Avoid division by zero and invalid depth values
                    origin_x = (x_c - cx_d) * z / fx_d
                    origin_y = (y_c - cy_d) * z / fy_d
                    origin_point_3D = (origin_x, origin_y, z)
                    print(f"Origin point detected at: {origin_point_3D}")

            cv2.putText(color_image, "Press 'o' to set origin point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check for key press to switch mode
            if cv2.waitKey(1) & 0xFF == ord('o'):
                if origin_point_3D is not None:
                    mode = 'track_targets'

        elif mode == 'track_targets':
            # Ensure we have exactly two bright spots (assuming they correspond to the retroreflective tape on the bottle)
            if len(centroids) < 2:
                continue

            # Sort centroids by area (assuming larger area corresponds to retroreflective tape)
            centroids.sort(key=lambda x: cv2.contourArea(np.array([x])), reverse=True)
            selected_centroids = centroids[:2]

            # Calculate 3D coordinates for each centroid relative to the origin point
            points_3D = []
            valid_centroids = []
            for c in selected_centroids:
                x_c, y_c = c
                # Get depth value at the corresponding point in the aligned depth image
                z = depth_frame.get_distance(x_c, y_c)
                if z > 0:  # Avoid division by zero and invalid depth values
                    x = (x_c - cx_d) * z / fx_d
                    y = (y_c - cy_d) * z / fy_d
                    relative_x = x - origin_point_3D[0]
                    relative_y = y - origin_point_3D[1]
                    relative_z = z - origin_point_3D[2]  # Keep the Z coordinate positive
                    if relative_z < 0:  # Ensure positive Z value
                        relative_z = -relative_z
                    points_3D.append((relative_x, relative_y, relative_z))
                    valid_centroids.append(c)

            if len(points_3D) == 2:
                # Assuming point 0 is the front (nozzle) and point 1 is the back
                front_point = points_3D[0]
                back_point = points_3D[1]

                # Calculate the differences to get the orientation
                diff_x = back_point[0] - front_point[0]
                diff_y = back_point[1] - front_point[1]
                diff_z = back_point[2] - front_point[2]

                # Calculate the angles
                vertical_angle = np.degrees(np.arctan2(diff_z, np.sqrt(diff_x**2 + diff_y**2)))
                horizontal_angle = np.degrees(np.arctan2(diff_y, diff_x))

                # Smooth the vertical angle using moving average
                smoothed_vertical_angle = calculate_smoothed_angle(vertical_angle, vertical_angles)

                # Predict the landing
                t_land, x_land, y_land = predict_landing(front_point[0], front_point[1], front_point[2], velocity, smoothed_vertical_angle, horizontal_angle)

                # Display the angles on the frame
                angle_text = f"Vertical Angle: {smoothed_vertical_angle:.2f} deg\n Horizontal Angle: {horizontal_angle:.2f} deg\nVelocity: {velocity:.2f} m/s"
                y0, dy = 30, 30
                for i, line in enumerate(angle_text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(color_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the predicted landing information
                landing_text = f"Time to Land: {t_land:.2f} s\nLanding Position: ({x_land:.2f} m, {y_land:.2f} m)"
                y0, dy = 150, 30
                #for i, line in enumerate(landing_text.split('\n')):
                   # y = y0 + i*dy
                   # cv2.putText(color_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the color image with centroids marked and coordinates drawn
            for i, c in enumerate(valid_centroids):
                cv2.circle(color_image, c, 5, (0, 255, 0), -1)
                text = f"({points_3D[i][0]:.2f}, {points_3D[i][1]:.2f}, {points_3D[i][2]:.2f})"
                cv2.putText(color_image, text, (c[0], c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Threshold Image', thresh)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    root.quit()
=======
import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from ctypes import c_uint8
import serial
import threading
import tkinter as tk

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Get depth camera intrinsics
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx_d, fy_d = depth_intrinsics.fx, depth_intrinsics.fy
cx_d, cy_d = depth_intrinsics.ppx, depth_intrinsics.ppy

# Create an align object to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize variables to store the origin point
origin_point_3D = None
mode = 'detect_origin'  # Start in origin detection mode

# Deque to store the last few vertical angles for smoothing
angle_window_size = 10
vertical_angles = deque(maxlen=angle_window_size)

# Initialize the serial connection (change the COM port as necessary)
ser = serial.Serial('COM3', 2000000)  # Replace 'COM5' with your actual COM port
velocity = 0.0

def calculate_smoothed_angle(new_angle, angle_deque):
    angle_deque.append(new_angle)
    return sum(angle_deque) / len(angle_deque)

def read_velocity():
    global velocity
    while True:
        try:
            # Read the serial data from Arduino
            line = ser.readline().decode('utf-8').strip()
            print(f"Received line: {line}")  # Debug print to verify serial data
            if "Velocity:" in line:
                velocity = float(line.split(" ")[1])
        except Exception as e:
            print(f"Error: {e}")

# Start the thread to read velocity
thread = threading.Thread(target=read_velocity)
thread.daemon = True
thread.start()

def update_velocity_label():
    velocity_label.config(text=f"Velocity: {velocity:.2f} m/s")
    root.after(100, update_velocity_label)

# Initialize Tkinter window
root = tk.Tk()
root.title("Velocity Display")
velocity_label = tk.Label(root, text="Velocity: 0.00 m/s", font=("Helvetica", 24))
velocity_label.pack(pady=20)

# Start updating the velocity label
root.after(100, update_velocity_label)

def predict_landing(real_x, real_y, real_z, v, vertical_angle, horizontal_angle):
    g = 9.81  # Acceleration due to gravity (m/s²)
    target_height = 0.33  # Height of the floor (m)

    # Decompose the velocity into its components
    v_x = v * np.cos(np.radians(vertical_angle)) * np.cos(np.radians(horizontal_angle))
    v_y = v * np.cos(np.radians(vertical_angle)) * np.sin(np.radians(horizontal_angle))
    v_z = v * np.sin(np.radians(vertical_angle))

    # Calculate the time of flight until landing
    t = (-v_z + np.sqrt(v_z**2 + 2 * g * (real_z - target_height))) / g

    # Calculate the landing position
    x_land = real_x + v_x * t
    y_land = real_y + v_y * t

    return t, x_land, y_land

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Mask the bottom half of the image
        height, width = gray_image.shape
        mask = np.zeros_like(gray_image)
        mask[:height // 2, :] = 255
        masked_gray_image = cv2.bitwise_and(gray_image, mask)

        # Thresholding to find bright spots (retroreflective tape)
        _, thresh = cv2.threshold(masked_gray_image, 252, 255, cv2.THRESH_BINARY)
        
        # Clean up
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), c_uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours and their centroids
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = [cv2.moments(contour) for contour in contours if cv2.moments(contour)['m00'] != 0]
        centroids = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in centroids]

        if mode == 'detect_origin':
            # Ensure there is exactly one bright spot
            if len(centroids) == 1:
                x_c, y_c = centroids[0]
                z = depth_frame.get_distance(x_c, y_c)
                if z > 0:  # Avoid division by zero and invalid depth values
                    origin_x = (x_c - cx_d) * z / fx_d
                    origin_y = (y_c - cy_d) * z / fy_d
                    origin_point_3D = (origin_x, origin_y, z)
                    print(f"Origin point detected at: {origin_point_3D}")

            cv2.putText(color_image, "Press 'o' to set origin point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check for key press to switch mode
            if cv2.waitKey(1) & 0xFF == ord('o'):
                if origin_point_3D is not None:
                    mode = 'track_targets'

        elif mode == 'track_targets':
            # Ensure we have exactly two bright spots (assuming they correspond to the retroreflective tape on the bottle)
            if len(centroids) < 2:
                continue

            # Sort centroids by area (assuming larger area corresponds to retroreflective tape)
            centroids.sort(key=lambda x: cv2.contourArea(np.array([x])), reverse=True)
            selected_centroids = centroids[:2]

            # Calculate 3D coordinates for each centroid relative to the origin point
            points_3D = []
            valid_centroids = []
            for c in selected_centroids:
                x_c, y_c = c
                # Get depth value at the corresponding point in the aligned depth image
                z = depth_frame.get_distance(x_c, y_c)
                if z > 0:  # Avoid division by zero and invalid depth values
                    x = (x_c - cx_d) * z / fx_d
                    y = (y_c - cy_d) * z / fy_d
                    relative_x = x - origin_point_3D[0]
                    relative_y = y - origin_point_3D[1]
                    relative_z = z - origin_point_3D[2]  # Keep the Z coordinate positive
                    if relative_z < 0:  # Ensure positive Z value
                        relative_z = -relative_z
                    points_3D.append((relative_x, relative_y, relative_z))
                    valid_centroids.append(c)

            if len(points_3D) == 2:
                # Assuming point 0 is the front (nozzle) and point 1 is the back
                front_point = points_3D[0]
                back_point = points_3D[1]

                # Calculate the differences to get the orientation
                diff_x = back_point[0] - front_point[0]
                diff_y = back_point[1] - front_point[1]
                diff_z = back_point[2] - front_point[2]

                # Calculate the angles
                vertical_angle = np.degrees(np.arctan2(diff_z, np.sqrt(diff_x**2 + diff_y**2)))
                horizontal_angle = np.degrees(np.arctan2(diff_y, diff_x))

                # Smooth the vertical angle using moving average
                smoothed_vertical_angle = calculate_smoothed_angle(vertical_angle, vertical_angles)

                # Predict the landing
                t_land, x_land, y_land = predict_landing(front_point[0], front_point[1], front_point[2], velocity, smoothed_vertical_angle, horizontal_angle)

                # Display the angles on the frame
                angle_text = f"Vertical Angle: {smoothed_vertical_angle:.2f} deg\n Horizontal Angle: {horizontal_angle:.2f} deg\nVelocity: {velocity:.2f} m/s"
                y0, dy = 30, 30
                for i, line in enumerate(angle_text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(color_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the predicted landing information
                landing_text = f"Time to Land: {t_land:.2f} s\nLanding Position: ({x_land:.2f} m, {y_land:.2f} m)"
                y0, dy = 150, 30
                #for i, line in enumerate(landing_text.split('\n')):
                   # y = y0 + i*dy
                   # cv2.putText(color_image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the color image with centroids marked and coordinates drawn
            for i, c in enumerate(valid_centroids):
                cv2.circle(color_image, c, 5, (0, 255, 0), -1)
                text = f"({points_3D[i][0]:.2f}, {points_3D[i][1]:.2f}, {points_3D[i][2]:.2f})"
                cv2.putText(color_image, text, (c[0], c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Color Image', color_image)
        cv2.imshow('Threshold Image', thresh)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    root.quit()
>>>>>>> c328291 (Added Project Files)
