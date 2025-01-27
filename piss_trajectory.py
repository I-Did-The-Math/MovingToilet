import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
from ctypes import c_uint8
import serial
import threading

#Start GoPro Camera
gopro_frame_width = 1280
gopro_frame_height = 720
gopro_obs_camera_index = 3 
gopro_feed = cv2.VideoCapture(gopro_obs_camera_index, cv2.CAP_DSHOW)
if not gopro_feed.isOpened():
    print(f"Error: Could not open virtual camera at index {gopro_obs_camera_index}.")
    exit()
print(f"Successfully opened virtual camera at index {gopro_obs_camera_index}.")
gopro_feed.set(cv2.CAP_PROP_FRAME_WIDTH, gopro_frame_width)
gopro_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, gopro_frame_height)

#Start Depth Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

#Grab Depth Camera Information
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx_d, fy_d = depth_intrinsics.fx, depth_intrinsics.fy
cx_d, cy_d = depth_intrinsics.ppx, depth_intrinsics.ppy

#Match Color Pixels To Depth Pixels
align_to = rs.stream.color
align = rs.align(align_to)

origin_point_3D = None
mode = 'detect_origin'

# Deque to store the last few vertical angles for smoothing
pee_vertical_angle_history_size = 10
pee_vertical_angle_history = deque(maxlen=pee_vertical_angle_history_size)

#Connect To Arduino
arduino_serial_port = 'COM3'
arduino_baud_rate = 2000000
arduino = serial.Serial(arduino_serial_port, arduino_baud_rate)
pee_initial_speed = 0.0

def calculate_smoothed_angle(new_angle, angle_deque):
    angle_deque.append(new_angle)
    return sum(angle_deque) / len(angle_deque)

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
thread = threading.Thread(target=read_velocity)
thread.daemon = True
thread.start()

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

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
        kernel_size = 7
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

            # Sort centroids by Y coordinate to identify the bottom point
            centroids.sort(key=lambda x: x[1])
            bottom_centroid = centroids[1]  # Assume the second centroid is the bottom point
            top_centroid = centroids[0]  # Assume the first centroid is the to` p point

            # Calculate 3D coordinates for the bottom centroid relative to the origin point
            x_c, y_c = bottom_centroid
            z = depth_frame.get_distance(x_c, y_c)
            if z > 0:  # Avoid division by zero and invalid depth values
                x = (x_c - cx_d) * z / fx_d
                y = (y_c - cy_d) * z / fy_d
                relative_x = x - origin_point_3D[0]
                relative_y = y - origin_point_3D[1]
                relative_z = z - origin_point_3D[2]  # Keep the Z coordinate positive
                if relative_z < 0:  # Ensure positive Z value
                    relative_z = -relative_z
                bottle_tip_real_coords = (relative_x, relative_y, relative_z)

                # Calculate 3D coordinates for the top centroid relative to the origin point
                x_c_top, y_c_top = top_centroid
                z_top = depth_frame.get_distance(x_c_top, y_c_top)
                if z_top > 0:  # Avoid division by zero and invalid depth values
                    x_top = (x_c_top - cx_d) * z_top / fx_d
                    y_top = (y_c_top - cy_d) * z_top / fy_d
                    relative_x_top = x_top - origin_point_3D[0]
                    relative_y_top = y_top - origin_point_3D[1]
                    relative_z_top = z_top - origin_point_3D[2]
                    if relative_z_top < 0:  # Ensure positive Z value
                        relative_z_top = -relative_z_top
                    bottle_mid_real_coords = (relative_x_top, relative_y_top, relative_z_top)

                    # Calculate the differences to get the orientation
                    diff_x = bottle_tip_real_coords[0] - bottle_mid_real_coords[0]
                    diff_y = bottle_tip_real_coords[1] - bottle_mid_real_coords[1]
                    diff_z = bottle_tip_real_coords[2] - bottle_mid_real_coords[2]

                    # Calculate the angles
                    vertical_angle = np.degrees(np.arctan2(diff_z, np.sqrt(diff_x**2 + diff_y**2)))
                    horizontal_angle = np.degrees(np.arctan2(diff_y,diff_x))

                    # Smooth the vertical angle using moving average
                    smoothed_vertical_angle = calculate_smoothed_angle(vertical_angle, pee_vertical_angle_history)

                    # Display the angles on the frame
                    angle_text = f"Vertical Angle: {smoothed_vertical_angle:.2f} deg\nHorizontal Angle: {horizontal_angle:.2f} deg"
                    y0, dy = 15, 15
                    for i, line in enumerate(angle_text.split('\n')):
                        y = y0 + i*dy 
                        cv2.putText(color_image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the 3D coordinates on the frame
                coord_text = f"Bottle Tip: ({bottle_tip_real_coords[0]:.2f}, {bottle_tip_real_coords[1]:.2f}, {bottle_tip_real_coords[2]:.2f})"
                cv2.putText(color_image, coord_text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Display the velocity on the frame
                velocity_text = f"Velocity: {pee_initial_speed:.2f} m/s"
                cv2.putText(color_image, velocity_text, (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw the bottom point on the frame
                cv2.circle(color_image, bottom_centroid, 5, (0, 255, 0), -1)


        # Show the current processed images
        cv2.imshow('Human Vision', color_image)
        cv2.imshow('Reflective Tape', thresh)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming and close all resources
    pipeline.stop()
    cv2.destroyAllWindows()
    arduino.close()
