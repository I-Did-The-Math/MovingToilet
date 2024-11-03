import cv2

# Replace with the correct index for the OBS virtual camera
virtual_camera_index = 3  # You may need to adjust this index

# Open the virtual camera
cap = cv2.VideoCapture(virtual_camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open virtual camera at index {virtual_camera_index}.")
    exit()

print(f"Successfully opened virtual camera at index {virtual_camera_index}.")

# Desired resolution
frame_width = 1280
frame_height = 720

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Display the video feed
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the frame
        cv2.imshow('OBS Virtual Camera Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
