<<<<<<< HEAD
import cv2
import numpy as np

def draw_optical_flow(frame, flow, step=16, scale=1.5):
    """ Draws arrows showing optical flow. """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    # Create line endpoints
    lines = np.vstack([x, y, x+fx*scale, y+fy*scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw lines and circle for current positions
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)
    return vis

# Open video file
cap = cv2.VideoCapture('60fpsshorttest.mp4')
ret, prev_frame = cap.read()
if not ret:
    print("Error: Video file could not be opened or is empty.")
    exit(1)

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Prepare video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_2.mp4', fourcc, 20.0, (prev_frame.shape[1], prev_frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file or error in reading frames.")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualize optical flow
    vis = draw_optical_flow(frame_gray, flow)

    # Write frame to output video
    out.write(vis)

    # Update previous frame
    prev_frame_gray = frame_gray

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
=======
import cv2
import numpy as np

def draw_optical_flow(frame, flow, step=16, scale=1.5):
    """ Draws arrows showing optical flow. """
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    # Create line endpoints
    lines = np.vstack([x, y, x+fx*scale, y+fy*scale]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw lines and circle for current positions
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (255, 0, 0), -1)
    return vis

# Open video file
cap = cv2.VideoCapture('60fpsshorttest.mp4')
ret, prev_frame = cap.read()
if not ret:
    print("Error: Video file could not be opened or is empty.")
    exit(1)

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Prepare video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_2.mp4', fourcc, 20.0, (prev_frame.shape[1], prev_frame.shape[0]))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file or error in reading frames.")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualize optical flow
    vis = draw_optical_flow(frame_gray, flow)

    # Write frame to output video
    out.write(vis)

    # Update previous frame
    prev_frame_gray = frame_gray

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
>>>>>>> c328291 (Added Project Files)
