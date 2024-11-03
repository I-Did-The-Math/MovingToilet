<<<<<<< HEAD
import math
import tkinter as tk
import serial

# Adjust these variables to match your setup
serial_port = 'COM3'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 2000000  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
current_speed = 0
directionX, directionY = 0, 0
speed = 0
is_moving = True

def send_command(directionX, directionY, speed):
    """Send command to the Arduino."""
    command = f"{directionX},{directionY},{speed}\n"
    ser.write(command.encode())

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude

def on_mouse_move(event):
    global directionX, directionY, speed, current_speed, is_moving

    # Calculate the center of the canvas
    center_x = canvas.winfo_width() // 2
    center_y = canvas.winfo_height() // 2

    # Calculate the vector from the center to the mouse position
    vector_x = event.x - center_x
    vector_x *= -1
    vector_y = center_y - event.y

    # Calculate the speed as the magnitude of the vector
    speed = int(math.sqrt(vector_x**2 + vector_y**2)) / 8
    current_speed = speed  # Store the current speed

    # Normalize the vector
    directionX, directionY = normalize_vector(vector_x, vector_y)

    if is_moving:
        # Display the current vector and speed
        vector_label.config(text=f"Vector: ({directionX:.2f}, {directionY:.2f}) Speed: {speed}")

        # Send the command to the Arduino
        send_command(directionX, directionY, speed)

def on_click(event):
    global is_moving, speed

    is_moving = not is_moving
    if not is_moving:
        speed = 0
    else:
        speed = current_speed

    # Display the updated vector and speed
    vector_label.config(text=f"Vector: ({directionX:.2f}, {directionY:.2f}) Speed: {speed}")

    # Send the command to the Arduino
    send_command(directionX, directionY, speed)

def on_quit():
    ser.close()
    root.quit()

# Set up the Tkinter window
root = tk.Tk()
root.title("CoreXY Gantry Control")

# Set up the canvas
canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack()

# Set up the label to display the vector and speed
vector_label = tk.Label(root, text="Vector: (0.00, 0.00) Speed: 0", font=("Helvetica", 14))
vector_label.pack(pady=20)

# Bind the mouse motion to the function
canvas.bind('<Motion>', on_mouse_move)

# Bind the mouse click to the function
canvas.bind('<Button-1>', on_click)

# Set up the quit button
quit_button = tk.Button(root, text="Quit", command=on_quit)
quit_button.pack(pady=20)

# Draw the coordinate grid
canvas.create_line(200, 0, 200, 400, fill="gray", dash=(4, 2))
canvas.create_line(0, 200, 400, 200, fill="gray", dash=(4, 2))

# Run the Tkinter event loop
root.mainloop()
=======
import math
import tkinter as tk
import serial

# Adjust these variables to match your setup
serial_port = 'COM3'  # Serial port Arduino is connected to (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)
baud_rate = 2000000  # Should match the baud rate in your Arduino sketch

# Initialize serial connection
ser = serial.Serial(serial_port, baud_rate)

# Initialize global variables
current_speed = 0
directionX, directionY = 0, 0
speed = 0
is_moving = True

def send_command(directionX, directionY, speed):
    """Send command to the Arduino."""
    command = f"{directionX},{directionY},{speed}\n"
    ser.write(command.encode())

def normalize_vector(x, y):
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude

def on_mouse_move(event):
    global directionX, directionY, speed, current_speed, is_moving

    # Calculate the center of the canvas
    center_x = canvas.winfo_width() // 2
    center_y = canvas.winfo_height() // 2

    # Calculate the vector from the center to the mouse position
    vector_x = event.x - center_x
    vector_x *= -1
    vector_y = center_y - event.y

    # Calculate the speed as the magnitude of the vector
    speed = int(math.sqrt(vector_x**2 + vector_y**2)) / 8
    current_speed = speed  # Store the current speed

    # Normalize the vector
    directionX, directionY = normalize_vector(vector_x, vector_y)

    if is_moving:
        # Display the current vector and speed
        vector_label.config(text=f"Vector: ({directionX:.2f}, {directionY:.2f}) Speed: {speed}")

        # Send the command to the Arduino
        send_command(directionX, directionY, speed)

def on_click(event):
    global is_moving, speed

    is_moving = not is_moving
    if not is_moving:
        speed = 0
    else:
        speed = current_speed

    # Display the updated vector and speed
    vector_label.config(text=f"Vector: ({directionX:.2f}, {directionY:.2f}) Speed: {speed}")

    # Send the command to the Arduino
    send_command(directionX, directionY, speed)

def on_quit():
    ser.close()
    root.quit()

# Set up the Tkinter window
root = tk.Tk()
root.title("CoreXY Gantry Control")

# Set up the canvas
canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack()

# Set up the label to display the vector and speed
vector_label = tk.Label(root, text="Vector: (0.00, 0.00) Speed: 0", font=("Helvetica", 14))
vector_label.pack(pady=20)

# Bind the mouse motion to the function
canvas.bind('<Motion>', on_mouse_move)

# Bind the mouse click to the function
canvas.bind('<Button-1>', on_click)

# Set up the quit button
quit_button = tk.Button(root, text="Quit", command=on_quit)
quit_button.pack(pady=20)

# Draw the coordinate grid
canvas.create_line(200, 0, 200, 400, fill="gray", dash=(4, 2))
canvas.create_line(0, 200, 400, 200, fill="gray", dash=(4, 2))

# Run the Tkinter event loop
root.mainloop()
>>>>>>> c328291 (Added Project Files)
