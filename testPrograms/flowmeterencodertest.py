<<<<<<< HEAD
import serial
import tkinter as tk
import threading

# Setup Serial
ser = serial.Serial('COM3', 2000000, timeout=0.1)  # Match the baud rate

# Setup Tkinter
root = tk.Tk()
root.title("Encoder and Flowmeter Tracker")

# Position labels
position_label_X = tk.Label(root, text="Position X: 0.00 mm", font=("Helvetica", 16))
position_label_X.pack(pady=10)
position_label_Y = tk.Label(root, text="Position Y: 0.00 mm", font=("Helvetica", 16))
position_label_Y.pack(pady=10)

# Flowmeter label
velocity_label = tk.Label(root, text="Flow Velocity: 0.00 m/s", font=("Helvetica", 16))
velocity_label.pack(pady=20)

# Global variables to store position and velocity data
positionX = 0.0
positionY = 0.0
velocity = 0.0

def read_serial():
    global positionX, positionY, velocity
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("X,"):
                try:
                    parts = line.split(",")
                    positionX = float(parts[1])
                    positionY = float(parts[3])
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
            elif line.startswith("Velocity,"):
                try:
                    velocity = float(line.split(",")[1])
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
        except UnicodeDecodeError:
            continue  # Skip any lines that cause a UnicodeDecodeError

def update_gui():
    position_label_X.config(text=f"Position X: {positionX:.2f} mm")
    position_label_Y.config(text=f"Position Y: {positionY:.2f} mm")
    velocity_label.config(text=f"Flow Velocity: {velocity:.2f} m/s")
    root.after(10, update_gui)  # Schedule this function to be called again after 10 ms

# Start the serial reading thread
thread = threading.Thread(target=read_serial)
thread.daemon = True
thread.start()

# Start the Tkinter update loop
update_gui()
root.mainloop()

import serial
import tkinter as tk
import threading

# Setup Serial
ser = serial.Serial('COM3', 2000000, timeout=0.1)  # Match the baud rate

# Setup Tkinter
root = tk.Tk()
root.title("Encoder and Flowmeter Tracker")

# Position labels
position_label_X = tk.Label(root, text="Position X: 0.00 mm", font=("Helvetica", 16))
position_label_X.pack(pady=10)
position_label_Y = tk.Label(root, text="Position Y: 0.00 mm", font=("Helvetica", 16))
position_label_Y.pack(pady=10)

# Flowmeter label
velocity_label = tk.Label(root, text="Flow Velocity: 0.00 m/s", font=("Helvetica", 16))
velocity_label.pack(pady=20)

# Global variables to store position and velocity data
positionX = 0.0
positionY = 0.0
velocity = 0.0

def read_serial():
    global positionX, positionY, velocity
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("X,"):
                try:
                    parts = line.split(",")
                    positionX = float(parts[1])
                    positionY = float(parts[3])
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
            elif line.startswith("Velocity,"):
                try:
                    velocity = float(line.split(",")[1])
                except ValueError:
                    pass  # Handle conversion error if data is incomplete or corrupted
        except UnicodeDecodeError:
            continue  # Skip any lines that cause a UnicodeDecodeError

def update_gui():
    position_label_X.config(text=f"Position X: {positionX:.2f} mm")
    position_label_Y.config(text=f"Position Y: {positionY:.2f} mm")
    velocity_label.config(text=f"Flow Velocity: {velocity:.2f} m/s")
    root.after(10, update_gui)  # Schedule this function to be called again after 10 ms

# Start the serial reading thread
thread = threading.Thread(target=read_serial)
thread.daemon = True
thread.start()

# Start the Tkinter update loop
update_gui()
root.mainloop()
>>>>>>> c328291 (Added Project Files)
