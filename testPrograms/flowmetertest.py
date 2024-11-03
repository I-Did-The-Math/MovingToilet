<<<<<<< HEAD
import serial
import threading
import time
import tkinter as tk

# Initialize the serial connection (change the COM port as necessary)
ser = serial.Serial('COM3', 2000000)  # Replace 'COM5' with your actual COM port
velocity = 0.0

def read_velocity():
    global velocity
    while True:
        try:
            # Read the serial data from Arduino
            line = ser.readline().decode('utf-8').strip()
            if "Velocity:" in line:
                velocity = float(line.split(" ")[1])
        except Exception as e:
            print(f"Error: {e}")

def update_velocity_label():
    velocity_label.config(text=f"Current Velocity: {velocity:.2f} m/s")
    root.after(100, update_velocity_label)  # Update every 100 ms

# Start the thread to read velocity
thread = threading.Thread(target=read_velocity)
thread.daemon = True
thread.start()

# Setup the Tkinter window
root = tk.Tk()
root.title("Velocity Display")

velocity_label = tk.Label(root, text="Current Velocity: 0.00 m/s", font=("Helvetica", 16))
velocity_label.pack(pady=20)

# Start the update loop
update_velocity_label()

# Run the Tkinter main loop
root.mainloop()

# Close the serial connection when the window is closed
ser.close()
=======
import serial
import threading
import time
import tkinter as tk

# Initialize the serial connection (change the COM port as necessary)
ser = serial.Serial('COM3', 2000000)  # Replace 'COM5' with your actual COM port
velocity = 0.0

def read_velocity():
    global velocity
    while True:
        try:
            # Read the serial data from Arduino
            line = ser.readline().decode('utf-8').strip()
            if "Velocity:" in line:
                velocity = float(line.split(" ")[1])
        except Exception as e:
            print(f"Error: {e}")

def update_velocity_label():
    velocity_label.config(text=f"Current Velocity: {velocity:.2f} m/s")
    root.after(100, update_velocity_label)  # Update every 100 ms

# Start the thread to read velocity
thread = threading.Thread(target=read_velocity)
thread.daemon = True
thread.start()

# Setup the Tkinter window
root = tk.Tk()
root.title("Velocity Display")

velocity_label = tk.Label(root, text="Current Velocity: 0.00 m/s", font=("Helvetica", 16))
velocity_label.pack(pady=20)

# Start the update loop
update_velocity_label()

# Run the Tkinter main loop
root.mainloop()

# Close the serial connection when the window is closed
ser.close()
>>>>>>> c328291 (Added Project Files)
