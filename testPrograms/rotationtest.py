<<<<<<< HEAD
import serial
import tkinter as tk
import threading

# Setup Serial
ser = serial.Serial('COM3', 2000000, timeout=0.1)

# Setup Tkinter
root = tk.Tk()
root.title("Encoder Distance Tracker")
distance_label_A = tk.Label(root, text="Distance A: 0.00 mm", font=("Helvetica", 16))
distance_label_A.pack(pady=10)
distance_label_B = tk.Label(root, text="Distance B: 0.00 mm", font=("Helvetica", 16))
distance_label_B.pack(pady=10)

# Global variables to store distance data
distanceA = 0.0
distanceB = 0.0

def read_serial():
    global distanceA, distanceB
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("A,"):
            try:
                distanceA = float(line.split(",")[1])
            except ValueError:
                pass  # Handle conversion error if data is incomplete or corrupted
        elif line.startswith("B,"):
            try:
                distanceB = float(line.split(",")[1])
            except ValueError:
                pass  # Handle conversion error if data is incomplete or corrupted

def update_gui():
    distance_label_A.config(text=f"Distance A: {distanceA:.2f} mm")
    distance_label_B.config(text=f"Distance B: {distanceB:.2f} mm")
    root.after(10, update_gui)  # Schedule this function to be called again after 10 ms

# Start the serial reading thread
thread = threading.Thread(target=read_serial)
thread.daemon = True
thread.start()

# Start the Tkinter update loop
update_gui()
root.mainloop()
=======
import serial
import tkinter as tk
import threading

# Setup Serial
ser = serial.Serial('COM3', 2000000, timeout=0.1)

# Setup Tkinter
root = tk.Tk()
root.title("Encoder Distance Tracker")
distance_label_A = tk.Label(root, text="Distance A: 0.00 mm", font=("Helvetica", 16))
distance_label_A.pack(pady=10)
distance_label_B = tk.Label(root, text="Distance B: 0.00 mm", font=("Helvetica", 16))
distance_label_B.pack(pady=10)

# Global variables to store distance data
distanceA = 0.0
distanceB = 0.0

def read_serial():
    global distanceA, distanceB
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line.startswith("A,"):
            try:
                distanceA = float(line.split(",")[1])
            except ValueError:
                pass  # Handle conversion error if data is incomplete or corrupted
        elif line.startswith("B,"):
            try:
                distanceB = float(line.split(",")[1])
            except ValueError:
                pass  # Handle conversion error if data is incomplete or corrupted

def update_gui():
    distance_label_A.config(text=f"Distance A: {distanceA:.2f} mm")
    distance_label_B.config(text=f"Distance B: {distanceB:.2f} mm")
    root.after(10, update_gui)  # Schedule this function to be called again after 10 ms

# Start the serial reading thread
thread = threading.Thread(target=read_serial)
thread.daemon = True
thread.start()

# Start the Tkinter update loop
update_gui()
root.mainloop()
>>>>>>> c328291 (Added Project Files)
