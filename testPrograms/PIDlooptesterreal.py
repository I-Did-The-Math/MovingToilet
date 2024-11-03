<<<<<<< HEAD
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Constants for PID control
Kp = 1
Ki = 0.1
Kd = 0.1

# Initial conditions
initial_distance = 0.0  # mm
target_distance = 200.0  # mm
distance = initial_distance
integral = 0.0
previousError = 0.0
previousTime = 0.0
speed = 0.0  # Initial speed

# Time step
time_step = 0.02  # 20 ms

# Object mass (kg)
mass = 2.2 / 2.205  # Convert pounds to kg

# Data lists for plotting
time_data = [0]
distance_data = [initial_distance]
speed_data = [0]
output_data = [0]
accelerationScaler = 10;

# Function to calculate force from PWM power
def power_to_force(power):
    sign = 1
    if power < 0:
        sign = -1
    power = abs(power)
    # Force from one motor
    force_per_motor = 1.25 / 0.0275  # Nm / m (torque / pulley radius)
    # Total force from both motors, scaled by power
    return sign * 2 * force_per_motor * (power / 255.0) * accelerationScaler # Linearly scale with PWM duty cycle

# Function to calculate max speed from PWM power
def power_to_max_speed(power):
    sign = 1
    if power < 0:
        sign = -1
    power = abs(power)
    return sign * (0.026 * power - 0.199) * 1000  # Convert m/s to mm/s

# Flag to control the simulation
continue_simulation = True

def update_simulation():
    global distance, integral, previousError, previousTime, continue_simulation, speed, after_id

    if not continue_simulation:
        return  # Stop further updates

    currentTime = previousTime + time_step
    elapsedTime = time_step

    error = target_distance - distance
    integral += error * elapsedTime
    derivative = (error - previousError) / elapsedTime

    output = Kp * error + Ki * integral + Kd * derivative
    previousError = error

    # Calculate force and resulting acceleration
    force = power_to_force(output)
    acceleration = force / mass

    # Update the speed using the acceleration
    speed += acceleration * elapsedTime

    # Calculate the maximum speed for the current power
    max_speed = power_to_max_speed(output)

    # Cap the speed based on the maximum speed for the given power
    if abs(speed) > abs(max_speed):
        speed = np.sign(speed) * abs(max_speed)

    # Adjust the distance
    distance += speed * elapsedTime

    # Append data for plotting
    time_data.append(currentTime)
    distance_data.append(distance)
    speed_data.append(speed)
    output_data.append(output)

    # Update the plot
    ax.clear()
    ax.plot(time_data, distance_data, label="Distance")
    ax.plot(time_data, speed_data, label="Speed")
    ax.plot(time_data, output_data, label="PID Output")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (mm) / Speed (mm/s) / PID Output')
    ax.set_ylim(-100, target_distance + 100)  # Adjust y-axis range
    ax.set_xlim(0, max(time_data) + 0.1)  # Adjust x-axis
    ax.legend()

    # Redraw the canvas
    canvas.draw()

    # Update previous time
    previousTime = currentTime

    if currentTime > 3.0:  # Stop after 3 seconds for visualization
        continue_simulation = False
    else:
        distance_label.config(text=f"Distance: {distance:.2f} mm")
        after_id = root.after(int(time_step * 1000), update_simulation)  # Continue simulation

    # Debug print statements
    print(f"Time: {currentTime}, Distance: {distance}, Speed: {speed}, Output: {output}, Force: {force}, Acceleration: {acceleration}")

# Set up the GUI
root = tk.Tk()
root.title("PID Control Simulation")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Distance label
distance_label = ttk.Label(frame, text=f"Distance: {initial_distance:.2f} mm")
distance_label.grid(row=0, column=0, pady=10)

# Matplotlib figure and canvas
fig, ax = plt.subplots()
ax.set_ylim(-50, target_distance + 50)
ax.set_xlim(0, 1)
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=1, column=0)
canvas.draw()

# Start the simulation
continue_simulation = True
after_id = root.after(int(time_step * 1000), update_simulation)

# Ensure clean exit
def on_closing():
    global continue_simulation, after_id
    continue_simulation = False
    if after_id:
        root.after_cancel(after_id)  # Cancel the after callback if it exists
    root.quit()  # Exit the Tkinter loop
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the Tkinter event loop
root.mainloop()
=======
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Constants for PID control
Kp = 1
Ki = 0.1
Kd = 0.1

# Initial conditions
initial_distance = 0.0  # mm
target_distance = 200.0  # mm
distance = initial_distance
integral = 0.0
previousError = 0.0
previousTime = 0.0
speed = 0.0  # Initial speed

# Time step
time_step = 0.02  # 20 ms

# Object mass (kg)
mass = 2.2 / 2.205  # Convert pounds to kg

# Data lists for plotting
time_data = [0]
distance_data = [initial_distance]
speed_data = [0]
output_data = [0]
accelerationScaler = 10;

# Function to calculate force from PWM power
def power_to_force(power):
    sign = 1
    if power < 0:
        sign = -1
    power = abs(power)
    # Force from one motor
    force_per_motor = 1.25 / 0.0275  # Nm / m (torque / pulley radius)
    # Total force from both motors, scaled by power
    return sign * 2 * force_per_motor * (power / 255.0) * accelerationScaler # Linearly scale with PWM duty cycle

# Function to calculate max speed from PWM power
def power_to_max_speed(power):
    sign = 1
    if power < 0:
        sign = -1
    power = abs(power)
    return sign * (0.026 * power - 0.199) * 1000  # Convert m/s to mm/s

# Flag to control the simulation
continue_simulation = True

def update_simulation():
    global distance, integral, previousError, previousTime, continue_simulation, speed, after_id

    if not continue_simulation:
        return  # Stop further updates

    currentTime = previousTime + time_step
    elapsedTime = time_step

    error = target_distance - distance
    integral += error * elapsedTime
    derivative = (error - previousError) / elapsedTime

    output = Kp * error + Ki * integral + Kd * derivative
    previousError = error

    # Calculate force and resulting acceleration
    force = power_to_force(output)
    acceleration = force / mass

    # Update the speed using the acceleration
    speed += acceleration * elapsedTime

    # Calculate the maximum speed for the current power
    max_speed = power_to_max_speed(output)

    # Cap the speed based on the maximum speed for the given power
    if abs(speed) > abs(max_speed):
        speed = np.sign(speed) * abs(max_speed)

    # Adjust the distance
    distance += speed * elapsedTime

    # Append data for plotting
    time_data.append(currentTime)
    distance_data.append(distance)
    speed_data.append(speed)
    output_data.append(output)

    # Update the plot
    ax.clear()
    ax.plot(time_data, distance_data, label="Distance")
    ax.plot(time_data, speed_data, label="Speed")
    ax.plot(time_data, output_data, label="PID Output")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (mm) / Speed (mm/s) / PID Output')
    ax.set_ylim(-100, target_distance + 100)  # Adjust y-axis range
    ax.set_xlim(0, max(time_data) + 0.1)  # Adjust x-axis
    ax.legend()

    # Redraw the canvas
    canvas.draw()

    # Update previous time
    previousTime = currentTime

    if currentTime > 3.0:  # Stop after 3 seconds for visualization
        continue_simulation = False
    else:
        distance_label.config(text=f"Distance: {distance:.2f} mm")
        after_id = root.after(int(time_step * 1000), update_simulation)  # Continue simulation

    # Debug print statements
    print(f"Time: {currentTime}, Distance: {distance}, Speed: {speed}, Output: {output}, Force: {force}, Acceleration: {acceleration}")

# Set up the GUI
root = tk.Tk()
root.title("PID Control Simulation")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Distance label
distance_label = ttk.Label(frame, text=f"Distance: {initial_distance:.2f} mm")
distance_label.grid(row=0, column=0, pady=10)

# Matplotlib figure and canvas
fig, ax = plt.subplots()
ax.set_ylim(-50, target_distance + 50)
ax.set_xlim(0, 1)
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=1, column=0)
canvas.draw()

# Start the simulation
continue_simulation = True
after_id = root.after(int(time_step * 1000), update_simulation)

# Ensure clean exit
def on_closing():
    global continue_simulation, after_id
    continue_simulation = False
    if after_id:
        root.after_cancel(after_id)  # Cancel the after callback if it exists
    root.quit()  # Exit the Tkinter loop
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the Tkinter event loop
root.mainloop()
>>>>>>> c328291 (Added Project Files)
