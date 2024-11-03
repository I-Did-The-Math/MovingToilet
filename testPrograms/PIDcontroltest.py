<<<<<<< HEAD
import serial
import time

# Function to send movement command to Arduino
def send_movement_command(ser, deltaX, deltaY):
    command = f"{deltaX},{deltaY}\n"
    ser.write(command.encode())
    print(f"Sent: {command}")

# Function to wait for movement completion
def wait_for_completion(ser):
    while True:
        response = ser.readline().decode().strip()
        if response:
            print(response)
            if "Movement completed in" in response:
                break

def main():
    # Serial port configuration
    port = 'COM3'  # Change this to your Arduino's COM port
    baudrate = 2000000

    # Open serial port
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(2)  # Wait for the serial connection to initialize

        while True:
            try:
                # Get user input
                deltaX = float(input("Enter deltaX (in mm): "))
                deltaY = float(input("Enter deltaY (in mm): "))

                # Send command to Arduino
                send_movement_command(ser, deltaX, deltaY)

                # Wait for movement completion
                wait_for_completion(ser)

            except KeyboardInterrupt:
                print("Exiting...")
                break
            except ValueError:
                print("Invalid input. Please enter numerical values.")

if __name__ == "__main__":
    main()
=======
import serial
import time

# Function to send movement command to Arduino
def send_movement_command(ser, deltaX, deltaY):
    command = f"{deltaX},{deltaY}\n"
    ser.write(command.encode())
    print(f"Sent: {command}")

# Function to wait for movement completion
def wait_for_completion(ser):
    while True:
        response = ser.readline().decode().strip()
        if response:
            print(response)
            if "Movement completed in" in response:
                break

def main():
    # Serial port configuration
    port = 'COM3'  # Change this to your Arduino's COM port
    baudrate = 2000000

    # Open serial port
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(2)  # Wait for the serial connection to initialize

        while True:
            try:
                # Get user input
                deltaX = float(input("Enter deltaX (in mm): "))
                deltaY = float(input("Enter deltaY (in mm): "))

                # Send command to Arduino
                send_movement_command(ser, deltaX, deltaY)

                # Wait for movement completion
                wait_for_completion(ser)

            except KeyboardInterrupt:
                print("Exiting...")
                break
            except ValueError:
                print("Invalid input. Please enter numerical values.")

if __name__ == "__main__":
    main()
>>>>>>> c328291 (Added Project Files)
100