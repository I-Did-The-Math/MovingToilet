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
        try:
            response = ser.readline().decode().strip()
            if response:
                print(f"Arduino response: {response}")
                if "Movement completed in" in response:
                    break
        except UnicodeDecodeError as e:
            print(f"Decoding error: {e}")
            raw_data = ser.readline()
            print(f"Raw data: {raw_data}")

def main():
    # Serial port configuration
    port = 'COM3'  # Change this to your Arduino's COM port
    baudrate = 2000000

    distance = 200
    # Movement pattern
    movements = [
        (distance, distance), (-distance, -distance), (distance, 0), (-distance, distance),
        (distance, -distance), (-distance, 0)
    ]

    # Open serial port
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(2)  # Wait for the serial connection to initialize

        while True:
            try:
                for deltaX, deltaY in movements:
                    # Send command to Arduino
                    send_movement_command(ser, deltaX, deltaY)

                    # Wait for movement completion
                    wait_for_completion(ser)

            except KeyboardInterrupt:
                print("Exiting...")
                break

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
        try:
            response = ser.readline().decode().strip()
            if response:
                print(f"Arduino response: {response}")
                if "Movement completed in" in response:
                    break
        except UnicodeDecodeError as e:
            print(f"Decoding error: {e}")
            raw_data = ser.readline()
            print(f"Raw data: {raw_data}")

def main():
    # Serial port configuration
    port = 'COM3'  # Change this to your Arduino's COM port
    baudrate = 2000000

    distance = 200
    # Movement pattern
    movements = [
        (distance, distance), (-distance, -distance), (distance, 0), (-distance, distance),
        (distance, -distance), (-distance, 0)
    ]

    # Open serial port
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(2)  # Wait for the serial connection to initialize

        while True:
            try:
                for deltaX, deltaY in movements:
                    # Send command to Arduino
                    send_movement_command(ser, deltaX, deltaY)

                    # Wait for movement completion
                    wait_for_completion(ser)

            except KeyboardInterrupt:
                print("Exiting...")
                break

if __name__ == "__main__":
    main()
>>>>>>> c328291 (Added Project Files)
