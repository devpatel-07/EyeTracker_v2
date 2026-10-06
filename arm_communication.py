import atexit
import serial
import time
from coord_convert import convert_coords

# Serial connection to the ESP32. Opened once on the first goToPoint call and kept open,
# since opening the port can reset the ESP32 and any message sent while it boots is lost
esp32 = None

# Opens the serial port without resetting the ESP32, then waits for it to be ready - function
def open_connection():
    global esp32
    if esp32 is not None and esp32.is_open:
        return esp32

    # DTR/RTS are set low before opening so the ESP32 auto-reset circuit is not triggered
    esp32 = serial.Serial()
    esp32.port = "COM3"
    esp32.baudrate = 115200
    esp32.timeout = 1
    esp32.dtr = False
    esp32.rts = False
    esp32.open()

    # Gives the ESP32 time to finish booting in case it reset anyway
    time.sleep(2)
    esp32.reset_input_buffer()
    return esp32

# Closes the serial port when the program exits - function
def close_connection():
    if esp32 is not None and esp32.is_open:
        esp32.close()

atexit.register(close_connection)

def goToPoint(gx, gy, gz,d,h,o):
    ser = open_connection()
    x, y, z = convert_coords(gx,gy,gz,d,h,o)
    message = f"{x}, {y}, {z}\n"
    ser.write(message.encode())

    # Waits until the whole message has been sent
    ser.flush()
