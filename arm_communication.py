import serial
import time

esp32 = serial.Serial("COM3", 115200)

time.sleep(2)

message = "5.5, 0.0, 0.5\n"
esp32.write(message.encode())

esp32.close()