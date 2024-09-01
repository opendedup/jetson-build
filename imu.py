import time
import board
import busio
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX
from micropython import const
i2c = busio.I2C(board.SCL_1, board.SDA_1)
print("-" * 40)
print("I2C SCAN")
print("-" * 40)

name = "i2c"

while not i2c.try_lock():
    pass

print(
    name,
    "addresses found:",
    [hex(device_address) for device_address in i2c.scan()],
)

i2c.unlock()

sensor = ISM330DHCX(i2c)
while True:
    print("Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (sensor.acceleration))
    print("Gyro X:%.2f, Y: %.2f, Z: %.2f radians/s" % (sensor.gyro))
    print("")
    time.sleep(0.5)
