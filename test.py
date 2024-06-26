import serial
import time
import json


smiley_face = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [255, 241, 232], [255, 241, 232], [255, 241, 232], [255, 241, 232], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [255, 241, 232], [255, 241, 232], [255, 241, 232], [255, 241, 232], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [255, 241, 232], [255, 241, 232], [255, 241, 232], [255, 241, 232], [194, 195, 199], [194, 195, 199], [194, 195, 199], [194, 195, 199], [255, 241, 232], [255, 241, 232], [255, 241, 232], [255, 241, 232]]

SerialObj = serial.Serial('/dev/ttyACM0') # COMxx  format on Windows
                  # ttyUSBx format on Linux
SerialObj.baudrate = 115200  # set Baud rate to 9600
SerialObj.bytesize = 8   # Number of data bits = 8
SerialObj.parity  ='N'   # No parity
SerialObj.stopbits = 1   # Number of Stop bits = 1
time.sleep(3)
obj = {"command":"power"}
data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
time.sleep(1)
obj = {"command":"led",'subcommand':"pattern","pattern":"rainbow"}
data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
time.sleep(1)
obj = {"command":"led",'subcommand':"fill","colors":[0,0,255],"start":0,"num":64}

data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
time.sleep(1)
obj = {"command":"led",'subcommand':"brightness","value":20}

data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
time.sleep(1)
obj = {"command":"led",'subcommand':"brightness","value":200}

data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
time.sleep(1)

obj = {"command":"led",'subcommand':"paint","pixels":smiley_face}
data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
time.sleep(1)
obj = {"command":"led",'subcommand':"turnoff"}

data=json.dumps(obj)
data = f"{data}\n"
SerialObj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
ReceivedString = SerialObj.readline()
robj = json.loads(ReceivedString)
print(robj)
SerialObj.close()      # Close the port