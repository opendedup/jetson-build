from chat_interfaces.srv import GetPower
from chat_interfaces.msg import PowerUsage
from chat_interfaces.msg import LedBrightness
from chat_interfaces.msg import LedColor
from chat_interfaces.msg import LedPattern
from chat_interfaces.msg import PaintLedColor

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor


import json
import threading
import time
import traceback

import serial

class MicroControllerrService(Node):

    def __init__(self,namespace=f'/shimmy_bot'):
        super().__init__('microcontroller_service')
        self.declare_parameter('serial_device', "")
        self.serial_device = self.get_parameter("serial_device").value
        self.serial_obj = None
        self.lock = threading.Lock()
        if len(self.serial_device) > 0:
            serial.Serial(self.serial_device)
        else:
            for i in range(0,9):
                try:
                    self.serial_device = f"/dev/ttyACM{i}"
                    self.serial_obj = serial.Serial(self.serial_device)
                    break
                except:
                    self.serial_obj = None
        if self.serial_obj is None:
            raise Exception("Serial Device not found")
        self.get_logger().info('Serial Device = %s'% (self.serial_device))
        
        self.serial_obj.baudrate = 115200  # set Baud rate to 9600
        self.serial_obj.bytesize = 8   # Number of data bits = 8
        self.serial_obj.parity  ='N'   # No parity
        self.serial_obj.stopbits = 1   # Number of Stop bits = 1
        
        
        self.srv = self.create_service(GetPower, f'{namespace}/get_power', self.get_power)
        self.led_brightness_subscription = self.create_subscription(
            LedBrightness,
            f'{namespace}/ledbrightness',
            self.ledbrightness_callback,
            10)
        self.led_color_subscription = self.create_subscription(
            LedColor,
            f'{namespace}/ledcolor',
            self.ledcolor_callback,
            10)
        self.led_pattern_subscription = self.create_subscription(
            LedPattern,
            f'{namespace}/ledpattern',
            self.ledpattern_callback,
            10)
        self.led_paint_subscription = self.create_subscription(
            PaintLedColor,  # Subscribe to the PaintLedColor message
            f'{namespace}/ledpaint',
            self.paint_leds,
            10)
        self.power_publisher_ = self.create_publisher(PowerUsage, f'{namespace}/powerusage', 10)
        obj = {"command":"led",'subcommand':"fill","colors":[0,0,0],"start":0,"num":64}
        data=json.dumps(obj)
        data = f"{data}\n"
        self.serial_obj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
        t = threading.Thread(target=self.tworker)
        t.start()
    
    def create_serial(self):
        if self.serial_obj is not None:
            self.serial_obj.close()
            
        self.serial_obj = None
        
        for i in range(0,9):
            try:
                self.serial_device = f"/dev/ttyACM{i}"
                self.serial_obj = serial.Serial(self.serial_device)
                break
            except:
                self.serial_obj = None
        if self.serial_obj is None:
            raise Exception("Serial Device not found")
        self.get_logger().info('Serial Device = %s'% (self.serial_device))
        
        self.serial_obj.baudrate = 115200  # set Baud rate to 9600
        self.serial_obj.bytesize = 8   # Number of data bits = 8
        self.serial_obj.parity  ='N'   # No parity
        self.serial_obj.stopbits = 1   # Number of Stop bits = 1
        
        
    def tworker(self):
        while True:
            try:
                time.sleep(30)
                obj = {"command":"power"}
                data=json.dumps(obj)
                data = f"{data}\n"
                with self.lock:
                    self.serial_obj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
                    rcv = self.serial_obj.readline()
                robj = json.loads(rcv)
                msg = PowerUsage()
                msg.busvoltage = float(robj["busvoltage"])
                msg.shuntvoltage = float(robj["shuntvoltage"])
                msg.currentma = float(robj["current_mA"])
                msg.powermw = int(robj["power_mW"])
                msg.loadvoltage = float(robj["loadvoltage"])
                self.power_publisher_.publish(msg)
                
            except:
                self.get_logger().error('Got an error while reading power info.')
                self.get_logger().error('%s' % traceback.format_exc())
                time.sleep(30)
                try:
                    self.create_serial()
                except:
                    self.get_logger().error('Got an error while trying to connect to serial device.')
                    self.get_logger().error('%s' % traceback.format_exc())
    
             
    def ledpattern_callback(self, msg: LedPattern):    
        with self.lock:
            obj = {"command":"led",'subcommand':"pattern","pattern":msg.pattern}
            data=json.dumps(obj)
            data = f"{data}\n"
            self.serial_obj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
            rcv = self.serial_obj.readline()
            # self.get_logger().info('Revieved %s' % (rcv))
    
    def ledcolor_callback(self, msg: LedColor):    
        with self.lock:
            obj = {"command":"led",'subcommand':"fill","colors":[msg.red,msg.blue,msg.green],"start":0,"num":64}
            data=json.dumps(obj)
            data = f"{data}\n"
            self.serial_obj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
            rcv = self.serial_obj.readline()
            # self.get_logger().info('Revieved %s' % (rcv))
    
    def paint_leds(self, msg: PaintLedColor):
        """Sends the "paint" command to the Arduino to set individual LED colors.

        Args:
            msg: A PaintLedColor message containing a list of LedColor messages.
        """
        pixels = [[led_color.red, led_color.blue, led_color.green] for led_color in msg.colors]

        with self.lock:
            obj = {"command": "led", 'subcommand': "paint", "pixels": pixels}
            data = json.dumps(obj)
            data = f"{data}\n"
            self.serial_obj.write(data.encode("utf-8"))
            rcv = self.serial_obj.readline()
            # self.get_logger().info('Received %s' % (rcv))
    
    def ledbrightness_callback(self, msg: LedBrightness):    
        with self.lock:
            obj = {"command":"led",'subcommand':"brightness","value":msg.brightness}
            data=json.dumps(obj)
            data = f"{data}\n"
            self.serial_obj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
            rcv = self.serial_obj.readline()
            # self.get_logger().info('Revieved %s' % (rcv))
        

    
    def get_power(self, request, response):
        with self.lock:
            obj = {"command":"power"}
            data=json.dumps(obj)
            data = f"{data}\n"
            self.serial_obj.write(data.encode("utf-8"))    #transmit 'A' (8bit) to micro/Arduino
            rcv = self.serial_obj.readline()
            robj = json.loads(rcv)
            self.get_logger().info('Revieved %s' % (rcv))
            response.powerusage.busvoltage = float(robj["busvoltage"])
            response.powerusage.shuntvoltage = float(robj["shuntvoltage"])
            response.powerusage.currentma = float(robj["current_mA"])
            response.powerusage.powermw = robj["power_mW"]
            response.powerusage.loadvoltage = float(robj["loadvoltage"])
            return response

    
        
    


def main():
    rclpy.init()

    m_service = MicroControllerrService()
    executor = MultiThreadedExecutor(num_threads=8)



    executor.add_node(m_service)

    executor.spin()

    executor.shutdown()
    m_service.serial_obj.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()