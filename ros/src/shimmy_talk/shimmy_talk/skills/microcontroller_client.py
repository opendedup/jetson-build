
import rclpy
from rclpy.node import Node
from chat_interfaces.srv import GetPower
from chat_interfaces.msg import PowerUsage
from chat_interfaces.msg import LedBrightness
from chat_interfaces.msg import LedColor
from chat_interfaces.msg import LedPattern
import json

class MicroControllerClientAsync(Node):

    def __init__(self,namespace=''):
        super().__init__('microcontroller_client')
        self.cli = self.create_client(GetPower, f"{namespace}/get_power")
        self.ledbrightness_publisher = self.create_publisher(LedBrightness, f'{namespace}/ledbrightness', 10)
        self.ledcolor_publisher = self.create_publisher(LedColor, f'{namespace}/ledcolor', 10)
        self.ledpattern_publisher = self.create_publisher(LedPattern, f'{namespace}/ledpattern', 10)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'microcontroller service {namespace} not available, waiting again...')
        
    def send_power_request(self):
        power_req = GetPower.Request()
        power_req.msg = ""
        future = self.cli.call_async(power_req)
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        return future.result()
    
    def publish_ledbrightness(self,brightness):
            msg = LedBrightness()
            msg.brightness = int(brightness)
            self.ledbrightness_publisher.publish(msg)
    
    def publish_ledcolor(self,red,green,blue):
            msg = LedColor()
            msg.red = int(red)
            msg.green = int(green)
            msg.blue = int(blue)
            self.ledcolor_publisher.publish(msg)
            
    def publish_ledpattern(self,pattern):
            msg = LedPattern()
            msg.pattern = pattern
            self.ledpattern_publisher.publish(msg)