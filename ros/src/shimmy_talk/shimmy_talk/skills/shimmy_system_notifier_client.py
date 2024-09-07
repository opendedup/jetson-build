
import rclpy
from rclpy.node import Node


from std_msgs.msg import String

class ShimmySystemNotifierAsync(Node):

    def __init__(self,namespace="/shimmy_bot"):
        super().__init__('shimmy_system_notifier_client')
        self.system_message_publisher  = self.create_publisher(String, f'{namespace}/system_message', 10)
    
    def publish_status(self,status):
        msg = String()
        msg.data = status
        self.system_message_publisher.publish(msg)

def main():
    rclpy.init()

    minimal_client = ShimmySystemNotifierAsync()
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

