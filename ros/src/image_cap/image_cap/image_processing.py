import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image as SImage
from cv_bridge import CvBridge, CvBridgeError
from chat_interfaces.srv import GetImage
import threading
from io import BytesIO
import numpy as np
from PIL import Image

class FIFOCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = []
        

    def set(self, value):
        if len(self.cache) == self.capacity:
            self.cache.pop(0)
        self.cache.append(value)

class ProcessImage_Subscriber(Node):

    def __init__(self,namespace="/shimmy_bot"):
        super().__init__('process_image_subscriber')
        self.bridge = CvBridge()
        self.fifo = FIFOCache(10)
        self.depth_fifo = FIFOCache(10)
        self.lock = threading.Lock()
        self.srv = self.create_service(GetImage, f'{namespace}/get_image', self.get_image)
        self.depth_srv = self.create_service(GetImage, f'{namespace}/get_depth_image', self.get_depth_image)
        self.image_subscription = self.create_subscription(
            SImage,
            'image_raw',
            self.img_callback,
            10)
        self.depth_subscription = self.create_subscription(
            SImage,
            'depth_image_raw',
            self.depth_callback,
            10)
    
    def img_callback(self,image_msg):
            with self.lock:
                self.fifo.set(image_msg)
    
    def depth_callback(self,image_msg):
            with self.lock:
                self.depth_fifo.set(image_msg)
            
    def get_image(self, request, response):
        with self.lock:
             img_data = self.fifo.cache[0]
        
        response.image = img_data
        self.get_logger().debug('done')
        return response
    
    def get_depth_image(self, request, response):
        with self.lock:
             img_data = self.depth_fifo.cache[0]
        
        response.image = img_data
        self.get_logger().debug('done')
        return response
    
def main(args=None):
    rclpy.init(args=args)
    image_sub = ProcessImage_Subscriber()
    

    rclpy.spin(image_sub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
            
        
        
        
    
    