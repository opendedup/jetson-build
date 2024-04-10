import sys

from chat_interfaces.srv import GetImage
import rclpy
from rclpy.node import Node
import os
import io
import PIL.Image as Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ImageClientAsync(Node):

    def __init__(self):
        super().__init__('image_client_async')
        self.cli = self.create_client(GetImage, 'get_image')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    def send_request(self):
        req = GetImage.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    rclpy.init()

    minimal_client = ImageClientAsync()
    response = minimal_client.send_request()
    cv_image = CvBridge().imgmsg_to_cv2(response.image, "rgb8")
    img_array = np.array(cv_image)
    img_pil = Image.fromarray(img_array)
    base_width= 720
    wpercent = (base_width / float(img_pil.size[0]))
    hsize = int((float(img_pil.size[1]) * float(wpercent)))
    img_pil = img_pil.resize((base_width, hsize))
    img_pil.save("/home/nvidia/git/jetson-build/myfile.jpg", format="JPEG")
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()