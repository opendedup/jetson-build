

from chat_interfaces.srv import GetImage
import rclpy
from rclpy.node import Node
import PIL.Image as Image
from cv_bridge import CvBridge
import numpy as np
import io
import json
import struct
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel

from vertexai.generative_models import (
    GenerativeModel,
    Part
)
import vertexai.preview.generative_models as generative_models

class ImageClientAsync(Node):

    def __init__(self,image_system_instructions=""):
        super().__init__('image_client_async')
        self.cli = self.create_client(GetImage, 'get_image')
        self.depth_cli = self.create_client(GetImage, 'get_depth_image')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.image_model = GenerativeModel("gemini-1.5-flash-001",system_instruction=[image_system_instructions])
        self.image_chat = self.image_model.start_chat()
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }
        self.config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
        self.subscription = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/depth/camera_info',
            self.camera_info_callback,
            10)
    
    def camera_info_callback(self,camera_info):
        self.cam_info = camera_info
        
    def send_request(self):
        req = GetImage.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def send_depth_request(self):
        req = GetImage.Request()
        future = self.depth_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def get_image(self,text_req):
        response = self.send_request()
        buffered,_ = convert_image(response.image)
        ipart = Part.from_data(data=buffered.getvalue(),mime_type="image/jpeg")
        response = self.image_chat.send_message([text_req,ipart],
                                generation_config=self.config,stream=False, safety_settings=self.safety_settings)
        return response.text
    
    def get_bounding_box(self,text_req,additional_context):
        if len(additional_context) > 0:
            additional_context = f"""Additional Context Regarding the object:
            {additional_context}
            """
        prompt = f"""
        What is the position of the {text_req} present in the image?
        
        {additional_context} 

Output objects in JSON format with both object names and positions as a JSON object: {{ name: [y_min, x_min, y_max, x_max] }}. Put the annswer in a JSON code block.
        """
        
        print(prompt)
        
        response = self.send_request()
        d_response = self.send_depth_request()
        buffered,dims = convert_image(response.image)
        
        print(dims)
        ipart = Part.from_data(data=buffered.getvalue(),mime_type="image/jpeg")
        response = self.image_chat.send_message([prompt,ipart],
                                generation_config=self.config,stream=False, safety_settings=self.safety_settings)
        coords = json.loads(response.text.replace("```json","").replace("```",""))
        print(coords)
        pt = [0,0,0]
        obj_coords = [0,0,0,0]
        for key in coords:
            coords[key][0] = int((coords[key][0]/1000)*dims[1])
            coords[key][1] = int((coords[key][1]/1000)*dims[0])
            coords[key][2] = int((coords[key][2]/1000)*dims[1])
            coords[key][3] = int((coords[key][3]/1000)*dims[0])
            obj_coords = coords[key]
            pt = convert_depth_image(d_response.image,coords[key],self.cam_info)
            crop_jpeg_from_buffer(buffered,coords[key][1],coords[key][0],coords[key][3],coords[key][2]).save("/root/jetson-build/myfile_crop.jpg", format="JPEG")
            break
        return (pt,obj_coords,buffered)
    
    

  
    
    
        

def convert_image(ros_image):
        cv_image = CvBridge().imgmsg_to_cv2(ros_image, "rgb8")
        img_array = np.array(cv_image)
        img_pil = Image.fromarray(img_array)
        buffered = io.BytesIO()
        #base_width= 1000
        #wpercent = (base_width / float(img_pil.size[0]))
        #hsize = int((float(img_pil.size[1]) * float(wpercent)))
        #img_pil = img_pil.resize((base_width, hsize))
        img_pil.save(buffered, format="JPEG")
        img_pil.save("/root/jetson-build/myfile.jpg", format="JPEG")
        return buffered,img_pil.size
    
def convert_depth_image(ros_image,coords,cam_info):
    #print(ros_image.encoding)
    mode_depth='F'
    depth_map = Image.frombytes(mode_depth, (ros_image.width, ros_image.height), ros_image.data)
    img_array = np.array(depth_map)
    cv_image = CvBridge().imgmsg_to_cv2(ros_image, "passthrough")
        # img_array = np.array(cv_image)
    #print("############################")
    img_array = img_array[coords[0]:coords[2], coords[1]:coords[3]]
    distance = np.nanmean(img_array)
    height, width = img_array.shape
    center_x = width // 2
    center_y = height // 2
    depth = img_array[center_y, center_x]
    #print(img_array.shape)
    #print(img_array)
    #print(distance)
    #print(depth)
    depth_image = np.array(cv_image, dtype=np.float32).reshape((ros_image.height, ros_image.width))
    #print(depth_image)
    target_y = int((coords[0]+coords[2])/2)
    target_x = int((coords[1]+coords[3])/2)
    depth = depth_image[target_y, target_x]
    
    
    ci = PinholeCameraModel()
    ci.fromCameraInfo(msg=cam_info)
    # Check for invalid depth values
    if np.isnan(depth) or np.isinf(depth):
        raise Exception('Invalid depth value at ({}, {})'.format(target_x, target_y))
    rp = ci.rectifyPoint((target_x, target_y))
    ray = ci.projectPixelTo3dRay(rp)
    point = [ray[0] * depth, ray[1] * depth, ray[2] * depth]
    point = [round(x, 2) for x in point]
    return point

def crop_jpeg_from_buffer(image_buffer, left, top, right, bottom):
        # Open the image from the buffer
        image = Image.open(image_buffer)
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

def main():
    rclpy.init()

    minimal_client = ImageClientAsync()
    response = minimal_client.send_request()
    cv_image = CvBridge().imgmsg_to_cv2(response.image, "rgb8")
    img_array = np.array(cv_image)
    img_pil = Image.fromarray(img_array)
    #base_width= 720
    #wpercent = (base_width / float(img_pil.size[0]))
    #hsize = int((float(img_pil.size[1]) * float(wpercent)))
    #img_pil = img_pil.resize((base_width, hsize))
    img_pil.save("/home/nvidia/jetson-build/myfile.jpg", format="JPEG")
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()