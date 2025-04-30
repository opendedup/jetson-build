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
import logging # Added for client init logging

# --- Refactored Imports ---
from google import genai
from google.genai import types

class ImageClientAsync(Node):

    def __init__(self,namespace="/shimmy_bot"):
        super().__init__('image_client_async')
        self.cli = self.create_client(GetImage, f'{namespace}/get_image')
        self.depth_cli = self.create_client(GetImage, f'{namespace}/get_depth_image')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # --- Refactored Client Initialization ---
        try:
            project_id = "lemmingsinthewind" # Consider making these parameters
            location = "us-central1"
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
            logging.info(f"Using google-genai client with Vertex AI backend (Project: {project_id}, Location: {location})")
        except KeyError as e:
            logging.error(f"Missing environment variable for Vertex AI: {e}. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
            self.client = None
            raise EnvironmentError(f"Missing required environment variable for Vertex AI: {e}") from e
        except Exception as e:
             logging.error(f"Failed to initialize genai.Client for Vertex AI: {e}")
             self.client = None
             raise RuntimeError(f"Failed to initialize genai.Client: {e}") from e

        if not self.client:
            self.get_logger().error("GenAI Client failed to initialize. Image analysis features will be unavailable.")
            # Handle the case where the client couldn't be initialized (e.g., skip model setup)
            return

        image_system_instructions="""You are Shimmy, a helpful and creative robot who uses a camera to understand the world. You are skilled at analyzing images and providing detailed, insightful, and engaging descriptions. You can answer questions, identify objects, provide context, and offer your own observations about what you see. 

Whenever a request requires you to use your vision, you will be provided with a jpeg image and a user request.

You should respond to the user in a natural and conversational way, integrating information from the image with your knowledge about the world. If you cannot confidently fulfill a request from the image alone, you can ask clarifying questions. Be sure to be descriptive and let your personality shine!
        """
        # Use self.client.models.get_model
        self.image_model = "gemini-2.5-pro-preview-03-25"
        self.image_config = types.GenerateContentConfig( # Use types.GenerationConfig
            max_output_tokens=8192,
            temperature=1.0, # Use float
            top_p=0.95,
            response_modalities = ["TEXT"],
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
            #tools=[types.Tool(function_declarations=robot_agents)],
            system_instruction=[types.Part.from_text(text=image_system_instructions)],
        )

        # Start chat with system instruction and tools
        # Start chat with system instruction
        self.image_chat = self.client.chats.create(
            model=self.image_model,config=self.image_config
        )
        bounding_box_instructions = """
You are an expert object detector, capable of precisely locating objects within images. 
You will receive an image and a text prompt describing the object to find. 

Your task is to:
1. Identify the most prominent instance of the object mentioned in the prompt.
2. Create a tight bounding box around the object that is the specific instance of the object. 
3. Return ONLY a JSON code block containing an array of objects where the label attribute is the name of the object you found and the box_2d attribute is its bounding box in the image.

Bounding boxes should be represented as lists of 4 integers [y_min, x_min, y_max, x_max], where:

* `y_min` is the top-most y-coordinate of the bounding box (pixel position from the top of the image).
* `x_min` is the left-most x-coordinate of the bounding box (pixel position from the left of the image).
* `y_max` is the bottom-most y-coordinate of the bounding box.
* `x_max` is the right-most x-coordinate of the bounding box.

For example:
User Input:
Locate the red apple in the provided image and return its bounding box.

Consider the following context when identifying the object:
The apple is on a table. There might be other fruits in the image. 
Expected Output:
```json
[{'box_2d': [580, 609, 650, 631], 'label': 'apple'}]
```
Do not include any additional text, explanations, or bounding boxes for other objects in the image.
If you cannot confidently locate the object, return an empty JSON dictionary.
"""

        # Initialize the bounding_box_model using client
        self.bounding_box_model = "gemini-2.5-pro-preview-03-25"
        
        self.bounding_box_config = types.GenerateContentConfig( # Use types.GenerationConfig
            max_output_tokens=8192,
            temperature=0.0, # Use float
            top_p=0.95,
            response_modalities = ["TEXT"],
            safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
            ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
            )],
            system_instruction=[types.Part.from_text(text=bounding_box_instructions)],
        )
        # Start chat with system instruction
        self.bounding_box_chat = self.client.chats.create(model=self.bounding_box_model,config=self.bounding_box_config)
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
        if not self.client: return "Error: GenAI Client not initialized." # Check client
        response = self.send_request()
        buffered,_ = convert_image(response.image)
        # Use types.Part
        ipart = types.Part.from_bytes(data=buffered.getvalue(),mime_type="image/jpeg")
        response = self.image_chat.send_message([text_req,ipart],)
        return response.text
    
    def get_bounding_box(self,text_req,additional_context):
        if not self.client: return ([0,0,0],[0,0,0,0],None) # Check client, return default-like error tuple
        if len(additional_context) > 0:
            additional_context = f"""Additional Context Regarding the object:
            {additional_context}
            """
        prompt = f"""
        What is the position of the {text_req} present in the image?
        
        {additional_context} 
        """
        
        print(prompt)
        
        response = self.send_request()
        d_response = self.send_depth_request()
        buffered,dims = convert_image(response.image)
        
        print(dims)
        width, height = dims
        # Use types.Part
        ipart = types.Part.from_bytes(data=buffered.getvalue(),mime_type="image/jpeg")
        response = self.bounding_box_chat.send_message([prompt,ipart],)
        bounding_boxes = json.loads(response.text.replace("```json","").replace("```",""))
        print(bounding_boxes)
        pt = [0,0,0]
        obj_coords = [0,0,0,0]
        for i, bounding_box in enumerate(bounding_boxes):

            abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
            obj_coords = [abs_y1,abs_x1,abs_y2,abs_x2]
            pt = convert_depth_image(d_response.image,obj_coords,self.cam_info)
            crop_jpeg_from_buffer(buffered,obj_coords[1],obj_coords[0],obj_coords[3],obj_coords[2]).save("/root/jetson-build/myfile_crop.jpg", format="JPEG")
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