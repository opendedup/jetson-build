from noaa_sdk import NOAA
from datetime import datetime

import vertexai

from vertexai.preview.generative_models import (
    grounding,
    Part,
    Tool,
    FunctionDeclaration,
)


from vertexai.vision_models import (
    Image as VXImage,
    MultiModalEmbeddingModel
)
import base64
import subprocess
import pytz 
import traceback
import logging
import sys
from .image_client import ImageClientAsync
from .faiss_client import FaissClientAsync
from .microcontroller_client import MicroControllerClientAsync

import io
import PIL.Image as Image
from cv_bridge import CvBridge
import numpy as np
import uuid
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Tool
)
import vertexai.preview.generative_models as generative_models


class AgentRunner:
    def __init__(self,image_system_instructions=""):
        self.image_client = ImageClientAsync()
        self.voice_emb_client = FaissClientAsync()
        self.microcontroller_client = MicroControllerClientAsync()
        self.image_emb_client = FaissClientAsync("images")
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        self.image_model = GenerativeModel("gemini-1.5-flash-001",system_instruction=[image_system_instructions])
        self.image_chat = self.image_model.start_chat()
        self.embmodel = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
        tools = [
            Tool.from_google_search_retrieval(
                google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval(disable_attribution=False)
            ),
        ]
        self.wbmodel = GenerativeModel(
            "gemini-1.5-flash-001",
            tools=tools,
        )
        self.config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        

    def get_weather(self,lat,lon):
        try:
            n = NOAA()
            fcs = n.points_forecast(lat,lon,hourly=False, type='forecast')['properties']['periods']
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            local_now = now.astimezone()
            local_tz = local_now.tzinfo
            local_tzname = local_tz.tzname(local_now)
            api_response = {"current_time": date_time, "time_zone": local_tzname,"weather":fcs[:3]}
            print(api_response)
            part = Part.from_function_response(
                name="get_weather",
                response={
                    "content": api_response,
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="get_weather",
                response={
                    "content": {"error":"cannot return weather at this time"},
                },
            )
        return part

    def add_voice(self,person,embedding):
        try:
            
            if person == "unknown":
                part = Part.from_function_response(
                name="remember_voice",
                response={
                    "content": {"error":"person is unknow. Tell us who you are first. Say something like \"Hey Shimmy, My name is Cindy. Remember my voice.\""},
                },
                )
            else:
                self.voice_emb_client.publish_embedding(person,embedding)
                part = Part.from_function_response(
                    name="remember_voice",
                    response={
                        "content": {"user_name":person},
                    },
                )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="store_voice",
                response={
                    "content": {"error":"cannot store voices at this time"},
                },
            )
        return part
    
    def change_led_color(self,red,green,blue):
        try:
            self.microcontroller_client.publish_ledcolor(red,green,blue)
            part = Part.from_function_response(
                name="change_led_color",
                response={
                    "content": {"status":"done"},
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="change_led_color",
                response={
                    "content": {"error":"cannot change color at this time"},
                },
            )
        return part
    
    def change_brightness(self,brightness):
        try:
            self.microcontroller_client.publish_ledbrightness(brightness)
            part = Part.from_function_response(
                name="change_brightness",
                response={
                    "content": {"status":"done"},
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="change_brightness",
                response={
                    "content": {"error":"cannot change color at this time"},
                },
            )
        return part
    
    def change_led_pattern(self,pattern):
        try:
            self.microcontroller_client.publish_ledpattern(pattern)
            part = Part.from_function_response(
                name="change_led_pattern",
                response={
                    "content": {"status":"done"},
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="change_led_pattern",
                response={
                    "content": {"error":"cannot change color at this time"},
                },
            )
        return part
    
    def get_power(self):
        try:
            response = self.microcontroller_client.send_power_request()
            part = Part.from_function_response(
                name="get_power",
                response={
                    "content": {"voltage":response.powerusage.loadvoltage,
                                "current_milli_amps": response.powerusage.currentma,
                                "milli_watts":response.powerusage.powermw
                                },
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="get_power",
                response={
                    "content": {"error":"cannot get power at this time"},
                },
            )
        return part

    def get_image(self,text_req):
        try:
            response = self.image_client.send_request()
            buffered = convert_image(response.image)
            ipart = Part.from_data(data=buffered.getvalue(),mime_type="image/jpeg")
            response = self.image_chat.send_message([text_req,ipart],
                                    generation_config=self.config,stream=False, safety_settings=self.safety_settings)
            part = Part.from_function_response(
                name="use_robot_eyes",
                response={
                    "content": {"description":response.text},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="use_robot_eyes",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part
    
    def use_web(self,text_req):
        try:
            
            response = self.wbmodel.generate_content([text_req],
                                    generation_config=self.config,stream=False, safety_settings=self.safety_settings)
            part = Part.from_function_response(
                name="use_web_browser",
                response={
                    "content": {"description":response.text},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="use_web_browser",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part

    def remember_image_objects(self,description):
        try:
            logging.info(description) 
            response = self.image_client.send_request()
            buffered = convert_image(response.image)
            vi = VXImage(image_bytes=buffered.getvalue())
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
            embeddings = model.get_embeddings(
                image=vi,
                contextual_text=description,
                dimension=1408,
            )
            img_data = base64.b64encode(buffered.getvalue())
            metadata = {
                "data":str(img_data, 'utf-8'),
                "mime_type":"image/jpeg",
                "description":description
            }
            self.image_emb_client.publish_embedding(str(uuid.uuid4()),embeddings.image_embedding,map=metadata)
            self.image_emb_client.publish_embedding(str(uuid.uuid4()),embeddings.text_embedding,map=metadata)
            part = Part.from_function_response(
                name="store_voice",
                response={
                    "content": {"description":description},
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="get_time",
                response={
                    "content": {"error":"cannot store image descriptions at this time"},
                },
            )
        return part
        

    def get_current_time(self,timezone):
        try:
            try:
                tz = pytz.timezone(timezone) 
                now = datetime.now(tz)
            except:
                now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M")
            api_response = {"current_time": date_time, "time_zone": timezone}
            part = Part.from_function_response(
                name="get_time",
                response={
                    "content": api_response,
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="get_time",
                response={
                    "content": {"error":"cannot return current time"},
                },
            )
        return part
    
    def change_voice_volume(self,percentage: float, increase: bool = True):
        """
        Adjusts the system volume by a given percentage.

        Args:
            percentage: The percentage to adjust the volume by.
            increase: Whether to increase (True) or decrease (False) the volume.
        """

        # Get current volume using amixer
        current_volume_output = subprocess.check_output(["amixer", "get", "Master"])
        current_volume = int(current_volume_output.decode().split()[4].split("%")[0])

        # Calculate the adjusted volume
        adjusted_volume = current_volume + (percentage * current_volume / 100) if increase else current_volume - (percentage * current_volume / 100)
        adjusted_volume = int(adjusted_volume)

        # Clamp the volume to the valid range (0-100)
        adjusted_volume = max(0, min(adjusted_volume, 100))

        # Set the new volume using amixer
        subprocess.run(["amixer", "set", "Master", str(adjusted_volume) + "%"])

def convert_image(ros_image):
        cv_image = CvBridge().imgmsg_to_cv2(ros_image, "rgb8")
        img_array = np.array(cv_image)
        img_pil = Image.fromarray(img_array)
        buffered = io.BytesIO()
        base_width= 720
        wpercent = (base_width / float(img_pil.size[0]))
        hsize = int((float(img_pil.size[1]) * float(wpercent)))
        img_pil = img_pil.resize((base_width, hsize))
        img_pil.save(buffered, format="JPEG")
        return buffered

def convert_string_array_to_float_array(string_array):
    """Converts a string array to a float array using a ternary expression.

    Args:
    string_array: A list of strings.

    Returns:
    A list of floats.
    """

    float_array = []
    for string in string_array:
        # Use a ternary expression to convert the string to a float.
        float_array.append(float(string) if string else None)
    return float_array

get_time_func = FunctionDeclaration(
            name="get_time",
            description="Get the current day, date, or time for a specific time zone. If someone asks for the day of the week, date, or time use this function.",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "time_zone": {"type": "string", "description": "A Valid Time Zone guessed from the request or system context. Its formated like Asia/Kolkata, America/New_York  so it can be used in a python api call."}
                },
                "required" : ["time_zone"],
            },
    )

change_neopixels_func = FunctionDeclaration(
            name="change_led_color",
            description="""Change the color of the rgb neopixel display on the front of your body. You can use this to turn on the display, change the color of the display, or turn off the display. 
            * If you want to turn off the display make red,green, and blue all 0.
            * If you want the color yellow red = 255, green=255, blue=0""",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "red": {"type": "number", "description": "an integer 0 to 255 that indicates the intensity of the red color."},
                    "green": {"type": "number", "description": "an integer 0 to 255 that indicates the intensity of the green color."},
                    "blue": {"type": "number", "description": "an integer 0 to 255 that indicates the intensity of the blue color."},
                },
                "required" : ["red","green","blue"],
            },
    )

change_brightness_func = FunctionDeclaration(
            name="change_brightness",
            description="""Change Brightness of the neopixel let. The brightness is a number between 0 and 100. 0 is the lowest.""",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "brightness": {"type": "number", "description": "an integer 0 to 100 that indicates the brightness."}
                },
                "required" : ["brightness"],
            },
    )

change_led_pattern = FunctionDeclaration(
            name="change_led_pattern",
            description="""Change the pattern for the neopixel array on the robot.""",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "the name of the pattern."}
                },
                "required" : ["pattern"],
            },
    )

get_power_func = FunctionDeclaration(
            name="get_power",
            description="""get the current wattage, voltage, and amperage usage on the robot.""",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                },
            },
    )

take_picture_function = FunctionDeclaration(
            name="use_robot_eyes",
            description="""Use a camera to take a pictures of surroundings and answer any questions that require vision or seeing stuff. 
This function can be used for anything that requires eyes or a camera including:
* describing what is in sight
* answering question that may required vision
* reading books or a sign.
* Anything that requires eyes
""",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "user_request": {"type": "string", "description": "The request from the user."},
                    "additional_context": {"type": "string", "description": "Any context from the conversation history that may be useful when analyzing what you see."},
                },
                "required" : ["user_request"],
            },

    )

store_voice_func = FunctionDeclaration(
        name="remember_voice",
        description="Capture introductions and remember someone by their voice using your robot hearing. If someone introduces themselves with statements like \"This is Sam\" or \"I'm Jenny\" or tells you to remember their voice use this function. This is useful for remembering people for later conversations or context.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The persons's name assosiated with the voice to remember."},
                
            },
            "required" : ["name"],
        },
        
)

change_volume_func = FunctionDeclaration(
    name="change_voice_volume",
        description="Lower or Raise the volume of your speaker using amixer.",
        parameters={
            "type": "object",
            "properties": {
                "volume_percent": {"type": "number", "description": """the percent the volume should be raised or lowered using amixer. The value should be returned as a positive float value.
                                   As an example, if the request is to lower the volume by 10% volume_percent would return .10"""},
                "increase_volume": {"type": "boolean", "description": """Return true if the volume should be raised, otherwise return false."""},
                
            },
            "required" : ["increase_volume"],
        },
)

store_image_func = FunctionDeclaration(
        name="remember_image_objects",
        description="Remember things in images.",
        parameters={
            "type": "object",
            "properties": {
                "picture_context": {"type": "string", "description": "Any context"},
                
            },
        },
)

web_browser = FunctionDeclaration(
        name="use_web_browser",
        description="""Use a web browser to consult the internet where the answer could be enhanced with additional context, you don know the answer, or you are not confident in your response. 
This can be used to ground responses with facts found in the internet. All the internet is available to this function. Some of the capabilites available to this function are:
* Get the Weather.
* Get useful information, facts, and figures.
* Help with Shopping and finding products
* Get Current Events and News
* Answer questions grounded in real data

""",
        parameters={
            "type": "object",
            "properties": {
                "search_txt": {"type": "string", "description": "The text used to do the web search or request. Any context that could be helpful for the web search should be added."},
            },
            "required":["search_txt"]
        },
)
google_search_tool = Tool.from_google_search_retrieval(
    google_search_retrieval=grounding.GoogleSearchRetrieval(disable_attribution=False)
)

robot_agents = Tool(
        function_declarations=[
            get_time_func,
            store_voice_func,
            take_picture_function,
            web_browser,
            change_volume_func,
            change_neopixels_func,
            change_brightness_func,
            get_power_func,
            change_led_pattern
            
        ],
)

