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
import pytz 
import traceback
import logging
import sys
from .image_client import ImageClientAsync
from .faiss_client import FaissClientAsync

import io
import PIL.Image as Image
from cv_bridge import CvBridge
import numpy as np
import uuid
logging.basicConfig(level=logging.INFO, stream=sys.stderr)


class AgentRunner:
    def __init__(self):
        self.image_client = ImageClientAsync()
        self.voice_emb_client = FaissClientAsync()
        self.image_emb_client = FaissClientAsync("images")
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        
        

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
            self.voice_emb_client.publish_embedding(person,embedding)
            part = Part.from_function_response(
                name="store_voice",
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

    def get_image(self):
        try:
            response = self.image_client.send_request()
            buffered = convert_image(response.image)
            part = Part.from_data(data=buffered.getvalue(),mime_type="image/jpeg")
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_text("An error occured while trying to get the image")
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
            tz = pytz.timezone(timezone) 
            now = datetime.now(tz)
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
            description="Get the current time and date for a specific time zone",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "time_zone": {"type": "string", "description": "A Valid Time Zone guessed from the request formated like Asia/Kolkata, America/New_York  so it can be used in a python api call."}
                },
            },
            
    )
    
get_weather_func = FunctionDeclaration(
        name="get_weather",
        description="Get the current weather for a specific area. The weather is returned in the local time zone of the area requested.",
        parameters={
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "The Valid latitude associated with the weather request.E.g. 38.8894"},
                "longitude": {"type": "number", "description": "The Valid longitude associated with the weather request E.g. -77.0352\" "}
                
            },
        },
)

take_picture_function = FunctionDeclaration(
            name="take_picture",
            description="Take a picture of your surroundings and describe what you see",
            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "The Subject to take the picture of"}
                },
            },
    )

store_voice_func = FunctionDeclaration(
        name="store_voice",
        description="Remember someone by their voice.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The persons's name assosiated with the voice to remember"},
                
            },
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

google_search_tool = Tool.from_google_search_retrieval(
    google_search_retrieval=grounding.GoogleSearchRetrieval(disable_attribution=False)
)

robot_agents = Tool(
        function_declarations=[
            get_time_func,
            store_voice_func,
            store_image_func,
            take_picture_function,
        ],
)

