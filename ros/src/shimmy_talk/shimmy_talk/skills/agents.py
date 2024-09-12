from noaa_sdk import NOAA
from datetime import datetime
import json

import vertexai

from vertexai.preview.generative_models import (
    grounding,
    Part,
    Tool,
    FunctionDeclaration,
)

import pytz 
import traceback
import logging
import sys
from .image_client import ImageClientAsync
from .shimmy_system_notifier_client import ShimmySystemNotifierAsync
from .microcontroller_client import MicroControllerClientAsync
from .shimmy_move_client import ShimmyMoveClientAsync

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
import re


class AgentRunner:
    def __init__(self,image_system_instructions=""):
        self.image_client = ImageClientAsync()
        #self.voice_emb_client = FaissClientAsync()
        self.microcontroller_client = MicroControllerClientAsync()
        self.shimmy_move_client = ShimmyMoveClientAsync()
        self.state_notifier = ShimmySystemNotifierAsync()
        vertexai.init(project="lemmingsinthewind", location="us-central1")
        tools = [
            Tool.from_google_search_retrieval(
                google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval()
            ),
        ]
        self.wbmodel = GenerativeModel(
            "gemini-1.5-pro-001",
            tools=tools,
            system_instruction=["""You are a world-class research agent, adept at using Google Search to find comprehensive and detailed information on any topic. Your goal is to provide the most informative and insightful answers possible.

When responding to a query:

1. **Thorough Research:** Conduct extensive searches using Google Search, exploring multiple relevant sources.
2. **Information Synthesis:** Combine information from different sources, synthesizing it into a coherent and comprehensive answer.
3. **Depth and Detail:** Prioritize providing as much relevant detail as possible, within a reasonable length.
4. **Accurate Attribution:** Clearly cite all sources you use in your responses using footnotes.

You will be provided with a user's query as input. Your task is to provide a detailed and well-researched response."""] 
        )
        self.config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

    def add_voice(self,person,embedding,training=False):
        try:
            
            if person == "unknown" and training is False:
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
    
    def move_shimmy(self,command):
        try:
            self.state_notifier.publish_status('Give me a few seconds while I figure out how to chart my path.')
            self.shimmy_move_client.publish_pose(command)
            part = Part.from_function_response(
                name="move_shimmy",
                response={
                    "content": {"status":"starting to move."},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="move_shimmy",
                response={
                    "content": {"error":"cannot move robot. "},
                },
            )
        return part
    
    def turn_shimmy(self,radians):
        try:
            self.shimmy_move_client.publish_turn(radians)
            part = Part.from_function_response(
                name="turn_inplace",
                response={
                    "content": {"status":"starting to turn"},
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="turn_inplace",
                response={
                    "content": {"error":"cannot turn robot."},
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
            self.state_notifier.publish_status('Give me a few seconds while I use my eyes.')
            rtxt = self.image_client.get_image(text_req)
            part = Part.from_function_response(
                name="use_robot_eyes",
                response={
                    "content": {"description":rtxt},
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
    def find_object(self,text_req,additional_context=""):
        try:
            self.state_notifier.publish_status(f'Give me a moment while I use my eyes to find {text_req} .')
            rtxt,_,_ = self.image_client.get_bounding_box(text_req,additional_context)
            direction = "left"
            if rtxt[0] > 0:
                direction = "right"
            print(rtxt)
            part = Part.from_function_response(
                name="find_object_with_eyes",
                response={
                    "content": {"description":f"{text_req} found about {rtxt[2]} meters in front and {abs(rtxt[0])} meters to the {direction}."}
                }
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="find_object_with_eyes",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part
    
    def move_to_object(self,text_req,additional_context="",move_command=""):
        try:
            self.state_notifier.publish_status(f'Give me a moment while I use my eyes to find {text_req} .')
            rtxt,obj_coords,_ = self.image_client.get_bounding_box(text_req,additional_context)
            direction = "left"
            if rtxt[0] > 0:
                direction = "right"
            if len(move_command) == 0:
                move_command = f"Move directly infront of the {text_req}."
            self.state_notifier.publish_status(f'Found the {text_req} {rtxt[2]} meters forward and to the {direction} {abs(rtxt[0])} meters. Charting my path')
            self.shimmy_move_client.publish_pose(f"""Move {rtxt[2]} meters forward and to the {direction} {abs(rtxt[0])} meters.
""")
            print(rtxt)
            part = Part.from_function_response(
                name="move_to_object_with_wheels",
                response={
                    "content": {"description":f"{text_req} found about {rtxt[2]} meters in front. Moving to {text_req} but not there yet."},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="move_to_object_with_wheels",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part
    
    def cancel_move(self):
        try:
           
            self.shimmy_move_client.publish_cancel()
            part = Part.from_function_response(
                name="stop_moving",
                response={
                    "content": {"description":f"Canceled Move"},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = Part.from_function_response(
                name="stop_moving",
                response={
                    "content": {"error":"Unable to stop moving"},
                },
            )
        return part
    
    def use_web(self,text_req):
        try:
            self.state_notifier.publish_status(f'Give me a moment while I look that up for you .')
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

    # def remember_image_objects(self,description):
    #     try:
    #         logging.info(description) 
    #         response = self.image_client.send_request()
    #         buffered = convert_image(response.image)
    #         vi = VXImage(image_bytes=buffered.getvalue())
    #         model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    #         embeddings = model.get_embeddings(
    #             image=vi,
    #             contextual_text=description,
    #             dimension=1408,
    #         )
    #         img_data = base64.b64encode(buffered.getvalue())
    #         metadata = {
    #             "data":str(img_data, 'utf-8'),
    #             "mime_type":"image/jpeg",
    #             "description":description
    #         }
    #         self.image_emb_client.publish_embedding(str(uuid.uuid4()),embeddings.image_embedding,map=metadata)
    #         self.image_emb_client.publish_embedding(str(uuid.uuid4()),embeddings.text_embedding,map=metadata)
    #         part = Part.from_function_response(
    #             name="store_voice",
    #             response={
    #                 "content": {"description":description},
    #             },
    #         )
    #     except:
    #         logging.error(traceback.format_exc()) 
    #         part = Part.from_function_response(
    #             name="get_time",
    #             response={
    #                 "content": {"error":"cannot store image descriptions at this time"},
    #             },
    #         )
    #     return part
        

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
    
    def change_voice_volume(self,percentage: float, increase: bool = True,current_volume: int = 0):
        """
        Adjusts the system volume by a given percentage.

        Args:
            percentage: The percentage to adjust the volume by.
            increase: Whether to increase (True) or decrease (False) the volume.
        """

        # Calculate the adjusted volume
        if increase is True:
            current_volume += int(percentage*40)
            if current_volume >= 20:
                current_volume = 20
        elif increase is False:
            current_volume -= int(percentage*30)
            if current_volume <= -30:
                current_volume = -30
        return current_volume



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
        * If you want to turn off the display make red,green, and blue all 0. You can also use this to show your emotions as well like blush by turning red, happy as blue.
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
        description="""Change the pattern for the neopixel array on the robot. """,
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
            description="""Use your camera to take a picture and provide a description and analysis of what you see.
This function is for understanding the overall scene and the objects present, rather than locating specific items.
Examples:
* Take a picture and tell me what's in front of you.
* Can you describe the scene you see?
* What objects are in the image?""",
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

find_object_function = FunctionDeclaration(
    name="find_object_with_eyes",
    description="""Use your camera to take a picture of your surroundings and **locate a specific object within the image**, providing its coordinates.
This function helps determine **where an object is positioned in your field of view**. 
Examples:
* Find the apple in the picture and tell me where it is.
* How far away is the door?
* Can you see a cat? If so, what are its coordinates in the image?""",

            # Function parameters are specified in OpenAPI JSON schema format
            parameters={
                "type": "object",
                "properties": {
                    "object": {"type": "string", "description": "The object to locate in the field of view."},
                    "additional_context": {"type": "string", "description": "Any context that will help identify the specific object."},
                },
                "required" : ["object"],
            },

)

move_to_object_function = FunctionDeclaration(
            name="move_to_object_with_wheels",
            description="""Move around using your wheels to an object in the field of view. 
This function can be used to find an object and then move to the object that was found. Examples are as follows:
* Move to a person
* Stroll over to a door way
* Get a closer look at an apple
* Move to the door and turn towards me
""",
        # Function parameters are specified in OpenAPI JSON schema format
        parameters={
            "type": "object",
            "properties": {
                "object": {"type": "string", "description": "The object to move to in the field of view."},
                "additional_context": {"type": "string", "description": "Any context that will help identify the object."},
                "movement_commands": {"type": "string", "description": "Any addition context regarding how to move or move commands. Examples are: Turn back towards me, Face where you came from, stop 1 foot in front of the shoe"},
            },
            "required" : ["object"],
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
        description="Lower or Raise the volume of your voice.",
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

move_shimmy = FunctionDeclaration(
        name="move_around",
        description="""Move around based on voice commands. Example commands are as follows:
* Move forward 3 feet.
* Come back 1 Meter and turn 180 degrees
* Turn around
* Rotate 90 degrees
""",
        parameters={
            "type": "object",
            "properties": {
                "move_instructions": {"type": "string", "description": "The instructions on how to move or navigate"},
            },
            "required":["move_instructions"]
        },
)

turn_shimmy = FunctionDeclaration(
        name="turn_inplace",
        description="""Turn in place based on voice commands. Example commands are as follows:
* Turn Around
* Turn 90 degrees
* Turn 6 radians
* Spin around 3 times
""",
        parameters={
            "type": "object",
            "properties": {
                "turn_instructions": {"type": "number", "description": "the float number in radians to turn. 1 turn or 360 degrees is 6.28319 radians. Clockwise is negative and counter clockwise is positive radians."},
            },
            "required":["turn_instructions"]
        },
)

stop_shimmy = FunctionDeclaration(
        name="stop_moving",
        description="""Stop moving or turning. This will cancel any current move or turn opperations.
""",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "boolean", "description": "whether or not to stop moving or turning. setting to true will stop turning"},
            },
            "required":["command"]
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








robot_agents = [
            get_time_func,
            #store_voice_func,
            take_picture_function,
            web_browser,
            change_volume_func,
            change_neopixels_func,
            change_brightness_func,
            get_power_func,
            change_led_pattern,
            move_shimmy,
            find_object_function,
            move_to_object_function,
            #turn_shimmy,
            stop_shimmy
            
        ]

