import traceback
from datetime import datetime
import json
import os # Added import for environment variables

from google import genai
from google.genai import types # types contains Part, FunctionDeclaration, etc.


import logging
import sys

import pytz
from .image_client import ImageClientAsync
from .shimmy_system_notifier_client import ShimmySystemNotifierAsync
from .microcontroller_client import MicroControllerClientAsync
from .shimmy_move_client import ShimmyMoveClientAsync


logging.basicConfig(level=logging.INFO, stream=sys.stderr)


# Removed imports from vertexai.generative_models
# import vertexai.preview.generative_models as generative_models 
import re


class AgentRunner:
    def __init__(self,image_system_instructions=""):
        self.image_client = ImageClientAsync()
        #self.voice_emb_client = FaissClientAsync()
        self.microcontroller_client = MicroControllerClientAsync()
        self.shimmy_move_client = ShimmyMoveClientAsync()
        self.state_notifier = ShimmySystemNotifierAsync()
        # --- Refactored Client Initialization for Vertex AI ---
        # Remove the incorrect genai.configure line
        # genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY")) 

        # Configure client to use Vertex AI backend
        # Assumes GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set as environment variables
        try:
            project_id = "lemmingsinthewind"
            location = "us-central1"
            # Initialize client using genai.Client for Vertex
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
            logging.info(f"Using google-genai client with Vertex AI backend (Project: {project_id}, Location: {location})")
        except KeyError as e:
            logging.error(f"Missing environment variable for Vertex AI: {e}. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
            self.client = None # Indicate client initialization failed
            raise EnvironmentError(f"Missing required environment variable for Vertex AI: {e}") from e
        except Exception as e:
             logging.error(f"Failed to initialize genai.Client for Vertex AI: {e}")
             self.client = None # Indicate client initialization failed
             raise RuntimeError(f"Failed to initialize genai.Client: {e}") from e

        # --- Web Search / Grounding Tool Section ---
        # Tool.from_google_search_retrieval and types.Grounding are not the correct way
        # to configure grounding with genai.Client, even for Vertex backend.
        # Remove the incompatible tool definition:
        # tools = [
        #     types.Tool.from_google_search_retrieval(
        #         google_search_retrieval=types.Grounding.GoogleSearchRetrieval()
        #     ),
        # ]
        # TODO: Implement web search/grounding. If using Vertex grounding features,
        # configure it via the appropriate parameters in client.models.generate_content calls,
        # or implement using external search APIs.

        # Comment out wbmodel as its tool setup was incorrect. Re-evaluate if needed with correct grounding setup.
        # self.wbmodel = genai.GenerativeModel(
        #     "gemini-1.5-pro-002", # Use appropriate Vertex model name if different
        #     # tools= ..., # Pass correctly configured tools if needed
        #     system_instruction=["""You are a world-class research agent... (rest of instruction)"""]
        # )
        system_instruction="""You are a world-class research agent, adept at using Google Search to find comprehensive and detailed information on any topic. Your goal is to provide the most informative and insightful answers possible.

When responding to a query:

1. **Thorough Research:** Conduct extensive searches using Google Search, exploring multiple relevant sources.
2. **Information Synthesis:** Combine information from different sources, synthesizing it into a coherent and comprehensive answer.
3. **Depth and Detail:** Prioritize providing as much relevant detail as possible, within a reasonable length.
4. **Accurate Attribution:** Clearly cite all sources you use in your responses using footnotes.

You will be provided with a user's query as input. Your task is to provide a detailed and well-researched response."""
        
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]

        self.config = types.GenerateContentConfig( # Use types.GenerationConfig
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
            tools = tools,
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )
        
        # You might want a reference to a model name for generate_content calls
        self.model_name = "gemini-2.5-flash-preview-04-17" # Adjust model name as needed for Vertex
        
    def add_voice(self, person, embedding, training=False):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            if person == "unknown" and training is False:
                part = types.Part.from_function_response(
                    name="remember_voice",
                    response={
                        "content": {"error":"person is unknown. Tell us who you are first. Say something like \"Hey Shimmy, My name is Cindy. Remember my voice.\""},
                    }
                )
            else:
                # self.voice_emb_client.publish_embedding(person,embedding) # Assuming this client is still valid
                part = types.Part.from_function_response(
                    name="remember_voice",
                    response={
                        "content": {"user_name":person},
                    }
                )
        except Exception as e:
            logging.error(f"Error in add_voice: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="remember_voice", # Return error under the expected function name
                response={
                    "content": {"error": f"cannot store voice at this time: {e}"},
                }
            )
        return part
    
    def change_led_color(self,red,green,blue):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            # Ensure types are integers
            red, green, blue = int(red), int(green), int(blue)
            self.microcontroller_client.publish_ledcolor(red, green, blue)
            part = types.Part.from_function_response(
                name="change_led_color",
                response={
                    "content": {"status":"done"},
                }
            )
        except Exception as e:
            logging.error(f"Error in change_led_color: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="change_led_color",
                response={
                    "content": {"error":f"cannot change color at this time: {e}"},
                }
            )
        return part
    
    def change_brightness(self,brightness):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            brightness = int(brightness) # Ensure type
            self.microcontroller_client.publish_ledbrightness(brightness)
            part = types.Part.from_function_response(
                name="change_brightness",
                response={
                    "content": {"status":"done"},
                }
            )
        except Exception as e:
            logging.error(f"Error in change_brightness: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="change_brightness",
                response={
                    "content": {"error":f"cannot change brightness at this time: {e}"},
                }
            )
        return part
    
    def move_shimmy(self,command):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            self.state_notifier.publish_status('Give me a few seconds while I figure out how to chart my path.')
            self.shimmy_move_client.publish_pose(str(command)) # Ensure command is string
            part = types.Part.from_function_response(
                name="move_shimmy",
                response={
                    "content": {"status":"starting to move."},
                }
            )
            
        except Exception as e:
            logging.error(f"Error in move_shimmy: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="move_shimmy",
                response={
                    "content": {"error":f"cannot move robot: {e}"},
                }
            )
        return part
    
    def turn_shimmy(self,radians):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            radians = float(radians) # Ensure type
            self.shimmy_move_client.publish_turn(radians)
            part = types.Part.from_function_response(
                name="turn_inplace",
                response={
                    "content": {"status":"starting to turn"},
                }
            )
        except Exception as e:
            logging.error(f"Error in turn_shimmy: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="turn_inplace",
                response={
                    "content": {"error":f"cannot turn robot: {e}"},
                }
            )
        return part
            
    
    def change_led_pattern(self,pattern):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            self.microcontroller_client.publish_ledpattern(str(pattern)) # Ensure type
            part = types.Part.from_function_response(
                name="change_led_pattern",
                response={
                    "content": {"status":"done"},
                }
            )
        except Exception as e:
            logging.error(f"Error in change_led_pattern: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="change_led_pattern",
                response={
                    "content": {"error":f"cannot change led pattern at this time: {e}"},
                }
            )
        return part
    
    def get_power(self):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            # Assuming send_power_request returns an object with .powerusage
            mcu_response = self.microcontroller_client.send_power_request()
            power_info = mcu_response.powerusage # Check attribute name
            part = types.Part.from_function_response(
                name="get_power",
                response={
                    "content": {"voltage": power_info.loadvoltage,
                                "current_milli_amps": power_info.currentma,
                                "milli_watts": power_info.powermw
                                },
                }
            )
        except AttributeError as ae:
             logging.error(f"Attribute error accessing power info: {ae} - Check response structure. Traceback: {traceback.format_exc()}")
             part = types.Part.from_function_response(name="get_power", response={"content": {"error":f"Internal error accessing power data: {ae}"}})
        except Exception as e:
            logging.error(f"Error in get_power: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="get_power",
                response={
                    "content": {"error":f"cannot get power at this time: {e}"},
                }
            )
        return part

    def get_image(self,text_req):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            self.state_notifier.publish_status('Give me a few seconds while I use my eyes.')
            rtxt = self.image_client.get_image(text_req)
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="use_robot_eyes",
                response={
                    "content": {"description":rtxt},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="use_robot_eyes",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part
    def find_object(self,text_req,additional_context=""):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
            self.state_notifier.publish_status(f'Give me a moment while I use my eyes to find {text_req} .')
            rtxt,_,_ = self.image_client.get_bounding_box(text_req,additional_context)
            direction = "left"
            if rtxt[0] > 0:
                direction = "right"
            print(rtxt)
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="find_object_with_eyes",
                response={
                    "content": {"description":f"{text_req} found about {rtxt[2]} meters in front and {abs(rtxt[0])} meters to the {direction}."}
                }
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="find_object_with_eyes",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part
    
    def move_to_object(self,text_req,additional_context="",move_command=""):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
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
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="move_to_object_with_wheels",
                response={
                    "content": {"description":f"{text_req} found about {rtxt[2]} meters in front. Moving to {text_req} but not there yet."},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="move_to_object_with_wheels",
                response={
                    "content": {"error":"cannot get image info at this time"},
                },
            )
        return part
    
    def cancel_move(self):
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        try:
           
            self.shimmy_move_client.publish_cancel()
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="stop_moving",
                response={
                    "content": {"description":f"Canceled Move"},
                },
            )
            
        except:
            logging.error(traceback.format_exc()) 
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="stop_moving",
                response={
                    "content": {"error":"Unable to stop moving"},
                },
            )
        return part
    
    def use_web(self,search_txt): # Renamed arg for clarity
        if not self.client: raise RuntimeError("Agent client not initialized.") # Check client
        # Needs reimplementation using an external search API or correct Vertex grounding config
        try:
            self.state_notifier.publish_status(f'Give me a moment while I look up: {search_txt}')

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=types.Part.from_text(text=search_txt),
                config=self.config,
            )
            response_text = response.text

            part = types.Part.from_function_response(
                name="use_web_browser",
                response={
                    "content": {"description": response_text},
                }
            )
        except Exception as e:
            logging.error(f"Error in use_web placeholder/implementation: {traceback.format_exc()}")
            part = types.Part.from_function_response(
                name="use_web_browser",
                response={
                    "content": {"error": f"Cannot get web info for '{search_txt}' at this time: {e}"},
                }
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
    #         part = types.Part.from_function_response(
    #             name="store_voice",
    #             response={
    #                 "content": {"description":description},
    #             },
    #         )
    #     except:
    #         logging.error(traceback.format_exc()) 
    #         part = types.Part.from_function_response(
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
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
                name="get_time",
                response={
                    "content": api_response,
                },
            )
        except:
            logging.error(traceback.format_exc()) 
            part = types.Part.from_function_response( # Uses Part from google.generativeai.types
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

get_time_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

change_neopixels_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

change_brightness_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

change_led_pattern = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

get_power_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
        name="get_power",
        description="""get the current wattage, voltage, and amperage usage on the robot.""",
        # Function parameters are specified in OpenAPI JSON schema format
        parameters={
            "type": "object",
            "properties": {
            },
        },
)

take_picture_function = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

find_object_function = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

move_to_object_function = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

store_voice_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

change_volume_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

store_image_func = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
        name="remember_image_objects",
        description="Remember things in images.",
        parameters={
            "type": "object",
            "properties": {
                "picture_context": {"type": "string", "description": "Any context"},
                
            },
        },
)

move_shimmy = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

turn_shimmy = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

stop_shimmy = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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

web_browser = types.FunctionDeclaration( # Uses FunctionDeclaration from google.generativeai.types
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
            types.Tool(function_declarations=[get_time_func]),
            # #store_voice_func,
            types.Tool(function_declarations=[take_picture_function]),
            types.Tool(function_declarations=[web_browser]),
            types.Tool(function_declarations=[change_volume_func]),
            types.Tool(function_declarations=[change_neopixels_func]),
            types.Tool(function_declarations=[change_brightness_func]),
            types.Tool(function_declarations=[get_power_func]),
            types.Tool(function_declarations=[change_led_pattern]),
            types.Tool(function_declarations=[move_shimmy]),
            types.Tool(function_declarations=[find_object_function]),
            types.Tool(function_declarations=[move_to_object_function]),
            # #turn_shimmy,
            types.Tool(function_declarations=[stop_shimmy])
            
        ]

