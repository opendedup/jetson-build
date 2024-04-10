from noaa_sdk import NOAA
from datetime import datetime
from vertexai.generative_models import (
    Part,
    Tool,
    FunctionDeclaration,
)
import pytz 

def get_weather(coords):
    n = NOAA()
    fcs = n.points_forecast(coords[0],coords[1],hourly=False, type='forecast')['properties']['periods']
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    local_now = now.astimezone()
    local_tz = local_now.tzinfo
    local_tzname = local_tz.tzname(local_now)
    api_response = {"current_time": date_time, "time_zone": local_tzname,"weather":fcs[:3]}
    part = Part.from_function_response(
        name="get_weather",
        response={
            "content": api_response,
        },
    )
    return part

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

def get_current_time(timezone):
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
    return part

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
                "coords": {"type": "string", "description": "The Valid latitude and longitude associated with the weather request.E.g. \"38.8894,-77.0352\" "},
                
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

robot_tools = Tool(
        function_declarations=[
            get_time_func,
            get_weather_func,
            store_voice_func,
            take_picture_function
        ],
)

