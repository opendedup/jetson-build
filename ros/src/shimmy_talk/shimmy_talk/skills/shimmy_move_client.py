
import rclpy
from rclpy.node import Node

import json
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from std_msgs.msg import (Float64, Bool)
import math
import numpy as np


from std_srvs.srv import Empty
from chat_interfaces.srv import GetPose

from google import genai
from google.genai import types

import logging
from logging.handlers import RotatingFileHandler

class ShimmyMoveClientAsync(Node):

    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('shimmy_move_client')
        
        self.shimmy_move_publisher = self.create_publisher(Pose, f'{namespace}/move', 10)
        self.shimmy_turn_publisher = self.create_publisher(Float64, f'{namespace}/target_angle', 10)
        self.shimmy_cancel_publisher = self.create_publisher(Bool, f'{namespace}/cancel_move', 10)
        self.zed_cli = self.create_client(Trigger, "/zed/zed_node/reset_pos_tracking")
        self.rtab_cli = self.create_client(Empty, "/rtabmap/reset")
        self.nav_model ="gemini-2.5-pro-preview-03-25"
        self.nav_config = types.GenerateContentConfig(
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
            tools=[types.Tool(function_declarations=[move_shimmy_func])],
            #system_instruction=[types.Part.from_text(text=bounding_box_instructions)],
        )
        self.pos_cli = self.create_client(GetPose, f"{namespace}/get_pose")
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
        
        
    
    
        
        
        
    
    def get_odom(self):
        req = GetPose.Request()
        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        ft = future.result()
        quad = ft.pose.pose.orientation
        pos = ft.pose.pose.position
    
        rpy = quaternion_to_rpy((quad.w,quad.x,quad.y,quad.z))
        return {
            "position": {
                "x": round(pos.x, 3),  # Round x to 4 decimal places
                "y": round(pos.y, 3),  # Round y to 4 decimal places
                "z": round(pos.z, 3)   # Round z to 4 decimal places 
            },
            "orientation": {
                "angle": round(rpy[2],4)
            }
        }
        
    
    def reset_odom(self):
        future = self.rtab_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        trigger_req = Trigger.Request()
        future = self.zed_cli.call_async(trigger_req)
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        print(future)
        return future.result()
        

    def publish_turn(self,radians):
        msg = Float64()
        msg.data = radians
        self.shimmy_turn_publisher.publish(msg)
    
    def publish_cancel(self):
        msg = Bool()
        msg.data = True
        self.shimmy_cancel_publisher.publish(msg)
    
    def publish_pose(self,command):
        self.get_logger().info("processing move command %s" % (command))
        response = self.client.models.generate_content(
            model=self.nav_model,
            contents=[command],
            config=self.nav_config,
        )
        logging.info(response)
        function_call = response.function_calls[0]
        logging.info(function_call)
        # Extract parameters from the function call
        forward_distance = function_call.args.get("forward_distance", 0)
        right_distance = function_call.args.get("right_distance", 0)
        rotation_angle = function_call.args.get("rotation_angle", 0)

        # Calculate the goal pose
        pose = calculate_goal_pose(
            self.get_odom(), forward_distance, right_distance, rotation_angle
        )
        print(pose)
        msg = Pose()
        msg.position.x = float(pose["position"]["x"])
        msg.position.y = float(pose["position"]["y"]*-1) # Left is positive
        msg.position.z = float(0)
        x, y, z, w = quaternion_from_euler(float(0),float(0),float(pose["orientation"]["angle"]))
        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w
        self.shimmy_move_publisher.publish(msg)
        
        
    

def main():
    rclpy.init()

    minimal_client = ShimmyMoveClientAsync()
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts Euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Uses ZYX convention (yaw-pitch-roll)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[2] = sy * cp * cr - cy * sp * sr  # z 
    q[1] = cy * sp * cr + sy * cp * sr  # y
    q[0] = cy * cp * sr - sy * sp * cr  # x
    q[3] = cy * cp * cr + sy * sp * sr  # w

    return q


def quaternion_to_rpy(quaternion):
  """
  Converts a quaternion to roll, pitch, and yaw angles.

  Args:
    quaternion: A numpy array representing the quaternion (w, x, y, z).

  Returns:
    A numpy array containing the roll, pitch, and yaw angles in radians.
  """

  # Extract quaternion components
  w, x, y, z = quaternion

  # Calculate roll, pitch, and yaw angles
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
  pitch = np.arcsin(2.0 * (w * y - z * x))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

  return np.array([roll, pitch, yaw])


def calculate_goal_pose(current_pose, forward_distance=0, right_distance=0, rotation_angle=0):
    """Calculates the target pose based on the current pose, forward/right distances, and rotation angle.

    Args:
        current_pose: A dictionary representing the robot's current position and orientation:
            {
                "position": {
                    "x": float,  # X position in meters
                    "y": float,  # Y position in meters
                    "z": float  # Z position in meters (always 0 for this example)
                },
                "orientation": {
                    "angle": float  # Angle in radians
                }
            }
        forward_distance: Distance to move forward (positive) or backward (negative) in meters.
        right_distance: Distance to move right (positive) or left (negative) in meters.
        rotation_angle: Angle to rotate in radians (counterclockwise is positive).

    Returns:
        A dictionary representing the target pose in the same format as current_pose.
    """

    target_pose = current_pose.copy()  # Start with the current pose

    # Extract current position and orientation
    current_x = current_pose["position"]["x"]
    current_y = current_pose["position"]["y"]
    current_angle = current_pose["orientation"]["angle"]

    # Calculate movement angle based on current orientation and right_distance
    movement_angle = current_angle - (math.pi / 2) * (right_distance / abs(right_distance)) if right_distance else current_angle

    # Calculate new position
    target_pose["position"]["x"] = current_x + forward_distance * math.cos(current_angle) + right_distance * math.cos(movement_angle)
    target_pose["position"]["y"] = current_y + forward_distance * math.sin(current_angle) + right_distance * math.sin(movement_angle)

    # Calculate new orientation
    target_pose["orientation"]["angle"] = (current_angle + rotation_angle + 2 * math.pi) % (2 * math.pi)

    return target_pose

move_shimmy_func = types.FunctionDeclaration(
    name="move_shimmy",
    description="""
    Calculate the target pose for the robot based on a natural language command 
    that includes forward/backward movement, left/right movement, and rotation.

    The command should be in the format: 
    "Move [forward_distance] meters [forward/back], [to the left/right] [right_distance] meters, and [turn around/rotate [angle] degrees [left/right]]"

    Examples:
    * "Move 1.25 meters forward, to the left 0.31 meters, and turn around"
    * "Move 2 meters back, to the right 1 meter, and rotate 90 degrees left"
    * "Move 0.5 meters forward and turn around"
    * "Move 1 meter to the right"
    * "Turn around"
    """,
    parameters={
        "type": "object",
        "properties": {
            "forward_distance": {
                "type": "number",
                "description": "Distance to move forward (positive) or backward (negative) in meters."
            },
            "right_distance": {
                "type": "number",
                "description": "Distance to move right (positive) or left (negative) in meters."
            },
            "rotation_angle": {
                "type": "number",
                "description": "Angle to rotate in radians (counterclockwise is positive)."
            }
        },
        "required": ["forward_distance", "right_distance", "rotation_angle"]
    },
)