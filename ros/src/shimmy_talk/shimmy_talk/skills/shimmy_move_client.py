
import rclpy
from rclpy.node import Node

import json
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from std_msgs.msg import (Float64, Bool)
import math
import numpy as np

import vertexai.preview.generative_models as generative_models

from vertexai.generative_models import (
    GenerativeModel,
    Part
)
from std_srvs.srv import Empty
from chat_interfaces.srv import GetPose

class ShimmyMoveClientAsync(Node):

    def __init__(self,namespace=''):
        super().__init__('shimmy_move_client')
        
        self.shimmy_move_publisher = self.create_publisher(Pose, f'{namespace}/shimmy_bot/move', 10)
        self.shimmy_turn_publisher = self.create_publisher(Float64, f'{namespace}/shimmy_bot/target_angle', 10)
        self.shimmy_cancel_publisher = self.create_publisher(Bool, f'{namespace}/shimmy_bot/cancel_move', 10)
        self.zed_cli = self.create_client(Trigger, "/zed/zed_node/reset_pos_tracking")
        self.rtab_cli = self.create_client(Empty, "/rtabmap/reset")
        nav_sys_intructions = """you are an expert on ROS 2. Your job is to use NAV2 to move a robot around from place to place or turn a robot around. You will be using geometry_msgs/msg/Pose messages to perform your actions.
This is what a typical pos message looks like

{
"position": {
"x":"X position in meters as float value positive is forward, backwards is negative",
"y":"Y position in meters as float value, left is positive, right is negative",
"z":"should always be zero"
},
"orientation": {
rpy:["roll angle in radians as float","pitch angle in radians as float","yaw angle in radians as float"]
}
}
for  the rpy values left is positive right is negative.
only return the message and nothing else."""
        self.nav_model = GenerativeModel(
            "gemini-1.5-pro-001",
            system_instruction=[nav_sys_intructions]
        )
        self.nav_chat = self.nav_model.start_chat()
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
        self.pos_cli = self.create_client(GetPose, f"/shimmy_bot/get_pose")
        
        
    
    
        
        
        
    
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
                "x": pos.x,
                "y": pos.y,
                "z": pos.z
            },
            "orientation": {
                "rpy": [rpy[0],rpy[1],rpy[2]]
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
        #self.reset_odom()
        codom = json.dumps(self.get_odom())
        c_msg = f"""The current position is:
        {codom}
        
        {command}
        """
        print(c_msg)
        response = self.nav_chat.send_message([c_msg],
                                generation_config=self.config,stream=False, safety_settings=self.safety_settings)
        
        print("###########################################")
        print(response.text)
        print(response.text.replace("```json","").replace("```",""))
        pose = json.loads(response.text.replace("```json","").replace("```",""))
        msg = Pose()
        msg.position.x = float(pose["position"]["x"])
        msg.position.y = float(pose["position"]["y"])
        msg.position.z = float(pose["position"]["z"])
        x, y, z, w = quaternion_from_euler(float(pose["orientation"]["rpy"][0]),float(pose["orientation"]["rpy"][1]),float(pose["orientation"]["rpy"][2]))
        msg.orientation.x = x
        msg.orientation.y = y
        msg.orientation.z = z
        msg.orientation.w = w
        self.shimmy_move_publisher.publish(msg)
        
    def publish_pose_object(self,command,image):
        #self.reset_odom()
        codom = json.dumps(self.get_odom())
        c_msg = f"""The current position is:
        {codom}
        
        {command}
        """
        print(c_msg)
        ipart = Part.from_data(data=image.getvalue(),mime_type="image/jpeg")
        response = self.nav_chat.send_message([c_msg,ipart],
                                generation_config=self.config,stream=False, safety_settings=self.safety_settings)
        
        print("###########################################")
        print(response.text)
        print(response.text.replace("```json","").replace("```",""))
        pose = json.loads(response.text.replace("```json","").replace("```",""))
        msg = Pose()
        msg.position.x = float(pose["position"]["x"])
        msg.position.y = float(pose["position"]["y"])
        msg.position.z = float(pose["position"]["z"])
        x, y, z, w = quaternion_from_euler(float(pose["orientation"]["rpy"][0]),float(pose["orientation"]["rpy"][1]),float(pose["orientation"]["rpy"][2]))
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
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

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

