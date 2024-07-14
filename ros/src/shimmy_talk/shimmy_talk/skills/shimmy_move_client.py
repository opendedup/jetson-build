
import rclpy
from rclpy.node import Node

import json
from geometry_msgs.msg import Pose
import math

class ShimmyMoveClientAsync(Node):

    def __init__(self,namespace=''):
        super().__init__('shimmy_move_client')
        self.shimmy_move_publisher = self.create_publisher(Pose, f'{namespace}/shimmy_move', 10)
        
    
    def publish_pose(self,pose):
            
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