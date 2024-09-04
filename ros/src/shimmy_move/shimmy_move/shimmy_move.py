import rclpy
from rclpy.node import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose

import json
import threading
import time
import traceback

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration

from geometry_msgs.msg import Twist
from std_msgs.msg import ( Float64, Bool)
from rclpy.time import Time

from chat_interfaces.srv import GetPose

class ShimmyMoveService(Node):

    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('shimmy_move_service')
        self.move_subscription = self.create_subscription(
            Pose,
            f'{namespace}/move',
            self.listener_callback,
            10)
        
        # Declare parameter for angular speed
        self.declare_parameter('angular_speed', 0.4)  # Angular speed in rad/s
        self.angular_speed = self.get_parameter('angular_speed').value
        
        # Create a publisher for Twist messages
        self.twist_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.moving_publisher = self.create_publisher(Bool, f'{namespace}/moving', 10)
        self.pose_svc = self.create_service(GetPose, f'{namespace}/get_pose', self.get_pose)

        # Create a subscriber for target angle
        self.target_angle_subscription = self.create_subscription(
            Float64,
            f'{namespace}/shimmy_bot/target_angle',
            self.angle_callback,
            10)
        
        self.cancel_move_subscription = self.create_subscription(
            Bool,
            f'{namespace}/shimmy_bot/cancel_move',
            self.cancel_move_callback,
            10)

        self.target_angle = None  # Initialize target_angle
        self.target_angle_remainder = None
        self.cancel_move = False
        self.odom_subscription = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.odom_callback,10)
        
    def get_pose(self,request, response):
        response.pose = self.pos
        return response
    
    def odom_callback(self,msg):
        self.pos = msg
    
    def cancel_move_callback(self,msg: Bool):
        self.cancel_move = True
    
    def angle_callback(self, msg):
        self.cancel_move = False
        self.target_angle = msg.data
        self.target_angle_remainder = abs(msg.data)
        self.get_logger().info('Received target angle: %f radians' % self.target_angle)

        # Start rotating
        self.timer_ = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        if self.target_angle is None:
            return  # Do nothing if no target angle is received

        # Create a Twist message
        twist = Twist()

        # Set the angular velocity
        twist.angular.z = self.angular_speed if self.target_angle > 0 else -self.angular_speed

        # Publish the Twist message
        self.twist_publisher.publish(twist)

        # Decrement the target angle
        self.target_angle_remainder -= self.angular_speed * 0.05 * 1  # Adjust based on timer period
        self.get_logger().info(f'ta = {self.target_angle_remainder}.')

        # Stop rotating when the target angle is reached
        if self.target_angle_remainder < 0.01 or self.cancel_move == True:  # Use a small threshold for comparison
            twist.angular.z = 0.0
            self.twist_publisher.publish(twist)
            self.get_logger().info('Rotation complete.')
            msg = Bool()
            msg.data = False
            self.moving_publisher.publish(msg)
            self.target_angle = None  # Reset target_angle
            self.destroy_timer(self.timer_)  # Stop the timer
            
        else:
            msg = Bool()
            msg.data = True
            self.moving_publisher.publish(msg)
            
             
    def listener_callback(self, msg: Pose):
        self.cancel_move = False
        try:
            navigator = BasicNavigator()

            # Set our demo's initial pose
            # initial_pose = PoseStamped()
            # initial_pose.header.frame_id = 'map'
            # initial_pose.header.stamp = navigator.get_clock().now().to_msg()
            # initial_pose.pose.position.x = 0.0
            # initial_pose.pose.position.y = 0.0
            # initial_pose.pose.orientation.z = 0.0
            # initial_pose.pose.orientation.w = 1.0
            # navigator.setInitialPose(initial_pose)
            # Activate navigation, if not autostarted. This should be called after setInitialPose()
            # or this will initialize at the origin of the map and update the costmap with bogus readings.
            # If autostart, you should `waitUntilNav2Active()` instead.
            # navigator.lifecycleStartup()

            # Wait for navigation to fully activate, since autostarting nav2
            #navigator.waitUntilNav2Active()

            # If desired, you can change or load the map as well
            # navigator.changeMap('/path/to/map.yaml')

            # You may use the navigator to clear or obtain costmaps
            # navigator.clearAllCostmaps()  # also have clearLocalCostmap() and clearGlobalCostmap()
            # global_costmap = navigator.getGlobalCostmap()
            # local_costmap = navigator.getLocalCostmap()

            # Go to our demos first goal pose
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'odom'
            goal_pose.header.stamp = Time(seconds=0, nanoseconds=0).to_msg()  # Set timestamp to 0
            goal_pose.pose = msg

            # sanity check a valid path exists
            # path = navigator.getPath(initial_pose, goal_pose)

            #navigator.goToPose(goal_pose)
            self.nav_publisher.publish(goal_pose)
            i = 0
            while not navigator.isTaskComplete():
                if self.cancel_move:
                    print("############################################")
                    navigator.cancelTask()
                else:
                    i = i + 1
                    feedback = navigator.getFeedback()
                    if feedback and i % 5 == 0:
                        self.get_logger().info(
                            'Estimated time of arrival: '
                            + '{0:.0f}'.format(
                                Duration.from_msg(feedback.estimated_time_remaining).nanoseconds
                                / 1e9
                            )
                            + ' seconds.'
                        )
                        # msg = Bool()
                        # msg.data = True
                        # self.moving_publisher.publish(msg)

                        # Some navigation timeout to demo cancellation
                        if Duration.from_msg(feedback.navigation_time) > Duration(seconds=600.0):
                            navigator.cancelTask()

                        # Some navigation request change to demo preemption
                        if Duration.from_msg(feedback.navigation_time) > Duration(seconds=18.0):
                            goal_pose.pose.position.x = 0.0
                            goal_pose.pose.position.y = 0.0
                            navigator.goToPose(goal_pose)

            # Do something depending on the return code
            result = navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info('Goal succeeded!')
            elif result == TaskResult.CANCELED:
                self.get_logger().info('Goal was canceled!')
            elif result == TaskResult.FAILED:
                self.get_logger().info('Goal failed!')
            else:
                self.get_logger().info('Goal has an invalid return status!')
            msg = Bool()
            msg.data = False
            self.moving_publisher.publish(msg)
            #navigator.lifecycleShutdown()
        except:
            self.get_logger().error('%s' % traceback.format_exc())
            
    
    
        

    
    

    
        
    


def main(args=None):
    rclpy.init(args=args)
    shimmy_move_service = ShimmyMoveService()
    

    rclpy.spin(shimmy_move_service)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    shimmy_move_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()