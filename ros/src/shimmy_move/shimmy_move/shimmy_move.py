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

class ShimmyMoveService(Node):

    def __init__(self,namespace=''):
        super().__init__('shimmy_move_service')
        self.subscription = self.create_subscription(
            Pose,
            f'{namespace}/shimmy_move',
            self.listener_callback,
            10)
    
    
    
             
    def listener_callback(self, msg: Pose):
        try:
            self.get_logger().info("Running!!!!")
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
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = navigator.get_clock().now().to_msg()
            goal_pose.pose = msg

            # sanity check a valid path exists
            # path = navigator.getPath(initial_pose, goal_pose)

            navigator.goToPose(goal_pose)

            i = 0
            while not navigator.isTaskComplete():
                ################################################
                #
                # Implement some code here for your application!
                #
                ################################################

                # Do something with the feedback
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