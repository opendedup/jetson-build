import rclpy
from rclpy.node import Node
from rclpy.lifecycle import Publisher
from rclpy.lifecycle import State
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose


import traceback

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration

from std_msgs.msg import ( Float64, Bool)
from rclpy.time import Time
from std_msgs.msg import String

from geometry_msgs.msg import Twist

from chat_interfaces.srv import GetPose

class ShimmyMoveService(Node):

    def __init__(self,namespace='/shimmy_bot'):
        super().__init__('shimmy_move_service')
        self.move_subscription = self.create_subscription(
            Pose,
            f'{namespace}/move',
            self.listener_callback,
            10)
        

        self.nav_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.moving_publisher = self.create_publisher(Bool, f'{namespace}/moving', 10)
        self.pose_svc = self.create_service(GetPose, f'{namespace}/get_pose', self.get_pose)
        self.system_message_publisher  = self.create_publisher(String, f'{namespace}/system_message', 10)

        
        self.cancel_move_subscription = self.create_subscription(
            Bool,
            f'{namespace}/shimmy_bot/cancel_move',
            self.cancel_move_callback,
            10)

        self.cancel_move = False
        self.odom_subscription = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.odom_callback,10)
        self.move_active = False  # Flag to track move status
        #self.wiggle()
        self.navigator = BasicNavigator()  # Initialize the navigator
        
    def publish_status(self,status):
        msg = String()
        msg.data = status
        self.system_message_publisher.publish(msg)
    
    def get_pose(self,request, response):
        response.pose = self.pos
        return response
    
    def odom_callback(self,msg):
        self.pos = msg
    
    def cancel_move_callback(self,msg: Bool):
        self.cancel_move = True
        
    
    def wiggle(self):
        """Moves the robot back and forth rapidly 4 times."""

        # Publisher for the /cmd_vel topic
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Initialize movement counter
        self.move_count = 0

        # Create a timer to handle back-and-forth motion
        self.timer = self.create_timer(1.0, self.wiggle_callback)  # 1.5 seconds for each movement

    def wiggle_callback(self):
        """Callback function for the timer, handles movement logic."""

        twist = Twist()  # Create a new Twist message each time

        if self.move_count < 9: # 8 movements (4 cycles of back and forth)
            if self.move_count % 2 == 0:
                twist.linear.x = 4.0  # Move forward
            else:
                twist.linear.x = -4.0  # Move backward

            self.cmd_vel_pub.publish(twist)
            self.move_count += 1

        else:
            # Stop the robot and cancel the timer
            twist.linear.x = 0.0
            self.cmd_vel_pub.publish(twist)
            self.destroy_timer(self.timer)

    
            
             
    def listener_callback(self, msg: Pose):
        self.cancel_move = False
        try:
            

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
            goal_pose.header.stamp = Time(seconds=0, nanoseconds=0).to_msg()  # Set timestamp to 0
            goal_pose.pose = msg

            # sanity check a valid path exists
            # path = navigator.getPath(initial_pose, goal_pose)

            self.navigator.goToPose(goal_pose)
            #self.nav_publisher.publish(goal_pose)
            i = 0
            while not self.navigator.isTaskComplete():
                if self.cancel_move:
                    self.navigator.cancelTask()
                    self.publish_status("Move Was Cancelled")
                else:
                    i = i + 1
                    feedback = self.navigator.getFeedback()
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
                            self.navigator.cancelTask()

                        # Some navigation request change to demo preemption
                        if Duration.from_msg(feedback.navigation_time) > Duration(seconds=300.0):
                            goal_pose.pose.position.x = 0.0
                            goal_pose.pose.position.y = 0.0
                            self.navigator.goToPose(goal_pose)

            # Do something depending on the return code
            result = self.navigator.getResult()
            self.get_logger().info(f"Result = {result}")
            if result == TaskResult.SUCCEEDED:
                self.publish_status("Finished Move")
                self.get_logger().info('Goal succeeded!')
            elif result == TaskResult.CANCELED:
                self.publish_status("Move Was Canceled")
                self.get_logger().info('Goal was canceled!')
            elif result == TaskResult.FAILED:
                self.publish_status("Move Failed")
                self.get_logger().info('Goal failed!')
            else:
                self.publish_status("Move Failed for some unkown reason.")
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