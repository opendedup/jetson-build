import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Bool
from std_srvs.srv import Trigger
from std_srvs.srv import Empty

class RotateClockwiseCounterclockwise(Node):

    def __init__(self):
        super().__init__('rotate_clockwise_counterclockwise')
        # Create a publisher for the target angle topic
        self.target_angle_publisher = self.create_publisher(Float64, '/shimmy_bot/target_angle', 10)

        # Create a subscriber for the moving topic
        self.moving_subscriber = self.create_subscription(Bool,
            '/shimmy_bot/moving',
            self.moving_callback,
            10)

        # Set the angles for clockwise and counterclockwise rotation
        self.clockwise_angle = 6.28319* 1.3 # 90 degrees in radians
        self.counterclockwise_angle = -6.28319*1.3  # -90 degrees in radians

        # Initialize the moving flag
        self.is_moving = False
        #self.zed_cli = self.create_client(Trigger, "/zed/zed_node/reset_pos_tracking")
        #self.rtab_cli = self.create_client(Empty, "/rtabmap/reset")
        
        # Create a timer to trigger the rotation sequence
        self.stage = 1
        self.get_logger().info(f"Step {self.stage} Started")
        self.publish_target_angle(self.clockwise_angle)

    def moving_callback(self, msg):
        self.is_moving = msg.data
        if self.is_moving is False and self.stage < 2:
            self.stage +=1
            #self.reset_odom()
            self.get_logger().info(f"Step {self.stage} Started")
            if self.stage == 2:
                self.publish_target_angle(self.counterclockwise_angle)
            
            self.get_logger().info(f"Step {self.stage} Ended")
            

    def publish_target_angle(self, angle):
        msg = Float64()
        msg.data = angle
        self.target_angle_publisher.publish(msg)
        self.get_logger().info('Publishing target angle: {}'.format(angle))

    def reset_odom(self):
        trigger_req = Trigger.Request()
        future = self.zed_cli.call_async(trigger_req)
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        print(future)
        future = self.rtab_cli.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future,timeout_sec=30)
        return future.result()


def main(args=None):
    rclpy.init(args=args)

    rotate_node = RotateClockwiseCounterclockwise()

    rclpy.spin(rotate_node)

    rotate_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()