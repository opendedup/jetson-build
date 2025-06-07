from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler,OpaqueFunction,SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import Command, FindExecutable, PythonExpression, PathJoinSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    config = PathJoinSubstitution(
        [
            FindPackageShare("shimmy_talk"),
            "config",
            "shimmy_talk.yaml",
        ]
    )
    
    
    
    m_service = Node(
            package='shimmy_microcontroller',
            executable='service',
            name="m_service",
            parameters=[
                config
            ]
        )
    
    gcp_agent_asr = Node(
            package='shimmy_talk',
            executable='gcp_agent_asr',
            name='asr_service',
            output='both',
            parameters=[config]
        )
    
    image_cap_service = Node(
            package='image_cap',
            executable='service',
            name='image_cap_service',
            remappings=[
            ('/image_raw', '/zed/zed_node/left/image_rect_color'),
            ('/depth_image_raw', '/zed/zed_node/depth/depth_registered'),
         ]
        )

    grpc_communication_node = Node(
        package='shimmy_cloud_communication',
        executable='grpc_communication_node',
        name='grpc_communication_node',
        output='both',
        parameters=[{'robot_id': 'shimmy_bot_001'}]
    )
    
    node_checker = Node(
        package='shimmy_bot_utils',
        executable='node_checker',
        name='shimmy_talk_node_checker',
        output='both',
        parameters=[{'target_nodes': ['voice_emb','m_service','image_cap_service','asr_listener','asr_service', 'grpc_communication_node']}]  # Pass the list of nodes to monitor
    )
    
    startup_feedback_node = Node(
        package='shimmy_bot_utils',
        executable='startup_controller',
        name='shimmy_startup_controller',
        output='both',
        parameters=[{'target_nodes': ['voice_emb','m_service','image_cap_service','asr_listener','asr_service', 'grpc_communication_node']}]  # Pass the list of nodes to monitor
    
    )
    
    # launch_init_after_talk = RegisterEventHandler(
    #     event_handler=OnProcessStart(
    #         target_action=asr_listener,
    #         on_start=[
    #             TimerAction(
    #                     period=30.0,
    #                     actions=[shimmy_move_init],
    #                 )],
    #     )
    # )
    
    return [
        # foxglove_bridge,
        image_cap_service,
        gcp_agent_asr,
        m_service,
        grpc_communication_node,
        node_checker,
        startup_feedback_node
        #launch_init_after_talk
    ]

def generate_launch_description():
    return LaunchDescription(
        [
            # SetEnvironmentVariable(name='RCUTILS_COLORIZED_OUTPUT', value='1'),
            # DeclareLaunchArgument(
            #     'use_zed_localization',
            #     default_value='true',
            #     description='Creates a TF tree with `camera_link` as root frame if `true`, otherwise the root is `base_ling`.',
            #     choices=['true', 'false']),
            OpaqueFunction(function=launch_setup)    
        ]
    )
    

