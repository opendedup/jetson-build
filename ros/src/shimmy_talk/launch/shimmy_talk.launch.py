from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler,OpaqueFunction,SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
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
    
    voice_emb = Node(
            package='faiss_service',
            name="voice_emb",
            executable='service',
            output='both',
            parameters=[
                {"dimensions": 1251}
            ]
        )
    
    img_emb = Node(
            package='faiss_service',
            executable='service',
            name="img_emb",
            namespace='images',
            output='both',
            parameters=[
                {"dimensions": 1408,
                 "embeddings_path":"/opt/shimmy/mmembeddings/"}
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
    
    riva_asr = Node(
            package='shimmy_talk',
            executable='riva_asr',
            name='riva_asr',
            output='both',
            parameters=[config]
        )
    
    asr_listener = Node(
            package='shimmy_talk',
            executable='listener',
            name='asr_listener',
            output='both',
            parameters=[
                config
            ]
            
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
    
    foxglove_bridge = Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
        )
    
    return [
        foxglove_bridge,
        image_cap_service,
        asr_listener,
        riva_asr,
        m_service,
        #img_emb,
        voice_emb,
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
    

