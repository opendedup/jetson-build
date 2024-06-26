from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='faiss_service',
            name="voice_emb",
            executable='service',
            parameters=[
                {"dimensions": 1251}
            ]
        ),
        Node(
            package='faiss_service',
            executable='service',
            name="img_emb",
            namespace='images',
            parameters=[
                {"dimensions": 1408,
                 "embeddings_path":"/opt/ros2/mmembeddings/"}
            ]
        ),
         Node(
            package='shimmy_microcontroller',
            executable='service',
            name="m_service",
            parameters=[
                {             "serial_device":"/dev/ttyACM0"}
            ]
        ),
        Node(
            package='riva_asr',
            executable='riva_asr',
            parameters=[{"sound_device": "respeaker"}]
        ),
        Node(
            package='riva_asr',
            executable='listener',
            name='asr_listener',
            output='both',
            parameters=[
                {"sound_device": "miniDSP",
                 "train_voice": False,
                 "train_voice_name": "unknown"}
            ]
            
        ),
        Node(
            package='image_cap',
            executable='service',
            remappings=[
            ('/image_raw', '/zed/zed_node/left/image_rect_color'),
         ]
        )
        ,
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
        ),
        
        
    ])