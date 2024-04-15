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
            package='riva_asr',
            executable='riva_asr',
            parameters=[
                {"device_number": 1,
                 "riva_uri": "172.17.0.3:50051"}
            ]
        ),
        Node(
            package='riva_asr',
            executable='listener',
            name='asr_listener',
            parameters=[
                {"sound_device": 0}
            ]
            
        ),
        Node(
            package='image_cap',
            executable='service',
            remappings=[
            ('/image_raw', '/zed/zed_node/left/image_rect_color'),
         ]
        ),
        
        
    ])