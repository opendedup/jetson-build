from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='faiss_service',
            executable='service',
            parameters=[
                {"dimensions": 1251}
            ]
        ),
        Node(
            package='riva_asr',
            executable='riva_asr',
            parameters=[
                {"device_number": 0}
            ]
        ),
        Node(
            package='riva_asr',
            executable='listener',
            name='asr_listener',
            parameters=[
                {"device_number": "iFi (by AMR) HD USB Audio (hw:0,0)"}
            ]
            
        ),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            parameters=[
                {"params-file": "usb_cam/params.yaml"}
            ]
        ),
        Node(
            package='image_cap',
            executable='service'
        ),
        
        
    ])