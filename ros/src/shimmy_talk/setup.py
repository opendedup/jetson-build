from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'shimmy_talk'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + "/services",package_name + "/skills"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam Silverberg',
    maintainer_email='sam.silverberg@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'riva_asr = shimmy_talk.riva_asr:main',
            'listener = shimmy_talk.google_asr_subscriber:main',
            'gcp_agent_asr = shimmy_talk.gcp_agent_asr:main',
            'walley_listener = shimmy_talk.walley_asr_subscriber:main',
            'whisper_asr = shimmy_talk.whisper_asr:main',
            'gcp_asr = shimmy_talk.gcp_asr:main',
            'gemini_asr = shimmy_talk.gemini_asr:main',
        ],
    },
)
