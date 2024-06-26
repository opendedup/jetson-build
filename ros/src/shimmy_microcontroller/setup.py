from setuptools import find_packages, setup

package_name = 'shimmy_microcontroller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam Silverberg',
    maintainer_email='sam.silverberg@gmail.com',
    description='Access the ESP32 micro-controller',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'service = shimmy_microcontroller.microcontroller:main',
        ],
    },
)
