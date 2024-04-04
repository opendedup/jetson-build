from setuptools import find_packages, setup

package_name = 'riva_asr'

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
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'riva_asr = riva_asr.riva_asr:main',
            'listener = riva_asr.asr_subscriber:main'
        ],
    },
)
