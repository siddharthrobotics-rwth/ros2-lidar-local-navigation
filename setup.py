from setuptools import find_packages, setup

package_name = 'obstacle_avoider'

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
    maintainer='sid',
    maintainer_email='sid@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'obstacle_avoider_node = obstacle_avoider.obstacle_avoider:main',
             'obstacle_avoider_phase3 = obstacle_avoider.obstacle_avoider_phase3:main',
             'obstacle_avoider_phase2_robust = obstacle_avoider.obstacle_avoider_phase2_robust_obstacle:main',
             'obstacle_avoider_phase2_5 = obstacle_avoider.obstacle_avoider_phase2_5_squeeze_corner:main',
             'obstacle_avoider_phase2_5_search_fix = obstacle_avoider.obstacle_avoider_phase2_5_search_fix:main',
             'obstacle_avoider_phase2_5_front_wall = obstacle_avoider.obstacle_avoider_phase2_5_front_wall_fix:main'
        ],
    },
)
