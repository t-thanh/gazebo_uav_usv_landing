cmake_minimum_required(VERSION 3.0.2)
project(uav_landing_evaluator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  gazebo_ros
  gazebo_msgs
  geometry_msgs
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS scripts/landing_evaluator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY config launch models worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
