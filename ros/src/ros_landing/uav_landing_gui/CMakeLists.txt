cmake_minimum_required(VERSION 3.0.2)
project(uav_landing_gui)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  mavros_msgs
  sensor_msgs
  nav_msgs
)

find_package(Qt5 REQUIRED COMPONENTS Widgets)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
