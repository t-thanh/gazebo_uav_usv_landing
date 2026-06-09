cmake_minimum_required(VERSION 3.0.2)
project(uav_traj_landing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  mavros_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs nav_msgs geometry_msgs mavros_msgs
)

catkin_install_python(PROGRAMS
  scripts/traj_landing_controller_node.py
  scripts/traj_landing_sm_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
