cmake_minimum_required(VERSION 3.0.2)
project(ar_code_landing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  mavros_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/gimbal_aruco_tracker_node.py
  scripts/ar_landing_controller_node.py
  scripts/ar_landing_sm_node.py
  scripts/spawn_model_idempotent.py
  scripts/trigger_drone_spawn.py
  scripts/climb_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Shell scripts also need installing so roslaunch can find them via $(find ...)
install(PROGRAMS
  scripts/cleanup_session.sh
  scripts/wait_and_launch.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
