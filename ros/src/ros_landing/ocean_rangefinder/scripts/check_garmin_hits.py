#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion

def odom_cb(msg):
    global current_z, tilt_angle
    current_z = msg.pose.pose.position.z
    q = msg.pose.pose.orientation
    roll, pitch, _ = euler_from_quaternion([q.x, q.y, q.z, q.w])
    tilt_angle = np.sqrt(roll**2 + pitch**2)

def range_cb(msg):
    if current_z is None:
        return
    
    # Calculate absolute Z of the hit point
    hit_z = current_z - (msg.range * np.cos(tilt_angle))
    
    status = "UNKNOWN"
    if hit_z < -0.2:
        status = "GRASS/FLOOR (Corrected: Water is 0.4m above this)"
    elif hit_z > 0.1:
        status = "USV / OBSTACLE"
    else:
        status = "WATER SURFACE (Z=0)"
        
    print(f"UAV_Z: {current_z:6.2f} | Range: {msg.range:6.2f} | Hit_Z: {hit_z:6.2f} | Target: {status}")

if __name__ == '__main__':
    rospy.init_node('garmin_diagnostic')
    current_z = None
    tilt_angle = 0.0
    
    # Check both potential topics (raw and redirected)
    topic = rospy.get_param('~topic', '/uav1/garmin/range')
    
    rospy.Subscriber('/uav1/odometry/odom_main', Odometry, odom_cb)
    rospy.Subscriber(topic, Range, range_cb)
    
    print(f"--- Monitoring {topic} ---")
    rospy.spin()
