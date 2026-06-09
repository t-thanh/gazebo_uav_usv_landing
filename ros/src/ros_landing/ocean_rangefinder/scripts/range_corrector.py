#!/usr/bin/env python3
"""
range_corrector.py
──────────────────
Virtual sensor that snaps rangefinder hits to Z=0 if they hit the grass plane (Z=-0.4).
"""

import rospy
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion

class OceanRangefinder:
    def __init__(self):
        rospy.init_node('ocean_rangefinder', anonymous=False)
        
        # --- Parameters ---
        self.water_z = 0.0
        self.grass_z = -0.4
        self.detection_threshold = -0.1  # If hit is below this, snap to water_z
        
        self.current_z = 0.0
        self.tilt_angle = 0.0
        
        # --- Subscribers ---
        # Using main odom for true height above world origin
        rospy.Subscriber('~odom', Odometry, self.odom_cb)
        rospy.Subscriber('~range_raw', Range, self.range_cb)
        
        # --- Publishers ---
        self.pub = rospy.Publisher('~range_filtered', Range, queue_size=10)
        
        rospy.loginfo("[ocean_rangefinder] Virtual surface initialized at Z=0.")

    def odom_cb(self, msg):
        self.current_z = msg.pose.pose.position.z
        
        # Extract tilt (pitch/roll) to correct for slanted ray casting
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # Total tilt from nadir
        self.tilt_angle = np.sqrt(roll**2 + pitch**2)

    def range_cb(self, msg):
        raw_range = msg.range
        
        # Calculate where the ray actually hit in world Z
        # Assume nadir rangefinder (pointing down)
        hit_z = self.current_z - (raw_range * np.cos(self.tilt_angle))
        
        corrected_msg = msg
        
        # If the hit is significantly below the water surface (hitting grass)
        if hit_z < self.detection_threshold:
            # New range = distance from current Z to Water Surface (Z=0)
            # Dividing by cos(tilt) accounts for the slant distance
            corrected_range = self.current_z / np.cos(self.tilt_angle)
            
            # Clamp to raw if we are somehow above (shouldn't happen)
            corrected_msg.range = min(raw_range, corrected_range)
            rospy.logdebug_throttle(1.0, f"[ocean_rangefinder] Hitting grass at Z={hit_z:.2f}. Corrected to water.")
        else:
            # Hitting USV or other object above water - no change
            corrected_msg.range = raw_range
            rospy.logdebug_throttle(1.0, f"[ocean_rangefinder] Hitting object above water at Z={hit_z:.2f}.")

        self.pub.publish(corrected_msg)

if __name__ == '__main__':
    try:
        node = OceanRangefinder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
