#!/usr/bin/env python3
"""
spawn_platform.py
─────────────────
Idempotent spawner for the validation_platform model.

  1. Waits for Gazebo spawn + delete services to become available.
  2. Deletes any existing 'validation_platform' model (silent no-op if absent).
  3. Spawns the SDF model fresh.

This replaces the bare gazebo_ros/spawn_model call so that repeated
roslaunch invocations never fail with "entity already exists".
"""

import os
import sys
import rospy
from gazebo_msgs.srv import (DeleteModel, DeleteModelRequest,
                              SpawnModel,  SpawnModelRequest)

MODEL_NAME = 'validation_platform'


def main():
    rospy.init_node('spawn_platform', anonymous=False)

    sdf_path = rospy.get_param('~sdf_path', '')
    if not sdf_path or not os.path.isfile(sdf_path):
        rospy.logfatal(f'[spawn_platform] sdf_path not set or file not found: {sdf_path!r}')
        sys.exit(1)

    rospy.loginfo(f'[spawn_platform] Waiting for Gazebo services …')
    rospy.wait_for_service('/gazebo/delete_model')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')

    delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    spawn_srv  = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # ── 1. Delete if already present ─────────────────────────────────────────
    try:
        resp = delete_srv(DeleteModelRequest(model_name=MODEL_NAME))
        if resp.success:
            rospy.loginfo(f'[spawn_platform] Deleted existing "{MODEL_NAME}".')
        # If it wasn't there resp.success is False — that's fine, continue.
    except rospy.ServiceException as e:
        rospy.logwarn(f'[spawn_platform] Delete call failed (ignored): {e}')

    # ── 2. Spawn fresh ────────────────────────────────────────────────────────
    with open(sdf_path) as f:
        sdf_xml = f.read()

    req = SpawnModelRequest()
    req.model_name      = MODEL_NAME
    req.model_xml       = sdf_xml
    req.robot_namespace = ''
    req.reference_frame = 'world'

    try:
        resp = spawn_srv(req)
    except rospy.ServiceException as e:
        rospy.logfatal(f'[spawn_platform] Spawn service exception: {e}')
        sys.exit(1)

    if resp.success:
        rospy.loginfo(f'[spawn_platform] "{MODEL_NAME}" spawned successfully.')
    else:
        rospy.logfatal(f'[spawn_platform] Spawn failed: {resp.status_message}')
        sys.exit(1)


if __name__ == '__main__':
    main()
