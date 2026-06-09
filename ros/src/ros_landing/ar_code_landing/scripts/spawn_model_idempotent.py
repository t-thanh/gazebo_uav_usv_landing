#!/usr/bin/env python3
"""
spawn_model_idempotent.py
─────────────────────────
Generic idempotent Gazebo model spawner.

Handles all three failure scenarios that occur on repeated roslaunch:

  A) Old Gazebo alive  → model already exists  → delete first, then spawn.
  B) Old Gazebo dies during spawn (kill_previous_session) → service exception
     → wait for new Gazebo → spawn fresh (nothing to delete).
  C) New Gazebo not ready yet → service timeout → retry until available.

Parameters (all via ROS ~param)
  model_name   str    Gazebo model name                              (required)
  model_type   str    "sdf" | "urdf"                                 (required)
  sdf_path     str    Absolute path to .sdf file     (required if model_type=sdf)
  urdf_param   str    ROS parameter holding URDF XML (required if model_type=urdf)
  spawn_x/y/z  float  World spawn position                           (default 0)
  spawn_roll/pitch/yaw  float  Spawn orientation                     (default 0)
  max_attempts int    How many delete+spawn cycles before giving up   (default 10)
"""

import sys
import rospy
from gazebo_msgs.srv import (DeleteModel,  DeleteModelRequest,
                              SpawnModel,   SpawnModelRequest)
from geometry_msgs.msg import Pose, Point, Quaternion
import tf.transformations as tft


def _make_pose(x, y, z, roll, pitch, yaw):
    q = tft.quaternion_from_euler(roll, pitch, yaw)
    pose = Pose()
    pose.position    = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return pose


def _get_xml(model_type, sdf_path, urdf_param):
    if model_type == 'sdf':
        import os
        if not os.path.isfile(sdf_path):
            rospy.logfatal(f'[spawn_idempotent] sdf_path not found: {sdf_path!r}')
            sys.exit(1)
        with open(sdf_path) as f:
            return f.read()
    else:  # urdf
        xml = rospy.get_param(urdf_param, None)
        if xml is None:
            rospy.logfatal(f'[spawn_idempotent] ROS param {urdf_param!r} not set')
            sys.exit(1)
        return xml


def main():
    rospy.init_node('spawn_model_idempotent', anonymous=True)

    model_name   = rospy.get_param('~model_name')
    model_type   = rospy.get_param('~model_type', 'sdf')       # "sdf" | "urdf"
    sdf_path     = rospy.get_param('~sdf_path',   '')
    urdf_param   = rospy.get_param('~urdf_param', '/robot_description')
    spawn_x      = float(rospy.get_param('~spawn_x',     0.0))
    spawn_y      = float(rospy.get_param('~spawn_y',     0.0))
    spawn_z      = float(rospy.get_param('~spawn_z',     0.0))
    spawn_roll   = float(rospy.get_param('~spawn_roll',  0.0))
    spawn_pitch  = float(rospy.get_param('~spawn_pitch', 0.0))
    spawn_yaw    = float(rospy.get_param('~spawn_yaw',   0.0))
    max_attempts = int(  rospy.get_param('~max_attempts', 10))

    spawn_service = ('/gazebo/spawn_urdf_model' if model_type == 'urdf'
                     else '/gazebo/spawn_sdf_model')
    pose = _make_pose(spawn_x, spawn_y, spawn_z, spawn_roll, spawn_pitch, spawn_yaw)

    rospy.loginfo(f'[spawn_idempotent] "{model_name}" ({model_type}) '
                  f'→ ({spawn_x},{spawn_y},{spawn_z})')

    for attempt in range(1, max_attempts + 1):
        if rospy.is_shutdown():
            return

        # ── Wait for Gazebo to be available ──────────────────────────────────
        try:
            rospy.wait_for_service('/gazebo/delete_model', timeout=15.0)
            rospy.wait_for_service(spawn_service,          timeout=15.0)
        except rospy.ROSException:
            rospy.logwarn(f'[spawn_idempotent] Gazebo not ready (attempt {attempt}), retrying …')
            rospy.sleep(2.0)
            continue

        delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        spawn_srv  = rospy.ServiceProxy(spawn_service,          SpawnModel)

        try:
            # ── Delete if present (idempotent) ────────────────────────────────
            del_resp = delete_srv(DeleteModelRequest(model_name=model_name))
            if del_resp.success:
                rospy.loginfo(f'[spawn_idempotent] Deleted existing "{model_name}".')

            # ── Fetch model XML (after Gazebo is ready so params are set) ─────
            xml = _get_xml(model_type, sdf_path, urdf_param)

            # ── Spawn ─────────────────────────────────────────────────────────
            req = SpawnModelRequest()
            req.model_name      = model_name
            req.model_xml       = xml
            req.robot_namespace = ''
            req.initial_pose    = pose
            req.reference_frame = 'world'

            sp_resp = spawn_srv(req)

        except rospy.ServiceException as exc:
            rospy.logwarn(f'[spawn_idempotent] Service exception (attempt {attempt}): {exc} — retrying …')
            rospy.sleep(2.0)
            continue

        if sp_resp.success:
            rospy.loginfo(f'[spawn_idempotent] "{model_name}" spawned successfully.')
            return

        rospy.logwarn(f'[spawn_idempotent] Spawn failed (attempt {attempt}): '
                      f'{sp_resp.status_message} — retrying …')
        rospy.sleep(2.0)

    rospy.logfatal(f'[spawn_idempotent] "{model_name}" failed after {max_attempts} attempts.')
    sys.exit(1)


if __name__ == '__main__':
    main()
