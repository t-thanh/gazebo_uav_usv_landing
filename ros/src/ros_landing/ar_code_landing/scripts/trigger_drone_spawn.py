#!/usr/bin/env python3
"""
trigger_drone_spawn.py
──────────────────────
Calls /mrs_drone_spawner/spawn and retries until it succeeds.

The plain `rosservice call --wait` used in the launch fires as soon as the
service is advertised, but mrs_drone_spawner checks /gazebo/model_states
internally and returns failure if Gazebo is not yet fully up.  This script
waits for Gazebo's model_states topic to arrive, then calls the spawn service,
retrying on any failure until the DroneSpawner accepts and queues the launch.

Parameters (ROS ~param)
  spawn_args   str   CLI args for mrs_drone_spawner/spawn
                     e.g. "1 x500 --enable-rangefinder --enable-ground-truth --pos 5 0 2 0"
  max_attempts int   Give up after this many attempts              (default 20)
"""

import sys
import rospy
from mrs_msgs.srv import String as MrsString, StringRequest


def main():
    rospy.init_node('trigger_drone_spawn', anonymous=True)

    spawn_args   = rospy.get_param('~spawn_args',
                                   '1 x500 --enable-rangefinder '
                                   '--enable-ground-truth --pos 5 0 2 0')
    max_attempts = int(rospy.get_param('~max_attempts', 20))

    rospy.loginfo(f'[trigger_drone_spawn] Waiting for Gazebo model_states …')
    try:
        rospy.wait_for_message('/gazebo/model_states',
                               __import__('gazebo_msgs.msg',
                                          fromlist=['ModelStates']).ModelStates,
                               timeout=60.0)
    except rospy.ROSException:
        rospy.logfatal('[trigger_drone_spawn] /gazebo/model_states never arrived. '
                       'Is Gazebo running?')
        sys.exit(1)

    rospy.loginfo('[trigger_drone_spawn] Gazebo ready. Waiting for spawn service …')

    for attempt in range(1, max_attempts + 1):
        if rospy.is_shutdown():
            return
        try:
            rospy.wait_for_service('/mrs_drone_spawner/spawn', timeout=15.0)
        except rospy.ROSException:
            rospy.logwarn(f'[trigger_drone_spawn] /mrs_drone_spawner/spawn not available '
                          f'(attempt {attempt}), retrying …')
            rospy.sleep(2.0)
            continue

        try:
            spawn_srv = rospy.ServiceProxy('/mrs_drone_spawner/spawn', MrsString)
            resp = spawn_srv(StringRequest(value=spawn_args))
        except rospy.ServiceException as exc:
            rospy.logwarn(f'[trigger_drone_spawn] Service exception (attempt {attempt}): '
                          f'{exc} — retrying …')
            rospy.sleep(2.0)
            continue

        if resp.success:
            rospy.loginfo(f'[trigger_drone_spawn] Spawn queued: {resp.message}')
            return

        rospy.logwarn(f'[trigger_drone_spawn] Spawn rejected (attempt {attempt}): '
                      f'{resp.message} — retrying …')
        rospy.sleep(3.0)

    rospy.logfatal(f'[trigger_drone_spawn] Drone spawn failed after {max_attempts} attempts.')
    sys.exit(1)


if __name__ == '__main__':
    main()
