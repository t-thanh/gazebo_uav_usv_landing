#!/usr/bin/env python3
"""tmp/px4_open_arming.py — open the PX4 (SITL, v1.13.2) arming + takeoff gate for the
heaving-deck takeoff (Mode B).  Sets a bundle of commander params at RUNTIME via mavros
(local-only, no firmware rebuild, no repo/Docker change) and exits.  Run AFTER mavros is
up and BEFORE the flight controller arms.

Why each one (root cause from the climb test on large waves):
  COM_RC_IN_MODE = 4   stick input disabled → kills "Arming denied! manual control lost"
  COM_RCL_EXCEPT = 7   exempt mission/hold/offboard from RC-loss failsafe
  NAV_RCL_ACT    = 0   RC-loss action = none (don't disarm/RTL on the deck)
  NAV_DLL_ACT    = 0   datalink-loss action = none
  COM_ARM_WO_GPS = 1   GPS-free arming
  COM_DISARM_PRFLT = -1  never auto-disarm for "no takeoff yet" (deck pins the marginal climb)
  COM_DISARM_LAND  = -1  never auto-disarm on "landing detected" — the land detector false-
                         fires under deck heave and was disarming the marginal takeoff before
                         the open-loop ramp could break free.  THE linchpin for Mode B.
  COM_ARM_IMU_ACC = 1.0  relax accel-consistency arming check (deck heave accel)
  COM_ARM_IMU_GYR = 0.5  relax gyro-consistency arming check
  COM_ARM_MAG_STR = 0    disable mag-strength arming check (irrelevant in SITL)
"""
import sys, rospy
from mavros_msgs.srv import ParamSet, ParamSetRequest, ParamGet, ParamGetRequest

NS = rospy.get_namespace().strip('/') or 'uav1'

# (name, value, is_integer)
PARAMS = [
    ('COM_RC_IN_MODE',  4,    True),
    ('COM_RCL_EXCEPT',  7,    True),
    ('NAV_RCL_ACT',     0,    True),
    ('NAV_DLL_ACT',     0,    True),
    ('COM_ARM_WO_GPS',  1,    True),
    ('COM_DISARM_PRFLT', -1.0, False),
    ('COM_DISARM_LAND',  -1.0, False),
    ('COM_ARM_IMU_ACC',  0.9, False),   # just under metadata max 1.0 (boundary is rejected)
    ('COM_ARM_IMU_GYR',  0.45, False),  # just under metadata max 0.5
    ('COM_ARM_MAG_STR',  0,   True),
]


def main():
    rospy.init_node('px4_open_arming', anonymous=True)
    ns = rospy.get_param('~ns', NS)
    set_srv = '/%s/mavros/param/set' % ns
    get_srv = '/%s/mavros/param/get' % ns
    rospy.loginfo('[open_arming] waiting for %s ...', set_srv)
    rospy.wait_for_service(set_srv, timeout=120)
    rospy.wait_for_service(get_srv, timeout=120)
    setp = rospy.ServiceProxy(set_srv, ParamSet)
    getp = rospy.ServiceProxy(get_srv, ParamGet)

    # mavros pulls params lazily — poll one known param until the table is ready
    for _ in range(60):
        try:
            if getp(ParamGetRequest(param_id='COM_RC_IN_MODE')).success:
                break
        except Exception:
            pass
        rospy.sleep(1.0)

    ok_all = True
    for name, val, is_int in PARAMS:
        done = False
        for attempt in range(8):
            try:
                req = ParamSetRequest(param_id=name)
                if is_int:
                    req.value.integer = int(val); req.value.real = 0.0
                else:
                    req.value.integer = 0; req.value.real = float(val)
                if setp(req).success:
                    rospy.loginfo('[open_arming] set %s = %s', name, val)
                    done = True
                    break
            except Exception as e:
                rospy.logwarn('[open_arming] %s set error: %s', name, e)
            rospy.sleep(0.8)
        if not done:
            ok_all = False
            rospy.logerr('[open_arming] FAILED to set %s', name)
    rospy.loginfo('[open_arming] done (all_ok=%s)', ok_all)
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
