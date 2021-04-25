import numpy as np
from utils.quaternion import Quaternion
from Kinematic.py.KinematicCore import ts_to_quat, quat_to_ts

def warp(zero, one, x, pow=2):
    dist = one - zero
    ratio = (x - zero) / dist
    return -np.power(ratio, pow)

class Limit:
    def __init__(self, low, high, preferred_low, preferred_high):
        self.low = low
        self.high = high
        self.preferred_low = preferred_low
        self.preferred_high = preferred_high

class AngleLimit:
    def __init__(self, limit, pow=2):
        self.limit = limit
        self.pow = pow
    
    def compute_rwd(self, angle, coeff):
        if angle < self.limit.low:
            r = -np.exp(coeff * (self.limit.low - angle))
        elif angle > self.limit.high:
            r = -np.exp(coeff * (angle - self.limit.high))
        elif angle >= self.limit.preferred_low and angle <= self.limit.preferred_high:
            r = 0
        elif angle < self.limit.preferred_low:
            r = warp(self.limit.preferred_low, self.limit.low, angle, self.pow)
        elif angle > self.limit.preferred_high:
            r = warp(self.limit.preferred_high, self.limit.high, angle, self.pow)

        return r
    
    def adjust_target(self, current, target):
        if current < self.limit.low and target < self.limit.low:
            return self.limit.low
        if current > self.limit.high and target > self.limit.high:
            return self.limit.high
        return target

class SphericalLimit:
    def __init__(self, swingxlimit, swingzlimit, twistlimit, pow=2):
        self.sxlimit = AngleLimit(swingxlimit)
        self.szlimit = AngleLimit(swingzlimit)
        self.tlimit = AngleLimit(twistlimit, pow=pow)
    
    # [w,x,y,z]
    def compute_rwd(self, quat, coeff):
        ts = quat_to_ts(quat)

        rwdx = self.sxlimit.compute_rwd(ts[0], coeff)
        rwdy = self.tlimit.compute_rwd(ts[1], coeff)
        rwdz = self.szlimit.compute_rwd(ts[2], coeff)

        return [rwdx, rwdy, rwdz]

    def adjust_target(self, current_quat, target_quat, remove_y=False):
        ts_current = quat_to_ts(current_quat)
        ts_target = quat_to_ts(target_quat)
        x = self.sxlimit.adjust_target(ts_current[0], ts_target[0])
        y = self.tlimit.adjust_target(ts_current[1], ts_target[1])
        z = self.szlimit.adjust_target(ts_current[2], ts_target[2])
        if remove_y:
            y = 0
        return ts_to_quat([x, y, z])

def test():
    lim = SphericalLimit(Limit(-1, 1, -0.4, 0.4), Limit(-1, 1, -0.4, 0.4), Limit(-1, 1, -0.4, 0.4))
    q = [0.1, 0.1, 0.1, 0.1]
    print(lim.compute_rwd(q))

    lim = AngleLimit(Limit(-0.5, 0, -0.3, 0))
    print(lim.compute_rwd(-0.4))
    print(lim.compute_rwd(0))

    ts = [0.5, -0.7, 0.9]
    q = ts_to_quat(ts)
    print(q)
    ts = quat_to_ts(q)
    print(ts)
