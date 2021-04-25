import numpy as np
from numba import float32
from numba import jit
from numba.experimental import jitclass

@jit(nopython=True)
def cross(x, y):
    """ Cross product
    """
    return np.array([x[1]*y[2] - x[2]*y[1], x[2]*y[0] - x[0]*y[2], x[0]*y[1] - x[1]*y[0]])

@jit(nopython=True)
def dot3(x, y):
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]

spec = [
        ('w', float32),
        ('v', float32[:]),
                ]

@jitclass(spec)
class Quaternion:
    def fromEulerZYX(angles):
        r1 = Quaternion.fromAngleAxis(angles[2], np.array([0,0,1]))
        r2 = Quaternion.fromAngleAxis(angles[1], np.array([0,1,0]))
        r3 = Quaternion.fromAngleAxis(angles[0], np.array([1,0,0]))
        return r1.mul(r2.mul(r3))

    def fromAngleAxis(angle, axis):
        w = np.cos(angle/2)
        if w == 1:
            return Quaternion(1, 0, 0, 0)
        v = axis / np.linalg.norm(axis)
        v *= np.sin(angle/2)
        return Quaternion(w, v[0], v[1], v[2])

    def fromWXYZ(q):
        return Quaternion(q[0], q[1], q[2], q[3])

    def fromXYZW(q):
        return Quaternion(q[3], q[0], q[1], q[2])

    def fromExpMap(exp):
        angle = np.linalg.norm(exp)
        return Quaternion.fromAngleAxis(angle, exp)

    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.v = np.array([x, y, z], dtype=np.float32)
        self.normlize()

    def mul(self, other):
        """
        return self \mul other
        """
        w_new = self.w * other.w - dot3(self.v, other.v)
        v_new = self.w * other.v + other.w * self.v + cross(self.v, other.v)

        return Quaternion(w_new, v_new[0], v_new[1], v_new[2])

    def normlize(self):
        l = np.sqrt(self.w*self.w + dot3(self.v, self.v))
        self.w = self.w / l
        self.v = self.v / l

        if self.w < 0:
            self.w *= -1
            self.v *= -1

    def rotate(self, r):
        """
            rotate a 3 dim vector
        """
        t = 2 * cross(self.v, r)
        r_new = r + self.w * t + cross(self.v, t)
        return r_new

    def conjugate(self):
        return Quaternion(self.w, -self.v[0], -self.v[1], -self.v[2])

    def wxyz(self):
        return np.array([self.w, self.v[0], self.v[1], self.v[2]])

    def pos_wxyz(self):
        out = self.wxyz()
        if out[0] < 0:
            out *= -1
        return out

    def xyzw(self):
        return np.array([self.v[0], self.v[1], self.v[2], self.w])

    def pos_xyzw(self):
        out = self.xyzw()
        if out[3] < 0:
            out *= -1
        return out

    def angle(self):
        l = np.linalg.norm(self.v)
        return 2 * np.arctan2(l, self.w)

    def axis(self):
        norm = np.linalg.norm(self.v)
        if norm < 1e-4:
            return self.v
        n = self.v / norm
        return n

    def angaxis(self):
        ang = self.angle()
        axis = self.axis()
        return np.array([ang, axis[0], axis[1], axis[2]])

    def expmap(self):
        ang = self.angle()
        axis = self.axis()
        omg = ang * axis
        return omg

def quatMul(q1, q2):
    return q1.mul(q2)

#@jit(nopython=True)
def computeAngDiffRel(q_s, q_e):
    """ Compute angluar difference between two rotations using quaternion slerp,
        represented in local frame of q_s
            q_s * q_diff = q_e

        Inputs:
            q_s   Quaternion, start quaternion
            q_e   Quaternion, end quaternion

        Outputs:
            angDiff     np.array of float, angular velocity vector in local frame of ornStart
    """
    q_diff = quatMul(q_s.conjugate(), q_e)
    angle = q_diff.angle()
    axis = q_diff.axis()
    angDiff = axis * angle

    return angDiff

#@jit(nopython=True)
def computeAngDiff(q_s, q_e):
    """ Compute angluar difference between two rotations using quaternion slerp,
        represented in world frame of q_s
            q_diff * q_s = q_e

        Inputs:
            q_s   Quaternion, start quaternion
            q_e   Quaternion, end quaternion

        Outputs:
            angDiff     np.array of float, angular velocity vector in local frame of ornStart
    """
    q_diff = q_e.mul(q_s.conjugate())
    angle = q_diff.angle()
    axis = q_diff.axis()
    angDiff = axis * angle

    return angDiff

#@jit(nopython=True)
def computeAngVel(q_s, q_e, dt):
    """ Compute angluar velocity between two rotations using quaternion slerp,
        represented in world frame of q_s
            q_diff * q_s = q_e

        Inputs:
            q_s   Quaternion, start quaternion
            q_e   Quaternion, end quaternion
            dt    float, delta t

        Outputs:
            angVel     np.array of float, angular velocity vector in local frame of ornStart
    """
    q_diff = quatMul(q_e, q_s.conjugate())
    omg = q_diff.angle() / dt
    axis = q_diff.axis()
    angVel = axis * omg

    return angVel

#@jit(nopython=True)
def computeAngVelRel(q_s, q_e, dt):
    """ Compute angluar velocity between two rotations using quaternion slerp,
        represented in local frame of q_s
            q_s * q_diff = q_e

        Inputs:
            q_s   Quaternion, start quaternion
            q_e   Quaternion, end quaternion
            dt    float, delta t

        Outputs:
            angVel     np.array of float, angular velocity vector in local frame of ornStart
    """
    q_diff = quatMul(q_s.conjugate(), q_e)
    omg = q_diff.angle() / dt
    axis = q_diff.axis()
    angVel = axis * omg

    return angVel

@jit(nopython=True)
def computeQuatSlerp(q0, q1, t):
    """ Compute spherical linear interpolation between quaternion q0 and q1

        Inputs:
            q0   np.array or list of float, quaternion (can be either wxyz, xyzw)
            q1   np.array or list of float, quaternion, same representation with q0
            t    float between [0, 1], interplation parameter

        Outputs:
            qt   np.array or list of float, quaternion, same representation with q1
                     slerp result of q0 and q1
    """
    dot = np.dot(q0, q1)

    if (dot < 0.0):
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        qt = q0 + t * (q1 - q0)
        qt = qt / np.linalg.norm(qt)
        return qt

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s1 = sin_theta / sin_theta_0
    s0 = np.cos(theta) - dot * s1
    qt = s0 * q0 + s1 * q1

    return qt

def assert_scalar_eq(x, y, threshold=1e-6):
    diff = abs(x-y)
    assert(diff < threshold), (x, y, diff)

def assert_vec_eq(x, y, threshold=1e-6):
    diff = abs(x-y).max()
    assert(diff < threshold), (x, y, diff)

if __name__=="__main__":
    TEST_NUM = 100

    # unit test Quaternion

    ## construction
    for i in range(TEST_NUM):
        data = np.random.rand(4)
        data /= np.linalg.norm(data)

        q0 = Quaternion.fromWXYZ(data)
        assert_scalar_eq(q0.w, data[0])
        assert_vec_eq(q0.v, data[1:])

        q1 = Quaternion.fromXYZW(data)
        assert_scalar_eq(q1.w, data[3])
        assert_vec_eq(q1.v, data[:-1])

        q2 = Quaternion.fromAngleAxis(data[0], data[1:])
        axis = data[1:].copy()
        axis /= np.linalg.norm(axis)
        assert_scalar_eq(q2.w, np.cos(data[0]/2))
        assert_vec_eq(q2.v, np.sin(data[0]/2) * axis)
    print("Quaternion::fromAngleAxis() pass")
    print("Quaternion::fromXYZW() pass")
    print("Quaternion::fromWXYZ() pass")

    for i in range(TEST_NUM):
        omega = np.random.rand(3)

        angle = np.linalg.norm(data)

        q1 = Quaternion.fromExpMap(data)
        q2 = Quaternion.fromAngleAxis(angle, data)
        assert_vec_eq(q1.wxyz(), q2.wxyz())
    print("Quaternion::fromExpMap() pass")

    ## test rotate
    for i in range(TEST_NUM):
        q = np.random.randn(4)
        r = np.random.randn(3)

        # calculate rotation
        t = q[0]
        n = q[1:]
        n = n / np.sqrt(np.dot(n, n))

        w = np.cos(t/2)
        v = n * np.sin(t/2)
        quat = Quaternion(w, v[0], v[1], v[2])

        # rotate
        r_n = n * np.dot(r, n)
        r_o = r - r_n
        r_rot = r_n + r_o * np.cos(t)  + np.cross(n, r) * np.sin(t)

        r_quat = quat.rotate(r)

        # compare
        assert_vec_eq(r_rot, r_quat)
    print("Quaternion::rotate() pass")

    ## test multiplication
    for i in range(TEST_NUM):
        q1 = np.random.randn(4)
        q2 = np.random.randn(4)
        r = np.random.randn(3)

        # calculate rotation
        quat1 = Quaternion(q1[0], q1[1], q1[2], q1[3])
        quat2 = Quaternion(q2[0], q2[1], q2[2], q2[3])

        quat3 = quat2.mul(quat1)

        # compare rotation results
        r_rot = quat2.rotate(quat1.rotate(r))
        r_3 = quat3.rotate(r)

        assert_vec_eq(r_rot, r_3)
    print("Quaternion:mul() pass")

    ## test magnitude
    for i in range(TEST_NUM):
        a = np.random.rand(4)
        q = Quaternion.fromAngleAxis(a[0], a[1:])
        assert_scalar_eq(a[0], q.angle(), 1e-4)
    print("Quaternion::angle() pass")

    # unittest ComputeAngDiffRel
    for i in range(TEST_NUM):
        q0 = Quaternion.fromWXYZ(np.random.rand(4))

        angle = abs(np.random.rand())
        axis  = np.random.rand(3)
        axis /= np.linalg.norm(axis)
        qr = Quaternion.fromAngleAxis(angle, axis)

        q1 = quatMul(q0, qr)

        angvel = computeAngDiffRel(q0, q1)
        angvel_new = angle * axis

        assert_vec_eq(angvel, angvel_new, 1e-4)
    print("computeAngDiffRel() Pass")

    # unittest ComputeAngDiff
    for i in range(TEST_NUM):
        q0 = Quaternion.fromWXYZ(np.random.rand(4))

        angle = abs(np.random.rand())
        axis  = np.random.rand(3)
        axis /= np.linalg.norm(axis)
        qr = Quaternion.fromAngleAxis(angle, axis)

        q1 = quatMul(qr, q0)

        angvel = computeAngDiff(q0, q1)
        angvel_new = angle * axis

        assert_vec_eq(angvel, angvel_new, 1e-4)
    print("computeAngDiff() Pass")

    # unittest ComputeAngVel
    for i in range(TEST_NUM):
        q0 = Quaternion.fromWXYZ(np.random.rand(4))

        angle = abs(np.random.rand())
        axis  = np.random.rand(3)
        axis /= np.linalg.norm(axis)
        qr = Quaternion.fromAngleAxis(angle, axis)

        q1 = quatMul(qr, q0)

        angvel = computeAngVel(q0, q1, 1.0)
        angvel_new = angle * axis

        assert_vec_eq(angvel, angvel_new, 1e-4)
    print("computeAngVel() Pass")

    # unittest getQuaternionSlerp
    for i in range(TEST_NUM):
        q0 = np.random.rand(4)
        q0 /= np.linalg.norm(q0)
        q0 = Quaternion.fromWXYZ(q0)

        angle = abs(np.random.rand())
        axis  = np.random.rand(3)
        axis /= np.linalg.norm(axis)
        qr = Quaternion.fromAngleAxis(angle, axis)

        q1 = quatMul(qr, q0)

        t = np.random.rand()
        qt = computeQuatSlerp(q0.wxyz(), q1.wxyz(), t)
        qrt = Quaternion.fromAngleAxis(angle * t, axis)
        qt_new = quatMul(qrt, q0).wxyz()

        assert_vec_eq(qt, qt_new)
    print("computeSlerp() Pass")

