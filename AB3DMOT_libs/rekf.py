import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.linalg import sqrtm


class Filter(object):
    def __init__(self, bbox3D, info, ID):

        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1  # number of total hits including the first detection
        self.info = info  # other information associated


def fx(states, dt):
    px, py, pz, phi, l, w, h, v, dz, a, omega = states
    _omega = omega
    _phi = phi + omega * dt
    _a = a
    _dz = dz
    _v = v + a * dt
    _px = px + (v * dt + 0.5 * a * dt * dt) * np.cos(phi)
    _py = py + (v * dt + 0.5 * a * dt * dt) * np.sin(phi)
    _pz = pz + dt * dz
    _l = l
    _w = w
    _h = h
    _states = np.array([_px, _py, _pz, _phi, _l, _w, _h, _v, _dz, _a, _omega])
    return _states


def fx1(states, dt):
    px, py, pz, phi, l, w, h, v, dz, a, omega = states
    if abs(omega) <= 1e4:
        _omega = omega
        _phi = phi + omega * dt
        _a = a
        _dz = dz
        _v = v + a * dt
        _px = px + (v * dt + 0.5 * a * dt * dt) * np.cos(phi)
        _py = py + (v * dt + 0.5 * a * dt * dt) * np.sin(phi)
        _pz = pz + dt * dz
        _l = l
        _w = w
        _h = h
        _states = np.array([_px, _py, _pz, _phi, _l, _w, _h, _v, _dz, _a, _omega])
        return _states
    else:
        _omega = omega
        _phi = phi + _omega * dt
        _a = a
        _dz = dz
        _v = v + a * dt
        delt_ax = (
            a
            * (np.cos(_phi) - np.cos(phi) + omega * dt * np.sin(_phi))
            / (omega * omega)
        )
        delt_ay = (
            a
            * (np.sin(_phi) - np.sin(phi) - omega * dt * np.cos(_phi))
            / (omega * omega)
        )
        _px = px + v * (np.sin(_phi) - np.sin(phi)) / omega + delt_ax
        _py = py + v * (-np.cos(_phi) - np.cos(phi)) / omega + delt_ay
        _pz = pz + dt * dz
        _l = l
        _w = w
        _h = h
        _states = np.array([_px, _py, _pz, _phi, _l, _w, _h, _v, _dz, _a, _omega])
        return _states


def hx(states):
    H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    return np.array(H @ states)


def jac_hx(states):
    H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    return H


def jac_fx(states, dt):
    px, py, pz, phi, l, w, h, v, dz, a, omega = states

    # 初始化雅可比矩阵为零矩阵
    J = np.zeros((11, 11))

    # 填充雅可比矩阵
    # _px 和 _py 关于 phi 的偏导数
    J[0, 3] = -(v * dt + 0.5 * a * dt**2) * np.sin(phi)  # _px 关于 phi
    J[1, 3] = (v * dt + 0.5 * a * dt**2) * np.cos(phi)  # _py 关于 phi

    # _px 和 _py 关于 v 的偏导数
    J[0, 7] = dt * np.cos(phi)  # _px 关于 v
    J[1, 7] = dt * np.sin(phi)  # _py 关于 v

    # _px 和 _py 关于 a 的偏导数
    J[0, 9] = 0.5 * dt**2 * np.cos(phi)  # _px 关于 a
    J[1, 9] = 0.5 * dt**2 * np.sin(phi)  # _py 关于 a

    # _phi 关于 omega 的偏导数
    J[3, 10] = dt  # _phi 关于 omega

    # _pz 关于 dz 的偏导数
    J[2, 8] = dt  # _pz 关于 dz

    # 对自己的偏导数
    J[np.diag_indices_from(J)] = 1

    return J


class RobustEKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, f_func, h_func, f_jac, h_jac, dt):
        super().__init__(
            dim_x=dim_x,
            dim_z=dim_z,
        )

        self.dim_y = self.dim_z
        self.f = f_func
        self.h = h_func
        self.jac_f = f_jac
        self.jac_h = h_jac
        self.dt = dt
        self.x = np.zeros((dim_x,))  # state
        self.V = self.R

    def predict(self, u=0, **fx_args):
        F = self.jac_f(self.x, self.dt)
        self.x = self.f(self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q
        self.x_prior = self.x.copy()

    def update(
        self, y, N_iteration=10, eta=0.01, sigma=0.85, be=0.95, N_PR=0, **hx_args
    ):
        R_polar_initial = self.R
        Q = self.Q
        H = self.jac_h(self.x)
        X10 = self.x
        Z_polar = y
        L = np.eye(self.dim_x)
        nx = self.dim_x
        fai_robust = 1
        P10 = self.P
        V = self.V
        XX = np.full((self.dim_x, N_iteration + 1), 1e6)
        while N_PR < N_iteration:
            R_sqrt = sqrtm(R_polar_initial)
            fai_inv = np.linalg.inv(fai_robust * np.eye(self.dim_z))
            R_polar = R_sqrt @ fai_inv
            R_polar = R_polar @ R_sqrt.T

            P10 = np.dot(np.dot(L, (P10 - Q)), L.T) + Q

            # Measurement update
            K = np.dot(
                np.dot(P10, H.T), np.linalg.inv(np.dot(np.dot(H, P10), H.T) + R_polar)
            )
            X = X10 + np.dot(K, (Z_polar - np.dot(H, X10)))

            e0 = np.dot(np.linalg.inv(R_polar_initial) ** 0.5, (Z_polar - np.dot(H, X)))
            fai_robust = (
                np.exp(-(e0**2) / (2 * sigma**2)) / np.sqrt(2 * np.pi) / sigma**3
            )

            # P_k|k-1 update
            V1 = np.dot((Z_polar - np.dot(H, X)), (Z_polar - np.dot(H, X)).T)
            V = (1 - be) * V + be * V1
            N_P = V - eta * R_polar - np.dot(np.dot(H, Q), H.T)
            G = np.dot(np.dot(np.linalg.pinv(H), N_P), np.linalg.pinv(H).T)
            J = P10 - Q

            for ix in range(nx):
                m = np.sqrt(G[ix, ix] / J[ix, ix])
                if m > 1:
                    L[ix, ix] = m
                else:
                    L[ix, ix] = 1

            XX[:, N_PR + 1] = X
            if (
                np.sqrt(
                    np.linalg.norm(XX[:, N_PR + 1] - XX[:, N_PR])
                    / np.linalg.norm(XX[:, N_PR])
                )
                <= 1e-4
            ):
                break

            N_PR += 1
        self.V = V
        self.K = K
        self.x = X
        I_KH = self._I - self.K @ H
        self.P = (I_KH @ self.P @ I_KH.T) + (self.K @ self.R @ self.K.T)


class REKF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)

        self.kf = RobustEKF(
            dim_x=11, dim_z=7, f_func=fx1, h_func=hx, f_jac=jac_fx, h_jac=jac_hx, dt=0.1
        )

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000.0
        self.kf.P *= 10.0

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01

        self.kf.Q = self.kf.Q * 0.3
        self.kf.R = np.eye(self.kf.dim_z) * 10

        # initialize data
        self.kf.x[:7] = self.initial_pos.reshape((7,))

    def compute_innovation_matrix(self):
        """compute the innovation matrix for association with mahalanobis distance"""
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

    def get_velocity(self):
        # return the object velocity in the state
        v = self.kf.x[7]
        dz = self.kf.x[8]
        phi = self.kf.x[3]
        dx = v * np.cos(phi)
        dy = v * np.sin(phi)
        vel = np.array([dx, dy, dz])
        return vel
        # return self.kf.x[7:10]
