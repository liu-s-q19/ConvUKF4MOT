import numpy as np
from filterpy.kalman import KalmanFilter
from numpy import eye, dot, isscalar, zeros
from scipy.linalg import sqrtm


class Filter(object):
    def __init__(self, bbox3D, info, ID):

        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1  # number of total hits including the first detection
        self.info = info  # other information associated


def STCR(x, P, v):
    nx = x.shape[0]
    nPts = 2 * nx
    CPtArray = np.sqrt(nPts / 2) * np.hstack((np.eye(nx), -np.eye(nx)))
    S = np.linalg.cholesky(v / (v - 2) * P).T
    x = x[:, np.newaxis]
    # print("np.repeat(x, nPts, axis=1)", np.repeat(x, nPts, axis=1).shape)
    # print("S @ CPtArray", S.shape, CPtArray.shape, (S @ CPtArray).shape)
    X = np.repeat(x, nPts, axis=1) + S @ CPtArray
    return X


def fx(states, dt=0.01):
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


def fx1(states, dt=0.01):
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


class ImprovedCubatureKF:
    def __init__(self, dim_x, dim_z, f, h, dt=0.01, v1=5, v2=5, v3=5):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_y = self.dim_z
        self.f = f
        self.h = h
        self.dt = dt
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.V = self.R

    def predict(self, dt=None, UT=None, fx=None, **fx_args):

        v1, v2, v3 = self.v1, self.v2, self.v3
        P = (v3 - 2) / v3 * self.P
        nx = self.dim_x
        nz = self.dim_z
        nPts = 2 * nx
        xkk = self.x
        Xk1k1 = STCR(xkk, P, v3)
        Xkk1 = np.array([self.f(Xk1k1[:, i]) for i in range(nPts)]).T
        xkk1 = np.sum(Xkk1, axis=1) / nPts
        self.x = xkk1
        self.P = (v3 - 2) / v3 * (Xkk1 @ Xkk1.T / nPts - np.outer(xkk1, xkk1)) + (
            (v3 - 2) / v3
        ) * (v1 / (v1 - 2)) * self.Q

    def _rho_function(self, x, r=1):
        for i in range(len(x)):
            if abs(x[i]) < r:
                x[i] = 0.5 * x[i] ** 2
            else:
                x[i] = r * abs(x[i]) - 0.5 * r * r
        return x

    def _phi_function(self, x, r=1):
        for i in range(len(x)):
            if abs(x[i]) < r:
                # x[i] = 1
                x[i] = 1
            else:
                x[i] = r * np.sign(x[i]) / x[i]
        return x

    def update(
        self,
        y,
        x=None,
        N_iteration=10,
        eta=0.01,
        sigma=0.85,
        be=0.95,
        N_PR=0,
        **hx_args
    ):
        v1, v2, v3 = self.v1, self.v2, self.v3
        nPts = 2 * self.dim_x
        xkk1, Pkk1 = self.x, self.P
        nz = self.dim_z
        z = y
        Xi = STCR(xkk1, Pkk1, v3)

        Zi = np.array([self.h(Xi[:, i]) for i in range(nPts)]).T
        zkk1 = np.sum(Zi, axis=1) / nPts

        Pzzkk1 = (v3 - 2) / v3 * (Zi @ Zi.T / nPts - np.outer(zkk1, zkk1)) + (
            (v3 - 2) / v3
        ) * (v2 / (v2 - 2)) * self.R

        Pxzkk1 = (v3 - 2) / v3 * (Xi @ Zi.T / nPts - np.outer(xkk1, zkk1))

        Wk = Pxzkk1 @ np.linalg.inv(Pzzkk1)

        xkk = xkk1 + Wk @ (z - zkk1)

        deta2 = (z - zkk1).T @ np.linalg.inv(Pzzkk1) @ (z - zkk1)

        Skk = (v3 + deta2) / (v3 + nz) * (Pkk1 - Wk @ Pzzkk1 @ Wk.T)

        vvv = v3 + nz

        Skk = vvv / (vvv - 2) * Skk

        R_polar_initial = self.R
        Q = self.Q
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
        X10 = self.x
        Z_polar = y
        L = np.eye(self.dim_x)
        nx = self.dim_x
        fai_robust = np.eye(self.dim_z)
        P10 = self.P
        V = self.V
        XX = np.full((self.dim_x, N_iteration + 1), 1e6)
        while N_PR < N_iteration:
            R_sqrt = sqrtm(R_polar_initial)
            fai_inv = np.linalg.inv(fai_robust)
            R_polar = R_sqrt @ fai_inv
            R_polar = R_polar @ R_sqrt.T

            P10 = np.dot(np.dot(L, (P10 - Q)), L.T) + Q

            # Measurement update
            K = np.dot(
                np.dot(P10, H.T), np.linalg.inv(np.dot(np.dot(H, P10), H.T) + R_polar)
            )
            X = X10 + np.dot(K, (Z_polar - zkk1))

            e0 = np.dot(np.linalg.inv(R_polar_initial) ** 0.5, (Z_polar - np.dot(H, X)))
            fai_robust = np.diag(self._phi_function(e0))

            # P_k|k-1 update
            V1 = np.dot((Z_polar - zkk1), (Z_polar - self.h(X)).T)
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
        I_KH = np.eye(self.dim_x) - self.K @ H
        self.P = (I_KH @ self.P @ I_KH.T) + (self.K @ self.R @ self.K.T)


class ICKF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)

        # point = MerweScaledSigmaPoints(10, alpha=0.1, beta=2.0, kappa=1.0)
        self.kf = ImprovedCubatureKF(dim_x=11, dim_z=7, dt=0.1, f=fx1, h=hx)

        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        # self.kf.R[0:,0:] *= 10.

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000.0
        self.kf.P *= 10.0

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01

        self.kf.Q = self.kf.Q * 1e-2
        self.kf.R = np.eye(self.kf.dim_z) * 10

        # initialize data
        self.kf.x[:7] = self.initial_pos.reshape((7,))

    # def predict(self, dt=None, UT=None, fx=None, **fx_args):
    #     super(HuberUKF, self).predict()
    #     self.x += self.ex
    #     self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)
    #     self.x_prior = self.x.copy()

    # def _rho_function(self, x, r=1):
    #     for i in range(len(x)):
    #         if abs(x[i]) < r:
    #             x[i] = 0.5 * x[i] ** 2
    #         else:
    #             x[i] = r * abs(x[i]) - 0.5 * r * r
    #     return x

    # def _phi_function(self, x, r=1):
    #     for i in range(len(x)):
    #         if abs(x[i]) < r:
    #             # x[i] = 1
    #             if self.jax_flag:
    #                 x = x.at[i].set(1)
    #             else:
    #                 x[i] = 1
    #         else:
    #             x[i] = r * np.sign(x[i]) / x[i]
    #     return x

    # def update(self, y, max_iter=1, epson=1e-2, **hx_args):
    #     UT = unscented_transform

    #     sigmas_h = []
    #     for s in self.sigmas_f:
    #         sigmas_h.append(self.hx(s, **hx_args))
    #     self.sigmas_h = np.atleast_2d(sigmas_h)
    #     hx, self.S = UT(self.sigmas_h, self.Wm, self.Wc, self.R, self.z_mean, self.residual_z)
    #     Pxz = self.cross_variance(self.x, hx, self.sigmas_f, self.sigmas_h)
    #     H = (np.linalg.inv(self.P) @ Pxz).T

    #     n, m = self.dim_y, self.dim_x
    #     zero_n_m = np.zeros((n, m))
    #     zero_m_n = np.zeros((m, n))
    #     SI = np.block([[self.R, zero_n_m],
    #                 [zero_m_n, self.P]])
    #     SI_inv_sqrt = np.linalg.inv(np.real(sqrtm(SI)))
    #     M = SI_inv_sqrt @ np.block([[H], [np.eye(self.dim_x)]])
    #     P_sqrt = SI_inv_sqrt[self.R.shape[0]:, self.R.shape[0]:]
    #     R_sqrt = SI_inv_sqrt[:self.R.shape[0], :self.R.shape[0]]

    #     x = self.x.copy()
    #     x_ = 100 * np.ones(self.dim_x)

    #     for i in range(max_iter):
    #         if(np.sum(abs(x_ - x)) < epson):
    #             break
    #         x_ = x
    #         z = SI_inv_sqrt @ np.block([y, x])

    #         e = z - M @ x
    #         phi = self._phi_function(e)
    #         Phi = np.diag(phi)

    #         # P_ = P_sqrt @ np.linalg.inv(Phi[self.R.shape[0]:, self.R.shape[0]:]) @ P_sqrt
    #         R_ = R_sqrt @ np.linalg.inv(Phi[:self.R.shape[0], :self.R.shape[0]]) @ R_sqrt
    #         hx, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R_, self.z_mean, self.residual_z)
    #         Pxz = self.cross_variance(x, hx, self.sigmas_f, self.sigmas_h)

    #         self.K = dot(Pxz, np.linalg.inv(self.S))        # Kalman gain
    #         x = x + self.K @ (y - hx)

    #     # update Gaussian state estimate (x, P)
    #     self.x = self.x + self.K @ (y - hx)
    #     self.P = self.P - dot(self.K, dot(self.S, self.K.T))

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
