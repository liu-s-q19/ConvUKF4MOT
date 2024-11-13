import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
# from AB3DMOT_libs.huber_ukf import HuberUKF
from filterpy.kalman import unscented_transform
from numpy import eye, dot, isscalar
from scipy.linalg import sqrtm

class Filter(object):
    def __init__(self, bbox3D, info, ID):

        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1           		# number of total hits including the first detection
        self.info = info        		# other information associated	

# def fx(states, dt):
#     F=np.array([[1,0,0,0,0,0,0,dt,0,0,0,0.5*dt*dt,0,0],      # state transition matrix, dim_x * dim_x
# 				[0,1,0,0,0,0,0,0,dt,0,0,0,0.5*dt*dt,0],
# 				[0,0,1,0,0,0,0,0,0,dt,0,0,0,0.5*dt*dt],
# 				[0,0,0,1,0,0,0,0,0,0,dt,0,0,0],  
# 				[0,0,0,0,1,0,0,0,0,0,0,0,0,0],
# 				[0,0,0,0,0,1,0,0,0,0,0,0,0,0],
# 				[0,0,0,0,0,0,1,0,0,0,0,0,0,0],
# 				[0,0,0,0,0,0,0,1,0,0,0,0,0,0],
# 				[0,0,0,0,0,0,0,0,1,0,0,0,0,0],
# 				[0,0,0,0,0,0,0,0,0,1,0,0,0,0],
# 				[0,0,0,0,0,0,0,0,0,0,1,0,0,0],
# 				[0,0,0,0,0,0,0,0,0,0,0,1,0,0],
# 				[0,0,0,0,0,0,0,0,0,0,0,0,1,0],
# 				[0,0,0,0,0,0,0,0,0,0,0,0,0,1]]) 
#     return np.array(F @ states) 

# def fx(states, dt):
#     F = np.array([[1,0,0,0,0,0,0,1,0,0],      
# 				[0,1,0,0,0,0,0,0,1,0],
# 				[0,0,1,0,0,0,0,0,0,1],
# 				[0,0,0,1,0,0,0,0,0,0],  
# 				[0,0,0,0,1,0,0,0,0,0],
# 				[0,0,0,0,0,1,0,0,0,0],
# 				[0,0,0,0,0,0,1,0,0,0],
# 				[0,0,0,0,0,0,0,1,0,0],
# 				[0,0,0,0,0,0,0,0,1,0],
# 				[0,0,0,0,0,0,0,0,0,1]])
#     return np.array(F @ states)
def fx(states, dt):
    px, py, pz, phi, l, w, h, v, dz, a, omega = states
    _omega = omega
    _phi = phi + omega * dt
    _a = a
    _dz = dz
    _v = v + a * dt
    _px = px + (v*dt+0.5*a*dt*dt)*np.cos(phi) 
    _py = py + (v*dt+0.5*a*dt*dt)*np.sin(phi)
    _pz = pz + dt * dz
    _l = l
    _w = w
    _h = h
    _states = np.array([_px, _py, _pz, _phi, _l, _w, _h, _v, _dz, _a, _omega])
    return _states

def fx1(states, dt):
    px, py, pz, phi, l, w, h, v, dz, a, omega = states
    if abs(omega)<=1e4:
        _omega = omega
        _phi = phi + omega * dt
        _a = a
        _dz = dz
        _v = v + a * dt
        _px = px + (v*dt+0.5*a*dt*dt)*np.cos(phi) 
        _py = py + (v*dt+0.5*a*dt*dt)*np.sin(phi)
        _pz = pz + dt * dz
        _l = l
        _w = w
        _h = h
        _states = np.array([_px, _py, _pz, _phi, _l, _w, _h, _v, _dz, _a, _omega])
        return _states
    else:
        _omega = omega
        _phi = phi +_omega * dt
        _a = a
        _dz = dz
        _v = v + a * dt
        delt_ax = a*(np.cos(_phi)-np.cos(phi)+omega*dt*np.sin(_phi))/(omega*omega)
        delt_ay = a*(np.sin(_phi)-np.sin(phi)-omega*dt*np.cos(_phi))/(omega*omega)
        _px = px + v * (np.sin(_phi) - np.sin(phi)) / omega + delt_ax
        _py = py + v * (-np.cos(_phi) - np.cos(phi)) / omega + delt_ay
        _pz = pz + dt * dz
        _l = l
        _w = w
        _h = h
        _states = np.array([_px, _py, _pz, _phi, _l, _w, _h, _v, _dz, _a, _omega])
        return _states 

def hx(states):
    H =   np.array([[1,0,0,0,0,0,0,0,0,0,0],      
                    [0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0]])
    return np.array(H @ states)

class HuberUnscentKF(UnscentedKalmanFilter):
    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        super(HuberUnscentKF, self).predict()
        self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)
        self.x_prior = self.x.copy()
        
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

    def update(self, y, x=None, max_iter=100, epson=1e-2, **hx_args):
        if x is not None:
            self.sigmas_f = self.points_fn.sigma_points(x, self.P)
        UT = unscented_transform
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(self.hx(s, **hx_args))
        self.sigmas_h = np.atleast_2d(sigmas_h)
        hx, self.S = UT(self.sigmas_h, self.Wm, self.Wc, self.R, self.z_mean, self.residual_z)
        Pxz = self.cross_variance(self.x, hx, self.sigmas_f, self.sigmas_h)
        H = (np.linalg.inv(self.P) @ Pxz).T
        
        n, m = self._dim_z, self._dim_x
        zero_n_m = np.zeros((n, m))
        zero_m_n = np.zeros((m, n))
        SI = np.block([[self.R, zero_n_m],
                    [zero_m_n, self.P]])
        SI_inv_sqrt = np.linalg.inv(np.real(sqrtm(SI)))
        M = SI_inv_sqrt @ np.block([[H], [np.eye(self._dim_x)]])
        P_sqrt = SI_inv_sqrt[self.R.shape[0]:, self.R.shape[0]:]
        R_sqrt = SI_inv_sqrt[:self.R.shape[0], :self.R.shape[0]]
        
                
        x = self.x.copy()
        x_ = 100 * np.ones(self._dim_x)
        
        
        for i in range(max_iter):
            if(np.sum(abs(x_ - x)) < epson):
                break
            x_ = x
            z = SI_inv_sqrt @ np.block([y, x]) 

            e = z - M @ x
            phi = self._phi_function(e)
            Phi = np.diag(phi)
            
            # P_ = P_sqrt @ np.linalg.inv(Phi[self.R.shape[0]:, self.R.shape[0]:]) @ P_sqrt
            R_ = R_sqrt @ np.linalg.inv(Phi[:self.R.shape[0], :self.R.shape[0]]) @ R_sqrt
            hx, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R_, self.z_mean, self.residual_z)
            Pxz = self.cross_variance(x, hx, self.sigmas_f, self.sigmas_h)
    
            self.K = dot(Pxz, np.linalg.inv(self.S))        # Kalman gain
            x = x + self.K @ (y - hx)
        
        # update Gaussian state estimate (x, P)
        self.x = self.x + self.K @ (y - hx)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))
        # print('huber')

class HuberUKF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)

        # point = MerweScaledSigmaPoints(10, alpha=0.1, beta=2.0, kappa=1.0)
        point = MerweScaledSigmaPoints(11, alpha=1e-3, beta=2, kappa=0)   
        # point =JulierSigmaPoints(10)  
        self.kf = HuberUnscentKF(dim_x=11, dim_z=7, dt=0.1, points=point, fx=fx1, hx=hx) 

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])

        # measurement uncertainty, uncomment if not super trust the measurement data due to detection noise
        # self.kf.R[0:,0:] *= 10.   

        # initial state uncertainty at time 0
        # Given a single data, the initial velocity is very uncertain, so giv a high uncertainty to start
        self.kf.P[7:, 7:] *= 1000. 	
        self.kf.P *= 10.

        # process uncertainty, make the constant velocity part more certain
        self.kf.Q[7:, 7:] *= 0.01 
        
        self.kf.Q = self.kf.Q * 1e-2
        self.kf.R = np.eye(self.kf._dim_z) * 10
        # self.kf.Q = self.kf.Q 
        # self.kf.R = np.eye(self.kf._dim_z)
        
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
        """ compute the innovation matrix for association with mahalanobis distance
        """
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