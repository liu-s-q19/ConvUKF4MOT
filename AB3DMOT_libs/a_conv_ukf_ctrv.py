import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints
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

class AdaptiveConvUKF(UnscentedKalmanFilter):
    
    def __init__(self, dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None):
        super().__init__(dim_x, dim_z, dt, hx, fx, points, sqrt_fn, x_mean_fn, z_mean_fn, residual_x, residual_z)
        
        self.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      
                    [0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,1,0,0,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0,0]])
        
        self.Cp = np.zeros((dim_z))
        self.varrho = 0
        self.alpha = 0.07
    
    def update(self, y, x=None, **hx_args):
        if x is not None:
            self.sigmas_f = self.points_fn.sigma_points(x, self.P)
        UT = unscented_transform
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, self.R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.PHT = Pxz
        
        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(y, zp)   # residual
        
        ### Adaptive
        # print('varrho:', self.varrho)
        
        self.varrho = (1-self.alpha)*self.varrho + self.alpha*self.varrho / (1 + np.exp(-2*self.varrho * (np.exp(-self.varrho) - np.mean(self.y))))
        self.R = (1-self.alpha)*self.R + self.alpha*self.R * self.varrho
        
        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))



class AConvUKF(Filter):
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)

        # point = MerweScaledSigmaPoints(10, alpha=0.1, beta=2.0, kappa=1.0)
        point = MerweScaledSigmaPoints(11, alpha=1e-3, beta=2, kappa=0)   
        # point =JulierSigmaPoints(10)  
        self.kf = AdaptiveConvUKF(dim_x=11, dim_z=7, dt=0.1, points=point, fx=fx1, hx=hx) 
        
        self.rho = 1e0
        self.varrho = 1e-3
        self.kf.varrho = self.varrho
        
        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
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
        self.kf.Q = self.kf.Q * 0.01
        self.kf.R = np.eye(self.kf._dim_z) * 10
        
        self.kf.Q = self.kf.Q * (self.rho) / (1 + self.rho)
        self.kf.R = self.kf.R * (self.varrho) / (1 + self.varrho)
        
        # initialize data
        self.kf.x[:7] = self.initial_pos.reshape((7,))
        

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