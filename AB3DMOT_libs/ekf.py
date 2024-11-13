import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, ExtendedKalmanFilter
# from AB3DMOT_libs.ukf import UnscentedKF
# from AB3DMOT_libs.ekf import ExtendedKF 
import jax
import jax.numpy as jnp
 
class Filter(object):
	def __init__(self, bbox3D, info, ID):

		self.initial_pos = bbox3D
		self.time_since_update = 0
		self.id = ID
		self.hits = 1           		# number of total hits including the first detection
		self.info = info        		# other information associated	

def fx(states, dt):
    F =np.array([[1,0,0,0,0,0,0,dt,0,0],      # state transition matrix, dim_x * dim_x
				[0,1,0,0,0,0,0,0,dt,0],
				[0,0,1,0,0,0,0,0,0,dt],
				[0,0,0,1,0,0,0,0,0,0],  
				[0,0,0,0,1,0,0,0,0,0],
				[0,0,0,0,0,1,0,0,0,0],
				[0,0,0,0,0,0,1,0,0,0],
				[0,0,0,0,0,0,0,1,0,0],
				[0,0,0,0,0,0,0,0,1,0],
				[0,0,0,0,0,0,0,0,0,1]]) 
    return np.array(F @ states) 

def hx(states):
    H =   np.array([[1,0,0,0,0,0,0,0,0,0],      
					[0,1,0,0,0,0,0,0,0,0],
					[0,0,1,0,0,0,0,0,0,0],
					[0,0,0,1,0,0,0,0,0,0],
					[0,0,0,0,1,0,0,0,0,0],
					[0,0,0,0,0,1,0,0,0,0],
					[0,0,0,0,0,0,1,0,0,0]])
    return np.array(H @ states)



class Ekf(Filter):
	def __init__(self, bbox3D, info, ID):
		super().__init__(bbox3D, info, ID)

		# point = MerweScaledSigmaPoints(10, alpha=0.1, beta=2.0, kappa=1.0)
		# point = MerweScaledSigmaPoints(10, alpha=1e-3, beta=2, kappa=0)   
		# self.kf = UnscentedKalmanFilter(dim_x=10, dim_z=7, dt=1., points=point, fx=fx, hx=hx) 
		self.kf = ExtendedKalmanFilter(dim_x=10, dim_z=7)
		# self.kf = KalmanFilter(dim_x=10, dim_z=7)
		# self.kf = ExtendedKF(dim_x=10, dim_z=7, fx=fx, hx=hx) 

	
		# self.kf = UnscentedKalmanFilter(dim_x=10, dim_z=7, dt=0.5, points=point, fx=fx, hx=hx) 
		# There is no need to use EKF here as the measurement and state are in the same space with linear relationship

		# state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
		# constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz 
		# while all others (theta, l, w, h, dx, dy, dz) remain the same
		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix, dim_x * dim_x
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

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

		# initialize data
		self.kf.x[:7] = self.initial_pos.reshape((7, 1))
  
		self.kf.hx = hx


	def compute_innovation_matrix(self):
		""" compute the innovation matrix for association with mahalanobis distance
		"""
		return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R

	def get_velocity(self):
		# return the object velocity in the state

		return self.kf.x[7:]