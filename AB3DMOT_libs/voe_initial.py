from __future__ import absolute_import, division

from copy import deepcopy
from math import log, exp, sqrt
import sys
import warnings
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z


import time
import jax
import haiku as hk
import pickle

from AB3DMOT_libs.modeling.voe import Voe_mlp as VoeMlp
from AB3DMOT_libs.utils_voe import Args
import jax.numpy as jnp

# path = r'AB3DMOT_libs/logs/2023-08-12_11-52-00'
# path = r'AB3DMOT_libs/logs/2023-07-08_15-52-53'
path = r'AB3DMOT_libs/logs/2023-10-08_20-30-14'
with open(path + r'/my_parm.pkl', 'rb') as f:
    params = pickle.load(f)
# _params = jax.device_put(params)
parm = jax.device_put(params)
config_path = path + '/config.json'
with open(config_path, 'r') as f1:
    args = Args.from_json(f1.read())
    
# args = args


class VoeFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))        # state
        self.P = eye(dim_x)               # uncertainty covariance
        self.Q = eye(dim_x)               # process uncertainty
        self.B = None                     # control transition matrix
        self.F = eye(dim_x)               # state transition matrix
        self.H = zeros((dim_z, dim_x))    # Measurement function
        self.R = eye(dim_z)               # state uncertainty
        self._alpha_sq = 1.               # fading memory control
        self.M = np.zeros((dim_z, dim_z)) # process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z)) # system uncertainty
        self.SI = np.zeros((dim_z, dim_z)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = np.linalg.inv

        self.state = self.x.copy()
    
        # self.path = r'AB3DMOT_libs/logs/2023-07-08_15-52-53'

        # with open(self.path + r'/my_parm.pkl', 'rb') as f:
        #     params = pickle.load(f)
        # # _params = jax.device_put(params)
        # self.parm = jax.device_put(params)
        # config_path = self.path + '/config.json'
        # with open(config_path, 'r') as f1:
        #     args = Args.from_json(f1.read())
        
        self.parm = parm 
        self.args = args


        def forward_fn(x):
            model = VoeMlp(args)
            return model(x)

        def rsample(mu: jax.Array, logvar: jax.Array) -> jax.Array:
            seed = int(time.time())
            n_key = jax.random.PRNGKey(seed)
            return mu + jnp.exp(0.5 * logvar) * jax.random.normal(key=n_key, shape=mu.shape)

        self.forward = hk.transform(forward_fn)
        self.rng_key = jax.random.PRNGKey(0)


    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        self.state = self.x.copy()
        
        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()


    def update(self, z, x_com, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # # x = x + Ky
        # # predict new x with residual scaled by the kalman gain
        # self.x = self.x + dot(self.K, self.y)
        state = x_com                   #shape(10,1)
        # state_ = self.x.copy().T
        state_ = jnp.array(np.dot(self.F, state))
        obs = z

        state_obs = jnp.concatenate([state.T, state_.T, obs.T], axis=-1)    #shape(1,10)
        
        # _state = jnp.array(np.dot(np.linalg.inv(self.F), state))
        # obs = z

        # state_obs = jnp.concatenate([_state.T, state.T, obs.T], axis=-1)    #shape(1,10)
        state_obs = state_obs.reshape(27,)
        q_w = self.forward.apply(params=self.parm, rng=self.rng_key, x=(state_obs))
        mu, logvar = jnp.split(q_w, 2, axis=-1)       #shape(1,10)\
        # mu, logvar = jnp.split(q_w[0:20], 2, axis=-1)
        # q_K = q_w[20:].reshape((10, 7))
        # print('voe_K:\n',q_K)
        mu = mu.reshape((10,1))
        state_for_test = state_ + mu          #shape(10,1)
        # print(state_for_test.shape)
        # print('afert update\n', state_for_test)
        # err = obs - self.H @ state_
        # state_for_test = state_ + q_K @ err
        self.x = np.array(state_for_test)
        
        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        # print("+++++++++\n",logvar.shape)
        # print('---------\n',np.diag(np.exp(logvar[0])))
        # print('P:\n',self.P)
        self.P = np.diag(np.exp(logvar)).copy()
        # print('P:\n',self.P)
        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_steadystate(self, u=0, B=None):
        """
        Predict state (prior) using the Kalman filter state propagation
        equations. Only x is updated, P is left unchanged. See
        update_steadstate() for a longer explanation of when to use this
        method.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        """

        if B is None:
            B = self.B

        # x = Fx + Bu
        if B is not None:
            self.x = dot(self.F, self.x) + dot(B, u)
        else:
            self.x = dot(self.F, self.x)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update_steadystate(self, z):
        """
        Add a new measurement (z) to the Kalman filter without recomputing
        the Kalman gain K, the state covariance P, or the system
        uncertainty S.

        You can use this for LTI systems since the Kalman gain and covariance
        converge to a fixed value. Precompute these and assign them explicitly,
        or run the Kalman filter using the normal predict()/update(0 cycle
        until they converge.

        The main advantage of this call is speed. We do significantly less
        computation, notably avoiding a costly matrix inversion.

        Use in conjunction with predict_steadystate(), otherwise P will grow
        without bound.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.


        Examples
        --------
        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> # let filter converge on representative data, then save k and P
        >>> for i in range(100):
        >>>     cv.predict()
        >>>     cv.update([i, i, i])
        >>> saved_k = np.copy(cv.K)
        >>> saved_P = np.copy(cv.P)

        later on:

        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> cv.K = np.copy(saved_K)
        >>> cv.P = np.copy(saved_P)
        >>> for i in range(100):
        >>>     cv.predict_steadystate()
        >>>     cv.update_steadystate([i, i, i])
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(self.H, self.x)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update_correlated(self, z, R=None, H=None):
        """ Add a new measurement (z) to the Kalman filter assuming that
        process noise and measurement noise are correlated as defined in
        the `self.M` matrix.

        If z is None, nothing is changed.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        if self.x.ndim == 1 and shape(z) == (1, 1):
            z = z[0]

        if shape(z) == (): # is it scalar, e.g. z=3 or z=np.array(3)
            z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + dot(H, self.M) + dot(self.M.T, H.T) + R
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT + self.M, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(H, self.P) + self.M.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def batch_filter(self, zs, Fs=None, Qs=None, Hs=None,
                     Rs=None, Bs=None, us=None, update_first=False,
                     saver=None):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self.dt`. Missing
            measurements must be represented by `None`.

        Fs : None, list-like, default=None
            optional value or list of values to use for the state transition
            matrix F.

            If Fs is None then self.F is used for all epochs.

            Otherwise it must contain a list-like list of F's, one for
            each epoch.  This allows you to have varying F per epoch.

        Qs : None, np.array or list-like, default=None
            optional value or list of values to use for the process error
            covariance Q.

            If Qs is None then self.Q is used for all epochs.

            Otherwise it must contain a list-like list of Q's, one for
            each epoch.  This allows you to have varying Q per epoch.

        Hs : None, np.array or list-like, default=None
            optional list of values to use for the measurement matrix H.

            If Hs is None then self.H is used for all epochs.

            If Hs contains a single matrix, then it is used as H for all
            epochs.

            Otherwise it must contain a list-like list of H's, one for
            each epoch.  This allows you to have varying H per epoch.

        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            Otherwise it must contain a list-like list of R's, one for
            each epoch.  This allows you to have varying R per epoch.

        Bs : None, np.array or list-like, default=None
            optional list of values to use for the control transition matrix B.

            If Bs is None then self.B is used for all epochs.

            Otherwise it must contain a list-like list of B's, one for
            each epoch.  This allows you to have varying B per epoch.

        us : None, np.array or list-like, default=None
            optional list of values to use for the control input vector;

            If us is None then None is used for all epochs (equivalent to 0,
            or no control input).

            Otherwise it must contain a list-like list of u's, one for
            each epoch.

       update_first : bool, optional, default=False
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

        Returns
        -------

        means : np.array((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        means_predictions : np.array((n,dim_x,1))
            array of the state for each time step after the predictions. Each
            entry is an np.array. In other words `means[k,:]` is the state at
            step `k`.

        covariance_predictions : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the prediction.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        Examples
        --------

        .. code-block:: Python

            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts. This requires
            # that F be recomputed for each epoch. The output is then smoothed
            # with an RTS smoother.

            zs = [t + random.randn()*4 for t in range (40)]
            Fs = [np.array([[1., dt], [0, 1]] for dt in dts]

            (mu, cov, _, _) = kf.batch_filter(zs, Fs=Fs)
            (xs, Ps, Ks) = kf.rts_smoother(mu, cov, Fs=Fs)
        """

        #pylint: disable=too-many-statements
        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, covariances, means_p, covariances_p)

    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None, inv=np.linalg.inv):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Fs : list-like collection of numpy.array, optional
            State transition matrix of the Kalman filter at each time step.
            Optional, if not provided the filter's self.F will be used

        Qs : list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        inv : function, default numpy.linalg.inv
            If you prefer another inverse function, such as the Moore-Penrose
            pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv


        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Pp : numpy.ndarray
           Predicted state covariances

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K, Pp) = rts_smoother(mu, cov, kf.F, kf.Q)

        """

        if len(Xs) != len(Ps):
            raise ValueError('length of Xs and Ps must be the same')

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n-2, -1, -1):
            Pp[k] = dot(dot(Fs[k+1], P[k]), Fs[k+1].T) + Qs[k+1]

            #pylint: disable=bad-whitespace
            K[k]  = dot(dot(P[k], Fs[k+1].T), inv(Pp[k]))
            x[k] += dot(K[k], x[k+1] - dot(Fs[k+1], x[k]))
            P[k] += dot(dot(K[k], P[k+1] - Pp[k]), K[k].T)

        return (x, P, K, Pp)

    def get_prediction(self, u=0):
        """
        Predicts the next state of the filter and returns it without
        altering the state of the filter.

        Parameters
        ----------

        u : np.array
            optional control input

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the prediction.
        """

        x = dot(self.F, self.x) + dot(self.B, u)
        P = self._alpha_sq * dot(dot(self.F, self.P), self.F.T) + self.Q
        return (x, P)

    def get_update(self, z=None):
        """
        Computes the new estimate based on measurement `z` and returns it
        without altering the state of the filter.

        Parameters
        ----------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the update.
       """

        if z is None:
            return self.x, self.P
        z = reshape_z(z, self.dim_z, self.x.ndim)

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        # error (residual) between measurement and prediction
        y = z - dot(H, x)

        # common subexpression for speed
        PHT = dot(P, H.T)

        # project system uncertainty into measurement space
        S = dot(H, PHT) + R

        # map system uncertainty into kalman gain
        K = dot(PHT, self.inv(S))

        # predict new x with residual scaled by the kalman gain
        x = x + dot(K, y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)

        return x, P

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return z - dot(self.H, self.x_prior)

    def measurement_of_state(self, x):
        """
        Helper function that converts a state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman state vector

        Returns
        -------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        """

        return dot(self.H, x)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq**.5

    def log_likelihood_of(self, z):
        """
        log likelihood of the measurement `z`. This should only be called
        after a call to update(). Calling after predict() will yield an
        incorrect result."""

        if z is None:
            return log(sys.float_info.min)
        return logpdf(z, dot(self.H, self.x), self.S)

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2

    def __repr__(self):
        return '\n'.join([
            'KalmanFilter object',
            pretty_str('dim_x', self.dim_x),
            pretty_str('dim_z', self.dim_z),
            pretty_str('dim_u', self.dim_u),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('x_post', self.x_post),
            pretty_str('P_post', self.P_post),
            pretty_str('F', self.F),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('H', self.H),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('S', self.S),
            pretty_str('SI', self.SI),
            pretty_str('M', self.M),
            pretty_str('B', self.B),
            pretty_str('z', self.z),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('likelihood', self.likelihood),
            pretty_str('mahalanobis', self.mahalanobis),
            pretty_str('alpha', self.alpha),
            pretty_str('inv', self.inv)
            ])

    def test_matrix_dimensions(self, z=None, H=None, R=None, F=None, Q=None):
        """
        Performs a series of asserts to check that the size of everything
        is what it should be. This can help you debug problems in your design.

        If you pass in H, R, F, Q those will be used instead of this object's
        value for those matrices.

        Testing `z` (the measurement) is problamatic. x is a vector, and can be
        implemented as either a 1D array or as a nx1 column vector. Thus Hx
        can be of different shapes. Then, if Hx is a single value, it can
        be either a 1D array or 2D vector. If either is true, z can reasonably
        be a scalar (either '3' or np.array('3') are scalars under this
        definition), a 1D, 1 element array, or a 2D, 1 element array. You are
        allowed to pass in any combination that works.
        """

        if H is None:
            H = self.H
        if R is None:
            R = self.R
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        x = self.x
        P = self.P

        assert x.ndim == 1 or x.ndim == 2, \
                "x must have one or two dimensions, but has {}".format(x.ndim)

        if x.ndim == 1:
            assert x.shape[0] == self.dim_x, \
                   "Shape of x must be ({},{}), but is {}".format(
                       self.dim_x, 1, x.shape)
        else:
            assert x.shape == (self.dim_x, 1), \
                   "Shape of x must be ({},{}), but is {}".format(
                       self.dim_x, 1, x.shape)

        assert P.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
                   self.dim_x, self.dim_x, P.shape)

        assert Q.shape == (self.dim_x, self.dim_x), \
               "Shape of P must be ({},{}), but is {}".format(
                   self.dim_x, self.dim_x, P.shape)

        assert F.shape == (self.dim_x, self.dim_x), \
               "Shape of F must be ({},{}), but is {}".format(
                   self.dim_x, self.dim_x, F.shape)

        assert np.ndim(H) == 2, \
               "Shape of H must be (dim_z, {}), but is {}".format(
                   P.shape[0], shape(H))

        assert H.shape[1] == P.shape[0], \
               "Shape of H must be (dim_z, {}), but is {}".format(
                   P.shape[0], H.shape)

        # shape of R must be the same as HPH'
        hph_shape = (H.shape[0], H.shape[0])
        r_shape = shape(R)

        if H.shape[0] == 1:
            # r can be scalar, 1D, or 2D in this case
            assert r_shape == () or r_shape == (1,) or r_shape == (1, 1), \
            "R must be scalar or one element array, but is shaped {}".format(
                r_shape)
        else:
            assert r_shape == hph_shape, \
            "shape of R should be {} but it is {}".format(hph_shape, r_shape)


        if z is not None:
            z_shape = shape(z)
        else:
            z_shape = (self.dim_z, 1)

        # H@x must have shape of z
        Hx = dot(H, x)

        if z_shape == (): # scalar or np.array(scalar)
            assert Hx.ndim == 1 or shape(Hx) == (1, 1), \
            "shape of z should be {}, not {} for the given H".format(
                shape(Hx), z_shape)

        elif shape(Hx) == (1,):
            assert z_shape[0] == 1, 'Shape of z must be {} for the given H'.format(shape(Hx))

        else:
            assert (z_shape == shape(Hx) or
                    (len(z_shape) == 1 and shape(Hx) == (z_shape[0], 1))), \
                    "shape of z should be {}, not {} for the given H".format(
                        shape(Hx), z_shape)

        if np.ndim(Hx) > 1 and shape(Hx) != (1, 1):
            assert shape(Hx) == z_shape, \
               'shape of z should be {} for the given H, but it is {}'.format(
                   shape(Hx), z_shape)


def update(x, P, z, R, H=None, return_all=False):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.

    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    update(1, 2, 1, 1, 1)  # univariate
    update(x, P, 1



    Parameters
    ----------

    x : numpy.array(dim_x, 1), or float
        State estimate vector

    P : numpy.array(dim_x, dim_x), or float
        Covariance matrix

    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

    R : numpy.array(dim_z, dim_z), or float
        Measurement noise matrix

    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.

    return_all : bool, default False
        If true, y, K, S, and log_likelihood are returned, otherwise
        only x and P are returned.

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    P : numpy.array
        Posterior covariance matrix

    y : numpy.array or scalar
        Residua. Difference between measurement and state in measurement space

    K : numpy.array
        Kalman gain

    S : numpy.array
        System uncertainty in measurement space

    log_likelihood : float
        log likelihood of the measurement
    """

    #pylint: disable=bare-except

    if z is None:
        if return_all:
            return x, P, None, None, None, None
        return x, P

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(dot(H, x))
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # project system uncertainty into measurement space
    S = dot(dot(H, P), H.T) + R


    # map system uncertainty into kalman gain
    try:
        K = dot(dot(P, H.T), linalg.inv(S))
    except:
        # can't invert a 1D array, annoyingly
        K = dot(dot(P, H.T), 1./S)


    # predict new x with residual scaled by the kalman gain
    x = x + dot(K, y)

    # P = (I-KH)P(I-KH)' + KRK'
    KH = dot(K, H)

    try:
        I_KH = np.eye(KH.shape[0]) - KH
    except:
        I_KH = np.array([1 - KH])
    P = dot(dot(I_KH, P), I_KH.T) + dot(dot(K, R), K.T)


    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(z, dot(H, x), S)
        return x, P, y, K, S, log_likelihood
    return x, P


def update_steadystate(x, z, K, H=None):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.


    Parameters
    ----------

    x : numpy.array(dim_x, 1), or float
        State estimate vector


    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

    K : numpy.array, or float
        Kalman gain matrix

    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    Examples
    --------

    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    >>> update_steadystate(1, 2, 1)  # univariate
    >>> update_steadystate(x, P, z, H)
    """


    if z is None:
        return x

    if H is None:
        H = np.array([1])

    if np.isscalar(H):
        H = np.array([H])

    Hx = np.atleast_1d(dot(H, x))
    z = reshape_z(z, Hx.shape[0], x.ndim)

    # error (residual) between measurement and prediction
    y = z - Hx

    # estimate new x with residual scaled by the kalman gain
    return x + dot(K, y)


def predict(x, P, F=1, Q=0, u=0, B=1, alpha=1.):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    Q : numpy.array, Optional
        Process noise matrix


    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, optional, default 0.
        Control transition matrix.

    alpha : float, Optional, default=1.0
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon

    Returns
    -------

    x : numpy.array
        Prior state estimate vector

    P : numpy.array
        Prior covariance matrix
    """

    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)
    P = (alpha * alpha) * dot(dot(F, P), F.T) + Q

    return x, P


def predict_steadystate(x, F=1, u=0, B=1):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations. This steady state form only computes x, assuming that the
    covariance is constant.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, optional, default 0.
        Control transition matrix.

    Returns
    -------

    x : numpy.array
        Prior state estimate vector
    """

    if np.isscalar(F):
        F = np.array(F)
    x = dot(F, x) + dot(B, u)

    return x



def batch_filter(x, P, zs, Fs, Qs, Hs, Rs, Bs=None, us=None,
                 update_first=False, saver=None):
    """
    Batch processes a sequences of measurements.

    Parameters
    ----------

    zs : list-like
        list of measurements at each time step. Missing measurements must be
        represented by None.

    Fs : list-like
        list of values to use for the state transition matrix matrix.

    Qs : list-like
        list of values to use for the process error
        covariance.

    Hs : list-like
        list of values to use for the measurement matrix.

    Rs : list-like
        list of values to use for the measurement error
        covariance.

    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.

    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.

    update_first : bool, optional
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

    Returns
    -------

    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.

    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.

    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.

    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.

    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]

        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)

    """

    n = np.size(zs, 0)
    dim_x = x.shape[0]

    # mean estimates from Kalman Filter
    if x.ndim == 1:
        means = zeros((n, dim_x))
        means_p = zeros((n, dim_x))
    else:
        means = zeros((n, dim_x, 1))
        means_p = zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances = zeros((n, dim_x, dim_x))
    covariances_p = zeros((n, dim_x, dim_x))

    if us is None:
        us = [0.] * n
        Bs = [0.] * n

    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            x, P = update(x, P, z, R=R, H=H)
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()

    return (means, covariances, means_p, covariances_p)



def rts_smoother(Xs, Ps, Fs, Qs):
    """
    Runs the Rauch-Tung-Striebal Kalman smoother on a set of
    means and covariances computed by a Kalman filter. The usual input
    would come from the output of `KalmanFilter.batch_filter()`.

    Parameters
    ----------

    Xs : numpy.array
       array of the means (state variable x) of the output of a Kalman
       filter.

    Ps : numpy.array
        array of the covariances of the output of a kalman filter.

    Fs : list-like collection of numpy.array
        State transition matrix of the Kalman filter at each time step.

    Qs : list-like collection of numpy.array, optional
        Process noise of the Kalman filter at each time step.

    Returns
    -------

    x : numpy.ndarray
       smoothed means

    P : numpy.ndarray
       smoothed state covariances

    K : numpy.ndarray
        smoother gain at each step

    pP : numpy.ndarray
       predicted state covariances

    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]

        (mu, cov, _, _) = kalman.batch_filter(zs)
        (x, P, K, pP) = rts_smoother(mu, cov, kf.F, kf.Q)
    """

    if len(Xs) != len(Ps):
        raise ValueError('length of Xs and Ps must be the same')

    n = Xs.shape[0]
    dim_x = Xs.shape[1]

    # smoother gain
    K = zeros((n, dim_x, dim_x))
    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n-2, -1, -1):
        pP[k] = dot(dot(Fs[k], P[k]), Fs[k].T) + Qs[k]

        #pylint: disable=bad-whitespace
        K[k]  = dot(dot(P[k], Fs[k].T), linalg.inv(pP[k]))
        x[k] += dot(K[k], x[k+1] - dot(Fs[k], x[k]))
        P[k] += dot(dot(K[k], P[k+1] - pP[k]), K[k].T)

    return (x, P, K, pP)
