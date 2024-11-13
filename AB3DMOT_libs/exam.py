from AB3DMOT_libs.kalman_filter import KF
from AB3DMOT_libs.ukf import Ukf
from AB3DMOT_libs.voe_filter import Voe
import numpy as np
np.set_printoptions(suppress=True, precision=5)
x0 = np.array([3.023, 1.684, 13.189, 3.827, 1.61, 1.562, -1.574])
z0 = np.array([3.003, 1.682, 12.024, 3.744, 1.584, 1.507, -1.574])
x_com = np.array([3.015, 1.697, 12.011, 3.827, 1.61, 1.562, -1.574, 0., 0., 0.])
# print(x0.shape)
# print(z0.shape)
voe = Voe(x0, [1,1], 0)
kf = KF(x0, [1,1], 0)
print('before predict:\nkf:\n',kf.kf.x.reshape((-1)))
print('voe:\n',voe.kf.x.reshape((-1)))
voe.kf.predict()
kf.kf.predict()
print('after predict:\nkf:\n',kf.kf.x.reshape((-1)))
print('voe:\n',voe.kf.x.reshape((-1)))
voe.kf.x = x_com
kf.kf.x = x_com
# print('after predict:\nkf:\n',kf.kf.x.reshape((-1)))
# print('ukf:\n',ukf.kf.x.reshape((-1)))
# voe.kf.update(z0, x_com)
kf.kf.update(z0)
print('after update:\nkf:\n',kf.kf.x.reshape((-1)))
print('voe:\n',voe.kf.x.reshape((-1)))
print('kf_K:\n',kf.kf.K)
print('voe_K:\n',voe.kf.K)

# ukf = Ukf(x0, [1,1], 0)
# kf = KF(x0, [1,1], 0)
# print('before predict:\nkf:\n',kf.kf.x.reshape((-1)))
# print('ukf:\n',ukf.kf.x.reshape((-1)))
# ukf.kf.predict()
# kf.kf.predict()
# print('after predict:\nkf:\n',kf.kf.x.reshape((-1)))
# print('ukf:\n',ukf.kf.x.reshape((-1)))
# ukf.kf.x = x_com
# kf.kf.x = x_com
# # print('after predict:\nkf:\n',kf.kf.x.reshape((-1)))
# # print('ukf:\n',ukf.kf.x.reshape((-1)))
# ukf.kf.update(z0)
# kf.kf.update(z0)
# print('after update:\nkf:\n',kf.kf.x.reshape((-1)))
# print('ukf:\n',ukf.kf.x.reshape((-1)))
# print(kf.kf.K)

