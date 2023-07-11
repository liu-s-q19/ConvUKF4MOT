import time
import jax
import haiku as hk
import pickle

from AB3DMOT_libs.modeling.voe import Voe_mlp as VoeMlp
from AB3DMOT_libs.utils_voe import Args
import jax.numpy as jnp

path = r'AB3DMOT_libs/logs/2023-07-08_15-52-53'

with open(path + r'/my_parm.pkl', 'rb') as f:
    params = pickle.load(f)