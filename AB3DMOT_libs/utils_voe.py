from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Callable, Tuple, Dict
import jax
from pathlib import Path
from datetime import datetime

Dynamic = Callable[[jax.Array], jax.Array]
ObsModel = Callable[[jax.Array, jax.Array], jax.Array]
Metrics = Dict[str, float]

@dataclass_json
@dataclass
class Args:

    exp_name: str = None

    # Modeling
    hidden_size: int = 256
    rnn_depth: int = 2
    decoder_depth: int = 2
    rnn_activation: bool = True

    dim_state: int = 2
    dim_obs: int = 2

    # Training
    lr: float = 1e-3
    params_factor: float = 1e-2
    max_epoch: int = 100
    batch_size: int = 128
    beta: float = 1.0
    beta_kl: float = 1.0
    lr_schedule: bool = False
    seed: int = 42

    # Misc
    log_root: str = "logs"

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self.log_dir = f"{self.log_root}/{self.exp_name}"
    
    def create_logdir(self):
        path = Path(self.log_dir)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"Created log directory: {self.log_dir}")
    
    def save_config(self):
        path = Path(self.log_dir) / "config.json"
        with open(path, 'w') as f:
            f.write(self.to_json(indent=2))
            
    # def from_json(self, summary):
    #     for key in summary.keys():
    #         self.

def create_iter_key_fn(key: jax.random.KeyArray) -> Callable[[int], Tuple[jax.random.KeyArray, jax.random.KeyArray]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        train_key, test_key = jax.random.split(iter_key)
        return train_key, test_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn