from os import PathLike
from pathlib import Path
from typing import Dict, NamedTuple
import numpy as np

from AB3DMOT_libs.data.loader import DataLoader

class Obs(NamedTuple):
    data: np.ndarray

    def __getitem__(self, index) -> "Obs":
        return Obs(self.data[index])

    def __len__(self):
        return self.data.shape[0]

class State(NamedTuple):
    data: np.ndarray

    def __getitem__(self, index) -> "State":
        return State(self.data[index])

    def __len__(self):
        return self.data.shape[0]


class Batch(NamedTuple):
    obs: Obs
    state: State

    def __getitem__(self, index) -> "Batch":
        return Batch(self.obs[index], self.state[index])

    def __len__(self):
        assert len(self.obs) == len(self.state)
        return len(self.obs)


def create_dataloader(data_dir: PathLike, mode: str, batch_size: int = 256, shuffle: bool = False, seed: int = 0, to_screen: bool=False) -> DataLoader[Batch]:
    data = load_data(data_dir, mode)
    obs = data['obs']
    state = data['state']
    batch = Batch(Obs(obs), State(state))

    return DataLoader(batch, batch_size, shuffle=shuffle, seed=seed, to_screen=to_screen)

def load_data(data_dir: PathLike, mode: str) -> Dict[str, np.ndarray]:
    data_path = Path(data_dir) / f"{mode}.npz"
    data = np.load(data_path)
    return {
        "obs": data["observations"],
        "state": data["states"],
    }
