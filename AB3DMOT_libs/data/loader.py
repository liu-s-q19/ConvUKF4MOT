from typing import Generic, TypeVar

import numpy as np
import jax

Batch = TypeVar("Batch")


class DataLoader(Generic[Batch]):
    def __init__(self, batch: Batch, batch_size: int = 256, shuffle: bool = False, seed: int = 0, to_screen: bool = False):
        self.data_size = len(batch)
        self.batch_size = batch_size
        self.batches = self.data_size // self.batch_size
        self.discard = self.data_size - batch_size * self.batches
        self.batch = batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        if to_screen:
            print(
                f"Data size: {self.data_size}; Batch size: {batch_size}; Batches: {self.batches}; Discard: {self.discard}; Shuffle: {shuffle}"
            )

    def reset_epoch(self):
        if self.shuffle:
            permutation = self.rng.permutation(self.data_size)
            if self.discard != 0:
                permutation = permutation[: -self.discard]
            self.permutation = permutation.reshape(self.batches, self.batch_size)

    def __len__(self):
        return self.batches

    def __getitem__(self, idx: int) -> Batch:
        if not 0 <= idx < self.batches:
            raise IndexError
        if self.shuffle:
            batch = self.batch[self.permutation[idx]]
        else:
            batch = self.batch[idx * self.batch_size : (idx + 1) * self.batch_size]
        return jax.device_put(batch)
