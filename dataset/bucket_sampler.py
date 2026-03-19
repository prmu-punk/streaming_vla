from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        indices: Sequence[int] | None = None,
        bucket_size_multiplier: int = 50,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if bucket_size_multiplier <= 0:
            raise ValueError(
                f"bucket_size_multiplier must be positive, got {bucket_size_multiplier}"
            )
        self.lengths = list(int(x) for x in lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.indices = (
            list(int(x) for x in indices) if indices is not None else list(range(len(self.lengths)))
        )
        self.bucket_size = int(batch_size * bucket_size_multiplier)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(self.indices)
        rng = random.Random(self.seed + self._epoch)
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            bucket_batches = [
                bucket[i : i + self.batch_size]
                for i in range(0, len(bucket), self.batch_size)
            ]
            if self.drop_last:
                bucket_batches = [b for b in bucket_batches if len(b) == self.batch_size]
            if self.shuffle:
                rng.shuffle(bucket_batches)
            batches.extend(bucket_batches)

        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)
