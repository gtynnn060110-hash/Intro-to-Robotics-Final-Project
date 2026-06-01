import numpy as np


class Normalizer:
    """Running mean/variance normalizer for vector observations.

    Usage:
        norm = Normalizer(size)
        x = norm.normalize(x, update=True)
    """

    def __init__(self, size, eps=1e-8, clip_range=10.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float64)
        self.m2 = np.zeros(size, dtype=np.float64)  # sum of squares of differences

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self.size,)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def var(self):
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return self.m2 / (self.count - 1)

    def normalize(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.update(x)
        std = np.sqrt(self.var() + self.eps)
        out = (x - self.mean) / std
        return np.clip(out, -self.clip_range, self.clip_range).astype(np.float32)
