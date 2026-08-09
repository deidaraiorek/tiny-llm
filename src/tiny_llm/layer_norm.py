import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        new_x = mx.array(x, dtype=mx.float32)
        new_x = new_x / \
            (mx.sqrt(mx.mean(new_x ** 2, axis=-1, keepdims=True) + self.eps))
        x = mx.array(new_x, dtype=x.dtype)
        return x * self.weight
