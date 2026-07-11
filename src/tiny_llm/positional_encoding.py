import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        pos = mx.arange(seq_len)
        theta = 1 / base ** (mx.arange(0, dims, 2) / dims)
        angles = mx.outer(pos, theta)
        self.sin_table = mx.sin(angles)
        self.cos_table = mx.cos(angles)
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape

        if offset is None:
            cos = self.cos_table[:L]
            sin = self.sin_table[:L]
        else:
            cos = self.cos_table[offset]
            sin = self.sin_table[offset]
        cos = cos.reshape(1, L, 1, D // 2)
        sin = sin.reshape(1, L, 1, D // 2)

        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)
            first = x[..., 0]
            second = x[..., 1]
            out_first = first * cos - second * sin
            out_second = first * sin + second * cos
            out = mx.stack([out_first, out_second], axis=-1)
            return out.reshape(N, L, H, D)
        else:
            first = x[..., : D // 2]
            second = x[..., D // 2 :]
            out_first = first * cos - second * sin
            out_second = first * sin + second * cos
            return mx.concatenate([out_first, out_second], axis=-1)
