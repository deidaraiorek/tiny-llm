import mlx.core as mx
from .basics import softmax, linear
import math 


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]
    factor = mx.rsqrt(d_k) if scale is None else scale
    scores = (query @ key.swapaxes(-1, -2)) * factor
    if mask is not None:
        scores = scores + mask
    return softmax(scores, -1) @ value

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int, 
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim = int(hidden_size / num_heads)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.dim).swapaxes(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.dim).swapaxes(-3,-2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.dim).swapaxes(-3, -2)
        attn = scaled_dot_product_attention_simple(q, k, v, None , mask)
        attn = attn.swapaxes(-3, -2).reshape(*query.shape[:-1], self.hidden_size)
        return linear(attn, self.wo)




def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    row_i = mx.arange(L).reshape(L, 1)
    col_j = mx.arange(S).reshape(1, S)
    allowed = col_j <= row_i + (S - L)
    return mx.where(allowed, mx.array(0.0), mx.array(-mx.inf)).astype(dtype)

def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    h_q = query.shape[-3]
    h = key.shape[-3]
    n_rep = int(h_q / h)
    query = query.reshape(*query.shape[:-3], h, n_rep, *query.shape[-2:])
    if mask == "causal":
        L = query.shape[-2]
        S = key.shape[-2]
        mask = causal_mask(L, S, query.dtype)
    elif isinstance(mask, mx.array):
        mask = mask.reshape(*mask.shape[:-3], h, n_rep, *mask.shape[-2:])
    key = mx.expand_dims(key, -3)
    value = mx.expand_dims(value, -3)
    attn = scaled_dot_product_attention_simple(query, key, value, scale, mask)
    return attn.reshape(*attn.shape[:-4], h_q, *attn.shape[-2:])
    



def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
