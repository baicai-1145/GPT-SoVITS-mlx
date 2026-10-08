"""T-major batch=1 AR KV state; active views must stay local to attention."""
import mlx.core as mx


class KVBuffer:
    __slots__ = ("buffer", "used")

    def __init__(self, values: mx.array, minimum_capacity: int = 256):
        b, t, d = values.shape
        capacity = max(minimum_capacity, 1 << (t - 1).bit_length())
        self.buffer = mx.contiguous(mx.full((b, capacity, d), 1.25, dtype=values.dtype))
        self.buffer[:, :t, :] = values
        self.used = t

    def append(self, values: mx.array):
        """Keep the full buffer as state; do not store or return a slice view."""
        dtype = mx.result_type(self.buffer.dtype, values.dtype)
        if dtype != self.buffer.dtype:
            self.buffer = self.buffer.astype(dtype)
        end = self.used + values.shape[1]
        if end > self.buffer.shape[1]:
            capacity = max(self.buffer.shape[1] * 2, 1 << (end - 1).bit_length())
            new = mx.contiguous(mx.full((self.buffer.shape[0], capacity, self.buffer.shape[2]),
                                       1.25, dtype=self.buffer.dtype))
            new[:, :self.used, :] = self.buffer[:, :self.used, :]
            self.buffer = new
        self.buffer[:, self.used:end, :] = values.astype(self.buffer.dtype)
        self.used = end

    def active(self) -> mx.array:
        """Return a step-local view; persist KVBuffer, never the returned array."""
        return self.buffer[:, :self.used, :]
