import torch
import triton
import triton.language as tl
from utils import bench_sweep

# Notes:
# 1. This is a simple fill kernel creates an empty tensor and fills it with a given value.
# 2. Performance would be comparable to torch fill since this is an single element-wise operation constrained by bandwidth.


@triton.jit
def fill_kernel(x_ptr, value, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    tl.store(x_ptr + offsets, value, mask=mask)


def fill(x: torch.Tensor, value: float):
    assert x.ndim == 1

    BLOCK_SIZE = 1024
    n = x.shape[0]
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    fill_kernel[grid](x, value, n, BLOCK_SIZE)
    return x


if __name__ == "__main__":
    value = 0.5

    def torch_func(x, value):
        x.fill_(value)
        return x

    x_test = torch.empty(100, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(torch_func(x_test, value), fill(x_test, value))
    print("correct!")

    bench_sweep(
        {"torch.fill_": torch_func, "triton": fill},
        [2**i for i in range(7, 16)],
        lambda n: (torch.empty(n, device="cuda", dtype=torch.float32), value),
        xlabel="size",
        title="Compare: Triton vs. torch.fill_",
        filename="1_fill",
    )
